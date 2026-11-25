# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""T018 measured geometry terms for a virtual try-on reward.

Every term is higher-is-better, so a defect can only lower the score. The
functions in the first half of this module are pure numpy/OpenCV operations on
images, masks and keypoints, so that the reward model, the offline evaluator and
the unit tests all compute the same numbers. The second half holds
:class:`GeometryScorer`, a thin holder for the three frozen models (pose, human
parser, patch features) plus the per-sample combination of the five terms.

Nothing here reads ground truth. The ground truth is used exactly once, by
``scripts/prepare_t018_geometry.py``, to measure the arm coverage of a garment.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

SCHEMA_VERSION = 6
TASK_ID = "T018"

# ---------------------------------------------------------------- term contract

TERM_NAMES: tuple[str, ...] = (
    "shoulder_proportion",
    "arm_coverage",
    "kept_area",
    "garment_patch",
    "lab_colour",
)
DEFAULT_WEIGHTS: dict[str, float] = {
    "shoulder_proportion": 1.0,
    "arm_coverage": 1.0,
    "kept_area": 1.0,
    "garment_patch": 1.0,
    "lab_colour": 0.5,
}
DEFAULT_BIAS = 0.0
# Penalty terms bottom out at -1, similarity terms at 0, so in every case the worst
# value is the lower bound of the range: one rule for the reward and the reports.
TERM_RANGES: dict[str, tuple[float, float]] = {
    "shoulder_proportion": (-1.0, 0.0),
    "arm_coverage": (-1.0, 0.0),
    "kept_area": (0.0, 1.0),
    "garment_patch": (0.0, 1.0),
    "lab_colour": (0.0, 1.0),
}
WORST_VALUE: dict[str, float] = {name: TERM_RANGES[name][0] for name in TERM_NAMES}

# ------------------------------------------------------------- frozen constants

# The changed-area mask is the pixel difference between the person photo and the
# agnostic photo that the dataset ships. The two files are separate JPEG encodings
# of the same frame, so untouched background differs by a level or two.
MASK_TOLERANCE = 10
MASK_CLOSE_KERNEL = 3
MASK_DILATE_KERNEL = 5
KEEP_DILATE_KERNEL = 5
MASK_AREA_RANGE = (0.20, 0.75)  # sanity band on the share of the frame that may change

# Structural similarity, computed on the whole frame and masked afterwards.
SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
# Every SSIM value is an 11 by 11 window average, so a kept pixel whose window still
# contains changed pixels reports the edit as a loss of structure. The changed-area mask
# is grown by less than this radius, so the kept mask is eroded by the full support
# radius before its SSIM is averaged.
SSIM_SUPPORT_RADIUS = SSIM_WINDOW // 2
# A mean CIE76 difference of this size is worth nothing, and the product form means a
# recoloured picture cannot buy its score back with structure. 25 is roughly "same
# object, different colour": a full hue rotation of a saturated garment lands far
# above it, so the hue-rotation ladder case reads 0.0 instead of the 0.70 ceiling the
# old additive 0.7 * SSIM + 0.3 * (1 - L1) could not go below.
KEEP_DELTA_E_MAX = 25.0
# CIE76 is 0 on an achromatic pixel however far its hue is rotated, so a mean over every
# kept pixel dilutes the defect the colour factor exists for: on the 160 graded pictures
# 87% of the kept mask is white studio background, and a full person hue rotation measured
# 0.869 against the spec's 0.35 ceiling. The colour reading is therefore taken over the
# kept pixels that carry colour (Lab chroma at or above this) as well, and the worse of the
# two readings scores -- the plain mean still catches a luminance drift on the background.
KEEP_CHROMATIC_MIN_CHROMA = 10.0

# A patch match on frozen DINOv2 tokens compares shape and print, and the final block is
# trained to ignore the very detail that separates a metal fabric from a flat grey one:
# panel-20 kept +0.698 with the sparkle smoothed away. Part of the term is spent on the
# high-frequency content of the aligned masked crops instead, at no model cost.
GARMENT_STRUCTURE_WEIGHT = 0.7
GARMENT_TEXTURE_WEIGHT = 0.3
TEXTURE_HIST_BINS = 16
TEXTURE_STD_WINDOW = 5
# Both texture channels are scaled onto the 0..255 axis the colour histograms already use,
# so one Wasserstein helper serves all three. The old scaling divided by the theoretical
# maximum of a binary checkerboard (4 * 255 * sqrt(2)), which put every real fabric
# gradient -- measured at 20 to 150 on the 160 graded pictures -- into bins 0 and 1 of a
# 16-bin histogram whose bin width is 15.9. Two terms then agreed at about 0.95 whatever
# the fabric, and only a scrambled texture moved them. Mapping gradients straight onto the
# axis fixed that but overshot in the other direction: a worn garment carries body-fold
# shadows and wrinkles a flat product shot cannot have, so full-scale Sobel turned the
# candidate's own drape into texture error and cost 0.009 AUROC. The current scales put the
# measured 20..150 gradient range across most of the axis and lift 20..40 levels of local
# variation off the first bin, without amplifying drape. Both axes are clipped at 255 by
# the histogram itself.
TEXTURE_GRADIENT_TO_255 = 1.0 / 5.66
TEXTURE_STD_TO_255 = 2.0
# Cosine similarities between DINOv2 tokens of two different garments -- a flat product
# shot and a worn garment -- sit in a narrow band well above zero, so averaging them raw
# compresses the whole term into 0.036 of its range. The band is rescaled onto [0, 1]
# before the position penalty, which is a monotone affine map: it cannot change an order,
# only the spread the weight fit has to work with.
GARMENT_SIMILARITY_FLOOR = 0.40
# Precision of what is visible on the person against coverage of what the flat shot shows.
# A tucked or partly occluded garment crops the worn mask, so whole flat patches have no
# worn counterpar, and the backward average then reads the occlusion as a defect: the
# ground-truth person photo scored 0.685 on flagged panel 3. The forward average asks only
# that what is visible is the right fabric, which is the question a candidate can answer.
GARMENT_FORWARD_WEIGHT = 0.65

# DINOv2 ViT-S/14 patch grid: 28 by 28 patches of 14 pixels. Both garment views are
# letterboxed onto one square canvas, so a patch centre normalised by the grid is the
# same coordinate in the flat garment and in the worn one.
PATCH_SIZE = 14
PATCH_GRID = (28, 28)  # (rows, columns) -> 392 by 392 pixels
GARMENT_CROP_MARGIN = 0.05
GARMENT_FILL = 128
# Below this share of the frame there is no garment to compare; the ladder's
# "garment erased" case reaches it long before the parser mask is empty.
GARMENT_MIN_AREA_SHARE = 0.01
# Block 6 of 12 keeps edges, prints and texture; the final block is trained towards
# invariance and drops exactly those. Neither alone ranks garment fidelity.
DINOV2_MID_BLOCK = 6
# Position tolerance of 0.25 of the canvas, applied as the spec's accepted
# alternative: subtract lambda * ||p_i - p_j||_2 from the cosine and floor at 0.
GARMENT_POSITION_PENALTY = 1.0
GARMENT_LAYER_WEIGHT = 0.7
GARMENT_QUADRANT_WEIGHT = 0.3
GARMENT_QUADRANT_GRID = 2
# The labels the ground-truth sleeve profiles were measured with, by
# ``prepare_t018_geometry.py``. The scarf rides along because it lies over the shoulder,
# and the cached profiles cannot be re-measured without re-running that script, so this
# tuple stays as it was even though the worn garment mask below no longer uses it.
PARSER_KEEP_LABELS: tuple[int, ...] = (3, 4, 10)  # top, dress, scarf
# Patch 2: the parser splits one top between ``upper_clothes`` and ``dress``, so a union
# of the two is the smallest set that keeps a top in one piece, and the scarf is not a
# garment. The union alone still swallows a skirt the parser calls a dress (panel-12), so
# it is intersected with the edited area.
WORN_MASK_GARMENT_LABELS: tuple[int, ...] = (3, 4)  # top, dress
# "Minus face, hair, skin/arms, hands and background" of the spec. On a single-label map
# this subtracts nothing -- a pixel carries one class -- and it is written out anyway so
# the mask stays correct if an overlaid or multi-label segmentation is ever fed in.
WORN_MASK_EXCLUDED_LABELS: tuple[int, ...] = (0, 1, 2, 12, 13)  # bg, face, hair, arms, hands
WORN_MASK_SKIRT_LABELS: tuple[int, ...] = (5, 6)  # skirt, pants: reported bleed, never kept
WORN_MASK_EDIT_DILATE = 5
PARSER_SKIN_LABEL = 12  # arms
PARSER_DROPPED_LABELS: tuple[int, ...] = (13, 17)  # hands, jewelry
PARSER_SIZE = (384, 576)  # (width, height) matching the parser's training input
# Encoder batch ceiling. A GRPO group is scored in one call, but the group can be as wide
# as the sampler makes it, and both the parser and DINOv2 allocate activations per image;
# chunking bounds the peak at eight pictures whatever the group size is.
BATCH_CHUNK = 8
PARSER_MEAN = (0.485, 0.456, 0.406)
PARSER_STD = (0.229, 0.224, 0.225)
PARSER_CLASSES: tuple[str, ...] = (
    "background",
    "face",
    "hair",
    "top",
    "dress",
    "skirt",
    "pants",
    "belt",
    "bag",
    "hat",
    "scarf",
    "glasses",
    "arms",
    "hands",
    "legs",
    "feet",
    "torso",
    "jewelry",
)

# Colour term: three one-dimensional Lab histogram distances.
HIST_BINS = 32
COLOUR_CHANNEL_WEIGHTS = (0.2, 0.4, 0.4)  # lightness, a, b
MIN_REGION_PIXELS = 200
# A 3 by 1 vertical split of the garment bbox catches the defect a global histogram
# cannot see: a top half and a bottom half swapped keep the histogram identical.
COLOUR_STRIP_COUNT = 3
COLOUR_GLOBAL_WEIGHT = 0.4
COLOUR_STRIP_WEIGHT = 0.6
COLOUR_MIN_STRIP_PIXELS = 20
# Similarity = 1 - D / COLOUR_ZERO_DISTANCE, clipped to [0, 1]: one number with an
# obvious meaning, "the distance at which the garment's colour scores nothing". The 160
# graded pictures put D in [0.0036, 0.0400] (median 0.0187, p95 0.0320), so 0.045 puts a
# perfect match at 1.0, the median candidate at 0.58 and the worst at 0.11. It replaces a
# shifted exponential whose slope and reference distance were both fitted inside each CV
# fold; on those 160 rows the two mappings rank identically (single-term AUROC 0.6022
# against 0.6024) and the linear one ties fewer pairs (296 of 495 within-input pairs
# correct against 291, 11 ties removed). The audit refits the one constant per fold as
# ``p95 / (1 - 1/3)``, the same rule the exponential's two constants came from.
COLOUR_ZERO_DISTANCE = 0.045

# Pose: COCO keypoint order produced by the container's pose detector.
KEYPOINT_CONFIDENCE = 0.3
COCO_KEYPOINT_ORDER: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# The frozen container has no scipy, so the ViTPose image processor (which calls
# scipy.linalg.inv) cannot be used. The crop is the whole photo, so its affine
# transform is a plain resize to the model input and the heatmaps decode back
# with the same scale.
POSE_MODEL_DIR_NAME = "vitpose-base-simple"
POSE_INPUT_SIZE = (192, 256)  # (width, height), the ViTPose checkpoint's size
POSE_IMAGE_MEAN = (0.485, 0.456, 0.406)
POSE_IMAGE_STD = (0.229, 0.224, 0.225)

SLEEVE_SAMPLES = 20
SLEEVE_T0 = 0.15
SLEEVE_T1 = 0.85
# Bin edges measured on 200 real wearing photos (see the sleeve_bin section of the
# manifest): forearm coverage is bimodal at 0 and 1, upper-arm coverage separates
# bare shoulders out, and the residue is three-quarter sleeves, which no bin
# describes. The plan's a-priori edges (0.8/0.8/0.3/0.3) sent 46% of garments to
# review; these edges send 10%.
SLEEVE_BIN_LONG_FOREARM = 0.6
SLEEVE_BIN_SHORT_FOREARM = 0.3
SLEEVE_BIN_SLEEVELESS_UPPER = 0.2
SLEEVE_ARM_DISAGREEMENT = 0.3
# A single line through two keypoints is brittle: a small keypoint error slides
# it off the arm. The fraction is averaged over five parallel lines instead.
SLEEVE_BAND_OFFSETS: tuple[float, ...] = (-6.0, -3.0, 0.0, 3.0, 6.0)
SLEEVE_BINS: tuple[str, ...] = ("long", "short", "sleeveless", "review")
# The two arm segments the coverage target is measured on, shoulder to elbow and
# elbow to wrist, in the order the details report them.
ARM_SEGMENTS: tuple[str, ...] = ("upper", "fore")

# The target result/person ratio. 1.0 means "the shoulders kept their proportion".
# The Phase 0 audit measures the median ratio on ground-truth wearing photos; if that
# median is more than 0.03 away from 1.0 the detector itself is biased and this
# constant is replaced by the measured median (or by its garment-bin median).
SHOULDER_TARGET_RATIO = 1.0


class GeometryTermError(RuntimeError):
    """A fail-closed geometry-term error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GeometryTermError(message)


# ------------------------------------------------------------------- primitives


def frame_shape(image: Any) -> tuple[int, int]:
    """``(height, width)`` of a picture, without decoding its pixels twice.

    A person photo and its edited result routinely have different resolutions -- the
    graded set is 1024 by 768 against 768 by 576 -- and keypoints live in the resolution
    they were detected at. Reading the frame size is the first half of putting the two
    sets of keypoints back into one frame.
    """
    size = getattr(image, "size", None)
    if isinstance(size, tuple) and len(size) == 2 and all(isinstance(v, int) for v in size):
        return (int(size[1]), int(size[0]))
    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            return (int(handle.size[1]), int(handle.size[0]))
    array = np.asarray(image)
    _require(array.ndim >= 2, "expected a picture to read a frame from")
    return (int(array.shape[0]), int(array.shape[1]))


def scale_points_to_frame(points: Mapping[str, Mapping[str, float]],
                          source: tuple[int, int],
                          target: tuple[int, int]) -> dict[str, dict[str, float]]:
    """Move keypoints from one frame's resolution onto another's, in place of a copy.

    The shoulder ratio divides one absolute width by another, so it is only meaningful
    when both are measured in the same pixels. Comparing a person pose detected at 1024 by
    768 against a result at 768 by 576 biases every ratio by three quarters and reads as a
    narrow-shouldered candidate rather than as a unit mismatch.
    """
    source_height, source_width = (int(source[0]), int(source[1]))
    target_height, target_width = (int(target[0]), int(target[1]))
    _require(source_height > 0 and source_width > 0, "source frame is empty")
    scale_y = target_height / float(source_height)
    scale_x = target_width / float(source_width)
    if scale_x == 1.0 and scale_y == 1.0:
        return {name: dict(point) for name, point in points.items()}
    scaled: dict[str, dict[str, float]] = {}
    for name, point in points.items():
        moved = dict(point)
        if "x" in moved:
            moved["x"] = float(moved["x"]) * scale_x
        if "y" in moved:
            moved["y"] = float(moved["y"]) * scale_y
        scaled[name] = moved
    return scaled


def to_rgb_array(image: Any, *, size: tuple[int, int] | None = None) -> np.ndarray:
    """Return a uint8 RGB numpy array from a PIL image, a path or an array."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            array = np.asarray(handle.convert("RGB"))
    elif isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.dtype != np.uint8:
            scale = 255.0 if float(np.nanmax(array, initial=0.0)) <= 1.0 else 1.0
            array = np.clip(array.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    else:  # PIL image
        array = np.asarray(image.convert("RGB"))
    _require(array.ndim == 3 and array.shape[2] == 3, "expected an RGB image")
    if size is not None and (array.shape[1], array.shape[0]) != size:
        array = cv2.resize(array, size, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(array)


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill enclosed holes of a uint8 binary mask (0/1) using a border flood fill."""
    height, width = binary.shape
    padded = np.zeros((height + 2, width + 2), np.uint8)
    padded[1:-1, 1:-1] = binary
    flood = padded.copy()
    flood_mask = np.zeros((height + 4, width + 4), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = (flood == 0).astype(np.uint8)[1:-1, 1:-1]
    return np.clip(binary | holes, 0, 1).astype(np.uint8)


def changed_area_mask(person: Any, agnostic: Any) -> np.ndarray:
    """Boolean changed-area mask from the person photo and the agnostic photo."""
    first = to_rgb_array(person).astype(np.int16)
    second = to_rgb_array(agnostic).astype(np.int16)
    _require(first.shape == second.shape, "person and agnostic photos differ in size")
    difference = np.abs(first - second).max(axis=2)
    binary = (difference > MASK_TOLERANCE).astype(np.uint8)
    binary = _fill_holes(binary)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((MASK_CLOSE_KERNEL, MASK_CLOSE_KERNEL), np.uint8)
    )
    binary = cv2.dilate(binary, np.ones((MASK_DILATE_KERNEL, MASK_DILATE_KERNEL), np.uint8))
    return binary.astype(bool)


def dilate_mask(mask: np.ndarray, kernel: int) -> np.ndarray:
    binary = np.ascontiguousarray(mask.astype(np.uint8))
    return cv2.dilate(binary, np.ones((kernel, kernel), np.uint8)).astype(bool)


def erode_mask(mask: np.ndarray, kernel: int) -> np.ndarray:
    binary = np.ascontiguousarray(mask.astype(np.uint8))
    return cv2.erode(binary, np.ones((kernel, kernel), np.uint8)).astype(bool)


def ssim_map(first: np.ndarray, second: np.ndarray, *, sigma: float = SSIM_SIGMA,
             window: int = SSIM_WINDOW) -> np.ndarray:
    """Closed-form structural similarity map on two RGB arrays, computed in grey."""
    _require(first.shape == second.shape, "SSIM operands differ in size")
    left = cv2.cvtColor(to_rgb_array(first), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    right = cv2.cvtColor(to_rgb_array(second), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mu_left = cv2.GaussianBlur(left, (window, window), sigma)
    mu_right = cv2.GaussianBlur(right, (window, window), sigma)
    left_sq = cv2.GaussianBlur(left * left, (window, window), sigma)
    right_sq = cv2.GaussianBlur(right * right, (window, window), sigma)
    cross = cv2.GaussianBlur(left * right, (window, window), sigma)
    var_left = left_sq - mu_left * mu_left
    var_right = right_sq - mu_right * mu_right
    covariance = cross - mu_left * mu_right
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_left * mu_right + c1) * (2 * covariance + c2)
    denominator = (mu_left * mu_left + mu_right * mu_right + c1) * (var_left + var_right + c2)
    return (numerator / denominator).astype(np.float32)


def centred_lab(lab8: np.ndarray) -> np.ndarray:
    """Turn OpenCV 8-bit Lab into the float units :func:`lab_array` returns.

    OpenCV stores L in 0..255 and a, b offset by +128, so both the scale of L and the
    offset of a and b have to be undone. Keep this in one place: the histogram terms bin
    the 8-bit array and the CIE76 difference needs the centred one, and a caller that
    derives the second from the first by hand gets the offset right and the scale wrong,
    which reads as a plausible difference of about 100 in every kept pixel.
    """
    out = np.asarray(lab8, dtype=np.float32).copy()
    out[:, :, 0] *= 100.0 / 255.0
    out[:, :, 1:] -= 128.0
    return out


def lab_array(image: Any) -> np.ndarray:
    """CIE Lab as float32 with the real units: L* in 0..100, a* and b* around 0."""
    return centred_lab(cv2.cvtColor(to_rgb_array(image), cv2.COLOR_RGB2LAB))


def delta_e_map(first: np.ndarray, second: np.ndarray, *, first_lab: np.ndarray | None = None,
                second_lab: np.ndarray | None = None) -> np.ndarray:
    """CIE76 colour difference per pixel, in Lab units.

    Three separate channel differences would let a large error on one channel hide
    behind two exact ones; the Euclidean distance in Lab is the standard way to say
    "how different do these two colours look". Precomputed Lab arrays can be passed in:
    the same person picture is compared against every candidate in its group, and
    converting it again per candidate was three redundant conversions of the same frame.
    """
    _require(first.shape == second.shape, "delta-E operands differ in size")
    left = lab_array(first) if first_lab is None else first_lab
    right = lab_array(second) if second_lab is None else second_lab
    difference = left - right
    return np.sqrt((difference * difference).sum(axis=2))


def kept_area_term(result: Any, person: Any, changed_mask: np.ndarray, *,
                   person_lab: np.ndarray | None = None,
                   person_chroma: np.ndarray | None = None,
                   result_lab: np.ndarray | None = None) -> dict[str, Any]:
    """Kept-area term, in [0, 1]: what the editor did not touch must not move.

    Structure and colour multiply instead of adding: an additive blend leaves half the
    range reachable by a picture whose colours are wrong but whose edges are intact,
    which is exactly what a hue-rotated person is, and the old 0.7 * SSIM + 0.3 *
    (1 - L1) could not go below 0.70 on that corruption. A product is 0 as soon as
    either factor is 0, so "kept the shapes, changed the colours" scores as the defect
    it is.

    The structure factor is averaged over the kept mask **eroded by the SSIM support
    radius**. The changed-area mask is grown by only ``MASK_DILATE_KERNEL`` pixels, which
    is narrower than the 11 by 11 window SSIM averages over, so without the erosion a ring
    of kept pixels is scored against windows that contain the edit and every candidate
    loses structure for having done the thing it was asked to do. Both means are reported,
    so the bleed stays visible in the offline table.
    """
    result_array = to_rgb_array(result)
    person_array = to_rgb_array(person, size=(result_array.shape[1], result_array.shape[0]))
    kept = ~dilate_mask(changed_mask, KEEP_DILATE_KERNEL)
    total = int(kept.sum())
    fallback = total < MIN_REGION_PIXELS
    if fallback:
        kept = ~changed_mask
        total = int(kept.sum())
    if total == 0:
        return {"value": WORST_VALUE["kept_area"], "flag": "kept_area_empty", "kept_pixels": 0,
                "ssim_mean": None, "ssim_mean_plain": None, "delta_e_mean": None,
                "delta_e_chromatic_mean": None, "delta_e_effective": None,
                "chromatic_min_chroma": KEEP_CHROMATIC_MIN_CHROMA, "chromatic_pixels": 0,
                "colour_fidelity": None}
    structure = ssim_map(result_array, person_array)
    difference = delta_e_map(result_array, person_array, first_lab=result_lab,
                             second_lab=person_lab)
    ssim_region = erode_mask(kept, 2 * SSIM_SUPPORT_RADIUS + 1)
    ssim_fallback = int(ssim_region.sum()) < MIN_REGION_PIXELS
    if ssim_fallback:
        ssim_region = kept
    ssim_mean = float(np.clip(structure[ssim_region].mean(), 0.0, 1.0))
    ssim_mean_plain = float(np.clip(structure[kept].mean(), 0.0, 1.0))
    person_lab = lab_array(person_array) if person_lab is None else person_lab
    chroma = (np.sqrt(person_lab[:, :, 1] ** 2 + person_lab[:, :, 2] ** 2)
              if person_chroma is None else person_chroma)
    # The chromatic reading is taken over the same eroded region as the structure one. It
    # used to be taken over the raw ``kept`` mask, whose inner edge is a ring of pixels the
    # changed-area dilation only partly covered: on the 87%-white studio photos the few
    # hundred chromatic pixels there are skin, hair and resampling fringe right beside the
    # edit, so a handful of boundary pixels decided the term and its own AUROC fell to
    # 0.46. The hue-rotation ladder case still reads 0.0, because a rotated person changes
    # the colour of every kept pixel the chroma threshold admits.
    chromatic = ssim_region & (chroma >= KEEP_CHROMATIC_MIN_CHROMA)
    chromatic_pixels = int(chromatic.sum())
    delta_e_mean = float(difference[kept].mean())
    delta_e_chromatic = (float(difference[chromatic].mean())
                         if chromatic_pixels >= MIN_REGION_PIXELS else None)
    delta_e_effective = delta_e_mean if delta_e_chromatic is None \
        else max(delta_e_mean, delta_e_chromatic)
    fidelity = max(0.0, 1.0 - delta_e_effective / KEEP_DELTA_E_MAX)
    low, high = TERM_RANGES["kept_area"]
    return {
        "value": float(min(max(ssim_mean * fidelity, low), high)),
        "flag": ("kept_ssim_plain_region" if ssim_fallback
                 else ("kept_area_plain_region" if fallback else "")),
        "kept_pixels": total,
        "ssim_pixels": int(ssim_region.sum()),
        "ssim_mean": ssim_mean,
        "ssim_mean_plain": ssim_mean_plain,
        "ssim_support_radius": SSIM_SUPPORT_RADIUS,
        "delta_e_mean": delta_e_mean,
        "delta_e_chromatic_mean": delta_e_chromatic,
        "delta_e_effective": delta_e_effective,
        "chromatic_min_chroma": KEEP_CHROMATIC_MIN_CHROMA,
        "chromatic_pixels": chromatic_pixels,
        "colour_fidelity": fidelity,
    }


# ------------------------------------------------------------------------- pose


def keypoint_dict(points: Sequence[Sequence[float]]) -> dict[str, dict[str, float]]:
    """Convert COCO-ordered (x, y, score) rows into a named mapping."""
    _require(len(points) == len(COCO_KEYPOINT_ORDER), "expected 17 COCO keypoints")
    named: dict[str, dict[str, float]] = {}
    for name, row in zip(COCO_KEYPOINT_ORDER, points, strict=True):
        x, y, score = (float(row[0]), float(row[1]), float(row[2]))
        named[name] = {"x": x, "y": y, "score": score, "visible": score >= KEYPOINT_CONFIDENCE}
    return named


def _quadratic_offset(heat: np.ndarray, row: int, column: int, axis: int) -> float:
    """Sub-pixel offset of a heatmap peak from a three-point parabola."""
    limit = heat.shape[axis]
    index = column if axis == 1 else row
    if index <= 0 or index >= limit - 1:
        return 0.0
    if axis == 1:
        values = (heat[row, column - 1], heat[row, column], heat[row, column + 1])
    else:
        values = (heat[row - 1, column], heat[row, column], heat[row + 1, column])
    before, peak, after = (float(value) for value in values)
    curvature = before - 2.0 * peak + after
    if abs(curvature) < 1e-9:
        return 0.0
    return float(np.clip(0.5 * (before - after) / curvature, -1.0, 1.0))


def _visible(points: Mapping[str, Mapping[str, float]], name: str) -> tuple[float, float] | None:
    row = points.get(name)
    if not row or not row.get("visible"):
        return None
    return float(row["x"]), float(row["y"])


def shoulder_width(points: Mapping[str, Mapping[str, float]]) -> float | None:
    """Pixel distance between the two shoulder keypoints, or None if one is missing."""
    left = _visible(points, "left_shoulder")
    right = _visible(points, "right_shoulder")
    if left is None or right is None:
        return None
    return math.dist(left, right)


def shoulder_proportion_term(result_points: Mapping[str, Mapping[str, float]],
                             person_points: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """Proportion change of shoulder width, in [-1, 0].

    The ratio is the result's shoulder width over the person's, both measured by the same
    detector on frames registered to each other, so zoom, crop and translation cancel.
    This ratio was chosen after Phase 0 measured four rulers on the 160 graded pictures:
    the person-anchored width won on single-term AUROC (0.778 against 0.722 for the best
    independent one), on within-input pair accuracy, and on the clean panels the
    independent rulers wrongly penalised, because a result hip keypoint that jumps under a
    generated hem appears in neither of its operands.

    The penalty is ``-min(1, |log(ratio / target)| / log 2)``: a ratio equal to the target
    scores 0 and a doubling or a halving scores -1. Working in log scale is what makes
    widening and narrowing cost the same, which a linear ``|ratio - 1|`` cannot do.

    A missing shoulder on the person photo says nothing about what the edit did -- the
    detector simply did not see it -- so the term abstains at 0.0. A missing shoulder on
    the result is a defect of the result and scores the worst value.
    """
    result_width = shoulder_width(result_points)
    person_width = shoulder_width(person_points)
    measured = {"ruler": "shoulder_over_shoulder", "result_ratio": result_width,
                "person_ratio": person_width}
    if person_width is None or person_width <= 1e-6:
        return {"value": 0.0, "flag": "shoulder_person_unmeasurable",
                "relative": None, "saturated": False, **measured}
    if result_width is None or result_width <= 0.0:
        return {"value": WORST_VALUE["shoulder_proportion"], "flag": "shoulder_unmeasurable",
                "relative": None, "saturated": True, **measured}
    relative = float(result_width / person_width)
    value = shoulder_penalty(relative)
    return {"value": value, "flag": "", "relative": relative,
            "saturated": value <= -1.0, **measured}


def shoulder_penalty(relative: float) -> float:
    """Log-scale shoulder penalty, in [-1, 0], for one result/person ratio."""
    if not math.isfinite(relative) or relative <= 0.0:
        return float(WORST_VALUE["shoulder_proportion"])
    error = abs(math.log(relative / SHOULDER_TARGET_RATIO)) / math.log(2.0)
    return float(-min(1.0, error))


def sample_segment(start: tuple[float, float], end: tuple[float, float], *,
                   samples: int = SLEEVE_SAMPLES, t0: float = SLEEVE_T0,
                   t1: float = SLEEVE_T1,
                   offsets: Sequence[float] = SLEEVE_BAND_OFFSETS) -> list[tuple[float, float]]:
    """Points along a segment, repeated on parallel lines either side of it."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    normal_x, normal_y = -dy / length, dx / length
    fractions = np.linspace(t0, t1, samples)
    points: list[tuple[float, float]] = []
    for offset in offsets:
        for t in fractions:
            points.append((start[0] + dx * float(t) + normal_x * offset,
                           start[1] + dy * float(t) + normal_y * offset))
    return points


def label_fractions(label_map: np.ndarray, points: Iterable[tuple[float, float]]) -> dict[str, int]:
    """Count garment, skin and dropped labels under a list of points."""
    height, width = label_map.shape[:2]
    counts = {"garment": 0, "skin": 0, "dropped": 0, "other": 0}
    for x, y in points:
        column = int(round(x))
        row = int(round(y))
        if not (0 <= row < height and 0 <= column < width):
            counts["other"] += 1
            continue
        label = int(label_map[row, column])
        if label in PARSER_KEEP_LABELS:
            counts["garment"] += 1
        elif label == PARSER_SKIN_LABEL:
            counts["skin"] += 1
        elif label in PARSER_DROPPED_LABELS:
            counts["dropped"] += 1
        else:
            counts["other"] += 1
    return counts


def sleeve_fraction(label_map: np.ndarray, elbow: tuple[float, float],
                    wrist: tuple[float, float]) -> dict[str, Any]:
    """Share of one arm segment covered by garment, skin, dropped labels or other."""
    counts = label_fractions(label_map, sample_segment(elbow, wrist))
    counted = counts["garment"] + counts["skin"]
    if counted == 0:
        return {"garment_fraction": None, "skin_fraction": None, "counted": 0, **counts}
    return {
        "garment_fraction": counts["garment"] / counted,
        "skin_fraction": counts["skin"] / counted,
        "counted": counted,
        **counts,
    }


def sleeve_bin(c_upper: float | None, c_fore: float | None) -> str:
    """Three-way sleeve length bin from the ground-truth measurement.

    Both arms wear the same sleeve, so the mean of the two measurements is the
    estimate and an arm disagreement no longer forces a hand check; the caller
    still records ``agree`` in the sleeve record for the report.
    """
    if c_upper is None or c_fore is None:
        return "review"
    if c_fore >= SLEEVE_BIN_LONG_FOREARM:
        return "long"
    if c_upper <= SLEEVE_BIN_SLEEVELESS_UPPER:
        return "sleeveless"
    if c_fore <= SLEEVE_BIN_SHORT_FOREARM:
        return "short"
    return "review"


def _arm_value(penalty: float) -> float:
    range_size = TERM_RANGES["arm_coverage"][1] - TERM_RANGES["arm_coverage"][0]
    return float(TERM_RANGES["arm_coverage"][1] - range_size * min(max(penalty, 0.0), 1.0))


def arm_coverage_term(result_points: Mapping[str, Mapping[str, float]], label_map: np.ndarray,
                      bin_label: str, person_points: Mapping[str, Mapping[str, float]],
                      expected: Mapping[str, float | None] | None = None) -> dict[str, Any]:
    """Arm coverage defect, in [-1, 0], scored against the garment's own sleeve.

    For each arm and each segment ``s`` in {upper arm, forearm} the error is two-sided,
    ``e = |measured - target|``, so a garment that grew sleeves it never had costs as
    much as one that lost them; the discrete sleeve bin cannot express that, because
    it names a length instead of measuring one. Per arm the two segments are averaged,
    and the arms are combined as ``0.5 * mean + 0.5 * max``: an arm that was destroyed
    outright must not be diluted by half by the arm that is still fine.

    ``expected`` carries the garment's own coverage, measured on the wearing photo by
    ``ground_truth_sleeve``: ``{"upper": c_upper, "fore": c_fore}``. The bin is kept
    for reporting only, on the ``bin``/``bin_value`` keys of the details; it never
    gates the score. Without a coverage profile in the cache the term cannot say
    anything about this garment and returns 0.0 with a flag.

    Patch 4: a ``review`` garment abstains. That bin is the classifier saying the wearing
    photo's own sleeve could not be placed -- the measurement is missing, or the forearm
    coverage landed between the short and long thresholds -- so the profile this term
    scores against is the very measurement that failed. Scoring it anyway damped a real
    sleeve defect five to eight times over (the same corruption reads -0.79 on a ``short``
    garment and -0.12 on a ``review`` one) and charged every clean ``review`` candidate a
    constant it had not earned. The term returns 0.0 with a flag instead: 0.0 is the
    perfect-match end of its range, so it contributes nothing to :func:`combine` and the
    reward simply does not vote on a sleeve it cannot define.

    Patch 3: an arm the person had and the result lost is a defect, not missing data. The
    old code dropped an unmeasurable arm and averaged whatever was left, so panel-11 --
    two arms on the person, one shoulder-to-wrist chain gone on the result, an obvious
    sleeve artifact -- scored -0.050. Each doubly-missing arm now contributes a full-arm
    error of 1.0 to the same ``0.5 * mean + 0.5 * max`` aggregation.
    """
    _require(bin_label in SLEEVE_BINS, f"unknown sleeve bin: {bin_label}")
    targets = {
        segment: None if expected is None else _expected_coverage(expected, segment)
        for segment in ARM_SEGMENTS
    }
    # Patch 3: per side, not per count. A three-quarter pose leaves one arm visible on
    # the person photo; the old rule only penalised a lost arm when the person had two, so
    # a candidate that destroyed the one visible arm scored 0.0 -- a perfect score for the
    # worst outcome. Each side is now measured against itself.
    person_sides = {
        side: (_visible(person_points, f"{side}_elbow") is not None
               and _visible(person_points, f"{side}_wrist") is not None)
        for side in ("left", "right")
    }
    person_arms = sum(person_sides.values())
    per_arm: list[dict[str, Any]] = []
    unmeasurable: list[str] = []
    for side in ("left", "right"):
        shoulder = _visible(result_points, f"{side}_shoulder")
        elbow = _visible(result_points, f"{side}_elbow")
        wrist = _visible(result_points, f"{side}_wrist")
        if elbow is None and wrist is None:
            unmeasurable.append(side)
            continue
        forearm = sleeve_fraction(label_map, elbow, wrist) if elbow and wrist else {
            "garment_fraction": None, "skin_fraction": None}
        upper = sleeve_fraction(label_map, shoulder, elbow) if shoulder and elbow else {
            "garment_fraction": None, "skin_fraction": None}
        measured = {"upper": upper["garment_fraction"], "fore": forearm["garment_fraction"]}
        if not any(value is not None for value in measured.values()):
            unmeasurable.append(side)
            continue
        per_arm.append({
            "side": side,
            **forearm,
            "upper_garment_fraction": upper["garment_fraction"],
            "upper_skin_fraction": upper["skin_fraction"],
            "measured": measured,
        })
    # Only an arm the person actually had can have been destroyed by the edit; a person
    # photo with one visible arm leaves the missing side unexplained rather than penalised.
    missing_arms = [side for side in unmeasurable if person_sides.get(side)]
    detail: dict[str, Any] = {"valid_arms_result": len(per_arm),
                              "valid_arms_person": person_arms,
                              "person_sides": person_sides,
                              "missing_arms": missing_arms,
                              "bin": bin_label, "per_arm": per_arm}
    if not per_arm:
        # Broken pose inside the shirt is a defect; a person photo with no visible
        # arms means nothing could have been destroyed.
        value = WORST_VALUE["arm_coverage"] if missing_arms else 0.0
        return {"value": float(value), "flag": "arm_coverage_missing", "mode": "missing",
                **detail}

    measured_mean = {
        segment: (float(np.mean([arm["measured"][segment] for arm in per_arm
                                 if arm["measured"][segment] is not None]))
                  if any(arm["measured"][segment] is not None for arm in per_arm) else None)
        for segment in ARM_SEGMENTS
    }
    segment_errors = {
        segment: (None if targets[segment] is None or measured_mean[segment] is None
                  else abs(measured_mean[segment] - float(targets[segment])))
        for segment in ARM_SEGMENTS
    }
    # Per-arm error uses only the segments that could be measured on that arm and
    # that the garment profile covers.
    arm_errors: dict[str, float] = {}
    for arm in per_arm:
        values = [abs(float(arm["measured"][segment]) - float(targets[segment]))
                  for segment in ARM_SEGMENTS
                  if targets[segment] is not None and arm["measured"][segment] is not None]
        if values:
            arm_errors[str(arm["side"])] = float(np.mean(values))
    if any(target is not None for target in targets.values()):
        # A full-arm error only means something against a garment whose own coverage is
        # known; without that profile the term returns 0.0 with a flag as before.
        for side in missing_arms:
            arm_errors[side] = 1.0
    bin_penalty = _bin_penalty(per_arm, bin_label)
    bin_value = 0.0 if bin_label == "review" else _arm_value(bin_penalty)
    detail.update({"measured": measured_mean, "expected": targets,
                   "segment_errors": segment_errors, "arm_errors": arm_errors,
                   "bin_penalty": bin_penalty, "bin_value": bin_value})

    if not arm_errors:
        detail.update({"mode": "no_profile", "penalty": 0.0})
        return {"value": 0.0, "flag": "arm_coverage_no_profile", **detail}

    errors = list(arm_errors.values())
    penalty = 0.5 * float(np.mean(errors)) + 0.5 * float(np.max(errors))
    detail.update({"mode": "coverage_profile", "penalty": penalty})
    if bin_label == "review":
        # Patch 4: no signal where the sleeve itself is undefined. The measured penalty
        # still rides along in the details, so a report can show what the term would
        # have said without letting it vote.
        detail.update({"mode": "abstain_bin_review", "withheld_value": _arm_value(penalty)})
        return {"value": 0.0, "flag": "arm_coverage_bin_review", **detail}
    return {"value": _arm_value(penalty), "flag": "", **detail}


def _expected_coverage(expected: Mapping[str, float | None], segment: str) -> float | None:
    """Read one segment of the garment profile, accepting either key spelling.

    ``ground_truth_sleeve`` returns ``c_upper`` and ``c_fore``; the geometry cache
    and the evaluator carry ``upper`` and ``fore``.
    """
    for key in (segment, f"c_{segment}"):
        value = expected.get(key)
        if value is not None:
            return float(value)
    return None


def _bin_penalty(per_arm: Sequence[Mapping[str, Any]], bin_label: str) -> float:
    """The retired rule, kept for the report: what must not be on the forearm.

    Nothing scores from this any more; it rides along so the offline table can still
    show what the old bin rule would have said about the same picture.
    """
    key = "skin_fraction" if bin_label == "long" else "garment_fraction"
    values = [float(arm[key]) for arm in per_arm if arm.get(key) is not None]
    return float(np.mean(values)) if values else 0.0


def ground_truth_sleeve(label_map: np.ndarray,
                        points: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """Upper-arm and forearm garment fractions on a real wearing photo."""
    upper: list[float] = []
    fore: list[float] = []
    side_fractions: dict[str, Any] = {}
    for side in ("left", "right"):
        shoulder = _visible(points, f"{side}_shoulder")
        elbow = _visible(points, f"{side}_elbow")
        wrist = _visible(points, f"{side}_wrist")
        if elbow is None or wrist is None:
            continue
        fore_arm = label_fractions(label_map, sample_segment(elbow, wrist))
        fore_counted = fore_arm["garment"] + fore_arm["skin"]
        fore_value = fore_arm["garment"] / fore_counted if fore_counted else None
        upper_value = None
        if shoulder is not None:
            upper_arm = label_fractions(label_map, sample_segment(shoulder, elbow))
            upper_counted = upper_arm["garment"] + upper_arm["skin"]
            upper_value = upper_arm["garment"] / upper_counted if upper_counted else None
        side_fractions[side] = {"upper": upper_value, "fore": fore_value}
        if upper_value is not None:
            upper.append(upper_value)
        if fore_value is not None:
            fore.append(fore_value)
    c_upper = float(np.mean(upper)) if upper else None
    c_fore = float(np.mean(fore)) if fore else None
    agree = True
    if len(fore) == 2:
        agree = abs(fore[0] - fore[1]) <= SLEEVE_ARM_DISAGREEMENT
    if len(upper) == 2:
        agree = agree and abs(upper[0] - upper[1]) <= SLEEVE_ARM_DISAGREEMENT
    return {
        "c_upper": c_upper,
        "c_fore": c_fore,
        "arms_measured": len(fore),
        "agree": agree,
        "per_side": side_fractions,
        "bin": sleeve_bin(c_upper, c_fore),
    }


# ----------------------------------------------------------------------- patches


def patch_grid_mask(mask: np.ndarray, *, grid: tuple[int, int] = PATCH_GRID,
                    threshold: float = 0.5, erode_patches: int = 1) -> np.ndarray:
    """Resample a pixel mask onto the patch grid and erode it by whole patches."""
    rows, columns = grid
    resized = cv2.resize(np.asarray(mask, dtype=np.uint8) * 255, (columns, rows),
                         interpolation=cv2.INTER_AREA)
    covered = (resized.astype(np.float32) / 255.0) >= threshold
    if erode_patches > 0:
        eroded = erode_mask(covered, 2 * erode_patches + 1)
        if eroded.any():
            return eroded
    return covered


@dataclass(frozen=True)
class GarmentTexture:
    """High-frequency summary of one masked garment canvas.

    ``gradient`` holds the Sobel magnitude of the luminance and ``local_std`` the local
    luminance standard deviation, both sampled only at pixels whose whole neighbourhood
    lies inside the garment mask (the mid-grey fill would otherwise contribute edges of
    its own). They are compared through their histograms, which is what makes the
    comparison blind to where a pixel sits and sensitive only to how much fine detail
    the garment carries: a smoothed metallic fabric keeps its silhouette and loses its
    sparkle, and only this half of the term sees that.
    """

    gradient: np.ndarray
    local_std: np.ndarray


def garment_texture(view: Mapping[str, Any]) -> GarmentTexture | None:
    """Sobel-magnitude and local luminance std of a letterboxed garment canvas."""
    mask = np.asarray(view["mask"], dtype=bool)
    interior = erode_mask(mask, TEXTURE_STD_WINDOW)
    if int(interior.sum()) < MIN_REGION_PIXELS:
        return None
    luminance = cv2.cvtColor(np.asarray(view["rgb"], np.uint8),
                             cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y) * TEXTURE_GRADIENT_TO_255
    window = (TEXTURE_STD_WINDOW, TEXTURE_STD_WINDOW)
    mean = cv2.blur(luminance, window)
    mean_square = cv2.blur(luminance * luminance, window)
    local_std = np.sqrt(np.clip(mean_square - mean * mean, 0.0, None)) * TEXTURE_STD_TO_255
    return GarmentTexture(gradient=magnitude[interior], local_std=local_std[interior])


def garment_texture_similarity(first: GarmentTexture, second: GarmentTexture) -> dict[str, float]:
    """Gradient-magnitude and local-std histogram agreement, each in [0, 1].

    ``histogram_wasserstein`` already reads a distribution over the 0..255 axis, so the
    two channels reuse it instead of multiplying the number of histogram helpers.
    """
    gradient = 1.0 - histogram_wasserstein(first.gradient, second.gradient, TEXTURE_HIST_BINS)
    local_std = 1.0 - histogram_wasserstein(first.local_std, second.local_std, TEXTURE_HIST_BINS)
    return {"gradient": float(np.clip(gradient, 0.0, 1.0)),
            "local_std": float(np.clip(local_std, 0.0, 1.0))}


@dataclass(frozen=True)
class GarmentPatchSet:
    """One garment view: patch tokens of two DINOv2 blocks, centres and texture.

    ``centres`` holds the normalised centre of every kept patch in [0, 1]^2 on the
    letterboxed canvas, so a flat patch and a worn patch are only allowed to match
    when they sit at the same place on the garment. ``texture`` is the same canvas seen
    as high-frequency content, which the two frozen blocks do not rank.
    """

    mid: np.ndarray
    final: np.ndarray
    centres: np.ndarray
    texture: GarmentTexture | None = None


def _region_bbox(region: np.ndarray, margin: float) -> tuple[int, int, int, int] | None:
    """Tight box around a region, grown by ``margin`` of its longer side."""
    rows, columns = np.nonzero(region)
    if rows.size == 0:
        return None
    height, width = region.shape[:2]
    pad = int(round(margin * max(rows.max() - rows.min() + 1, columns.max() - columns.min() + 1)))
    return (max(0, int(columns.min()) - pad), max(0, int(rows.min()) - pad),
            min(width, int(columns.max()) + 1 + pad), min(height, int(rows.max()) + 1 + pad))


def letterbox_view(image: Any, region: np.ndarray, *, grid: tuple[int, int] = PATCH_GRID,
                   margins: float = GARMENT_CROP_MARGIN,
                   fill: int = GARMENT_FILL) -> dict[str, Any] | None:
    """Crop one garment view to its own bbox, mask it grey and pad it to a square.

    A plain resize to the patch grid stretched the flat garment to the worn garment's
    proportions, so a print that was simply narrowed by perspective scored as a
    different print. Letterbox padding keeps the garment's aspect ratio, and filling
    the non-garment pixels with mid-grey stops the background of the flat product
    shot from matching the trousers of the wearing photo.
    """
    region = np.asarray(region, dtype=bool)
    height, width = region.shape[:2]
    if int(region.sum()) < GARMENT_MIN_AREA_SHARE * height * width:
        return None
    box = _region_bbox(region, margins)
    if box is None:
        return None
    left, top, right, bottom = box
    crop = to_rgb_array(image)[top:bottom, left:right].copy()
    mask = region[top:bottom, left:right]
    crop[~mask] = fill
    side = min(grid) * PATCH_SIZE
    scale = min(side / crop.shape[0], side / crop.shape[1])
    size = (max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale))))
    canvas = np.full((side, side, 3), fill, np.uint8)
    canvas_mask = np.zeros((side, side), bool)
    offset = ((side - size[0]) // 2, (side - size[1]) // 2)
    canvas[offset[1]:offset[1] + size[1], offset[0]:offset[0] + size[0]] = cv2.resize(
        crop, size, interpolation=cv2.INTER_AREA)
    canvas_mask[offset[1]:offset[1] + size[1], offset[0]:offset[0] + size[0]] = cv2.resize(
        mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    return {"rgb": np.ascontiguousarray(canvas), "mask": canvas_mask, "bbox": box,
            "grid": grid}


def worn_garment_mask(image: Any, labels: np.ndarray,
                      changed_mask: np.ndarray) -> dict[str, Any]:
    """The worn garment: parser top-or-dress, intersected with the edited area.

    The parser is not a garment segmenter on tops. It splits one top between
    ``upper_clothes`` and ``dress``, which cut the mask in two at the waist
    (panel-09/19/20), and on panel-12 it labelled an entire floral skirt ``dress``, so the
    old top+dress+scarf mask swallowed the skirt as garment. The union of the two garment
    classes closes the waist split, and intersecting it with the dilated changed area
    keeps the parts the editor actually replaced, which drops both the skirt and any
    trouser the parser called a dress. The scarf is no longer a garment class.

    ``counts`` records the two constructions side by side -- patch counts on the same
    letterboxed grid the term matches on, plus the pixel counts that explain them -- so
    the Phase 0 audit can quote both numbers for the same picture.
    """
    labels = np.asarray(labels)
    changed = np.asarray(changed_mask, dtype=bool)
    _require(labels.shape[:2] == changed.shape[:2], "labels and changed mask differ in size")
    garment = np.isin(labels, WORN_MASK_GARMENT_LABELS)
    scarf = labels == PARSER_KEEP_LABELS[2]
    # What the old ``np.isin(labels, PARSER_KEEP_LABELS)`` mask kept, kept for comparison.
    parser_only = garment | scarf
    edited = dilate_mask(changed, WORN_MASK_EDIT_DILATE)
    mask = garment & edited & ~np.isin(labels, WORN_MASK_EXCLUDED_LABELS)

    def patch_count(region: np.ndarray) -> int:
        view = letterbox_view(image, region)
        return 0 if view is None else int(patch_grid_mask(view["mask"]).sum())

    counts = {
        "parser_patches": patch_count(parser_only),
        "mask_patches": patch_count(mask),
        "parser_pixels": int(parser_only.sum()),
        "mask_pixels": int(mask.sum()),
        "scarf_pixels_excluded": int((scarf & ~mask).sum()),
        "edit_clipped_pixels": int((garment & ~edited).sum()),
        "skirt_pixels_excluded": int((np.isin(labels, WORN_MASK_SKIRT_LABELS) & ~mask).sum()),
    }
    return {"mask": mask, "parser_mask": parser_only, "counts": counts}


def patch_centres(grid_mask: np.ndarray, *, grid: tuple[int, int] = PATCH_GRID) -> np.ndarray:
    """Normalised centre of every selected patch cell, in [0, 1]^2."""
    rows, columns = np.nonzero(np.asarray(grid_mask, dtype=bool).reshape(grid))
    return np.stack([(columns + 0.5) / grid[1], (rows + 0.5) / grid[0]], axis=1).astype(np.float32)


def _unit(features: np.ndarray) -> np.ndarray:
    """L2-normalise feature rows so a cosine is a plain dot product."""
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)


def spatial_cosine_match(first: np.ndarray, first_centres: np.ndarray,
                         second: np.ndarray, second_centres: np.ndarray) -> float:
    """Two-direction mean of position-limited maximum cosine similarity, in [0, 1].

    Unconstrained max-pooling let any flat patch match its best worn patch anywhere, so
    a collar patch could answer a cuff patch and a scrambled garment still scored
    well. Subtracting ``lambda * ||p_i - p_j||_2`` before the max (the spec's accepted
    alternative to a hard 0.25 box) forces every match to be local, and flooring at 0
    keeps the term's "higher is better" direction and its bound.
    """
    _require(first.ndim == 2 and second.ndim == 2, "patch features must be matrices")
    _require(first.shape[1] == second.shape[1], "patch features differ in width")
    separation = np.linalg.norm(first_centres[:, None, :] - second_centres[None, :, :], axis=2)
    similarity = _unit(first) @ _unit(second).T
    forward = np.clip(similarity - GARMENT_POSITION_PENALTY * separation, 0.0, None)
    backward = np.clip(similarity.T - GARMENT_POSITION_PENALTY * separation.T, 0.0, None)
    return (GARMENT_FORWARD_WEIGHT * float(forward.max(axis=1).mean())
            + (1.0 - GARMENT_FORWARD_WEIGHT) * float(backward.max(axis=1).mean()))


def block_similarity(first: GarmentPatchSet, second: GarmentPatchSet) -> float:
    """Mean of the spatially constrained match on the mid and on the final block."""
    mid = spatial_cosine_match(first.mid, first.centres, second.mid, second.centres)
    final = spatial_cosine_match(first.final, first.centres, second.final, second.centres)
    return 0.5 * mid + 0.5 * final


def quadrant_similarity(first: GarmentPatchSet, second: GarmentPatchSet) -> float | None:
    """Cosine between the mean features of each 2 by 2 quadrant of the two views.

    Per-patch matching says nothing about layout: a garment sewn back together with its
    quadrants swapped keeps every local patch. Comparing quadrant means is the cheapest
    place-sensitive summary the same forward pass can pay for.
    """
    left = 0.5 * (_unit(first.mid) + _unit(first.final))
    right = 0.5 * (_unit(second.mid) + _unit(second.final))
    scores = []
    step = 1.0 / GARMENT_QUADRANT_GRID
    for row in range(GARMENT_QUADRANT_GRID):
        for column in range(GARMENT_QUADRANT_GRID):
            low = np.asarray([column * step, row * step], dtype=np.float32)
            high = low + step
            left_members = np.all((first.centres >= low) & (first.centres < high), axis=1)
            right_members = np.all((second.centres >= low) & (second.centres < high), axis=1)
            if not left_members.any() or not right_members.any():
                continue
            left_mean = _unit(left[left_members].mean(axis=0, keepdims=True))[0]
            right_mean = _unit(right[right_members].mean(axis=0, keepdims=True))[0]
            scores.append(float(np.clip(left_mean @ right_mean, 0.0, 1.0)))
    return float(np.mean(scores)) if scores else None


def scaled_similarity(value: float, *, floor: float = GARMENT_SIMILARITY_FLOOR) -> float:
    """Map a cosine band ``[floor, 1]`` onto ``[0, 1]`` without reordering anything."""
    if not math.isfinite(value):
        return 0.0
    span = max(1.0 - float(floor), 1e-6)
    return float(min(1.0, max(0.0, (float(value) - float(floor)) / span)))


def garment_patch_term(worn: GarmentPatchSet | None, flat: GarmentPatchSet | None, *,
                      worn_mask_counts: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Garment patch-match term, in [0, 1], higher meaning the garment survived.

    Two halves. The structural half is the spatially constrained match on an intermediate
    and the final DINOv2 block plus the 2 by 2 quadrant means, which ranks shape, print
    and layout. The texture half is the gradient and local-std agreement of the aligned
    masked crops, which ranks the detail the frozen blocks ignore: panel-13 kept generic
    stripes and lost the cuff trim, panel-20 kept the silhouette and smoothed the metallic
    sparkle, and both escaped above 0.69 on structure alone. The two halves are combined
    as a weighted mean with the module constants rather than a product: smoothing the
    fabric is no reason to stop paying for the silhouette that survived, and the reverse
    holds just as well, so neither half should be able to erase the other.
    """
    counts = dict(worn_mask_counts or {})
    if worn is None or flat is None:
        return {"value": WORST_VALUE["garment_patch"], "flag": "garment_region_empty",
                "worn_patches": 0, "flat_patches": 0, "block_similarity": None,
                "quadrant_similarity": None, "structure_similarity": None,
                "texture_similarity": None, "gradient_similarity": None,
                "local_std_similarity": None, **counts}
    blocks = block_similarity(worn, flat)
    quadrants = quadrant_similarity(worn, flat)
    structure = blocks if quadrants is None else \
        GARMENT_LAYER_WEIGHT * blocks + GARMENT_QUADRANT_WEIGHT * quadrants
    # Cosines between DINOv2 tokens of a flat shot and a worn garment live in a narrow
    # band above ``GARMENT_SIMILARITY_FLOOR``. Rescaling that band onto [0, 1] is a
    # monotone affine map, so no order changes; what changes is the spread the weight fit
    # sees, which was std 0.036 against 0.11-0.17 for every other term, and the fit paid for
    # that narrowness by handing this one 74% of the weight.
    structure = scaled_similarity(structure)
    texture: dict[str, float] | None = None
    if worn.texture is not None and flat.texture is not None:
        channels = garment_texture_similarity(worn.texture, flat.texture)
        texture = {**channels, "value": 0.5 * (channels["gradient"] + channels["local_std"])}
    if texture is None:
        # The texture half needs the masked crops. A view built without them can only
        # report structure, and says so instead of passing off half a term as the whole.
        value = structure
        flag = "garment_texture_unmeasurable"
    else:
        value = GARMENT_STRUCTURE_WEIGHT * structure + GARMENT_TEXTURE_WEIGHT * texture["value"]
        flag = ""
    low, high = TERM_RANGES["garment_patch"]
    return {
        "value": float(min(max(value, low), high)),
        "flag": flag,
        "worn_patches": int(worn.final.shape[0]),
        "flat_patches": int(flat.final.shape[0]),
        "block_similarity": float(blocks),
        "quadrant_similarity": None if quadrants is None else float(quadrants),
        "structure_similarity": float(structure),
        "texture_similarity": None if texture is None else float(texture["value"]),
        "gradient_similarity": None if texture is None else float(texture["gradient"]),
        "local_std_similarity": None if texture is None else float(texture["local_std"]),
        **counts,
    }


# ------------------------------------------------------------------------ colour


def _channel_histogram(values: np.ndarray, bins: int) -> np.ndarray:
    histogram, _ = np.histogram(values, bins=bins, range=(0, 255))
    total = float(histogram.sum())
    _require(total > 0, "empty histogram")
    return histogram.astype(np.float64) / total


def histogram_wasserstein(first: np.ndarray, second: np.ndarray, bins: int = HIST_BINS) -> float:
    """One-dimensional Wasserstein distance between two histograms, in [0, 1]."""
    left = np.cumsum(_channel_histogram(first, bins))
    right = np.cumsum(_channel_histogram(second, bins))
    width = 255.0 / bins
    return float(np.abs(left - right).sum() * width / 255.0)


def colour_distance(flat_lab: np.ndarray, flat_region: np.ndarray, worn_lab: np.ndarray,
                    worn_region: np.ndarray) -> tuple[float, list[float]]:
    """Weighted (0.2 L, 0.4 a, 0.4 b) Wasserstein distance and its three channels."""
    distances = [
        histogram_wasserstein(flat_lab[:, :, channel][flat_region],
                              worn_lab[:, :, channel][worn_region])
        for channel in range(3)
    ]
    total = float(sum(weight * value for weight, value in zip(COLOUR_CHANNEL_WEIGHTS, distances)))
    return total, distances


def vertical_strips(region: np.ndarray) -> list[np.ndarray]:
    """Split a region's bounding box into equal horizontal thirds, top to bottom."""
    box = _region_bbox(np.asarray(region, dtype=bool), 0.0)
    if box is None:
        return []
    top, bottom = box[1], box[3]
    edges = np.linspace(top, bottom, COLOUR_STRIP_COUNT + 1).round().astype(int)
    strips = []
    for index in range(COLOUR_STRIP_COUNT):
        band = np.zeros_like(np.asarray(region, dtype=bool))
        band[edges[index]:edges[index + 1], box[0]:box[2]] = True
        strips.append(band & np.asarray(region, dtype=bool))
    return strips


def colour_similarity(distance: float, *, zero_distance: float | None = None) -> float:
    """The linear ramp that maps a Lab distance onto [0, 1].

    Kept apart from :func:`lab_colour_term` because the audit refits the one constant
    inside each training fold: the term scores with the frozen distance, the report
    re-scores the same stored raw distances with the fold's own percentile.
    """
    zero = COLOUR_ZERO_DISTANCE if zero_distance is None else zero_distance
    if zero <= 0.0:
        return float(WORST_VALUE["lab_colour"])
    return float(min(1.0, max(0.0, 1.0 - float(distance) / float(zero))))


def lab_colour_term(flat_rgb: np.ndarray, flat_region: np.ndarray, worn_rgb: np.ndarray,
                    worn_region: np.ndarray, *, zero_distance: float | None = None,
                    flat_lab: np.ndarray | None = None,
                    worn_lab: np.ndarray | None = None) -> dict[str, Any]:
    """Lab-histogram colour term, in [0, 1], on the whole garment and on vertical strips.

    A global histogram is blind to any rearrangement of the same pixels, so a garment
    whose top and bottom halves were swapped scores a perfect 0. Splitting the garment
    box into three horizontal thirds and weighting the strip distance at 0.6 catches
    that, while the global term keeps the plain "wrong colour" case.

    The distance maps to a similarity through ``clip(1 - D / zero_distance, 0, 1)``: one
    constant with a readable meaning, the distance at which the colour scores nothing. The
    frozen constant and the raw distances ride along in the details, so the Phase 0 run can
    report the percentile it calibrated it from.
    """
    flat_region = np.asarray(flat_region, dtype=bool)
    worn_region = np.asarray(worn_region, dtype=bool)
    if flat_region.sum() < MIN_REGION_PIXELS or worn_region.sum() < MIN_REGION_PIXELS:
        return {"value": WORST_VALUE["lab_colour"], "flag": "colour_region_empty",
                "flat_pixels": int(flat_region.sum()), "worn_pixels": int(worn_region.sum()),
                "channel_distances": None, "distance_global": None,
                "distance_strips": None, "distance": None, "strips_used": 0}
    flat_lab = (cv2.cvtColor(to_rgb_array(flat_rgb), cv2.COLOR_RGB2LAB)
                if flat_lab is None else flat_lab)
    worn_lab = (cv2.cvtColor(to_rgb_array(worn_rgb), cv2.COLOR_RGB2LAB)
                if worn_lab is None else worn_lab)
    global_distance, channel_distances = colour_distance(flat_lab, flat_region,
                                                        worn_lab, worn_region)
    strip_distances = []
    for flat_strip, worn_strip in zip(vertical_strips(flat_region), vertical_strips(worn_region)):
        if min(int(flat_strip.sum()), int(worn_strip.sum())) < COLOUR_MIN_STRIP_PIXELS:
            continue
        strip_distances.append(colour_distance(flat_lab, flat_strip, worn_lab, worn_strip)[0])
    strip_distance = (float(np.mean(strip_distances)) if strip_distances else global_distance)
    distance = COLOUR_GLOBAL_WEIGHT * global_distance + COLOUR_STRIP_WEIGHT * strip_distance
    zero = COLOUR_ZERO_DISTANCE if zero_distance is None else zero_distance
    value = colour_similarity(distance, zero_distance=zero)
    low, high = TERM_RANGES["lab_colour"]
    return {
        "value": float(min(max(value, low), high)),
        "flag": "",
        "channel_distances": channel_distances,
        "distance_global": global_distance,
        "distance_strips": strip_distance,
        "distance": distance,
        "zero_distance": zero,
        "strips_used": len(strip_distances),
        "flat_pixels": int(flat_region.sum()),
        "worn_pixels": int(worn_region.sum()),
    }


# ----------------------------------------------------------------------- combine


def validate_weights(weights: Mapping[str, float], bias: float = DEFAULT_BIAS) -> dict[str, float]:
    _require(set(weights) == set(TERM_NAMES), f"weights must cover exactly {TERM_NAMES}")
    result: dict[str, float] = {}
    for name in TERM_NAMES:
        value = float(weights[name])
        _require(math.isfinite(value), f"weight {name} is not finite")
        _require(value >= 0.0, f"weight {name} must be non-negative")
        result[name] = value
    _require(math.isfinite(float(bias)), "bias is not finite")
    return result


def combine(terms: Mapping[str, float], weights: Mapping[str, float],
            bias: float = DEFAULT_BIAS) -> float:
    """R = b + sum_i w_i f_i, with non-negative weights."""
    checked = validate_weights(weights, bias)
    missing = [name for name in TERM_NAMES if name not in terms]
    _require(not missing, f"missing term values: {missing}")
    total = float(bias)
    for name in TERM_NAMES:
        value = float(terms[name])
        _require(math.isfinite(value), f"term {name} is not finite")
        total += checked[name] * value
    return float(total)


def term_contract() -> dict[str, Any]:
    """Definitions, constants and the default weights, hashed for the contract file."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK_ID,
        "terms": list(TERM_NAMES),
        "ranges": {name: list(TERM_RANGES[name]) for name in TERM_NAMES},
        "worst_values": dict(WORST_VALUE),
        "default_weights": dict(DEFAULT_WEIGHTS),
        "default_bias": DEFAULT_BIAS,
        "combination": "R = b + sum_i w_i f_i with w_i >= 0",
        "changed_area_mask": {
            "rule": "abs(person - agnostic).max(axis=-1) > tolerance, holes filled, closed, dilated",
            "tolerance": MASK_TOLERANCE,
            "close_kernel": MASK_CLOSE_KERNEL,
            "dilate_kernel": MASK_DILATE_KERNEL,
            "expected_area_share": list(MASK_AREA_RANGE),
        },
        "shoulder_proportion": {
            "ruler": "shoulder width on the result / shoulder width on the person, both from "
                      "the same detector on frames registered to each other",
            "target_ratio": SHOULDER_TARGET_RATIO,
            "penalty": "-min(1, |log(ratio / target)| / log(2))",
            "missing_value": "-1 when the result's shoulders are missing, 0 when the person "
                              "photo's are",
        },
        "arm_coverage": {
            "samples_per_arm": SLEEVE_SAMPLES,
            "band_offsets_px": list(SLEEVE_BAND_OFFSETS),
            "t_range": [SLEEVE_T0, SLEEVE_T1],
            "garment_labels": list(PARSER_KEEP_LABELS),
            "skin_label": PARSER_SKIN_LABEL,
            "dropped_labels": list(PARSER_DROPPED_LABELS),
            "bins": list(SLEEVE_BINS),
            "bin_thresholds": {
                "long_forearm": SLEEVE_BIN_LONG_FOREARM,
                "short_forearm": SLEEVE_BIN_SHORT_FOREARM,
                "sleeveless_upper": SLEEVE_BIN_SLEEVELESS_UPPER,
                "arm_disagreement": SLEEVE_ARM_DISAGREEMENT,
            },
            "bins_are_scoring": False,
            "abstain_bins": ["review"],
            "abstain_value": "0.0 with the flag arm_coverage_bin_review: the wearing photo's "
                             "sleeve could not be classified, so the profile this term scores "
                             "against is the measurement that failed",
            "segments": ["shoulder_to_elbow", "elbow_to_wrist"],
            "segments_compared": list(ARM_SEGMENTS),
            "error": "two-sided: |measured garment coverage - the garment's own coverage|",
            "combination": "-min(1, 0.5 * mean(e_L, e_R) + 0.5 * max(e_L, e_R))",
            "missing_arm_rule": "each side visible on the person photo and unmeasurable on the "
                                "result contributes e_a = 1.0 to the same aggregation, whether the "
                                "person showed one arm or two",
            "missing_value": "worst when the person photo had a visible arm on a lost side, 0 otherwise",
            "no_profile_value": "0 with a flag: the cache carries no arm coverage for this garment",
        },
        "kept_area": {
            "ssim_window": SSIM_WINDOW,
            "ssim_sigma": SSIM_SIGMA,
            "ssim_support_radius": SSIM_SUPPORT_RADIUS,
            "keep_mask": f"complement of the changed-area mask dilated by {KEEP_DILATE_KERNEL}",
            "ssim_mask": "the kept mask eroded by the SSIM support radius, so no kept pixel is "
                          "scored against a window that contains changed pixels; both means "
                          "are reported (ssim_mean, ssim_mean_plain)",
            "delta_e_max": KEEP_DELTA_E_MAX,
            "delta_e_chromatic_min_chroma": KEEP_CHROMATIC_MIN_CHROMA,
            "delta_e_aggregation": "max(mean_K(delta-E), mean_Keroded & chroma>=min_chroma(delta-E)); "
                                   "the chromatic reading is skipped when fewer than "
                                   f"{MIN_REGION_PIXELS} kept pixels carry colour, and "
                                   "both readings are reported (delta_e_mean, "
                                   "delta_e_chromatic_mean)",
            "formula": "mean_Keroded(ssim_map) * max(0, 1 - delta_e_effective / delta_e_max)",
        },
        "garment_patch": {
            "patch_size": PATCH_SIZE,
            "patch_grid": list(PATCH_GRID),
            "canvas": [PATCH_GRID[1] * PATCH_SIZE, PATCH_GRID[0] * PATCH_SIZE],
            "preprocess": "tight bbox plus 5% margin, non-garment pixels mid-grey 128, "
                          "aspect-preserving letterbox, INTER_NEAREST for masks",
            "min_area_share": GARMENT_MIN_AREA_SHARE,
            "worn_region": "(parser top | parser dress) & dilate(changed area, "
                           f"{WORN_MASK_EDIT_DILATE} px), minus background, face, hair, "
                           "arms and hands",
            "worn_mask_counts": "parser_patches vs mask_patches on the same letterboxed grid, "
                                "plus the pixel counts that explain them",
            "flat_region": "cloth-mask",
            "matching": f"{GARMENT_FORWARD_WEIGHT} * mean_worn max_flat + "
                        f"{1.0 - GARMENT_FORWARD_WEIGHT:.2f} * mean_flat max_worn, each max "
                        "taken over clip(cos - lambda_pos * ||p_i - p_j||_2, 0, None)",
            "similarity_floor": GARMENT_SIMILARITY_FLOOR,
            "structure_rescale": "(structure - floor) / (1 - floor), a monotone affine map ",
            "position_penalty": GARMENT_POSITION_PENALTY,
            "blocks": {"mid": DINOV2_MID_BLOCK, "final": "last"},
            "structure": f"{GARMENT_LAYER_WEIGHT} * mean(mid, final) + "
                         f"{GARMENT_QUADRANT_WEIGHT} * 2x2 quadrant mean-feature cosine",
            "texture": {
                "channels": "Sobel gradient-magnitude histogram and local luminance "
                            "standard-deviation map histogram of the aligned masked crops",
                "bins": TEXTURE_HIST_BINS,
                "local_std_window": TEXTURE_STD_WINDOW,
                "channel_weights": [0.5, 0.5],
            },
            "combination": f"{GARMENT_STRUCTURE_WEIGHT} * structure + "
                           f"{GARMENT_TEXTURE_WEIGHT} * texture",
        },
        "lab_colour": {
            "space": "OpenCV 8-bit CIE Lab",
            "bins": HIST_BINS,
            "channel_weights": list(COLOUR_CHANNEL_WEIGHTS),
            "distance": "per-channel 1D Wasserstein on normalised histograms",
            "strips": COLOUR_STRIP_COUNT,
            "strip_weights": [COLOUR_GLOBAL_WEIGHT, COLOUR_STRIP_WEIGHT],
            "min_strip_pixels": COLOUR_MIN_STRIP_PIXELS,
            "zero_distance": COLOUR_ZERO_DISTANCE,
            "similarity": "clip(1 - D / zero_distance, 0, 1)",
            "calibration": "zero_distance = training-fold p95 / (1 - 1/3); fitted inside each "
                           "CV fold from the stored raw distances",
        },
        "parser": {"classes": list(PARSER_CLASSES), "input_size": list(PARSER_SIZE)},
        "pose": {
            "model": POSE_MODEL_DIR_NAME,
            "keypoint_confidence": KEYPOINT_CONFIDENCE,
            "order": list(COCO_KEYPOINT_ORDER),
            "input_size": list(POSE_INPUT_SIZE),
            "box": "whole photo",
            "decode": "bilinear heatmap upsampling to input size, arg-max plus quadratic refinement",
        },
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


# ------------------------------------------------------------------------ models


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GeometryScorer:
    """Holds the three frozen models and scores one sample end to end.

    Models are loaded once per process. Pose keypoints of the person photo are not
    computed here: they come from the cached geometry prepared once per dataset
    sample, so that the two ratios always come from the same detector.
    """

    def __init__(self, models_dir: str | Path, *, device: str = "cuda:0",
                 weights: Mapping[str, float] | None = None, bias: float = DEFAULT_BIAS,
                 allow_cpu_fallback: bool = False) -> None:
        import torch
        from transformers import (
            Dinov2Model,
            SegformerForSemanticSegmentation,
            VitPoseForPoseEstimation,
        )

        self.torch = torch
        self.models_dir = Path(models_dir)
        self.device = device
        if device.startswith("cuda") and not torch.cuda.is_available():
            if not allow_cpu_fallback:
                raise GeometryTermError(f"CUDA device {device} requested but unavailable")
            self.device = "cpu"
        self.weights = validate_weights(weights or DEFAULT_WEIGHTS, bias)
        self.bias = float(bias)

        dinov2_dir = self.models_dir / "dinov2-small"
        parser_dir = self.models_dir / "fashn-human-parser"
        pose_dir = self.models_dir / POSE_MODEL_DIR_NAME
        for path in (dinov2_dir, parser_dir, pose_dir):
            _require(Path(path).exists(), f"missing model at {path}")

        self.dinov2 = Dinov2Model.from_pretrained(str(dinov2_dir), local_files_only=True)
        self.dinov2.eval().to(self.device)
        _require(int(self.dinov2.config.patch_size) == PATCH_SIZE,
                 f"unexpected patch size {self.dinov2.config.patch_size}")
        self.parser = SegformerForSemanticSegmentation.from_pretrained(
            str(parser_dir), local_files_only=True
        )
        self.parser.eval().to(self.device)

        self.pose = VitPoseForPoseEstimation.from_pretrained(
            str(pose_dir), local_files_only=True
        )
        self.pose.eval().to(self.device)
        self.model_paths = {
            "dinov2": {"path": str(dinov2_dir),
                       "sha256": _sha256_file(dinov2_dir / "model.safetensors")},
            "parser": {"path": str(parser_dir),
                       "sha256": _sha256_file(parser_dir / "model.safetensors")},
            "pose": {"path": str(pose_dir),
                     "sha256": _sha256_file(pose_dir / "model.safetensors")},
        }

    # -- model calls -------------------------------------------------------

    # The three encoders run on fixed-size inputs (the parser at ``PARSER_SIZE``, the pose
    # detector at ``POSE_INPUT_SIZE``, DINOv2 on its letterboxed 392-pixel canvas), so a
    # group of candidates can share one forward pass each. ``score_batch`` does exactly
    # that; the single-image entry points below survive as thin wrappers so the evaluator,
    # the ladder and the tests keep calling the same names.

    def _stack(self, arrays: Sequence[np.ndarray], size: tuple[int, int],
               mean: Sequence[float], std: Sequence[float],
               interpolation: int) -> Any:
        """Normalise and stack RGB arrays into one ``(n, 3, H, W)`` float tensor."""
        torch = self.torch
        resized = [cv2.resize(np.asarray(array, np.uint8), size,
                              interpolation=interpolation).astype(np.float32) / 255.0
                   for array in arrays]
        stacked = np.stack(resized).transpose(0, 3, 1, 2)
        tensor = torch.from_numpy(np.ascontiguousarray(stacked)).float()
        mean_t = torch.tensor(list(mean), dtype=torch.float32).view(1, 3, 1, 1)
        std_t = torch.tensor(list(std), dtype=torch.float32).view(1, 3, 1, 1)
        return ((tensor - mean_t) / std_t).to(self.device)

    def parse_labels(self, image: Any) -> np.ndarray:
        """Per-pixel parser labels at the input image resolution."""
        return self.parse_many([to_rgb_array(image)])[0][0]

    def parse_labels_with_confidence(self, image: Any) -> tuple[np.ndarray, np.ndarray]:
        """Labels plus the parser's own maximum softmax probability, at full resolution."""
        return self.parse_many([to_rgb_array(image)])[0]

    def parse_many(self, arrays: Sequence[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Batched parser pass; one (labels, confidence) pair per input picture."""
        if len(arrays) > BATCH_CHUNK:
            output: list[tuple[np.ndarray, np.ndarray]] = []
            for start in range(0, len(arrays), BATCH_CHUNK):
                output.extend(self.parse_many(arrays[start:start + BATCH_CHUNK]))
            return output
        torch = self.torch
        if not arrays:
            return []
        sizes = [tuple(np.asarray(array).shape[:2]) for array in arrays]
        tensor = self._stack(arrays, PARSER_SIZE, PARSER_MEAN, PARSER_STD, cv2.INTER_AREA)
        with torch.no_grad():
            logits = self.parser(pixel_values=tensor).logits
        probabilities = torch.softmax(logits, dim=1)
        confidence, labels = probabilities.max(dim=1)
        output: list[tuple[np.ndarray, np.ndarray]] = []
        for index, (height, width) in enumerate(sizes):
            label_map = torch.nn.functional.interpolate(
                labels[index][None, None].float(), size=(height, width), mode="nearest")
            margin = torch.nn.functional.interpolate(
                confidence[index][None, None], size=(height, width), mode="bilinear",
                align_corners=False)
            output.append((label_map[0, 0].to(torch.uint8).cpu().numpy(),
                           margin[0, 0].cpu().numpy().astype(np.float32)))
        return output

    def detect_pose(self, image: Any) -> dict[str, dict[str, float]]:
        """Run the pose detector and return named COCO keypoints in pixel units."""
        return self.detect_poses([to_rgb_array(image)])[0]

    def detect_poses(self, arrays: Sequence[np.ndarray]) -> list[dict[str, dict[str, float]]]:
        """Batched pose pass over whole photos, one keypoint dict per picture.

        The whole photo is the detection box, so the model input is the photo resized to
        ``POSE_INPUT_SIZE`` and the heatmap coordinates map back with the inverse of that
        resize.
        """
        if len(arrays) > BATCH_CHUNK:
            output: list[dict[str, dict[str, float]]] = []
            for start in range(0, len(arrays), BATCH_CHUNK):
                output.extend(self.detect_poses(arrays[start:start + BATCH_CHUNK]))
            return output
        torch = self.torch
        if not arrays:
            return []
        sizes = [tuple(np.asarray(array).shape[:2]) for array in arrays]
        width_in, height_in = POSE_INPUT_SIZE
        tensor = self._stack(arrays, POSE_INPUT_SIZE, POSE_IMAGE_MEAN, POSE_IMAGE_STD,
                             cv2.INTER_LINEAR)
        with torch.no_grad():
            heatmaps = self.pose(pixel_values=tensor).heatmaps.detach().cpu().numpy()
        _require(heatmaps.ndim == 4 and heatmaps.shape[1] == len(COCO_KEYPOINT_ORDER),
                 f"unexpected heatmap shape {heatmaps.shape}")
        output: list[dict[str, dict[str, float]]] = []
        for index, (height, width) in enumerate(sizes):
            points = np.zeros((heatmaps.shape[1], 3), dtype=np.float64)
            for joint in range(heatmaps.shape[1]):
                peak = heatmaps[index, joint].max()
                upsampled = cv2.resize(heatmaps[index, joint], (width_in, height_in),
                                       interpolation=cv2.INTER_LINEAR)
                row, column = np.unravel_index(int(np.argmax(upsampled)), upsampled.shape)
                offset_x = _quadratic_offset(upsampled, row, column, axis=1)
                offset_y = _quadratic_offset(upsampled, row, column, axis=0)
                pixel_x = (column + offset_x + 0.5) * width / width_in
                pixel_y = (row + offset_y + 0.5) * height / height_in
                points[joint] = (pixel_x, pixel_y, float(peak))
            output.append(keypoint_dict(points))
        return output

    def garment_patches(self, image: Any, region: np.ndarray) -> GarmentPatchSet | None:
        """DINOv2 tokens of one garment view, at two blocks and at known patch centres."""
        return self.garment_patches_many([letterbox_view(image, region)])[0]

    def garment_patches_many(self, views: Sequence[Mapping[str, Any] | None]
                             ) -> list[GarmentPatchSet | None]:
        """Batched DINOv2 pass over letterboxed garment canvases.

        One forward pass over the whole group asks for the intermediate hidden states, so
        the mid block and the final block cost one evaluation and two slices; only the
        kept patches leave the device. A view whose own mask covers no patch cell keeps
        its ``None`` and is skipped rather than encoded as an empty canvas.
        """
        if len(views) > BATCH_CHUNK:
            output: list[GarmentPatchSet | None] = []
            for start in range(0, len(views), BATCH_CHUNK):
                output.extend(self.garment_patches_many(views[start:start + BATCH_CHUNK]))
            return output
        torch = self.torch
        prepared: list[tuple[int, np.ndarray]] = []
        for index, view in enumerate(views):
            if view is None:
                continue
            cells = patch_grid_mask(view["mask"])
            if not cells.any():
                continue
            prepared.append((index, cells))
        output: list[GarmentPatchSet | None] = [None] * len(views)
        if not prepared:
            return output
        tensor = self._stack([views[index]["rgb"] for index, _ in prepared],
                             (PATCH_GRID[1] * PATCH_SIZE, PATCH_GRID[0] * PATCH_SIZE),
                             PARSER_MEAN, PARSER_STD, cv2.INTER_AREA)
        with torch.no_grad():
            # The pretrained grid is 37 by 37; our square 28 by 28 canvas needs explicit
            # interpolation of the position embeddings.
            result = self.dinov2(pixel_values=tensor, interpolate_pos_encoding=True,
                                 output_hidden_states=True)
        _require(result.hidden_states is not None and len(result.hidden_states) > DINOV2_MID_BLOCK,
                 "DINOv2 returned no intermediate hidden states")
        final_states = result.last_hidden_state
        mid_states = result.hidden_states[DINOV2_MID_BLOCK]
        for position, (index, cells) in enumerate(prepared):
            keep = torch.from_numpy(cells.reshape(-1)).to(self.device)
            final_tokens = final_states[position, 1:, :][keep].cpu().numpy()
            mid_tokens = mid_states[position, 1:, :][keep].cpu().numpy()
            expected = int(keep.sum())
            _require(final_tokens.shape[0] == expected,
                     f"patch token count {final_tokens.shape[0]} != {expected}; the grid changed")
            output[index] = GarmentPatchSet(mid=mid_tokens, final=final_tokens,
                                            centres=patch_centres(cells),
                                            texture=garment_texture(views[index]))
        return output

    def global_embedding(self, image: Any) -> np.ndarray:
        """Mean of the final DINOv2 patch tokens of the whole picture, L2-normalised.

        One square resize onto the patch grid keeps the token grid the encoder was
        trained on, and averaging the patch tokens keeps the read global instead of tied
        to whichever crop a term happens to use. The encoder is already resident for the
        garment views, so this costs one forward pass and no extra memory. The resize
        ignores aspect ratio on purpose: the picture's own framing is what a learned
        read-out is allowed to see, and letterboxing it would be a second framing decision
        layered on top of the one the generator already made.
        """
        torch = self.torch
        side = PATCH_GRID[0] * PATCH_SIZE
        array = to_rgb_array(image)
        resized = cv2.resize(array, (side, side), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self.device)
        with torch.no_grad():
            result = self.dinov2(pixel_values=tensor, interpolate_pos_encoding=True)
        vector = result.last_hidden_state[0, 1:, :].mean(dim=0).cpu().numpy()
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0.0 else vector

    # -- terms -------------------------------------------------------------

    def build_group_state(self, *, person: Any, garment: Any, cloth_mask: np.ndarray,
                          person_points: Mapping[str, Mapping[str, float]],
                          frame: tuple[int, int]) -> dict[str, Any]:
        """Everything one source input contributes, computed once for its whole group.

        Every candidate in a GRPO group shares the person photo, the flat garment, the
        cloth mask and the cached pose, and the five terms above read them unchanged on
        every sample. Recomputing them per sample was most of the 103 ms ``terms`` share,
        which was more than all three networks together.
        """
        height, width = frame
        person_array = to_rgb_array(person, size=(width, height))
        person_lab = lab_array(person_array)
        cloth = np.asarray(cloth_mask, dtype=bool)
        flat_region = erode_mask(cloth, PATCH_SIZE + 1)
        if not flat_region.any():
            flat_region = cloth
        garment_array = to_rgb_array(garment)
        flat_view = self.garment_patches_many([letterbox_view(garment_array, cloth)])[0]
        return {
            "frame": frame,
            "person_array": person_array,
            "person_lab": person_lab,
            "person_chroma": np.sqrt(person_lab[:, :, 1] ** 2 + person_lab[:, :, 2] ** 2),
            "person_points": scale_points_to_frame(person_points,
                                                   frame_shape(person), frame),
            "garment_array": garment_array,
            "cloth_mask": cloth,
            "flat_region": flat_region,
            "flat_lab": cv2.cvtColor(garment_array, cv2.COLOR_RGB2LAB),
            "flat_view": flat_view,
        }

    def score_sample(self, *, result: Any, person: Any, garment: Any,
                     changed_mask: np.ndarray, cloth_mask: np.ndarray,
                     sleeve_bin_label: str, person_points: Mapping[str, Mapping[str, float]],
                     sleeve_expected: Mapping[str, float | None] | None = None,
                     person_available: bool = True) -> dict[str, Any]:
        """Compute the five terms and the combined score for one candidate."""
        return self.score_batch(
            [{"result": result, "changed_mask": changed_mask,
              "sleeve_bin_label": sleeve_bin_label, "sleeve_expected": sleeve_expected,
              "person_available": person_available}],
            person=person, garment=garment, cloth_mask=cloth_mask,
            person_points=person_points)[0]

    def score_batch(self, samples: Sequence[Mapping[str, Any]], *, person: Any, garment: Any,
                    cloth_mask: np.ndarray,
                    person_points: Mapping[str, Mapping[str, float]]
                    ) -> list[dict[str, Any]]:
        """Score every candidate of one group with one forward pass per encoder."""
        if not samples:
            return []
        frames = [tuple(np.asarray(to_rgb_array(sample["result"])).shape[:2])
                  for sample in samples]
        # One person photo resized onto the result grid cannot serve two grids, so a group
        # that mixes frame sizes is scored in one pass per size. In practice a group holds
        # one source input and one size, so this is the single-pass case.
        if len(set(frames)) > 1:
            output: list[dict[str, Any]] = [dict() for _ in samples]
            for frame in dict.fromkeys(frames):
                members = [index for index, value in enumerate(frames) if value == frame]
                batch = self.score_batch([samples[index] for index in members], person=person,
                                         garment=garment, cloth_mask=cloth_mask,
                                         person_points=person_points)
                for index, result in zip(members, batch, strict=True):
                    output[index] = result
            return output

        frame = frames[0]
        height, width = frame
        started = time.perf_counter()
        state = self.build_group_state(person=person, garment=garment, cloth_mask=cloth_mask,
                                       person_points=person_points, frame=frame)
        prepared = time.perf_counter()

        result_arrays = [to_rgb_array(sample["result"]) for sample in samples]
        masks = []
        for sample in samples:
            mask = np.asarray(sample["changed_mask"], dtype=bool)
            if mask.shape[:2] != frame:
                mask = cv2.resize(mask.astype(np.uint8), (width, height),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            masks.append(mask)
        parsed = self.parse_many(result_arrays)
        parse_done = time.perf_counter()
        available = [index for index, sample in enumerate(samples)
                     if sample.get("person_available", True)]
        poses: list[dict[str, dict[str, float]]] = [{} for _ in samples]
        if available:
            detected = self.detect_poses([result_arrays[index] for index in available])
            for index, points_out in zip(available, detected, strict=True):
                poses[index] = points_out
        pose_done = time.perf_counter()

        worn: list[dict[str, Any]] = [worn_garment_mask(result_arrays[index], parsed[index][0],
                                                        masks[index])
                                      for index in range(len(samples))]
        worn_views = [letterbox_view(result_arrays[index], worn[index]["mask"])
                      for index in range(len(samples))]
        worn_views = self.garment_patches_many(worn_views)
        encoded = time.perf_counter()

        count = len(samples)
        rows: list[dict[str, Any]] = []
        for index, sample in enumerate(samples):
            labels = parsed[index][0]
            changed = masks[index]
            result_array = result_arrays[index]
            result_points = poses[index]
            shoulder = shoulder_proportion_term(result_points, state["person_points"])
            arms = arm_coverage_term(result_points, labels, str(sample["sleeve_bin_label"]),
                                     state["person_points"],
                                     expected=sample.get("sleeve_expected"))

            garment_region = worn[index]["mask"]
            eroded = erode_mask(garment_region, 2 * PATCH_SIZE + 1)
            worn_region = eroded if eroded.any() else garment_region
            garment_result = garment_patch_term(worn_views[index], state["flat_view"],
                                                worn_mask_counts=worn[index]["counts"])

            # Two Lab conventions meet here and must not be mixed. ``lab_array`` returns
            # CIE76-difference inputs with a and b centred on zero; the histogram terms
            # read OpenCV's 8-bit Lab, whose a and b carry the +128 offset and therefore
            # cover the same 0..255 axis the Wasserstein helper bins over. One conversion
            # feeds both, so the shift is applied to a copy instead of converting twice.
            colour_lab = cv2.cvtColor(result_array, cv2.COLOR_RGB2LAB)
            result_lab = centred_lab(colour_lab)
            keep = kept_area_term(result_array, state["person_array"], changed,
                                  person_lab=state["person_lab"],
                                  person_chroma=state["person_chroma"],
                                  result_lab=result_lab)
            colour = lab_colour_term(state["garment_array"], state["flat_region"],
                                     result_array, worn_region,
                                     flat_lab=state["flat_lab"], worn_lab=colour_lab)

            values = {
                "shoulder_proportion": shoulder["value"],
                "arm_coverage": arms["value"],
                "kept_area": keep["value"],
                "garment_patch": garment_result["value"],
                "lab_colour": colour["value"],
            }
            details = {
                "shoulder_proportion": shoulder,
                "arm_coverage": arms,
                "kept_area": keep,
                "garment_patch": garment_result,
                "lab_colour": colour,
            }
            rows.append({
                "terms": values,
                "score": combine(values, self.weights, self.bias),
                # The runtime bar is 0.30 s per sample and the audit measures 0.353, so the
                # split of that number is reported rather than guessed at: the parser, the
                # pose detector, the two DINOv2 views and the numpy terms are timed apart.
                # A batched pass is divided evenly over the candidates it scored, so the
                # audit's per-sample mean stays comparable with the single-sample runs.
                "timings": {
                    "parse": (parse_done - prepared) / count,
                    "pose": (pose_done - parse_done) / count,
                    "garment_views": (encoded - pose_done) / count,
                    "terms": (time.perf_counter() - encoded) / count,
                },
                "details": details,
                "flags": {name: detail.get("flag", "") for name, detail in details.items()},
                "parser_label_share": {
                    "garment": float(garment_region.mean()),
                    "skin": float((labels == PARSER_SKIN_LABEL).mean()),
                },
            })
        return rows


# ------------------------------------------------------------------- cache I/O


def load_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("L"))
    return array > 127


def load_points(path: str | Path) -> dict[str, dict[str, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    points = payload.get("keypoints", payload)
    _require(isinstance(points, dict), f"malformed keypoint cache: {path}")
    return points

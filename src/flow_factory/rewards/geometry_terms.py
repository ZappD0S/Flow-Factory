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
``scripts/prepare_t018_geometry.py``, to measure the sleeve length of a garment.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

SCHEMA_VERSION = 1
TASK_ID = "T018"

# ---------------------------------------------------------------- term contract

TERM_NAMES: tuple[str, ...] = ("shoulder", "sleeve", "keep", "garment", "colour")
DEFAULT_WEIGHTS: dict[str, float] = {
    "shoulder": 1.0,
    "sleeve": 1.0,
    "keep": 1.0,
    "garment": 1.0,
    "colour": 0.5,
}
DEFAULT_BIAS = 0.0
TERM_RANGES: dict[str, tuple[float, float]] = {
    "shoulder": (-1.0, 0.0),
    "sleeve": (-1.0, 0.0),
    "keep": (-0.3, 0.7),
    "garment": (-1.0, 1.0),
    "colour": (-1.0, 0.0),
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
KEEP_SSIM_WEIGHT = 0.7
KEEP_ABSOLUTE_WEIGHT = 0.3

# DINOv2 ViT-S/14 patch grid: 28 by 37 patches of 14 pixels keeps the 3:4 frame.
PATCH_SIZE = 14
PATCH_GRID = (37, 28)  # (rows, columns) -> 518 by 392 pixels
PARSER_KEEP_LABELS: tuple[int, ...] = (3, 4, 10)  # top, dress, scarf
PARSER_SKIN_LABEL = 12  # arms
PARSER_DROPPED_LABELS: tuple[int, ...] = (13, 17)  # hands, jewelry
PARSER_SIZE = (384, 576)  # (width, height) matching the parser's training input
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

# Pose: COCO keypoint order produced by the container's pose detector.
KEYPOINT_CONFIDENCE = 0.3
COCO_KEYPOINT_INDEX: dict[str, int] = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
}
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

# A half again as wide or half as wide reaches the floor. Measured on the 160
# graded candidates, the ratio spans 0.90 to 1.54, so a slope of 3 clamped 12 of
# them for no extra ranking power: every slope from 0.75 to 2 orders the grades
# identically (AUROC 0.722), while 3 clamps 12 candidates and 6 clamps 39 and
# loses ordering (0.685).
SHOULDER_RATIO_SLOPE = 2.0
# Three rulers, tried in this order; each is also reported as a non-scoring
# variant so the order is a measured claim. The head ruler (neck midpoint to nose)
# is the oldest and the worst: the nose is a face keypoint, the pose detector
# slides it on generated faces, and on the 160 graded pictures that drove the ratio
# far enough to pin 57% of the candidates at the -1 floor. Hip width is a body
# keypoint but scales with the whole body, so a horizontally widened body leaves the
# ratio unchanged and the corruption ladder caught exactly that. The torso length
# (shoulder midpoint to hip midpoint) is vertical, so width changes move one operand
# only, and it never reads the face.
SHOULDER_RULERS: tuple[str, ...] = ("torso", "hips", "head")


class GeometryTermError(RuntimeError):
    """A fail-closed geometry-term error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GeometryTermError(message)


# ------------------------------------------------------------------- primitives


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


def float01(array: np.ndarray) -> np.ndarray:
    return array.astype(np.float32) / 255.0


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


def keep_term(result: Any, person: Any, changed_mask: np.ndarray) -> dict[str, Any]:
    """Kept-area term: structural and colour preservation outside the editable area."""
    result_array = to_rgb_array(result)
    person_array = to_rgb_array(person, size=(result_array.shape[1], result_array.shape[0]))
    kept = ~dilate_mask(changed_mask, KEEP_DILATE_KERNEL)
    total = int(kept.sum())
    fallback = total < MIN_REGION_PIXELS
    if fallback:
        kept = ~changed_mask
        total = int(kept.sum())
    if total == 0:
        return {"value": WORST_VALUE["keep"], "flag": "keep_empty", "kept_pixels": 0,
                "ssim_mean": None, "absolute_mean": None}
    structure = ssim_map(result_array, person_array)
    absolute = np.abs(float01(result_array) - float01(person_array)).mean(axis=2)
    ssim_mean = float(structure[kept].mean())
    absolute_mean = float(absolute[kept].mean())
    value = KEEP_SSIM_WEIGHT * ssim_mean - KEEP_ABSOLUTE_WEIGHT * absolute_mean
    low, high = TERM_RANGES["keep"]
    return {
        "value": float(min(max(value, low), high)),
        "flag": "keep_plain_region" if fallback else "",
        "kept_pixels": total,
        "ssim_mean": ssim_mean,
        "absolute_mean": absolute_mean,
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


def pose_ratio(points: Mapping[str, Mapping[str, float]],
               ruler: str = "head") -> dict[str, Any]:
    """Scale-free shoulder ratio against a built-in ruler.

    ``ruler="torso"`` divides the shoulder width by the length from the shoulder
    midpoint to the hip midpoint, ``ruler="hips"`` by the hip width, ``ruler="head"``
    by the distance from the shoulder midpoint to the nose. Every one of them
    cancels zoom, crop and translation. They differ in what they are blind to: hip
    width scales with the whole body, so a picture whose body was widened
    horizontally moves both operands and the ratio stays put, while the torso
    length is a vertical measure and does not; the head ruler reads face keypoints,
    which the detector slides on generated faces. Both operands must come from the same
    detector: the person ratio is cached by ``prepare_t018_geometry.py`` from a run
    of the same detector on the person photo.
    """
    left = _visible(points, "left_shoulder")
    right = _visible(points, "right_shoulder")
    if left is None or right is None:
        return {"ratio": None, "ruler": None, "reason": "missing_shoulders"}
    shoulder = math.dist(left, right)
    if ruler == "torso":
        left_hip = _visible(points, "left_hip")
        right_hip = _visible(points, "right_hip")
        if left_hip is None or right_hip is None:
            return {"ratio": None, "ruler": None, "reason": "missing_hips"}
        shoulder_mid = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
        hip_mid = ((left_hip[0] + right_hip[0]) / 2.0, (left_hip[1] + right_hip[1]) / 2.0)
        torso = math.dist(shoulder_mid, hip_mid)
        if shoulder <= 0 or torso <= 1e-6:
            return {"ratio": None, "ruler": None, "reason": "degenerate_ruler"}
        return {"ratio": shoulder / torso, "ruler": "shoulder_over_torso", "reason": ""}
    if ruler == "hips":
        left_hip = _visible(points, "left_hip")
        right_hip = _visible(points, "right_hip")
        if left_hip is None or right_hip is None:
            return {"ratio": None, "ruler": None, "reason": "missing_hips"}
        width = math.dist(left_hip, right_hip)
        if shoulder <= 0 or width <= 1e-6:
            return {"ratio": None, "ruler": None, "reason": "degenerate_ruler"}
        return {"ratio": shoulder / width, "ruler": "shoulder_over_hip", "reason": ""}
    nose = _visible(points, "nose")
    if nose is None:
        return {"ratio": None, "ruler": None, "reason": "missing_nose"}
    neck = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
    head = math.dist(neck, nose)
    if shoulder <= 0 or head <= 1e-6:
        return {"ratio": None, "ruler": None, "reason": "degenerate_ruler"}
    return {"ratio": shoulder / head, "ruler": "shoulder_over_neck_nose", "reason": ""}


def shoulder_term(result_points: Mapping[str, Mapping[str, float]],
                  person_points: Mapping[str, Mapping[str, float]],
                  ruler: str | None = None) -> dict[str, Any]:
    """Proportion change of shoulder width, in [-1, 0].

    Every ruler is measured on both frames; ``variants`` keeps all of them so the
    offline report can show what each one would have scored. ``ruler=None`` picks
    the first usable entry of ``SHOULDER_RULERS``; naming a ruler forces it and
    makes the term unmeasurable when that ruler alone is unavailable.
    """
    variants: dict[str, dict[str, Any]] = {}
    for name in SHOULDER_RULERS:
        result_ratio = pose_ratio(result_points, name)
        person_ratio = pose_ratio(person_points, name)
        entry: dict[str, Any] = {
            "ruler": result_ratio["ruler"] or person_ratio["ruler"],
            "result_ratio": result_ratio["ratio"],
            "person_ratio": person_ratio["ratio"],
            "available": False,
            "relative": None,
            "value": None,
            "reason": result_ratio["reason"] or person_ratio["reason"],
        }
        if entry["result_ratio"] is not None and entry["person_ratio"] is not None:
            relative = entry["result_ratio"] / entry["person_ratio"]
            entry["relative"] = float(relative)
            entry["value"] = float(-min(1.0, SHOULDER_RATIO_SLOPE * abs(relative - 1.0)))
            entry["available"] = True
        variants[name] = entry

    order = (ruler,) if ruler is not None else SHOULDER_RULERS
    chosen = next((name for name in order if variants[name]["available"]), None)
    if chosen is None:
        return {"value": WORST_VALUE["shoulder"], "flag": "shoulder_unmeasurable",
                "ruler": None, "result_ratio": None, "person_ratio": None,
                "relative": None, "saturated": True, "variants": variants}
    entry = variants[chosen]
    return {"value": entry["value"], "flag": "", "ruler": entry["ruler"],
            "ruler_name": chosen, "result_ratio": entry["result_ratio"],
            "person_ratio": entry["person_ratio"], "relative": entry["relative"],
            "saturated": entry["value"] <= -1.0, "variants": variants}


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


def sleeve_bin(c_upper: float | None, c_fore: float | None, agree: bool = True) -> str:
    """Three-way sleeve length bin from the ground-truth measurement.

    ``agree`` records whether the two arms measured alike; a disagreement is
    flagged by the caller but no longer forces a hand check, because both arms
    wear the same sleeve and the mean is the better estimate of it.
    """
    del agree
    if c_upper is None or c_fore is None:
        return "review"
    if c_fore >= SLEEVE_BIN_LONG_FOREARM:
        return "long"
    if c_upper <= SLEEVE_BIN_SLEEVELESS_UPPER:
        return "sleeveless"
    if c_fore <= SLEEVE_BIN_SHORT_FOREARM:
        return "short"
    return "review"


def _sleeve_value(penalty: float) -> float:
    range_size = TERM_RANGES["sleeve"][1] - TERM_RANGES["sleeve"][0]
    return float(TERM_RANGES["sleeve"][1] - range_size * min(max(penalty, 0.0), 1.0))


def sleeve_term(result_points: Mapping[str, Mapping[str, float]], label_map: np.ndarray,
                bin_label: str, person_points: Mapping[str, Mapping[str, float]],
                expected: Mapping[str, float | None] | None = None) -> dict[str, Any]:
    """Arm coverage defect, in [-1, 0], never missing.

    The upper arm (shoulder to elbow) is measured as well as the forearm (elbow to
    wrist). Previously only the forearm was scored, so a garment whose sleeve
    vanished above the elbow while the forearm stayed bare produced no penalty at
    all: the bin rule never looked there.

    ``expected`` carries the garment's own coverage, measured on the wearing photo
    by ``ground_truth_sleeve``: ``{"upper": c_upper, "fore": c_fore}``. When both
    segments can be compared, the penalty is the mean shortfall against that
    profile, and the bin no longer gates the score -- that matters because the
    ``review`` bin (three-quarter sleeves) covers 30% of the graded candidates and
    used to return exactly 0.0. Without ``expected`` the term falls back to the old
    bin rule, one segment per bin.
    """
    _require(bin_label in SLEEVE_BINS, f"unknown sleeve bin: {bin_label}")
    per_arm: list[dict[str, Any]] = []
    for side in ("left", "right"):
        shoulder = _visible(result_points, f"{side}_shoulder")
        elbow = _visible(result_points, f"{side}_elbow")
        wrist = _visible(result_points, f"{side}_wrist")
        if elbow is None or wrist is None:
            continue
        forearm = sleeve_fraction(label_map, elbow, wrist)
        if forearm["garment_fraction"] is None:
            continue
        upper_arm = sleeve_fraction(label_map, shoulder, elbow) if shoulder else {
            "garment_fraction": None, "skin_fraction": None}
        per_arm.append({
            "side": side,
            **forearm,
            "upper_garment_fraction": upper_arm["garment_fraction"],
            "upper_skin_fraction": upper_arm["skin_fraction"],
        })
    person_arms = sum(
        _visible(person_points, f"{side}_elbow") is not None
        and _visible(person_points, f"{side}_wrist") is not None
        for side in ("left", "right")
    )
    detail: dict[str, Any] = {"arms_measured": len(per_arm), "person_arms": person_arms,
                             "bin": bin_label, "per_arm": per_arm}
    if not per_arm:
        # Broken pose inside the shirt is a defect; a person photo with no visible
        # arms means nothing could have been destroyed.
        value = WORST_VALUE["sleeve"] if person_arms >= 2 else 0.0
        return {"value": float(value), "flag": "sleeve_missing", "mode": "missing", **detail}

    upper_values = [arm["upper_garment_fraction"] for arm in per_arm
                    if arm["upper_garment_fraction"] is not None]
    measured = {
        "upper": float(np.mean(upper_values)) if upper_values else None,
        "fore": float(np.mean([arm["garment_fraction"] for arm in per_arm])),
    }
    forecasted = {
        segment: None if expected is None else _expected_coverage(expected, segment)
        for segment in ("upper", "fore")
    }
    deficit = {
        segment: max(0.0, float(target) - float(measured[segment]))
        for segment, target in forecasted.items()
        if target is not None and measured[segment] is not None
    }
    bin_penalty = _bin_penalty(per_arm, bin_label)
    bin_value = 0.0 if bin_label == "review" else _sleeve_value(bin_penalty)
    detail.update({"measured": measured, "expected": forecasted, "deficit": deficit,
                   "bin_penalty": bin_penalty, "bin_value": bin_value})

    if deficit:
        penalty = float(np.mean(list(deficit.values())))
        detail.update({"mode": "coverage_profile", "penalty": penalty})
        # The review bin is measured now, so it is not a fallback and raises no flag;
        # the bin itself is reported in the details for anyone reading the figure.
        return {"value": _sleeve_value(penalty), "flag": "", **detail}

    detail.update({"mode": "bin", "penalty": bin_penalty})
    if bin_label == "review":
        return {"value": 0.0, "flag": "sleeve_bin_review", **detail}
    return {"value": bin_value, "flag": "", **detail}


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
    """Old rule: what must not be seen on the forearm, given the garment's length."""
    if bin_label == "long":
        return float(np.mean([arm["skin_fraction"] for arm in per_arm]))
    return float(np.mean([arm["garment_fraction"] for arm in per_arm]))


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
        "bin": sleeve_bin(c_upper, c_fore, agree),
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


def cosine_match(first: np.ndarray, second: np.ndarray) -> float:
    """Two-direction mean of maximum cosine similarity between two patch sets."""
    _require(first.ndim == 2 and second.ndim == 2, "patch features must be matrices")
    _require(first.shape[1] == second.shape[1], "patch features differ in width")
    left = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-8)
    right = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-8)
    similarity = left @ right.T
    forward = float(similarity.max(axis=1).mean())
    backward = float(similarity.max(axis=0).mean())
    return 0.5 * forward + 0.5 * backward


def garment_term(worn_features: np.ndarray | None, flat_features: np.ndarray | None) -> dict[str, Any]:
    """Garment patch-match term, in [-1, 1]."""
    if worn_features is None or flat_features is None:
        return {"value": WORST_VALUE["garment"], "flag": "garment_region_empty",
                "worn_patches": 0, "flat_patches": 0}
    value = cosine_match(worn_features, flat_features)
    low, high = TERM_RANGES["garment"]
    return {
        "value": float(min(max(value, low), high)),
        "flag": "",
        "worn_patches": int(worn_features.shape[0]),
        "flat_patches": int(flat_features.shape[0]),
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


def colour_term(flat_rgb: np.ndarray, flat_region: np.ndarray, worn_rgb: np.ndarray,
                worn_region: np.ndarray) -> dict[str, Any]:
    """Lab-histogram colour term, in [-1, 0]."""
    flat_region = np.asarray(flat_region, dtype=bool)
    worn_region = np.asarray(worn_region, dtype=bool)
    if flat_region.sum() < MIN_REGION_PIXELS or worn_region.sum() < MIN_REGION_PIXELS:
        return {"value": WORST_VALUE["colour"], "flag": "colour_region_empty",
                "flat_pixels": int(flat_region.sum()), "worn_pixels": int(worn_region.sum())}
    flat_lab = cv2.cvtColor(to_rgb_array(flat_rgb), cv2.COLOR_RGB2LAB)
    worn_lab = cv2.cvtColor(to_rgb_array(worn_rgb), cv2.COLOR_RGB2LAB)
    distances = [
        histogram_wasserstein(flat_lab[:, :, channel][flat_region],
                              worn_lab[:, :, channel][worn_region])
        for channel in range(3)
    ]
    distance = float(sum(weight * value for weight, value in zip(COLOUR_CHANNEL_WEIGHTS, distances)))
    return {
        "value": float(-min(1.0, distance)),
        "flag": "",
        "channel_distances": distances,
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
        "shoulder": {
            "rulers_in_order": [
                "euclidean(left_shoulder, right_shoulder) / euclidean(left_hip, right_hip)",
                "euclidean(left_shoulder, right_shoulder) / euclidean(mid_shoulder, nose)",
            ],
            "ruler_policy": "hip width when both frames show both hips, else neck-to-nose; "
                            "every ruler is also reported as a non-scoring variant",
            "penalty": f"-min(1, {SHOULDER_RATIO_SLOPE} * |result/person - 1|)",
        },
        "sleeve": {
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
            "segments": ["shoulder_to_elbow", "elbow_to_wrist"],
            "simplified_rule": "mean shortfall of measured garment coverage below the garment's own "
                               "upper-arm and forearm coverage; falls back to the forearm-only bin rule",
            "uses_bin_as_gate": False,
            "missing_value": "worst when the person photo had two visible arms, 0 otherwise",
            "review_bin_value": "measured, not 0.0",
            "upper_arm_gap": "measured since T018 repair; the bin rule ignores the upper arm entirely",
        },
        "keep": {
            "ssim_window": SSIM_WINDOW,
            "ssim_sigma": SSIM_SIGMA,
            "weights": [KEEP_SSIM_WEIGHT, KEEP_ABSOLUTE_WEIGHT],
            "keep_mask": f"complement of the changed-area mask dilated by {KEEP_DILATE_KERNEL}",
            "formula": "0.7 * mean_K(ssim_map) - 0.3 * mean_K(|r - p|)",
        },
        "garment": {
            "patch_size": PATCH_SIZE,
            "patch_grid": list(PATCH_GRID),
            "resize": [PATCH_GRID[1] * PATCH_SIZE, PATCH_GRID[0] * PATCH_SIZE],
            "worn_region": "parser garment labels eroded by 1 patch",
            "flat_region": "cloth-mask eroded by 1 patch",
            "matching": "0.5 * mean_worn max_flat cos + 0.5 * mean_flat max_worn cos",
        },
        "colour": {
            "space": "OpenCV 8-bit CIE Lab",
            "bins": HIST_BINS,
            "channel_weights": list(COLOUR_CHANNEL_WEIGHTS),
            "distance": "per-channel 1D Wasserstein on normalised histograms",
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

    def detect_pose(self, image: Any) -> dict[str, dict[str, float]]:
        """Run the pose detector and return named COCO keypoints in pixel units.

        The whole photo is the detection box, so the model input is the photo
        resized to ``POSE_INPUT_SIZE`` and the heatmap coordinates map back with
        the inverse of that resize.
        """
        torch = self.torch
        array = to_rgb_array(image)
        height, width = array.shape[:2]
        width_in, height_in = POSE_INPUT_SIZE
        resized = cv2.resize(array, (width_in, height_in), interpolation=cv2.INTER_LINEAR)
        scaled = resized.astype(np.float32) / 255.0
        scaled = (scaled - np.asarray(POSE_IMAGE_MEAN, np.float32)) / np.asarray(
            POSE_IMAGE_STD, np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(scaled.transpose(2, 0, 1))[None])
        with torch.no_grad():
            heatmaps = self.pose(pixel_values=tensor.float().to(self.device)).heatmaps[0]
        heatmaps = heatmaps.detach().cpu().numpy()
        _require(heatmaps.ndim == 3 and heatmaps.shape[0] == len(COCO_KEYPOINT_ORDER),
                 f"unexpected heatmap shape {heatmaps.shape}")
        points = np.zeros((heatmaps.shape[0], 3), dtype=np.float64)
        for index in range(heatmaps.shape[0]):
            peak = heatmaps[index].max()
            upsampled = cv2.resize(heatmaps[index], (width_in, height_in),
                                   interpolation=cv2.INTER_LINEAR)
            row, column = np.unravel_index(int(np.argmax(upsampled)), upsampled.shape)
            offset_x = _quadratic_offset(upsampled, row, column, axis=1)
            offset_y = _quadratic_offset(upsampled, row, column, axis=0)
            pixel_x = (column + offset_x + 0.5) * width / width_in
            pixel_y = (row + offset_y + 0.5) * height / height_in
            points[index] = (pixel_x, pixel_y, float(peak))
        return keypoint_dict(points)

    def parse_labels(self, image: Any) -> np.ndarray:
        """Per-pixel parser labels at the input image resolution."""
        torch = self.torch
        array = to_rgb_array(image)
        height, width = array.shape[:2]
        resized = cv2.resize(array, PARSER_SIZE, interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.parser(pixel_values=tensor).logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=(height, width), mode="bilinear", align_corners=False
        )
        return upsampled.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()

    def patch_features(self, image: Any, region: np.ndarray) -> np.ndarray | None:
        """DINOv2 patch tokens inside a pixel-level region, or None when empty."""
        torch = self.torch
        grid_mask = patch_grid_mask(region)
        if not grid_mask.any():
            return None
        array = to_rgb_array(image)
        resized = cv2.resize(array, (PATCH_GRID[1] * PATCH_SIZE, PATCH_GRID[0] * PATCH_SIZE),
                             interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # The pretrained grid is square; our 37x28 grid needs explicit
            # interpolation of the position embeddings.
            output = self.dinov2(pixel_values=tensor, interpolate_pos_encoding=True)
        tokens = output.last_hidden_state[0, 1:, :].cpu().numpy()
        expected = PATCH_GRID[0] * PATCH_GRID[1]
        _require(tokens.shape[0] == expected,
                 f"patch token count {tokens.shape[0]} != {expected}; position embeddings changed the grid")
        return tokens[grid_mask.reshape(-1)]

    # -- terms -------------------------------------------------------------

    def score_sample(self, *, result: Any, person: Any, garment: Any,
                     changed_mask: np.ndarray, cloth_mask: np.ndarray,
                     sleeve_bin_label: str, person_points: Mapping[str, Mapping[str, float]],
                     sleeve_expected: Mapping[str, float | None] | None = None,
                     person_available: bool = True) -> dict[str, Any]:
        """Compute the five terms and the combined score for one candidate."""
        result_array = to_rgb_array(result)
        person_array = to_rgb_array(person, size=(result_array.shape[1], result_array.shape[0]))
        garment_array = to_rgb_array(garment)
        if changed_mask.shape != result_array.shape[:2]:
            changed_mask = cv2.resize(
                changed_mask.astype(np.uint8), (result_array.shape[1], result_array.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        labels = self.parse_labels(result_array)
        result_points = self.detect_pose(result_array) if person_available else {}

        shoulder = shoulder_term(result_points, person_points) if result_points else {
            "value": WORST_VALUE["shoulder"], "flag": "shoulder_unmeasurable",
            "result_ratio": None, "person_ratio": pose_ratio(person_points)["ratio"], "ruler": None}
        sleeve = sleeve_term(result_points, labels, sleeve_bin_label, person_points,
                             expected=sleeve_expected)

        garment_region = np.isin(labels, PARSER_KEEP_LABELS)
        eroded = erode_mask(garment_region, 2 * PATCH_SIZE + 1)
        worn_region = eroded if eroded.any() else garment_region
        flat_region = np.asarray(cloth_mask, dtype=bool)
        flat_eroded = erode_mask(flat_region, PATCH_SIZE + 1)
        flat_region = flat_eroded if flat_eroded.any() else flat_region
        worn_features = self.patch_features(result_array, worn_region)
        flat_features = self.patch_features(garment_array, flat_region)
        garment_result = garment_term(worn_features, flat_features)

        keep = keep_term(result_array, person_array, changed_mask)
        colour = colour_term(garment_array, flat_region, result_array, worn_region)

        values = {
            "shoulder": shoulder["value"],
            "sleeve": sleeve["value"],
            "keep": keep["value"],
            "garment": garment_result["value"],
            "colour": colour["value"],
        }
        return {
            "terms": values,
            "score": combine(values, self.weights, self.bias),
            "details": {
                "shoulder": shoulder,
                "sleeve": sleeve,
                "keep": keep,
                "garment": garment_result,
                "colour": colour,
            },
            "flags": {
                name: detail.get("flag", "")
                for name, detail in (
                    ("shoulder", shoulder),
                    ("sleeve", sleeve),
                    ("keep", keep),
                    ("garment", garment_result),
                    ("colour", colour),
                )
            },
            "parser_label_share": {
                "garment": float(garment_region.mean()),
                "skin": float((labels == PARSER_SKIN_LABEL).mean()),
            },
        }


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

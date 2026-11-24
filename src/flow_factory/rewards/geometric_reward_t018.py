# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""T018 geometric virtual try-on reward: five measured terms, one weighted sum.

The reward is reference-free at scoring time. It reads the candidate picture
(``image``), the person photo and the flat garment photo (``condition_images``),
the changed-area mask and the garment's own arm-coverage profile (from the cached
geometry named in the sample ``metadata``). Ground truth was used once, offline, to
measure that coverage; the sleeve-length bin is read too, but only for reporting.

R = b + w1 * shoulder_proportion + w2 * arm_coverage + w3 * kept_area
      + w4 * garment_patch + w5 * lab_colour,
with every weight enforced non-negative.

See ``geometry_terms.py`` for the term definitions and the unit tests in
``scripts/test_geometric_vto_reward.py`` for the contracts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

from ..hparams.reward_args import RewardArguments
from ..utils.logger_utils import setup_logger
from . import geometry_terms
from .abc import GroupwiseRewardModel, RewardModelOutput

logger = setup_logger(__name__)

REQUIRED_METADATA_KEYS = ("stem",)


class GeometricRewardError(RuntimeError):
    """A fail-closed T018 reward error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GeometricRewardError(message)


def _parse_metadata(value: Any, index: int) -> dict[str, Any]:
    if value is None:
        raise GeometricRewardError(f"sample {index} carries no dataset metadata")
    if isinstance(value, dict):
        return dict(value)
    _require(isinstance(value, str), f"sample {index} metadata is not JSON text")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise GeometricRewardError(f"sample {index} metadata is not valid JSON: {exc}") from exc
    _require(isinstance(payload, dict), f"sample {index} metadata is not an object")
    return payload


def _load_weights_file(path: str | Path) -> dict[str, float]:
    """Read term weights out of a contract or gate report.

    ``reward_contract.json`` nests them one level down and ``gate_report.json`` nests
    them beside a bias and a convention string, so passing either file used to fail
    ``validate_weights`` on a key set of ``{"terms"}``.

    Order matters: both files also carry ``relative``, the standardized shares the fit
    works in, while ``terms`` holds the production weights for the raw term values. A
    share is a few hundredths where its weight is a few tenths, so reading ``relative``
    first scaled the reward by the wrong factor -- silently, and only on ``anatomy``
    in practice, whose share is 0.075 against a weight of 0.407.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    candidate: Any = payload.get("weights", payload) if isinstance(payload, dict) else payload
    for key in ("terms", "default_weights", "relative"):
        if isinstance(candidate, dict) and isinstance(candidate.get(key), dict):
            candidate = candidate[key]
            break
    _require(isinstance(candidate, dict), f"weights file carries no weight mapping: {path}")
    return {str(name): float(value) for name, value in candidate.items()}


class T018GeometricRewardModel(GroupwiseRewardModel):
    """Groupwise geometric reward: five measured terms and non-negative weights."""

    required_fields = ("image",)
    use_tensor_inputs = False

    def __init__(self, config: RewardArguments, accelerator: Any):
        super().__init__(config, accelerator)
        extras = dict(config.extra_kwargs or {})
        self.models_dir = extras.get("models_dir")
        _require(bool(self.models_dir), "geometric reward requires extra_kwargs.models_dir")
        self.geometry_dir = extras.get("geometry_dir")
        _require(bool(self.geometry_dir), "geometric reward requires extra_kwargs.geometry_dir")
        self.device = str(extras.get("device") or "cuda:0")
        weights = extras.get("weights")
        bias = float(extras.get("bias", geometry_terms.DEFAULT_BIAS))
        if isinstance(weights, str) and weights:
            weights = _load_weights_file(weights)
        self.weights = geometry_terms.validate_weights(weights or geometry_terms.DEFAULT_WEIGHTS, bias)
        self.scorer = geometry_terms.GeometryScorer(
            self.models_dir, device=self.device, weights=self.weights, bias=bias
        )
        self.telemetry: list[dict[str, Any]] = []
        self.contract = geometry_terms.term_contract()
        logger.info(
            "T018 geometric reward ready: weights=%s contract=%s",
            self.weights,
            self.contract["contract_sha256"][:12],
        )

    # ------------------------------------------------------------------ helpers

    def _geometry_paths(self, metadata: dict[str, Any], index: int) -> dict[str, Any]:
        for key in REQUIRED_METADATA_KEYS:
            _require(key in metadata, f"sample {index} metadata is missing '{key}'")
        root = Path(metadata.get("geometry_dir") or self.geometry_dir)
        stem = str(metadata["stem"])
        split = metadata.get("split")
        _require(bool(split), f"sample {index} metadata is missing 'split'")
        mask_path = Path(metadata.get("mask_path") or root / "mask" / str(split) / f"{stem}.png")
        pose_path = Path(metadata.get("pose_path") or root / "pose" / str(split) / f"{stem}.json")
        record = self._sleeve_profile()[str(split)].get(stem, {})
        bin_label = metadata.get("sleeve_bin") or record.get("sleeve.bin")
        coverage = {
            "upper": metadata.get("sleeve_c_upper", record.get("sleeve.c_upper")),
            "fore": metadata.get("sleeve_c_fore", record.get("sleeve.c_fore")),
        }
        _require(mask_path.is_file(), f"missing changed-area mask: {mask_path}")
        _require(pose_path.is_file(), f"missing cached pose: {pose_path}")
        _require(bin_label in geometry_terms.SLEEVE_BINS,
                 f"sample {index} sleeve bin is invalid: {bin_label}")
        return {"mask_path": mask_path, "pose_path": pose_path, "bin": str(bin_label),
                "coverage": coverage}

    def _sleeve_profile(self) -> dict[str, dict[str, dict[str, Any]]]:
        if not hasattr(self, "_bins_cache"):
            path = Path(self.geometry_dir) / "sleeve.json"
            _require(path.is_file(), f"missing sleeve cache: {path}")
            self._bins_cache = json.loads(path.read_text(encoding="utf-8"))["garments"]
        return self._bins_cache

    def _cloth_mask(self, metadata: dict[str, Any]) -> Any:
        path = metadata.get("cloth_mask_path")
        _require(bool(path), "sample metadata is missing 'cloth_mask_path'")
        with Image.open(path) as handle:
            import numpy as np

            return np.asarray(handle.convert("L")) > 127

    # --------------------------------------------------------------- scoring

    def score_group(self, images: list[Any], conditions: list[list[Any]],
                    metadata: list[Any], *, group_id: str | int) -> list[dict[str, Any]]:
        _require(len(images) == len(conditions) == len(metadata),
                 "candidates, conditions and metadata must have equal length")
        samples: list[dict[str, Any]] = []
        for index, (image, condition, raw_metadata) in enumerate(zip(images, conditions, metadata)):
            parsed = _parse_metadata(raw_metadata, index)
            _require(isinstance(condition, (list, tuple)) and len(condition) >= 2,
                     f"group {group_id} sample {index} needs person and garment conditions")
            paths = self._geometry_paths(parsed, index)
            samples.append({
                "result": image,
                "changed_mask": geometry_terms.load_mask(paths["mask_path"]),
                "sleeve_bin_label": paths["bin"],
                "sleeve_expected": paths["coverage"],
                "person": condition[0],
                "garment": condition[1],
                "person_available": True,
                "stem": parsed["stem"],
                "points": geometry_terms.load_points(paths["pose_path"]),
                "cloth_mask": self._cloth_mask(parsed),
            })
        # One GRPO group usually shares one person photo and one flat garment, which is what
        # lets the parsers run once for the whole group. A group that mixes sources is split
        # by (person, garment) identity and scored one batch per source, in the original order.
        partitions: dict[tuple[int, int], list[int]] = {}
        for index, sample in enumerate(samples):
            partitions.setdefault((id(sample["person"]), id(sample["garment"])), []).append(index)
        rows: list[dict[str, Any] | None] = [None] * len(samples)
        for members in partitions.values():
            head = samples[members[0]]
            scored = self.scorer.score_batch([samples[index] for index in members],
                                             person=head["person"], garment=head["garment"],
                                             cloth_mask=head["cloth_mask"],
                                             person_points=head["points"])
            for index, row in zip(members, scored, strict=True):
                rows[index] = {"candidate_index": index, "stem": samples[index]["stem"], **row}
        _require(all(row is not None for row in rows), "a candidate was never scored")
        return [row for row in rows if row is not None]

    def __call__(
        self,
        image: list[Any] | None = None,
        condition_images: list[list[Any]] | None = None,
        metadata: list[Any] | None = None,
        **_: Any,
    ) -> RewardModelOutput:
        import torch

        _require(image is not None, "geometric reward requires candidate images")
        _require(condition_images is not None, "geometric reward requires condition images")
        _require(metadata is not None, "geometric reward requires dataset metadata")
        group_id = f"rank-{getattr(self.accelerator, 'process_index', 0)}-group-{len(self.telemetry)}"
        rows = self.score_group(image, condition_images, metadata, group_id=group_id)
        rewards = [float(row["score"]) for row in rows]
        _require(all(math.isfinite(value) for value in rewards), "geometric reward produced a non-finite value")
        _require(len(rewards) == len(image), "geometric reward length mismatch")
        self.telemetry.append(
            {
                "group_id": group_id,
                "candidate_count": len(rows),
                "weights": dict(self.weights),
                "rows": rows,
            }
        )
        extra_info: dict[str, list[float]] = {}
        for name in geometry_terms.TERM_NAMES:
            extra_info[f"t018_{name}"] = [float(row["terms"][name]) for row in rows]
        for name in geometry_terms.TERM_NAMES:
            extra_info[f"t018_{name}_flag"] = [str(row["flags"][name]) for row in rows]
        return RewardModelOutput(
            rewards=torch.tensor(rewards, dtype=torch.float32), extra_info=extra_info
        )

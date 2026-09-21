"""Frozen T009 anatomy-plus-geometry direct-logit reward.

The production reward is served by a dedicated Transformers process so training
workers do not load the 27B judge.  Every request is group-atomic: a malformed,
missing, or non-finite candidate fails the complete GRPO group.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import threading
import time
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
import torch
from accelerate import Accelerator
from PIL import Image

from ..hparams import RewardArguments
from .abc import GroupwiseRewardModel, RewardModelOutput

MODEL_NAME = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ARM = "defect_specific_semantic"
MAPPING = "yes_no"
COMPONENTS = {
    "anatomy": "broken or impossible anatomy",
    "garment_geometry_drape": "grossly implausible garment geometry or drape",
}
POLARITIES = ("positive", "negative")
BASE_CONTEXT = """The image is a virtual try-on candidate.

Judge only whether it contains an obvious severe physical generation
defect.

Count an issue as severe when at least one clearly visible catastrophic
or major failure is present, including:
- impossible, broken, missing, duplicated, or fused anatomy;
- broken person-garment boundaries;
- grossly implausible garment geometry or drape; or
- a major occlusion failure.

Do not count garment identity or style mismatch, source-person identity
differences, harmless pixel-level artifacts, or minor cosmetic flaws.

"""
FROZEN_PROMPT_CONTRACT = {
    "anatomy:positive": {
        "prompt_sha256": "d69678ad0e48c76059bac8a5801652e9fd168926f2c0cfb95c1c68a932725dae",
        "rendered_prompt_sha256": "0630acdea179e99b2c4969198e3f1a8cad33b6543d7160d2912fc38b4e84280b",
        "tokens": {"Yes": 9175, "No": 2665},
    },
    "anatomy:negative": {
        "prompt_sha256": "4a3742e6bfa7d860584dc22aff922572b4afd4cfa5a03eade74e34e0acf36f6f",
        "rendered_prompt_sha256": "c38e2df7120f71b62e976ab51638282df3d967e5cc5aba10dde024ad825ec98e",
        "tokens": {"Yes": 9175, "No": 2665},
    },
    "garment_geometry_drape:positive": {
        "prompt_sha256": "6178bf418d888c9f0fe8a73cd519458fb1994f53663fcf604097d53ada941328",
        "rendered_prompt_sha256": "c994fabd68b1f0a989b36ba9efef2bdb14c959270bdbca74f58d694df3c40dfd",
        "tokens": {"Yes": 9175, "No": 2665},
    },
    "garment_geometry_drape:negative": {
        "prompt_sha256": "cd411ad2197bf2b3ed0f2cb46c688158b38faca2558232e06e05de54dc985771",
        "rendered_prompt_sha256": "2306a8e029a7b3c3386fd0b50ae81137dade714a5086b7f08e2bd66968d77f3f",
        "tokens": {"Yes": 9175, "No": 2665},
    },
}


class PhysicalRewardError(RuntimeError):
    """The frozen contract or a complete group could not be scored."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhysicalRewardError(message)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prompt_for(component: str, polarity: str) -> str:
    _require(component in COMPONENTS, f"unknown component: {component}")
    _require(polarity in POLARITIES, f"unknown polarity: {polarity}")
    phrase = COMPONENTS[component]
    if polarity == "positive":
        question = f"Does the candidate contain {phrase}?"
    else:
        question = f"Is the candidate free of {phrase.removeprefix('a ')}?"
    return f"{BASE_CONTEXT}Question: {question}\n\nAnswer with exactly one word: Yes or No.\nAnswer:\n"


def canonical_margin(polarity: str, yes_logit: float, no_logit: float) -> float:
    _require(polarity in POLARITIES, f"unknown polarity: {polarity}")
    _require(math.isfinite(yes_logit) and math.isfinite(no_logit), "non-finite continuation logit")
    score = yes_logit - no_logit if polarity == "positive" else no_logit - yes_logit
    _require(math.isfinite(score), "non-finite canonical severity margin")
    return score


def aggregate_candidate(polarity_scores: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    _require(set(polarity_scores) == set(COMPONENTS), "component score set mismatch")
    components: dict[str, float] = {}
    for component in COMPONENTS:
        values = polarity_scores[component]
        _require(set(values) == set(POLARITIES), f"polarity score set mismatch: {component}")
        pair = [float(values[polarity]) for polarity in POLARITIES]
        _require(all(math.isfinite(value) for value in pair), f"non-finite component score: {component}")
        components[component] = sum(pair) / 2.0
    severity = sum(components.values()) / 2.0
    reward = -severity
    _require(math.isfinite(reward), "non-finite aggregate reward")
    return {**components, "severity": severity, "reward": reward}


def reward_contract() -> dict[str, Any]:
    prompts = {}
    for component in COMPONENTS:
        for polarity in POLARITIES:
            key = f"{component}:{polarity}"
            prompt = prompt_for(component, polarity)
            frozen = FROZEN_PROMPT_CONTRACT[key]
            _require(_sha256_text(prompt) == frozen["prompt_sha256"], f"embedded prompt drift: {key}")
            prompts[key] = {"prompt": prompt, **frozen}
    payload: dict[str, Any] = {
        "schema_version": 1,
        "source_task": "T009",
        "identifier": "t009_anatomy_geometry_direct_logit_v1",
        "model": {"name": MODEL_NAME, "revision": MODEL_REVISION},
        "runtime": {
            "dtype": "bfloat16",
            "attention": "sdpa",
            "enable_thinking": False,
            "preserve_thinking": False,
            "forward_only": True,
            "image_mode": "RGB",
            "view": "full",
        },
        "request": {
            "arm": ARM,
            "mapping": MAPPING,
            "components": dict(COMPONENTS),
            "polarities": list(POLARITIES),
            "prompts": prompts,
        },
        "orientation": "positive values are more severe; reward is higher when defects are lower",
        "aggregation": {
            "component": "mean canonical severity margin over positive and negative polarity",
            "severity": "(anatomy + garment_geometry_drape) / 2",
            "reward": "-severity",
            "normalization": None,
            "fitted_weights": None,
        },
        "group_contract": {
            "expected_size": 16,
            "preserve_input_order": True,
            "tie_rule": "exactly equal float32 rewards tie and retain candidate input order",
            "failure": "fail the complete group on missing, extra, malformed, or non-finite output",
            "transport_retries": 0,
        },
        "output_precision": "Python float logits aggregated then converted to float32 reward tensor",
    }
    payload["contract_sha256"] = _canonical_json_hash(payload)
    return payload


def _template_text(processor: Any, prompt: str) -> str:
    try:
        return processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            preserve_thinking=False,
        )
    except TypeError as exc:
        raise PhysicalRewardError("processor does not support the frozen thinking-disabled template") from exc


def verify_processor_contract(processor: Any) -> dict[str, Any]:
    tokenizer = processor.tokenizer
    observed: dict[str, Any] = {}
    for component in COMPONENTS:
        for polarity in POLARITIES:
            key = f"{component}:{polarity}"
            prompt = prompt_for(component, polarity)
            rendered = _template_text(processor, prompt)
            base_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            if base_ids and isinstance(base_ids[0], list):
                base_ids = base_ids[0]
            tokens: dict[str, int] = {}
            for name in ("Yes", "No"):
                full_ids = tokenizer(rendered + name, add_special_tokens=False)["input_ids"]
                if full_ids and isinstance(full_ids[0], list):
                    full_ids = full_ids[0]
                _require(full_ids[: len(base_ids)] == base_ids and len(full_ids) == len(base_ids) + 1,
                         f"continuation is not one token: {key}/{name}")
                tokens[name] = int(full_ids[-1])
            entry = {
                "prompt_sha256": _sha256_text(prompt),
                "rendered_prompt_sha256": _sha256_text(rendered),
                "tokens": tokens,
            }
            _require(entry == FROZEN_PROMPT_CONTRACT[key], f"processor contract drift: {key}")
            observed[key] = entry
    return observed


def model_revision(model_path: Path) -> str:
    marker = model_path / ".model-revision"
    if marker.is_file():
        revision = marker.read_text(encoding="utf-8").strip()
    elif model_path.name == MODEL_REVISION:
        revision = MODEL_REVISION
    else:
        tree_dir = model_path / ".cache" / "huggingface" / "trees"
        revisions = sorted(path.stem for path in tree_dir.glob("*.json")) if tree_dir.is_dir() else []
        revision = revisions[0] if len(revisions) == 1 else ""
    _require(revision == MODEL_REVISION, f"model revision mismatch: expected {MODEL_REVISION}, found {revision or 'none'}")
    return revision


class DirectLogitScorer:
    """Exact frozen Transformers evaluator used behind the online service."""

    def __init__(self, model_path: str | Path, device: str = "cuda:0") -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_path = Path(model_path).resolve()
        model_revision(self.model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        verify_processor_contract(self.processor)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.device = device

    def _score_request(self, image: Image.Image, component: str, polarity: str) -> float:
        prompt = prompt_for(component, polarity)
        content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
            preserve_thinking=False,
        )
        inputs = {name: value.to(self.device) if hasattr(value, "to") else value for name, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1]
        tokens = FROZEN_PROMPT_CONTRACT[f"{component}:{polarity}"]["tokens"]
        yes = float(logits[int(tokens["Yes"])].float().cpu())
        no = float(logits[int(tokens["No"])].float().cpu())
        return canonical_margin(polarity, yes, no)

    def score_images(self, images: Sequence[Image.Image]) -> list[dict[str, Any]]:
        _require(bool(images), "empty image group")
        results = []
        for index, source in enumerate(images):
            _require(isinstance(source, Image.Image), f"candidate {index} is not a PIL image")
            image = source.convert("RGB")
            try:
                polarity_scores = {
                    component: {
                        polarity: self._score_request(image, component, polarity)
                        for polarity in POLARITIES
                    }
                    for component in COMPONENTS
                }
            finally:
                image.close()
            aggregate = aggregate_candidate(polarity_scores)
            results.append({"candidate_index": index, "polarity_scores": polarity_scores, **aggregate})
        return results


def _encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_image(value: Any) -> Image.Image:
    _require(isinstance(value, str) and value, "image payload must be nonempty base64")
    try:
        raw = base64.b64decode(value, validate=True)
        with Image.open(io.BytesIO(raw)) as source:
            return source.convert("RGB")
    except Exception as exc:
        raise PhysicalRewardError(f"invalid image payload: {exc}") from exc


def validate_scores(rows: Any, expected_count: int) -> list[dict[str, Any]]:
    _require(isinstance(rows, list) and len(rows) == expected_count,
             f"score count mismatch: expected {expected_count}, got {len(rows) if isinstance(rows, list) else type(rows).__name__}")
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        _require(isinstance(raw, Mapping), f"candidate {index} output is not an object")
        _require(raw.get("candidate_index") == index, f"candidate ordering mismatch at position {index}")
        numeric = {key: float(raw[key]) for key in (*COMPONENTS, "severity", "reward")}
        _require(all(math.isfinite(value) for value in numeric.values()), f"candidate {index} has non-finite output")
        _require(math.isclose(numeric["severity"], (numeric["anatomy"] + numeric["garment_geometry_drape"]) / 2, abs_tol=1e-7),
                 f"candidate {index} severity arithmetic mismatch")
        _require(math.isclose(numeric["reward"], -numeric["severity"], abs_tol=1e-7),
                 f"candidate {index} reward sign mismatch")
        validated.append(dict(raw))
    return validated


class T009PhysicalRewardModel(GroupwiseRewardModel):
    """Groupwise HTTP client for the frozen T009 scorer."""

    required_fields = ("image",)
    use_tensor_inputs = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        extras = config.extra_kwargs
        self.endpoint = str(extras.get("endpoint", "http://127.0.0.1:8890/")).rstrip("/") + "/"
        self.timeout = float(extras.get("timeout", 1800))
        self.expected_group_size = int(extras.get("expected_group_size", 16))
        _require(self.expected_group_size > 0, "expected_group_size must be positive")
        self.telemetry: list[dict[str, Any]] = []
        health = self.health()
        _require(health.get("contract_sha256") == reward_contract()["contract_sha256"], "reward server contract mismatch")

    def health(self) -> dict[str, Any]:
        try:
            response = requests.get(urljoin(self.endpoint, "health"), timeout=min(self.timeout, 10.0))
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise PhysicalRewardError(f"physical reward server health check failed: {exc}") from exc
        _require(isinstance(payload, dict), "physical reward health response is not an object")
        return payload

    def score_images(self, images: Sequence[Image.Image], group_id: str | int, *, enforce_group_size: bool = True) -> list[dict[str, Any]]:
        count = len(images)
        if enforce_group_size:
            _require(count == self.expected_group_size,
                     f"group {group_id} has {count} candidates; expected {self.expected_group_size}")
        _require(count > 0, f"group {group_id} is empty")
        payload = {"group_id": str(group_id), "images": [_encode_image(image) for image in images]}
        started = time.monotonic()
        try:
            response = requests.post(urljoin(self.endpoint, "score"), json=payload, timeout=self.timeout)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            raise PhysicalRewardError(f"group {group_id} scoring failed: {exc}") from exc
        _require(isinstance(body, dict), f"group {group_id} response is not an object")
        rows = validate_scores(body.get("scores"), count)
        telemetry = {
            "group_id": str(group_id),
            "candidate_count": count,
            "client_latency_seconds": time.monotonic() - started,
            "server_latency_seconds": float(body.get("latency_seconds", float("nan"))),
            "model_call_latency_seconds": float(body.get("model_call_latency_seconds", float("nan"))),
            "peak_gpu_memory_bytes": int(body.get("peak_gpu_memory_bytes", 0)),
            "scores": rows,
        }
        _require(math.isfinite(telemetry["server_latency_seconds"]), "server latency is non-finite")
        _require(math.isfinite(telemetry["model_call_latency_seconds"]), "model-call latency is non-finite")
        self.telemetry.append(telemetry)
        return rows

    def __call__(
        self,
        image: list[Image.Image] | None = None,
        **_: Any,
    ) -> RewardModelOutput:
        _require(image is not None, "physical reward requires candidate images")
        call_id = f"rank-{getattr(self.accelerator, 'process_index', 0)}-group-{len(self.telemetry)}"
        rows = self.score_images(image, call_id, enforce_group_size=True)
        rewards = [float(row["reward"]) for row in rows]
        _require(len(rewards) == len(image) and all(math.isfinite(value) for value in rewards),
                 "invalid complete-group rewards")
        return RewardModelOutput(
            rewards=torch.tensor(rewards, dtype=torch.float32),
            extra_info={
                "physical_anatomy": [float(row["anatomy"]) for row in rows],
                "physical_geometry": [float(row["garment_geometry_drape"]) for row in rows],
                "physical_severity": [float(row["severity"]) for row in rows],
            },
        )


class _RewardHTTPServer(ThreadingHTTPServer):
    scorer: DirectLogitScorer
    score_lock: threading.Lock


class _Handler(BaseHTTPRequestHandler):
    server: _RewardHTTPServer

    def _json(self, status: int, payload: Mapping[str, Any]) -> None:
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._json(404, {"error": "not found"})
            return
        contract = reward_contract()
        self._json(200, {"status": "ok", "contract_sha256": contract["contract_sha256"]})

    def do_POST(self) -> None:
        if self.path != "/score":
            self._json(404, {"error": "not found"})
            return
        images: list[Image.Image] = []
        try:
            length = int(self.headers.get("Content-Length", "0"))
            _require(0 < length <= 512 * 1024 * 1024, "invalid request size")
            payload = json.loads(self.rfile.read(length))
            _require(isinstance(payload, dict), "request is not an object")
            encoded = payload.get("images")
            _require(isinstance(encoded, list) and encoded, "request images must be a nonempty list")
            images = [_decode_image(item) for item in encoded]
            started = time.monotonic()
            with self.server.score_lock:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                model_started = time.monotonic()
                scores = self.server.scorer.score_images(images)
                model_latency = time.monotonic() - model_started
                peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            self._json(200, {
                "group_id": str(payload.get("group_id", "")),
                "scores": scores,
                "latency_seconds": time.monotonic() - started,
                "model_call_latency_seconds": model_latency,
                "peak_gpu_memory_bytes": peak,
            })
        except Exception as exc:  # noqa: BLE001 - return one fail-closed response
            self._json(422, {"error": f"{type(exc).__name__}: {exc}"})
        finally:
            for image in images:
                image.close()

    def log_message(self, fmt: str, *args: Any) -> None:
        print(json.dumps({"http": fmt % args}), flush=True)


def serve(model_path: str | Path, host: str, port: int, device: str) -> None:
    server = _RewardHTTPServer((host, port), _Handler)
    server.scorer = DirectLogitScorer(model_path, device=device)
    server.score_lock = threading.Lock()
    print(json.dumps({"status": "ready", "host": host, "port": port,
                      "contract_sha256": reward_contract()["contract_sha256"]}), flush=True)
    server.serve_forever()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    serve(args.model_path, args.host, args.port, args.device)


if __name__ == "__main__":
    main()

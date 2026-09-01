# src/flow_factory/rewards/rational_rewards_vto.py
import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from PIL import Image

from ..hparams import RewardArguments
from ..utils.image import pil_image_to_base64

# Recycle necessary components from existing reward models
from .abc import RewardModelOutput
from .rational_rewards_edit import (
    RationalRewardsEditRewardModel,
    _extract_score_from_block,
    _get_condition_image,
)
from .rational_rewards_t2i import (
    _clip_vlm_text_for_log,
    aggregate_aspect_scores,
)

logger = logging.getLogger(__name__)

# VTO-specific evaluation axes based on "When Rubrics Fail" paper
VTO_SUPPORTED_ASPECTS: Tuple[str, ...] = (
    "garment_transfer",
    "attribute_preservation",
    "physical_quality",
    "source_integrity",
)

RATIONAL_VTO_SYSTEM_PROMPT = (
    "You are an expert virtual try-on (VTO) evaluator. Your task is to evaluate the quality of a generated "
    "VTO image by comparing it directly to a real ground-truth image of the same person wearing the target garment. "
    "Reduce the level of detail and verbosity by 50%. Prioritize brevity and high-level summaries."
)

VTO_TASK_GUIDELINE = """Assess the generated image against the ground truth on four critical aspects.
Provide justifications and absolute scores on a 1-4 scale.

**CRITICAL:** Compress your evaluations toward the center of the 1-4 scale. Avoid assigning the absolute highest (4.0) or lowest (1.0) scores. Map extreme perfection toward 3.5, and extreme failures toward 1.5.

### Scoring Rubric
**1. Garment Transfer Accuracy** (Placement, shape, and structure)
- **4:** Perfect match in garment placement, sleeve/hem length, collar/neckline, and overall shape.
- **3:** Minor deviations in placement or proportions.
- **2:** Noticeable structural errors (e.g., wrong sleeve length).
- **1:** Catastrophic placement or structural failures.

**2. Attribute Preservation** (Colors, patterns, and textures)
- **4:** Colors, dominant patterns, and fabric textures perfectly match the ground truth.
- **3:** Slight color shifts or minor pattern misalignments.
- **2:** Noticeable differences in texture or missing pattern details.
- **1:** Completely different color, pattern, or fabric texture.

**3. Physical and Visual Quality** (Realism, drape, and artifacts)
- **4:** Natural drape, correct shadows, and clean boundaries (no halos, leaks).
- **3:** Small artifacts, slightly unnatural wrinkles.
- **2:** Clear errors like visible seams, halos, bleeding.
- **1:** Severe artifacts, impossible physics, or garbled textures.

**4. Source Integrity** (Preservation of identity, pose, and background)
- **4:** Person's face, hair, skin color, pose, and background perfectly match ground truth.
- **3:** Slight background shift or minor change in hair/expression.
- **2:** Noticeable changes in identity, pose, or background corruption.
- **1:** Completely different person, severely broken pose, or destroyed background.

Output format exactly as follows:
# Detailed Judgement
1. Garment Transfer Accuracy:
## Justification: [ Brief analysis ]
## Score: [ float score ]
2. Attribute Preservation:
## Justification: [ Brief analysis ]
## Score: [ float score ]
3. Physical and Visual Quality:
## Justification: [ Brief analysis ]
## Score: [ float score ]
4. Source Integrity:
## Justification: [ Brief analysis ]
## Score: [ float score ]
# Summary: [ Brief summary ]
"""

VTO_ASPECT_TITLES = {
    "garment_transfer": "Garment Transfer Accuracy",
    "attribute_preservation": "Attribute Preservation",
    "physical_quality": "Physical and Visual Quality",
    "source_integrity": "Source Integrity",
}

VTO_ASPECT_RUBRICS = {
    "garment_transfer": """Assess only garment placement, shape, and structure.
- 4: Perfect match in placement, sleeve/hem length, collar/neckline, and overall shape.
- 3: Minor deviations in placement or proportions.
- 2: Noticeable structural errors, such as the wrong sleeve length.
- 1: Catastrophic placement or structural failures.""",
    "attribute_preservation": """Assess only garment colors, patterns, and textures.
- 4: Colors, dominant patterns, and fabric textures perfectly match the ground truth.
- 3: Slight color shifts or minor pattern misalignments.
- 2: Noticeable texture differences or missing pattern details.
- 1: Completely different color, pattern, or fabric texture.""",
    "physical_quality": """Assess only realism, drape, and visual artifacts.
- 4: Natural drape, correct shadows, and clean boundaries with no halos or leaks.
- 3: Small artifacts or slightly unnatural wrinkles.
- 2: Clear errors such as visible seams, halos, or bleeding.
- 1: Severe artifacts, impossible physics, or garbled textures.""",
    "source_integrity": """Assess only preservation of identity, pose, and background.
- 4: Face, hair, skin color, pose, and background perfectly match the ground truth.
- 3: Slight background shift or a minor change in hair or expression.
- 2: Noticeable identity or pose changes, or background corruption.
- 1: A different person, severely broken pose, or destroyed background.""",
}


def parse_scores_from_detailed_judgement_vto(
    detailed_judgement: str,
) -> Dict[str, Optional[Union[float, str]]]:
    """Parse four VTO aspect scores from the judge reply."""
    result: Dict[str, Optional[Union[float, str]]] = {
        k: None for k in VTO_SUPPORTED_ASPECTS
    }
    content_body = detailed_judgement.split("# Summary:")[0]

    sections = {
        "1. Garment Transfer": "garment_transfer",
        "2. Attribute Preservation": "attribute_preservation",
        "3. Physical and Visual Quality": "physical_quality",
        "4. Source Integrity": "source_integrity",
    }

    current_section = None
    section_blocks = {}

    for line in content_body.split("\n"):
        stripped = line.strip()
        matched = False
        for prefix, key in sections.items():
            if stripped.startswith(prefix):
                current_section = key
                section_blocks[current_section] = [line]
                matched = True
                break

        if not matched and current_section:
            section_blocks[current_section].append(line)

    for key, block_lines in section_blocks.items():
        extracted = _extract_score_from_block("\n".join(block_lines))
        if extracted is not None:
            result[key] = extracted

    return result


def build_scoring_messages_vto(
    prompt: str,
    ground_truth_url: str,
    generated_url: str,
) -> List[dict]:
    return [
        {"role": "system", "content": RATIONAL_VTO_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"User Instruction: {prompt}\n\n1. Ground Truth Image ",
                },
                {"type": "image_url", "image_url": {"url": ground_truth_url}},
                {"type": "text", "text": "\n2. Generated Image "},
                {"type": "image_url", "image_url": {"url": generated_url}},
                {"type": "text", "text": f"\n\n{VTO_TASK_GUIDELINE}"},
            ],
        },
    ]


def build_aspect_scoring_messages_vto(
    prompt: str,
    ground_truth_url: str,
    generated_url: str,
    aspect: str,
) -> List[dict]:
    if aspect not in VTO_SUPPORTED_ASPECTS:
        raise ValueError(f"Unsupported VTO aspect: {aspect!r}")

    title = VTO_ASPECT_TITLES[aspect]
    guideline = f"""Evaluate exactly one aspect: **{title}**.

{VTO_ASPECT_RUBRICS[aspect]}

Compress scores toward the center of the 1-4 scale. Map extreme perfection toward 3.5 and extreme
failure toward 1.5. Do not evaluate or mention the other VTO aspects. Return exactly:
# Judgement
## Justification: [one concise sentence]
## Score: [float score from 1 to 4]"""
    return [
        {
            "role": "system",
            "content": (
                f"{RATIONAL_VTO_SYSTEM_PROMPT} Evaluate only {title}; do not let other quality "
                "dimensions influence the score."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"User Instruction: {prompt}\n\n1. Ground Truth Image ",
                },
                {"type": "image_url", "image_url": {"url": ground_truth_url}},
                {"type": "text", "text": "\n2. Generated Image "},
                {"type": "image_url", "image_url": {"url": generated_url}},
                {"type": "text", "text": f"\n\n{guideline}"},
            ],
        },
    ]


def parse_aspect_judgement_vto(judgement: str) -> Tuple[float, str]:
    score = _extract_score_from_block(judgement)
    if score is None:
        bracketed = re.search(
            r"(?:##\s*)?Score\s*:\s*\[?\s*(\d+(?:\.\d+)?)",
            judgement,
            re.IGNORECASE,
        )
        score = float(bracketed.group(1)) if bracketed else None
    if not isinstance(score, (int, float)):
        raise ValueError(f"missing numeric VTO aspect score: {score!r}")
    score = float(score)
    if not 1.0 <= score <= 4.0:
        raise ValueError(f"VTO aspect score outside [1, 4]: {score}")

    match = re.search(
        r"(?:##\s*)?Justification\s*:\s*(.+?)(?=\n\s*(?:##\s*)?Score\s*:|\Z)",
        judgement,
        re.IGNORECASE | re.DOTALL,
    )
    rationale = " ".join(match.group(1).strip().split()) if match else ""
    return score, rationale


class RationalRewardsVTORewardModel(RationalRewardsEditRewardModel):
    """
    VTO reward using one focused VLM call per selected aspect by default.

    Set ``scoring_mode="joint"`` only for paired evaluation against the original
    four-aspect prompt. Both modes return a normalized reward in [0, 1] and expose
    the raw rubric mean in ``extra_info``.
    """

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        # Intercept config aspects to bypass parent's hardcoded text-edit aspect validation
        raw_aspects = config.extra_kwargs.pop("aspects", None)
        super().__init__(config, accelerator)

        # Re-apply VTO specific aspect validation
        if raw_aspects is None:
            self.aspects = VTO_SUPPORTED_ASPECTS
        else:
            if not isinstance(raw_aspects, (list, tuple)) or not raw_aspects:
                raise TypeError("VTO aspects must be a non-empty list or tuple")
            self.aspects = tuple(str(a) for a in raw_aspects)

        unknown = [a for a in self.aspects if a not in VTO_SUPPORTED_ASPECTS]
        if unknown:
            raise ValueError(
                f"Unsupported VTO aspect(s) {unknown!r}; allowed: {list(VTO_SUPPORTED_ASPECTS)}"
            )

        self.scoring_mode = str(config.extra_kwargs.get("scoring_mode", "split"))
        if self.scoring_mode not in ("split", "joint"):
            raise ValueError("VTO scoring_mode must be 'split' or 'joint'")

    async def _request_completion(
        self,
        client: Any,
        semaphore: asyncio.Semaphore,
        messages: List[dict],
        label: str,
    ) -> Optional[str]:
        from openai import APIConnectionError, APITimeoutError, RateLimitError

        last_err: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                async with semaphore:
                    completion = await client.chat.completions.create(
                        model=self.vlm_model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=self.timeout,
                    )
            except (
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
                asyncio.TimeoutError,
            ) as e:
                last_err = e
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(2**attempt)
                    continue
                break

            content = completion.choices[0].message.content
            if content and str(content).strip():
                return str(content)
            last_err = ValueError("empty assistant content")
            if attempt + 1 < self.max_retries:
                await asyncio.sleep(2**attempt)

        logger.warning("VTO %s request failed. Reward 0.0. Last error: %s", label, last_err)
        return None

    async def _score_single_joint(
        self,
        client: Any,
        semaphore: asyncio.Semaphore,
        prompt: str,
        gt_url: str,
        gen_url: str,
    ) -> Tuple[float, Dict[str, Any]]:
        content = await self._request_completion(
            client,
            semaphore,
            build_scoring_messages_vto(prompt, gt_url, gen_url),
            "joint",
        )
        if content is None:
            return 0.0, {"scoring_mode": "joint"}

        try:
            parsed = parse_scores_from_detailed_judgement_vto(content)
            reward = aggregate_aspect_scores(
                parsed,
                self.aspects,
                supported_aspects=VTO_SUPPORTED_ASPECTS,
            )
            return reward, {
                "scoring_mode": "joint",
                "vlm_content": content,
                "parsed_scores": parsed,
                "aggregate_score": 1.0 + 3.0 * reward,
                "aggregate_reward": reward,
            }
        except (TypeError, ValueError) as e:
            logger.warning(
                "VTO joint parse error. Reward 0.0: %s. Output: %s",
                e,
                _clip_vlm_text_for_log(content),
            )
            return 0.0, {
                "scoring_mode": "joint",
                "vlm_content": content,
                "parsed_scores": {},
            }

    async def _score_aspect(
        self,
        client: Any,
        semaphore: asyncio.Semaphore,
        prompt: str,
        gt_url: str,
        gen_url: str,
        aspect: str,
    ) -> Tuple[str, Optional[float], str, str]:
        content = await self._request_completion(
            client,
            semaphore,
            build_aspect_scoring_messages_vto(prompt, gt_url, gen_url, aspect),
            aspect,
        )
        if content is None:
            return aspect, None, "", ""
        try:
            score, rationale = parse_aspect_judgement_vto(content)
            return aspect, score, rationale, content
        except (TypeError, ValueError) as e:
            logger.warning(
                "VTO %s parse error. Reward 0.0: %s. Output: %s",
                aspect,
                e,
                _clip_vlm_text_for_log(content),
            )
            return aspect, None, "", content

    async def _score_single_split(
        self,
        client: Any,
        semaphore: asyncio.Semaphore,
        prompt: str,
        gt_url: str,
        gen_url: str,
    ) -> Tuple[float, Dict[str, Any]]:
        results = await asyncio.gather(
            *[
                self._score_aspect(
                    client, semaphore, prompt, gt_url, gen_url, aspect
                )
                for aspect in self.aspects
            ]
        )
        parsed_scores = {aspect: score for aspect, score, _, _ in results}
        rationales = {aspect: rationale for aspect, _, rationale, _ in results}
        contents = {aspect: content for aspect, _, _, content in results}
        missing = [aspect for aspect in self.aspects if parsed_scores[aspect] is None]
        extra = {
            "scoring_mode": "split",
            "vlm_contents": contents,
            "parsed_scores": parsed_scores,
            "rationales": rationales,
            "parse_failures": missing,
        }
        if missing:
            logger.warning("VTO split scoring missing aspects %s. Reward 0.0.", missing)
            return 0.0, extra

        reward = aggregate_aspect_scores(
            parsed_scores,
            self.aspects,
            supported_aspects=VTO_SUPPORTED_ASPECTS,
        )
        aggregate_score = sum(float(parsed_scores[a]) for a in self.aspects) / len(
            self.aspects
        )
        extra.update(
            {
                "aggregate_score": aggregate_score,
                "aggregate_reward": reward,
            }
        )
        return reward, extra

    async def _score_single(
        self,
        client: Any,
        semaphore: asyncio.Semaphore,
        prompt: str,
        source: Image.Image,  # Treating condition_images as Ground Truth
        edited: Image.Image,
    ) -> Tuple[float, Dict[str, Any]]:
        gt_url = pil_image_to_base64(source, format="PNG")
        gen_url = pil_image_to_base64(edited, format="PNG")
        if self.scoring_mode == "joint":
            return await self._score_single_joint(
                client, semaphore, prompt, gt_url, gen_url
            )
        return await self._score_single_split(
            client, semaphore, prompt, gt_url, gen_url
        )

    async def _run_batch(
        self,
        prompts: List[str],
        sources: List[Image.Image],
        edited: List[Image.Image],
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Score a batch and return (scores, vlm_extra_infos)."""
        async with self._async_openai_cls(
            base_url=self.api_base_url, api_key=self.api_key
        ) as client:
            semaphore = asyncio.Semaphore(max(1, self.max_concurrent))
            results = await self._async_score_batch(
                client, semaphore, prompts, sources, edited
            )
        scores = [r[0] for r in results]
        extras = [r[1] for r in results]
        return scores, extras

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        if image is None and video is not None:
            image = [frames[0] for frames in video]
        if image is None:
            raise ValueError(
                "Either 'image' or 'video' must be provided for RationalRewardsVTORewardModel"
            )
        if condition_images is None:
            raise ValueError(
                "condition_images (source) is required for RationalRewardsVTORewardModel"
            )
        if len(prompt) != len(image) or len(prompt) != len(condition_images):
            raise ValueError(
                f"expected len(prompt)==len(image)==len(condition_images), got "
                f"{len(prompt)}, {len(image)}, {len(condition_images)}"
            )

        source_images = [
            _get_condition_image(c, self.source_image_index) for c in condition_images
        ]
        scores, vlm_infos = asyncio.run(self._run_batch(prompt, source_images, image))
        rewards = torch.tensor(scores, dtype=torch.float32, device=self.device)
        return RewardModelOutput(
            rewards=rewards, extra_info={"vlm_judgments": vlm_infos}
        )

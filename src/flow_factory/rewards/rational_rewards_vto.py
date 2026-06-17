# src/flow_factory/rewards/rational_rewards_vto.py
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from accelerate import Accelerator
from PIL import Image

from ..hparams import RewardArguments
from ..utils.image import pil_image_to_base64

# Recycle necessary components from existing reward models
from .rational_rewards_edit import (
    RationalRewardsEditRewardModel,
    _extract_score_from_block,
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


class RationalRewardsVTORewardModel(RationalRewardsEditRewardModel):
    """
    Subclasses the Rational Edit reward to swap prompts and parsers for VTO tasks
    while preserving async LLM batching, scaling, and retry logic.
    """

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        # Intercept config aspects to bypass parent's hardcoded text-edit aspect validation
        raw_aspects = config.extra_kwargs.pop("aspects", None)
        super().__init__(config, accelerator)

        # Re-apply VTO specific aspect validation
        if raw_aspects is None:
            self.aspects = VTO_SUPPORTED_ASPECTS
        else:
            self.aspects = tuple(str(a) for a in raw_aspects)

        unknown = [a for a in self.aspects if a not in VTO_SUPPORTED_ASPECTS]
        if unknown:
            raise ValueError(
                f"Unsupported VTO aspect(s) {unknown!r}; allowed: {list(VTO_SUPPORTED_ASPECTS)}"
            )

    async def _score_single(
        self,
        client: Any,
        semaphore: asyncio.Semaphore,
        prompt: str,
        source: Image.Image,  # Treating condition_images as Ground Truth
        edited: Image.Image,
    ) -> float:
        from openai import APIConnectionError, APITimeoutError, RateLimitError

        gt_url = pil_image_to_base64(source, format="PNG")
        gen_url = pil_image_to_base64(edited, format="PNG")
        messages = build_scoring_messages_vto(prompt, gt_url, gen_url)

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
                if attempt + 1 >= self.max_retries:
                    break
                await asyncio.sleep(2**attempt)
                continue

            content = completion.choices[0].message.content
            if not content or not str(content).strip():
                return 0.0

            try:
                parsed = parse_scores_from_detailed_judgement_vto(str(content))
                return aggregate_aspect_scores(
                    parsed,
                    self.aspects,
                    supported_aspects=VTO_SUPPORTED_ASPECTS,
                )
            except (TypeError, ValueError) as e:
                logger.warning(
                    f"VTO Parse error. Reward 0.0: {e}. Output: {_clip_vlm_text_for_log(str(content))}"
                )
                return 0.0

        logger.warning(f"VTO API failed. Reward 0.0. Last error: {last_err}")
        return 0.0

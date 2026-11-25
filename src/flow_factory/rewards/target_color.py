# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/rewards/target_color.py
"""
Target Color Reward Model for RL debugging and smoke testing.

Computes a deterministic, computationally instantaneous reward based on image/video
color statistics (e.g. greenness, redness, blueness, or L1 distance to a target RGB color).
Useful for fast verification of policy gradient flow, trajectory log-prob accumulation,
and group relative advantage computation without reward model noise or external dependencies.
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

from ..hparams import RewardArguments
from .abc import PointwiseRewardModel, RewardModelOutput


class TargetColorRewardModel(PointwiseRewardModel):
    """
    Target Color Reward Model.

    Evaluates generated images or videos against a target color rule.
    By default, computes the "pure green" reward:
        reward = g - 0.5 * (r + b)
    where r, g, b are spatial channel means in [0, 1].

    Supported `target_color` settings (passed via RewardArguments/extra_kwargs):
        - "green" / "pure_green" (default): g - 0.5 * (r + b)
        - "red" / "pure_red": r - 0.5 * (g + b)
        - "blue" / "pure_blue": b - 0.5 * (r + g)
        - "yellow": 0.5 * (r + g) - b
        - "cyan": 0.5 * (g + b) - r
        - "magenta": 0.5 * (r + b) - g
        - "white": (r + g + b) / 3.0
        - "black": -(r + g + b) / 3.0
        - Tuple/List [R, G, B] in [0, 1] or [0, 255]: 1.0 - L1_distance(image_mean, target_rgb)
    """

    required_fields = ("prompt", "image", "video")
    use_tensor_inputs = True

    def __init__(
        self, config: RewardArguments, accelerator: Optional[Accelerator] = None
    ):
        super().__init__(config, accelerator)
        self.target_color = getattr(self.config, "target_color", "green")

    def _extract_rgb_means_from_image(
        self, image: Union[torch.Tensor, list]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(image, torch.Tensor):
            tensor = image.to(self.device).float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)  # (1, C, H, W)
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            elif tensor.min() < 0.0:
                tensor = (tensor + 1.0) / 2.0

            if tensor.shape[1] == 1:
                tensor = tensor.repeat(1, 3, 1, 1)

            r = tensor[:, 0, :, :].mean(dim=[-2, -1])
            g = tensor[:, 1, :, :].mean(dim=[-2, -1])
            b = tensor[:, 2, :, :].mean(dim=[-2, -1])
            return r, g, b

        elif isinstance(image, list):
            r_list, g_list, b_list = [], [], []
            for img in image:
                if isinstance(img, Image.Image):
                    arr = torch.from_numpy(np.array(img.convert("RGB"))).float() / 255.0
                    arr = arr.to(self.device)
                    r_list.append(arr[:, :, 0].mean())
                    g_list.append(arr[:, :, 1].mean())
                    b_list.append(arr[:, :, 2].mean())
                elif isinstance(img, torch.Tensor):
                    t = img.to(self.device).float()
                    if t.ndim == 4 and t.shape[0] == 1:
                        t = t.squeeze(0)
                    if t.max() > 1.0:
                        t = t / 255.0
                    elif t.min() < 0.0:
                        t = (t + 1.0) / 2.0
                    if t.shape[0] == 1:
                        t = t.repeat(3, 1, 1)
                    r_list.append(t[0, :, :].mean())
                    g_list.append(t[1, :, :].mean())
                    b_list.append(t[2, :, :].mean())
                else:
                    raise TypeError(
                        f"Unsupported image element type in list: {type(img)}"
                    )

            r = torch.stack(r_list)
            g = torch.stack(g_list)
            b = torch.stack(b_list)
            return r, g, b
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def _extract_rgb_means_from_video(
        self, video: Union[torch.Tensor, list]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(video, torch.Tensor):
            tensor = video.to(self.device).float()
            if tensor.ndim == 4:
                tensor = tensor.unsqueeze(0)  # (1, T, C, H, W)
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            elif tensor.min() < 0.0:
                tensor = (tensor + 1.0) / 2.0

            if tensor.shape[2] == 1:
                tensor = tensor.repeat(1, 1, 3, 1, 1)

            r = tensor[:, :, 0, :, :].mean(dim=[1, 2, 3])
            g = tensor[:, :, 1, :, :].mean(dim=[1, 2, 3])
            b = tensor[:, :, 2, :, :].mean(dim=[1, 2, 3])
            return r, g, b

        elif isinstance(video, list):
            r_list, g_list, b_list = [], [], []
            for item in video:
                if (
                    isinstance(item, list)
                    and len(item) > 0
                    and isinstance(item[0], Image.Image)
                ):
                    frame_r, frame_g, frame_b = [], [], []
                    for frame in item:
                        arr = (
                            torch.from_numpy(np.array(frame.convert("RGB"))).float()
                            / 255.0
                        )
                        arr = arr.to(self.device)
                        frame_r.append(arr[:, :, 0].mean())
                        frame_g.append(arr[:, :, 1].mean())
                        frame_b.append(arr[:, :, 2].mean())
                    r_list.append(torch.stack(frame_r).mean())
                    g_list.append(torch.stack(frame_g).mean())
                    b_list.append(torch.stack(frame_b).mean())
                elif isinstance(item, torch.Tensor):
                    t = item.to(self.device).float()
                    if t.ndim == 5 and t.shape[0] == 1:
                        t = t.squeeze(0)
                    if t.max() > 1.0:
                        t = t / 255.0
                    elif t.min() < 0.0:
                        t = (t + 1.0) / 2.0
                    if t.shape[1] == 1:
                        t = t.repeat(1, 3, 1, 1)
                    r_list.append(t[:, 0, :, :].mean())
                    g_list.append(t[:, 1, :, :].mean())
                    b_list.append(t[:, 2, :, :].mean())
                else:
                    raise TypeError(
                        f"Unsupported video element type in list: {type(item)}"
                    )

            r = torch.stack(r_list)
            g = torch.stack(g_list)
            b = torch.stack(b_list)
            return r, g, b
        else:
            raise TypeError(f"Unsupported video type: {type(video)}")

    def _extract_rgb_means(
        self,
        image: Optional[Union[torch.Tensor, list]] = None,
        video: Optional[Union[torch.Tensor, list]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image is not None and video is not None:
            raise ValueError("Only one of image or video should be provided.")
        if image is None and video is None:
            raise ValueError("Either image or video must be provided.")

        if image is not None:
            return self._extract_rgb_means_from_image(image)
        else:
            return self._extract_rgb_means_from_video(video)

    def _compute_color_score(
        self, r: torch.Tensor, g: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        color = self.target_color
        if isinstance(color, str):
            color = color.lower().strip()
            if color in ("green", "pure_green"):
                return g - 0.5 * (r + b)
            elif color in ("red", "pure_red"):
                return r - 0.5 * (g + b)
            elif color in ("blue", "pure_blue"):
                return b - 0.5 * (r + g)
            elif color == "yellow":
                return 0.5 * (r + g) - b
            elif color == "cyan":
                return 0.5 * (g + b) - r
            elif color == "magenta":
                return 0.5 * (r + b) - g
            elif color == "white":
                return (r + g + b) / 3.0
            elif color == "black":
                return -(r + g + b) / 3.0
            else:
                raise ValueError(
                    f"Unsupported target_color string '{color}'. Supported names: "
                    "['green', 'red', 'blue', 'yellow', 'cyan', 'magenta', 'white', 'black'] "
                    "or a 3-element tuple/list [R, G, B]."
                )
        elif isinstance(color, (list, tuple, torch.Tensor, np.ndarray)):
            if len(color) != 3:
                raise ValueError(
                    f"target_color sequence must have 3 RGB components, got {color}"
                )
            target_rgb = [float(c) for c in color]
            if any(c > 1.0 for c in target_rgb):
                target_rgb = [c / 255.0 for c in target_rgb]
            rt, gt, bt = target_rgb
            l1_dist = (torch.abs(r - rt) + torch.abs(g - gt) + torch.abs(b - bt)) / 3.0
            return 1.0 - l1_dist
        else:
            raise TypeError(
                f"Invalid target_color type {type(color)}. Expected str or 3-element sequence."
            )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[list[str]] = None,
        image: Optional[Union[torch.Tensor, list]] = None,
        video: Optional[Union[torch.Tensor, list]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        r, g, b = self._extract_rgb_means(image=image, video=video)
        scores = self._compute_color_score(r, g, b)
        return RewardModelOutput(rewards=scores, extra_info={})

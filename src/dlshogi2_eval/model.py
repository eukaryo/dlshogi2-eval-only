from __future__ import annotations

"""Policy/value network extracted from python-dlshogi2."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import FEATURES_NUM, MOVE_LABELS_NUM, MOVE_PLANES_NUM


class Bias(nn.Module):
    def __init__(self, shape: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.bias


class ResNetBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(out + x)


class PolicyValueNetwork(nn.Module):
    """Reference policy/value CNN used by python-dlshogi2.

    Output contract:
      - policy: [N, MOVE_LABELS_NUM] dense logits
      - value:  [N, 1] raw value logit
    """

    def __init__(self, blocks: int = 10, channels: int = 192, fcl: int = 256) -> None:
        super().__init__()
        self.blocks_count = blocks
        self.channels = channels
        self.fcl = fcl

        self.conv1 = nn.Conv2d(
            in_channels=FEATURES_NUM,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(channels)
        self.blocks = nn.Sequential(*[ResNetBlock(channels) for _ in range(blocks)])

        self.policy_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=MOVE_PLANES_NUM,
            kernel_size=1,
            bias=False,
        )
        self.policy_bias = Bias(MOVE_LABELS_NUM)

        self.value_conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=MOVE_PLANES_NUM,
            kernel_size=1,
            bias=False,
        )
        self.value_norm1 = nn.BatchNorm2d(MOVE_PLANES_NUM)
        self.value_fc1 = nn.Linear(MOVE_LABELS_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = F.relu(self.norm1(x))
        x = self.blocks(x)

        policy = self.policy_conv(x)
        policy = self.policy_bias(torch.flatten(policy, 1))

        value = F.relu(self.value_norm1(self.value_conv1(x)))
        value = F.relu(self.value_fc1(torch.flatten(value, 1)))
        value = self.value_fc2(value)
        return policy, value

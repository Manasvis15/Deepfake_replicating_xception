"""Xception backbone, identical in structure and parameter names to DeepfakeBench's
training/networks/xception.py (mode='original'), so xception_best.pth loads strictly.
"""

import torch
import torch.nn.functional as F
from torch import nn


class SeparableConv2d(nn.Module):
    """Depthwise k x k conv (one filter per channel) followed by a 1x1 pointwise conv."""

    def __init__(
        self,
        cin: int,
        cout: int,
        k: int = 1,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cin, k, stride, pad, dilation, groups=cin, bias=bias)
        self.pointwise = nn.Conv2d(cin, cout, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.conv1(x))


class Block(nn.Module):
    """Residual unit: [ReLU -> SepConv -> BN] x reps (+ MaxPool if strided) + skip."""

    def __init__(
        self,
        cin: int,
        cout: int,
        reps: int,
        stride: int = 1,
        start_with_relu: bool = True,
        grow_first: bool = True,
    ) -> None:
        super().__init__()
        if cout != cin or stride != 1:
            self.skip: nn.Module | None = nn.Conv2d(cin, cout, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(cout)
        else:
            self.skip = None

        relu = nn.ReLU(inplace=True)
        rep: list[nn.Module] = []
        filters = cin
        if grow_first:
            rep += [relu, SeparableConv2d(cin, cout, 3, 1, 1), nn.BatchNorm2d(cout)]
            filters = cout
        for _ in range(reps - 1):
            rep += [relu, SeparableConv2d(filters, filters, 3, 1, 1), nn.BatchNorm2d(filters)]
        if not grow_first:
            rep += [relu, SeparableConv2d(cin, cout, 3, 1, 1), nn.BatchNorm2d(cout)]

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)  # must not overwrite the input used by the skip
        if stride != 1:
            rep.append(nn.MaxPool2d(3, stride, 1))
        # Positions in this Sequential are part of the checkpoint key names.
        self.rep = nn.Sequential(*rep)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.rep(inp)
        skip = self.skipbn(self.skip(inp)) if self.skip is not None else inp
        return x + skip


class Xception(nn.Module):
    def __init__(self, num_classes: int = 2, inc: int = 3) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        # Entry flow
        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2)
        self.block3 = Block(256, 728, 2, 2)
        # Middle flow: block4 ... block11
        for i in range(4, 12):
            setattr(self, f"block{i}", Block(728, 728, 3, 1))
        # Exit flow
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.last_linear = nn.Linear(2048, num_classes)
        # Unused in 'original' mode but present in the checkpoint; kept for strict loading.
        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=False)
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))  # [B,   32, 127, 127]
        x = self.relu(self.bn2(self.conv2(x)))  # [B,   64, 125, 125]
        for i in range(1, 13):
            x = getattr(self, f"block{i}")(x)  # -> [B, 1024, 8, 8]
        x = self.relu(self.bn3(self.conv3(x)))  # [B, 1536, 8, 8]
        return self.bn4(self.conv4(x))  # [B, 2048, 8, 8]

    def classifier(self, feat: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(self.relu(feat), 1).flatten(1)  # [B, 2048]
        return self.last_linear(x)  # [B, 2] logits

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        return self.classifier(feat), feat

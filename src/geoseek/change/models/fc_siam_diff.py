"""FC-Siam-diff: fully-convolutional Siamese change-detection network.

Reference architecture: Daudt, Le Saux, Boulch, *"Fully Convolutional Siamese
Networks for Change Detection"*, ICIP 2018 (and the OSCD paper, IGARSS 2018).

Structure
---------
* A single convolutional **encoder** whose weights are **shared** across the two
  dates (a true Siamese branch - the same ``nn.Module`` objects process date 1
  and date 2).
* At every resolution level the two encoded feature maps are combined by
  **feature differencing** and handed to the decoder as that level's skip
  connection.
* A **U-Net style decoder**: transposed-conv upsampling, concatenate the
  differenced skip, two 3x3 convs.
* A final 1x1 conv producing **one change logit per pixel** (binary change).

Design choice vs the original paper: the skip / bottleneck combination here is
the **absolute** difference ``|f1 - f2|`` rather than the signed ``f1 - f2``.
That makes the whole network invariant to the order of the two dates - a change
is a change whether the pixel went bare->built or built->bare - which is the
right inductive bias for OSCD's binary label and for feeding it a
``TemporalObservationMatcher`` pair without worrying which observation is
"earlier". :func:`FCSiamDiff.forward` is symmetric: ``forward(a, b)`` and
``forward(b, a)`` return identical logits.

Kept deliberately small (~1-2M parameters, printed by
:func:`count_parameters`) so it trains from scratch on the 14-pair OSCD train
split on an 8 GB GPU and transfers to geoseek's 5-band Ayodhya data.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# geoseek production bands, in the channel order the model expects.
DEFAULT_BANDS = ("B02", "B03", "B04", "B08", "B11")


def count_parameters(module: nn.Module, *, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


class _ConvBlock(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x2, same spatial size."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FCSiamDiff(nn.Module):
    """Siamese FCN change detector. ``forward(x1, x2) -> (B, 1, H, W)`` change logits.

    Args:
        in_channels: spectral bands per date (5 for geoseek: B02/B03/B04/B08/B11).
        base_channels: encoder width at the top level; doubles each level.
        depth: number of encoder levels (``depth - 1`` pool / upsample steps).
        dropout: Dropout2d after each conv block (0 disables).
    """

    def __init__(
        self,
        in_channels: int = 5,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depth = depth
        self._stride = 2 ** (depth - 1)  # input H,W must be a multiple of this (padded internally)

        chs = [base_channels * (2 ** i) for i in range(depth)]  # e.g. [32, 64, 128, 256]
        self.channels = chs

        # shared Siamese encoder
        self.enc_blocks = nn.ModuleList()
        prev = in_channels
        for i, c in enumerate(chs):
            self.enc_blocks.append(_ConvBlock(prev, c, dropout if i > 0 else 0.0))
            prev = c
        self.pool = nn.MaxPool2d(2)

        # U-Net decoder over the differenced skips
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.upconvs.append(nn.ConvTranspose2d(chs[i + 1], chs[i], kernel_size=2, stride=2))
            self.dec_blocks.append(_ConvBlock(chs[i] * 2, chs[i], dropout if i > 0 else 0.0))

        self.classifier = nn.Conv2d(chs[0], 1, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # -- encoder (shared) --------------------------------------------------

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return the per-level feature maps (pre-pool), deepest last."""
        feats: list[torch.Tensor] = []
        h = x
        for i, block in enumerate(self.enc_blocks):
            h = block(h)
            feats.append(h)
            if i < self.depth - 1:
                h = self.pool(h)
        return feats

    # -- forward ---------------------------------------------------------

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if x1.shape != x2.shape:
            raise ValueError(f"x1 {tuple(x1.shape)} and x2 {tuple(x2.shape)} must match")
        _, _, h0, w0 = x1.shape
        ph = (self._stride - h0 % self._stride) % self._stride
        pw = (self._stride - w0 % self._stride) % self._stride
        if ph or pw:
            x1 = F.pad(x1, (0, pw, 0, ph), mode="reflect")
            x2 = F.pad(x2, (0, pw, 0, ph), mode="reflect")

        f1 = self.encode(x1)
        f2 = self.encode(x2)
        # order-invariant combination at every level
        diffs = [torch.abs(a - b) for a, b in zip(f1, f2)]

        d = diffs[-1]  # bottleneck difference
        for k, (up, dec) in enumerate(zip(self.upconvs, self.dec_blocks)):
            d = up(d)
            skip = diffs[self.depth - 2 - k]
            # transposed-conv can be a pixel off vs the skip on odd sizes - align
            if d.shape[-2:] != skip.shape[-2:]:
                d = F.interpolate(d, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            d = dec(torch.cat([d, skip], dim=1))

        logits = self.classifier(d)
        if ph or pw:
            logits = logits[..., :h0, :w0]
        return logits

    # -- convenience ---------------------------------------------------

    @torch.no_grad()
    def predict_proba(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x1, x2))

    def num_parameters(self) -> int:
        return count_parameters(self)

    def extra_repr(self) -> str:
        return (f"in_channels={self.in_channels}, base_channels={self.base_channels}, "
                f"depth={self.depth}, channels={self.channels}, params={self.num_parameters():,}")

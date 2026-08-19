"""Masked policy/value network (encoding_v2, catalog v1).

Default architecture ``pvn_v1`` (Phase 3.3): sheet MLP + 1D CNN over dice +
context MLP, fused into policy and value heads. The older flat ``mlp`` trunk is
kept so encoding-matched mlp checkpoints still load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from doppelt.actions.catalog_v1 import ACTION_SPACE_SIZE, CATALOG_VERSION
from doppelt.actions.space import ILLEGAL_LOGIT
from doppelt.ml.encoding import (
    CONTEXT_SIZE,
    DICE_CHANNELS,
    DICE_COUNT,
    ENCODING_VERSION,
    FEATURE_SIZE,
    GLOBAL_SLICE,
    SHEET_SIZE,
    SHEET_SLICE,
    DICE_SLICE,
)

DEFAULT_HIDDEN = 256
DEFAULT_ARCHITECTURE = "pvn_v1"
ARCHITECTURES = ("pvn_v1", "mlp")
MIN_PARAM_TARGET = 100_000
MAX_PARAM_TARGET = 1_000_000


def require_torch():
    """Import torch or raise a CLI-friendly error."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError(
            'PyTorch is required for training. Install with: pip install -e ".[ml]"'
        ) from exc
    return torch, nn, F


def infer_architecture(state_dict: dict) -> str:
    """Detect mlp vs pvn_v1 from weight names (older files may tag the wrong arch)."""
    keys = list(state_dict)
    if any(key.startswith("sheet_encoder.") for key in keys):
        return "pvn_v1"
    if any(key.startswith("backbone.") for key in keys):
        return "mlp"
    raise ValueError("cannot infer architecture from checkpoint state_dict")


def count_parameters(net) -> int:
    return sum(int(param.numel()) for param in net.parameters() if param.requires_grad)


def build_net(hidden: int = DEFAULT_HIDDEN, architecture: str = DEFAULT_ARCHITECTURE):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {architecture!r}; expected one of {ARCHITECTURES}")
    torch, nn, _F = require_torch()
    if architecture == "mlp":
        return _build_mlp(nn, hidden)
    return _build_pvn_v1(torch, nn, hidden)


def _build_mlp(nn, hidden: int):
    class FlatMlpNet(nn.Module):
        architecture = "mlp"

        def __init__(self) -> None:
            super().__init__()
            self.hidden = hidden
            self.backbone = nn.Sequential(
                nn.Linear(FEATURE_SIZE, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.policy_head = nn.Linear(hidden, ACTION_SPACE_SIZE)
            self.value_head = nn.Linear(hidden, 1)

        def forward(self, features, mask):
            trunk = self.backbone(features)
            logits = self.policy_head(trunk).masked_fill(~mask, ILLEGAL_LOGIT)
            value = self.value_head(trunk).squeeze(-1)
            return logits, value

    return FlatMlpNet()


def _build_pvn_v1(torch, nn, hidden: int):
    sheet_width = max(hidden // 2, 16)
    dice_width = max(hidden // 4, 8)
    context_width = max(hidden // 4, 8)
    conv_channels = max(hidden // 8, 8)

    class PolicyValueNetV1(nn.Module):
        architecture = "pvn_v1"

        def __init__(self) -> None:
            super().__init__()
            self.hidden = hidden
            self.sheet_encoder = nn.Sequential(
                nn.Linear(SHEET_SIZE, sheet_width),
                nn.ReLU(),
                nn.Linear(sheet_width, sheet_width),
                nn.ReLU(),
            )
            self.dice_conv = nn.Sequential(
                nn.Conv1d(DICE_CHANNELS, conv_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.dice_proj = nn.Sequential(
                nn.Linear(conv_channels * DICE_COUNT, dice_width),
                nn.ReLU(),
            )
            self.context_encoder = nn.Sequential(
                nn.Linear(CONTEXT_SIZE, context_width),
                nn.ReLU(),
            )
            fused = sheet_width + dice_width + context_width
            self.trunk = nn.Sequential(
                nn.Linear(fused, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.policy_head = nn.Linear(hidden, ACTION_SPACE_SIZE)
            self.value_head = nn.Linear(hidden, 1)

        def forward(self, features, mask):
            sheet = self.sheet_encoder(features[:, SHEET_SLICE])
            dice = features[:, DICE_SLICE].reshape(-1, DICE_COUNT, DICE_CHANNELS)
            dice = self.dice_conv(dice.transpose(1, 2)).flatten(1)
            dice = self.dice_proj(dice)
            context = self.context_encoder(features[:, GLOBAL_SLICE.start :])
            hidden_out = self.trunk(torch.cat((sheet, dice, context), dim=-1))
            logits = self.policy_head(hidden_out).masked_fill(~mask, ILLEGAL_LOGIT)
            value = self.value_head(hidden_out).squeeze(-1)
            return logits, value

    return PolicyValueNetV1()


def save_checkpoint(
    path: Path,
    net,
    *,
    hidden: int,
    architecture: str = DEFAULT_ARCHITECTURE,
    extra: dict[str, Any] | None = None,
) -> None:
    torch, _nn, _F = require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "encoding_version": ENCODING_VERSION,
        "catalog_version": CATALOG_VERSION,
        "feature_size": FEATURE_SIZE,
        "action_space": ACTION_SPACE_SIZE,
        "architecture": architecture,
        "hidden": hidden,
        "model_state": net.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: Path, *, map_location: str = "cpu"):
    torch, _nn, _F = require_torch()
    try:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(Path(path), map_location=map_location)
    if payload.get("encoding_version") != ENCODING_VERSION:
        raise ValueError(
            f"checkpoint encoding v{payload.get('encoding_version')} != v{ENCODING_VERSION}"
        )
    if payload.get("catalog_version") != CATALOG_VERSION:
        raise ValueError(
            f"checkpoint catalog v{payload.get('catalog_version')} != v{CATALOG_VERSION}"
        )
    hidden = int(payload.get("hidden", DEFAULT_HIDDEN))
    state = payload["model_state"]
    architecture = infer_architecture(state)
    tagged = payload.get("architecture")
    if tagged not in (None, architecture):
        payload = dict(payload)
        payload["architecture"] = architecture
    net = build_net(hidden=hidden, architecture=architecture)
    net.load_state_dict(state)
    net.eval()
    return net, payload

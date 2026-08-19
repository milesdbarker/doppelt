"""Policy that selects catalog IDs from a trained PolicyValueNet."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from doppelt.actions.catalog_v1 import ACTION_SPACE_SIZE
from doppelt.core.phases import Phase
from doppelt.core.scoring import total_score
from doppelt.core.state import GameState
from doppelt.engine.game import legal_action_ids
from doppelt.ml.encoding import SCORE_SCALE, encode_state
from doppelt.ml.model import load_checkpoint, require_torch


class NeuralPolicy:
    """Greedy, sampled, or PUCT-search policy over catalog v1."""

    name = "neural"

    def __init__(
        self,
        checkpoint: Path | None = None,
        seed: int = 0,
        *,
        sample: bool = False,
        mcts_sims: int = 0,
        device: str = "cpu",
        net: Any = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        torch, _nn, _F = require_torch()
        self._torch = torch
        self._rng_seed = seed
        self.sample = sample
        self.mcts_sims = mcts_sims
        self.device = device
        if net is None:
            if checkpoint is None:
                raise ValueError("NeuralPolicy requires a checkpoint path or an in-memory net")
            net, payload = load_checkpoint(checkpoint, map_location=device)
        self._payload = payload or {}
        self._net = net.to(device)
        self._net.eval()
        self.score_scale = float(self._payload.get("score_scale", SCORE_SCALE))
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)
        self._search_rng = random.Random(seed ^ 0x4C75)

    def _forward(self, state: GameState, legal: list[int]):
        torch = self._torch
        encoded = encode_state(state, legal)
        features = torch.tensor([encoded.features], dtype=torch.float32, device=self.device)
        mask = torch.tensor([encoded.mask], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits, value = self._net(features, mask)
        return logits, float(value.item())

    def _leaf_value(self, state: GameState) -> float:
        if state.phase is Phase.GAME_OVER:
            return total_score(state.sheet) / self.score_scale
        legal = legal_action_ids(state)
        if not legal:
            return total_score(state.sheet) / self.score_scale
        _logits, value = self._forward(state, legal)
        return value

    def select(self, state: GameState, legal: list[int]) -> int:
        if not legal:
            raise ValueError("NeuralPolicy received no legal actions")
        if len(legal) == 1:
            return legal[0]
        torch = self._torch
        logits, _value = self._forward(state, legal)
        if self.mcts_sims > 0:
            from doppelt.ml.mcts import puct_select

            probs = torch.softmax(logits.cpu(), dim=-1).squeeze(0)
            priors = {action_id: float(probs[action_id].item()) for action_id in legal}
            return puct_select(
                state,
                legal,
                priors,
                self._leaf_value,
                self._search_rng,
                n_sims=self.mcts_sims,
            )
        if self.sample:
            probs = torch.softmax(logits.cpu(), dim=-1)
            return int(torch.multinomial(probs, 1, generator=self._generator).item())
        return int(torch.argmax(logits, dim=-1).item())

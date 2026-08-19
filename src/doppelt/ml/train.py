"""Behavioral cloning training loop."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from doppelt.ml.bc_data import BCExample
from doppelt.ml.model import build_net, require_torch, save_checkpoint


@dataclass(frozen=True)
class TrainResult:
    checkpoint_path: Path
    epochs: int
    best_val_loss: float
    train_loss: float
    n_train: int
    n_val: int


def _to_tensors(examples: Sequence[BCExample], torch):
    features = torch.tensor([ex.features for ex in examples], dtype=torch.float32)
    mask = torch.tensor([ex.mask for ex in examples], dtype=torch.bool)
    actions = torch.tensor([ex.action_id for ex in examples], dtype=torch.long)
    values = torch.tensor([ex.return_score for ex in examples], dtype=torch.float32)
    return features, mask, actions, values


def _epoch_loss(net, loader, torch, F, *, optimizer=None, value_weight: float = 0.5) -> float:
    total = 0.0
    count = 0
    train = optimizer is not None
    net.train(train)
    for features, mask, actions, values in loader:
        logits, pred_value = net(features, mask)
        policy_loss = F.nll_loss(F.log_softmax(logits, dim=-1), actions)
        value_loss = F.mse_loss(pred_value, values)
        loss = policy_loss + value_weight * value_loss
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch = features.size(0)
        total += float(loss.item()) * batch
        count += batch
    return total / count if count else 0.0


def train_bc(
    train_examples: Sequence[BCExample],
    val_examples: Sequence[BCExample],
    *,
    out_path: Path,
    epochs: int = 8,
    batch_size: int = 64,
    hidden: int = 256,
    architecture: str = "pvn_v1",
    lr: float = 1e-3,
    value_weight: float = 0.5,
    seed: int = 0,
) -> TrainResult:
    """Train a masked policy/value net to clone labeled actions."""
    if not train_examples:
        raise ValueError("no training examples")
    torch, _nn, F = require_torch()
    torch.manual_seed(seed)

    net = build_net(hidden=hidden, architecture=architecture)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train_tensors = _to_tensors(train_examples, torch)
    train_ds = torch.utils.data.TensorDataset(*train_tensors)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=min(batch_size, len(train_examples)), shuffle=True
    )
    val_loader = None
    if val_examples:
        val_tensors = _to_tensors(val_examples, torch)
        val_ds = torch.utils.data.TensorDataset(*val_tensors)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=min(batch_size, len(val_examples)), shuffle=False
        )

    best_val = float("inf")
    last_train = 0.0
    for epoch in range(epochs):
        last_train = _epoch_loss(
            net, train_loader, torch, F, optimizer=optimizer, value_weight=value_weight
        )
        if val_loader is None:
            best_val = last_train
            save_checkpoint(
                out_path,
                net,
                hidden=hidden,
                architecture=architecture,
                extra={"epoch": epoch + 1, "train_loss": last_train, "val_loss": last_train},
            )
            continue
        val_loss = _epoch_loss(net, val_loader, torch, F, optimizer=None, value_weight=value_weight)
        if val_loss <= best_val:
            best_val = val_loss
            save_checkpoint(
                out_path,
                net,
                hidden=hidden,
                architecture=architecture,
                extra={"epoch": epoch + 1, "train_loss": last_train, "val_loss": val_loss},
            )

    if best_val == float("inf"):
        save_checkpoint(
            out_path,
            net,
            hidden=hidden,
            architecture=architecture,
            extra={"epoch": epochs, "train_loss": last_train},
        )
        best_val = last_train

    return TrainResult(
        checkpoint_path=Path(out_path),
        epochs=epochs,
        best_val_loss=best_val,
        train_loss=last_train,
        n_train=len(train_examples),
        n_val=len(val_examples),
    )


def format_train_result(result: TrainResult) -> str:
    return (
        f"BC train: {result.n_train} examples, val {result.n_val}, "
        f"epochs={result.epochs}, train_loss={result.train_loss:.4f}, "
        f"best_val_loss={result.best_val_loss:.4f}\n"
        f"checkpoint: {result.checkpoint_path}"
    )

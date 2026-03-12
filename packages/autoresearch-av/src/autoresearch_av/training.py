from __future__ import annotations

import json
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class TrainingArtifacts:
    train_seconds: float
    best_val_metric: float
    best_epoch: int
    peak_memory_mb: float
    checkpoint_path: Path
    early_stopped: bool


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_params_m(model: torch.nn.Module) -> float:
    return sum(param.numel() for param in model.parameters()) / 1_000_000.0


def peak_memory_mb(device: torch.device) -> float:
    import torch

    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    if device.type == "mps":
        current = torch.mps.current_allocated_memory()
        driver = torch.mps.driver_allocated_memory()
        return max(current, driver) / (1024 * 1024)
    return 0.0


def reset_peak_memory(device: torch.device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def save_json_log(log_path: Path, payload: dict) -> None:
    log_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_text_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")
    print(message, flush=True)


def append_json_log(log_path: Path, payload: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def maybe_autocast(device: torch.device):
    import torch

    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_classifier(
    *,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    epochs: int,
    device: torch.device,
    checkpoint_path: Path,
    patience: int = 0,
    progress_callback: Callable[[str], None] | None = None,
) -> TrainingArtifacts:
    import torch

    model.to(device)
    reset_peak_memory(device)
    best_val_metric = 0.0
    best_epoch = 0
    stale_epochs = 0
    early_stopped = False
    started = time.perf_counter()
    if progress_callback:
        progress_callback(f"starting training on {device} for {epochs} epochs")

    for epoch_index in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with maybe_autocast(device):
                logits = model(inputs)
                loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1

        val_metric = evaluate_classifier(model=model, data_loader=val_loader, device=device)
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        elapsed = time.perf_counter() - started
        if progress_callback:
            progress_callback(
                f"epoch {epoch_index + 1}/{epochs} train_loss={avg_loss:.4f} val_metric={val_metric:.4f} elapsed={elapsed:.1f}s",
            )
        if val_metric >= best_val_metric:
            best_val_metric = val_metric
            best_epoch = epoch_index + 1
            stale_epochs = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            if progress_callback:
                progress_callback(f"checkpoint updated val_metric={best_val_metric:.4f}")
        else:
            stale_epochs += 1
            if patience > 0 and stale_epochs >= patience and epoch_index + 1 < epochs:
                early_stopped = True
                if progress_callback:
                    progress_callback(
                        f"early stopping triggered at epoch {epoch_index + 1}/{epochs} "
                        f"best_epoch={best_epoch} best_val_metric={best_val_metric:.4f}",
                    )
                break

    finished = time.perf_counter()
    if progress_callback:
        progress_callback(
            f"training completed in {finished - started:.1f}s "
            f"best_epoch={best_epoch} best_val_metric={best_val_metric:.4f} "
            f"early_stopped={str(early_stopped).lower()}",
        )
    return TrainingArtifacts(
        train_seconds=finished - started,
        best_val_metric=best_val_metric,
        best_epoch=best_epoch,
        peak_memory_mb=peak_memory_mb(device),
        checkpoint_path=checkpoint_path,
        early_stopped=early_stopped,
    )


def evaluate_classifier(*, model: torch.nn.Module, data_loader, device: torch.device) -> float:
    import torch

    @torch.no_grad()
    def _evaluate() -> float:
        model.eval()
        correct = 0
        total = 0
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == targets).sum().item())
            total += int(targets.numel())
        return correct / total if total > 0 else 0.0

    return _evaluate()

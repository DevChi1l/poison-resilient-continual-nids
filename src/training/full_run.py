"""Durable epoch-boundary training for the disk-backed Transformer path."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Callable

import numpy as np

from .disk_backed import DiskBackedFlowDataset


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _save_torch(path: Path, value: dict, torch: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _cpu_state(network: Any) -> dict:
    return {name: tensor.detach().cpu().clone() for name, tensor in network.state_dict().items()}


def _runtime_signature(torch: Any, model: Any) -> dict:
    return {
        "torch": torch.__version__, "device_type": model.device.type,
        "cuda_build": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else None,
        "cuda_device_count": torch.cuda.device_count() if model.device.type == "cuda" else 0,
    }


def _resume_header(config: dict, class_ids: tuple[int, ...],
                   data_identity: dict, runtime: dict) -> dict:
    return {"format_version": 1, "config": config, "class_ids": list(class_ids),
            "data_identity": data_identity, "runtime": runtime}


def validate_resume_header(saved: dict, expected: dict) -> None:
    """Reject mismatched config, data/preprocessor identity, and runtime."""

    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f"Resume checkpoint {key} does not match this run")


def _rng_snapshot(torch: Any) -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(torch: Any, saved: dict, device_type: str) -> None:
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch_cpu"])
    if device_type == "cuda":
        cuda_states = saved.get("torch_cuda")
        if cuda_states is None or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count changed; exact resume is unavailable")
        torch.cuda.set_rng_state_all(cuda_states)


def _sync(torch: Any, model: Any) -> None:
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)


def _best_checkpoint(model: Any, best_state: dict, best_epoch: int,
                     identity: dict, config: dict) -> dict:
    return {"config": config, "class_ids": list(model.class_ids),
            "state_dict": best_state, "best_epoch": best_epoch,
            "data_identity": identity}


def train_full_disk_backed(
    model: Any,
    train: DiskBackedFlowDataset,
    validation: DiskBackedFlowDataset,
    *,
    output_dir: str | Path,
    data_identity: dict,
    resume: bool = False,
    max_epochs_this_call: int | None = None,
    progress_every_batches: int = 1000,
    report: Callable[[str], None] = print,
) -> dict:
    """Train full partitions with atomic latest/best checkpoints each epoch.

    Latest is authoritative. It includes model, optimizer, scaler, Python/
    NumPy/Torch CPU/CUDA RNG, history, completed epoch, best weights, and
    patience. A torn best checkpoint is regenerated from latest on resume.
    No mid-epoch resume is claimed: an interrupted epoch repeats from its
    previous completed boundary.
    """

    import torch

    if train.partition != "train" or validation.partition != "validation":
        raise ValueError("Training and validation views must use their own partitions")
    if train.class_ids != model.class_ids or validation.class_ids != model.class_ids:
        raise ValueError("Model and disk views must have identical external class IDs")
    if train.features.shape[1] != model.config.num_features:
        raise ValueError("Feature width does not match model configuration")
    if not isinstance(data_identity, dict) or not data_identity:
        raise ValueError("data_identity must identify the prepared input and preprocessing")
    if progress_every_batches < 1 or (max_epochs_this_call is not None and max_epochs_this_call < 1):
        raise ValueError("Progress interval and optional epoch limit must be positive")

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    latest_path = folder / "latest.pt"
    best_path = folder / "best.pt"
    history_path = folder / "history.json"
    manifest_path = folder / "training_manifest.json"
    config = asdict(model.config)
    runtime = _runtime_signature(torch, model)
    header = _resume_header(config, model.class_ids, data_identity, runtime)
    optimizer = torch.optim.AdamW(model.network.parameters(), lr=model.config.learning_rate,
                                  weight_decay=model.config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    use_amp = model.config.mixed_precision and model.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    seed = model.config.seed if model.config.sampler_seed is None else model.config.sampler_seed
    columns = {class_id: index for index, class_id in enumerate(model.class_ids)}

    if resume:
        if not latest_path.is_file():
            raise FileNotFoundError("Resume requested but latest.pt is absent")
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        validate_resume_header(checkpoint, header)
        model.network.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint["history"]
        completed = int(checkpoint["completed_epoch"])
        best_loss = float(checkpoint["best_validation_loss"])
        best_epoch = checkpoint["best_epoch"]
        best_state = checkpoint["best_state_dict"]
        stale = int(checkpoint["stale_epochs"])
        early_stopped = bool(checkpoint["early_stopped"])
        if len(history["epochs"]) != completed or history["optimizer_steps"] < completed:
            raise ValueError("Resume checkpoint history is inconsistent")
        _restore_rng(torch, checkpoint["rng"], model.device.type)
        report(f"Resumed exact completed epoch {completed}; optimizer and RNG restored")
        # Latest is authoritative; also repairs a crash between the two saves.
        if best_state is not None:
            _save_torch(best_path, _best_checkpoint(model, best_state, best_epoch,
                                                    data_identity, config), torch)
    else:
        if latest_path.exists() or best_path.exists():
            raise FileExistsError("Training checkpoints exist; request resume explicitly")
        history = {"epochs": [], "optimizer_steps": 0, "training_rows_processed": 0,
                   "validation_rows_processed": 0}
        completed = 0
        best_loss = float("inf")
        best_epoch = None
        best_state = None
        stale = 0
        early_stopped = False

    target = model.config.epochs
    end_epoch = min(target, completed + max_epochs_this_call) if max_epochs_this_call else target
    for epoch in range(completed, end_epoch):
        if early_stopped:
            break
        model.network.train()
        _sync(torch, model)
        train_start = perf_counter()
        loss_sum = 0.0
        actual_rows = 0
        steps = 0
        for x, labels, _ in train.iter_epoch(batch_size=model.config.batch_size,
                                             seed=seed, epoch=epoch,
                                             mode=model.config.training_sampler):
            encoded = np.fromiter((columns[int(value)] for value in labels),
                                  dtype=np.int64, count=len(labels))
            inputs = torch.as_tensor(x, dtype=torch.float32, device=model.device)
            targets = torch.as_tensor(encoded, dtype=torch.long, device=model.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=model.device.type, enabled=use_amp):
                loss = criterion(model.network(inputs), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; no epoch checkpoint written")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().item()) * len(labels)
            actual_rows += len(labels)
            steps += 1
            if steps % progress_every_batches == 0 or actual_rows == len(train):
                _sync(torch, model)
                elapsed = perf_counter() - train_start
                rate = actual_rows / elapsed
                report(f"epoch {epoch + 1}/{target} batch {steps}: "
                       f"{actual_rows:,}/{len(train):,} train rows; "
                       f"{rate:,.0f} rows/s; measured train remaining "
                       f"{(len(train) - actual_rows) / rate / 60:.1f} min")
        _sync(torch, model)
        train_seconds = perf_counter() - train_start
        if actual_rows != len(train):
            raise RuntimeError("Training epoch did not cover the full selected partition")

        model.network.eval()
        _sync(torch, model)
        val_start = perf_counter()
        validation_rows = 0
        val_loss_sum = 0.0
        val_correct = 0
        with torch.no_grad():
            for start in range(0, len(validation), model.config.batch_size):
                x, labels, _ = validation.batch(
                    validation.indices[start:start + model.config.batch_size])
                encoded = np.fromiter((columns[int(value)] for value in labels),
                                      dtype=np.int64, count=len(labels))
                inputs = torch.as_tensor(x, dtype=torch.float32, device=model.device)
                targets = torch.as_tensor(encoded, dtype=torch.long, device=model.device)
                logits = model.network(inputs)
                batch_loss = criterion(logits, targets)
                if not torch.isfinite(batch_loss):
                    raise FloatingPointError("Non-finite validation loss")
                val_loss_sum += float(batch_loss.item()) * len(labels)
                val_correct += int((logits.argmax(dim=1) == targets).sum().item())
                validation_rows += len(labels)
        _sync(torch, model)
        validation_seconds = perf_counter() - val_start
        if validation_rows != len(validation):
            raise RuntimeError("Validation did not cover the full partition")
        val_loss = val_loss_sum / validation_rows
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = _cpu_state(model.network)
            stale = 0
        else:
            stale += 1
        history["optimizer_steps"] += steps
        history["training_rows_processed"] += actual_rows
        history["validation_rows_processed"] += validation_rows
        history["epochs"].append({
            "epoch": epoch + 1, "train_loss": loss_sum / actual_rows,
            "validation_loss": val_loss, "validation_accuracy": val_correct / validation_rows,
            "train_rows": actual_rows, "validation_rows": validation_rows,
            "optimizer_steps": steps, "cumulative_optimizer_steps": history["optimizer_steps"],
            "train_seconds": train_seconds, "validation_seconds": validation_seconds,
            "best_epoch_so_far": best_epoch, "stale_epochs": stale,
        })
        completed = epoch + 1
        early_stopped = (model.config.early_stopping_patience is not None and
                         stale >= model.config.early_stopping_patience)
        checkpoint = {
            **header, "completed_epoch": completed,
            "model_state_dict": _cpu_state(model.network),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "rng": _rng_snapshot(torch), "history": history,
            "best_validation_loss": best_loss, "best_epoch": best_epoch,
            "best_state_dict": best_state, "stale_epochs": stale,
            "early_stopped": early_stopped,
        }
        _save_torch(latest_path, checkpoint, torch)
        _save_torch(best_path, _best_checkpoint(model, best_state, best_epoch,
                                                data_identity, config), torch)
        _write_json(history_path, history)
        _write_json(manifest_path, {
            **header, "completed_epoch": completed, "best_epoch": best_epoch,
            "best_validation_loss": best_loss, "stale_epochs": stale,
            "early_stopped": early_stopped, "latest_checkpoint": latest_path.name,
            "best_checkpoint": best_path.name,
            "measured_total_train_seconds": sum(row["train_seconds"] for row in history["epochs"]),
            "measured_total_validation_seconds": sum(row["validation_seconds"] for row in history["epochs"]),
        })
        observed_epoch_seconds = sum(row["train_seconds"] + row["validation_seconds"]
                                     for row in history["epochs"])
        remaining_hours = ((target - completed) * observed_epoch_seconds / completed) / 3600
        report(f"epoch {completed}: train_loss={loss_sum / actual_rows:.6f}, "
               f"full_val_loss={val_loss:.6f}, full_val_acc={val_correct / validation_rows:.4f}; "
               f"train={train_seconds:.1f}s, val={validation_seconds:.1f}s, "
               f"rows={actual_rows:,}/{validation_rows:,}, steps={steps}; "
               f"measured remaining estimate={remaining_hours:.2f}h; best={best_epoch}")
        if early_stopped:
            report(f"Early stopping after epoch {completed}; best epoch {best_epoch}")

    return {"completed_epoch": completed, "best_epoch": best_epoch,
            "best_validation_loss": best_loss, "early_stopped": early_stopped,
            "history": history, "latest_checkpoint": str(latest_path),
            "best_checkpoint": str(best_path),
            "paused": completed < target and not early_stopped}

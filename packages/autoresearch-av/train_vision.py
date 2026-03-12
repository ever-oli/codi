#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autoresearch_av.results import append_result_row, initialize_results_file
from autoresearch_av.storage import create_run_dir, ensure_storage_layout, get_storage_paths, write_run_config
from autoresearch_av.tasks import VISION_TASKS, get_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or scaffold a vision autoresearch baseline.")
    parser.add_argument("--task", help="Vision task name")
    parser.add_argument("--root", help="Override AUTORESEARCH_AV_ROOT")
    parser.add_argument("--model-name", default="resnet18", help="Baseline model identifier")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Epoch count")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    parser.set_defaults(pretrained=True)
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="Initialize supported vision models with pretrained ImageNet weights (default)",
    )
    pretrained_group.add_argument(
        "--random-init",
        dest="pretrained",
        action="store_false",
        help="Disable pretrained weights and train from random initialization",
    )
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience in epochs. Default: disabled")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional cap on train samples")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Optional cap on validation samples")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader workers")
    parser.add_argument("--dry-run", action="store_true", help="Create run scaffolding without training")
    parser.add_argument("--list-tasks", action="store_true", help="List available vision tasks")
    return parser.parse_args()


def make_beans_loaders(*, batch_size: int, image_size: int, data_root: Path, max_train_samples: int, max_val_samples: int, num_workers: int):
    try:
        from datasets import load_dataset
        import torch
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError(
            "Vision training requires datasets and torchvision. Install package dependencies from packages/autoresearch-av."
        ) from exc

    train_dataset = load_dataset("AI-Lab-Makerere/beans", split="train", cache_dir=str(data_root))
    val_dataset = load_dataset("AI-Lab-Makerere/beans", split="validation", cache_dir=str(data_root))

    if max_train_samples > 0:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
    if max_val_samples > 0:
        val_dataset = val_dataset.select(range(min(max_val_samples, len(val_dataset))))

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    class BeansTorchDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset) -> None:
            self.hf_dataset = hf_dataset

        def __len__(self) -> int:
            return len(self.hf_dataset)

        def __getitem__(self, index: int):
            row = self.hf_dataset[index]
            image = row["image"].convert("RGB")
            label = int(row["labels"])
            return transform(image), label

    train_loader = torch.utils.data.DataLoader(
        BeansTorchDataset(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        BeansTorchDataset(val_dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, 3


def make_trashnet_loaders(*, batch_size: int, image_size: int, data_root: Path, max_train_samples: int, max_val_samples: int, num_workers: int):
    try:
        import torch
        from huggingface_hub import snapshot_download
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
    except ImportError as exc:
        raise RuntimeError(
            "Vision training requires torchvision and huggingface_hub. Install package dependencies from packages/autoresearch-av."
        ) from exc

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    snapshot_dir = Path(
        snapshot_download(
            repo_id="garythung/trashnet",
            repo_type="dataset",
            local_dir=str(data_root / "hf"),
        ),
    )
    extracted_root = data_root / "extracted"
    extracted_root.mkdir(parents=True, exist_ok=True)
    zip_files = sorted(snapshot_dir.rglob("*.zip"))
    if not zip_files:
        raise RuntimeError(f"No zip archive found in TrashNet snapshot: {snapshot_dir}")
    sentinel = extracted_root / ".extracted"
    if not sentinel.exists():
        with ZipFile(zip_files[0]) as archive:
            archive.extractall(extracted_root)
        sentinel.write_text("ok\n", encoding="utf-8")

    class_dirs = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
    image_root: Path | None = None
    for candidate in [extracted_root, *sorted(extracted_root.rglob("*"))]:
        if not candidate.is_dir():
            continue
        if "__MACOSX" in candidate.parts:
            continue
        children = {child.name for child in candidate.iterdir() if child.is_dir()}
        if class_dirs.issubset(children):
            image_root = candidate
            break
    if image_root is None:
        raise RuntimeError(f"Could not locate extracted TrashNet class folders under {extracted_root}")

    def is_valid_file(path: str) -> bool:
        path_obj = Path(path)
        if "__MACOSX" in path_obj.parts:
            return False
        if any(part.startswith(".") for part in path_obj.parts):
            return False
        return path_obj.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

    full_dataset = ImageFolder(root=str(image_root), transform=transform, is_valid_file=is_valid_file)
    total_size = len(full_dataset)
    train_size = int(total_size * 0.8)
    val_size = total_size - train_size
    generator = torch.Generator().manual_seed(17)
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)

    if max_train_samples > 0:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(max_train_samples, len(train_dataset))))
    if max_val_samples > 0:
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(max_val_samples, len(val_dataset))))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, len(full_dataset.classes)


def make_flowers102_loaders(*, batch_size: int, image_size: int, data_root: Path, max_train_samples: int, max_val_samples: int, num_workers: int):
    try:
        import torch
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError(
            "Vision training requires torchvision. Install package dependencies from packages/autoresearch-av."
        ) from exc

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    try:
        train_dataset = datasets.Flowers102(root=str(data_root), split="train", transform=transform, download=True)
        val_dataset = datasets.Flowers102(root=str(data_root), split="val", transform=transform, download=True)
    except ModuleNotFoundError as exc:
        if exc.name == "scipy":
            raise RuntimeError(
                "Flowers102 requires scipy for split metadata loading. Run `uv sync` in packages/autoresearch-av first."
            ) from exc
        raise

    if max_train_samples > 0:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(max_train_samples, len(train_dataset))))
    if max_val_samples > 0:
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(max_val_samples, len(val_dataset))))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, 102


def make_food101_loaders(*, batch_size: int, image_size: int, data_root: Path, max_train_samples: int, max_val_samples: int, num_workers: int):
    try:
        import torch
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError(
            "Vision training requires torchvision. Install package dependencies from packages/autoresearch-av."
        ) from exc

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    train_dataset = datasets.Food101(root=str(data_root), split="train", transform=transform, download=True)
    val_dataset = datasets.Food101(root=str(data_root), split="test", transform=transform, download=True)

    if max_train_samples > 0:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(max_train_samples, len(train_dataset))))
    if max_val_samples > 0:
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(max_val_samples, len(val_dataset))))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, 101


def make_oxford_pet_loaders(*, batch_size: int, image_size: int, data_root: Path, max_train_samples: int, max_val_samples: int, num_workers: int):
    try:
        import torch
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError(
            "Vision training requires torchvision. Install package dependencies from packages/autoresearch-av."
        ) from exc

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    train_dataset = datasets.OxfordIIITPet(
        root=str(data_root),
        split="trainval",
        target_types="category",
        transform=transform,
        download=True,
    )
    val_dataset = datasets.OxfordIIITPet(
        root=str(data_root),
        split="test",
        target_types="category",
        transform=transform,
        download=True,
    )

    if max_train_samples > 0:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(max_train_samples, len(train_dataset))))
    if max_val_samples > 0:
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(max_val_samples, len(val_dataset))))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, 37


def make_cub200_loaders(*, batch_size: int, image_size: int, data_root: Path, max_train_samples: int, max_val_samples: int, num_workers: int):
    try:
        from datasets import load_dataset
        import torch
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError(
            "Vision training requires datasets and torchvision. Install package dependencies from packages/autoresearch-av."
        ) from exc

    train_dataset = load_dataset("bentrevett/caltech-ucsd-birds-200-2011", split="train", cache_dir=str(data_root))
    val_dataset = load_dataset("bentrevett/caltech-ucsd-birds-200-2011", split="test", cache_dir=str(data_root))

    if max_train_samples > 0:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
    if max_val_samples > 0:
        val_dataset = val_dataset.select(range(min(max_val_samples, len(val_dataset))))

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    def resolve_label_key(hf_dataset) -> str:
        for key in ("label", "labels"):
            if key in hf_dataset.column_names:
                return key
        raise RuntimeError(f"Unable to locate CUB label column. Found: {hf_dataset.column_names}")

    label_key = resolve_label_key(train_dataset)

    class CubTorchDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset) -> None:
            self.hf_dataset = hf_dataset

        def __len__(self) -> int:
            return len(self.hf_dataset)

        def __getitem__(self, index: int):
            row = self.hf_dataset[index]
            image = row["image"].convert("RGB")
            label = int(row[label_key])
            return transform(image), label

    train_loader = torch.utils.data.DataLoader(
        CubTorchDataset(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        CubTorchDataset(val_dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, 200


def build_vision_model(model_name: str, num_classes: int, pretrained: bool):
    try:
        import torch
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError("Vision training requires torchvision. Install package dependencies from packages/autoresearch-av.") from exc

    if model_name != "resnet18":
        raise RuntimeError(f"Only model_name=resnet18 is implemented right now, got '{model_name}'.")
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def main() -> int:
    args = parse_args()
    if args.list_tasks:
        for task_name, task in sorted(VISION_TASKS.items()):
            print(f"{task_name}: {task.problem} [{task.dataset}] metric={task.metric}")
        return 0

    if not args.task:
        print("--task is required unless --list-tasks is set", file=sys.stderr)
        return 2

    task = get_task(args.task)
    if task.modality != "vision":
        print(f"Task '{args.task}' is modality '{task.modality}', not vision", file=sys.stderr)
        return 2

    paths = get_storage_paths(args.root)
    ensure_storage_layout(paths)
    results_path = paths.results / "results.tsv"
    initialize_results_file(results_path)

    run_id, run_dir = create_run_dir(paths, task.name, args.model_name, args.dry_run)
    log_path = run_dir / "train.log"
    checkpoint_path = run_dir / "checkpoint"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "run_id": run_id,
        "task": task.name,
        "modality": task.modality,
        "problem": task.problem,
        "dataset": task.dataset,
        "dataset_url": task.dataset_url,
        "metric": task.metric,
        "storage_root": str(paths.root),
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "image_size": args.image_size,
        "pretrained": args.pretrained,
        "patience": args.patience,
        "dry_run": args.dry_run,
    }
    config_path = write_run_config(run_dir, config_payload)

    if args.dry_run:
        from autoresearch_av.training import save_json_log

        save_json_log(
            log_path,
            {
                "message": "Vision baseline scaffold created.",
                "task": task.name,
                "dataset": task.dataset,
                "metric": task.metric,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        append_result_row(
            results_path,
            {
                "run_id": run_id,
                "task": task.name,
                "dataset": task.dataset,
                "modality": task.modality,
                "model_name": args.model_name,
                "batch_size": str(args.batch_size),
                "seed": str(args.seed),
                "status": "dry-run",
                "notes": f"Baseline vision scaffold created{' pretrained' if args.pretrained else ' random-init'}",
                "config_path": str(config_path),
                "checkpoint_path": str(checkpoint_path),
                "log_path": str(log_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
    else:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Vision training requires torch. Run `uv sync` in packages/autoresearch-av first.") from exc
        from autoresearch_av.training import (
            append_json_log,
            append_text_log,
            count_params_m,
            resolve_device,
            set_seed,
            train_classifier,
        )
        if task.name not in {"beans", "trashnet", "flowers102", "food101", "oxford_pet", "cub200"}:
            print(
                "Real vision training is currently implemented only for tasks "
                "'beans', 'trashnet', 'flowers102', 'food101', 'oxford_pet', and 'cub200'; got "
                f"'{task.name}'.",
                file=sys.stderr,
            )
            return 2
        set_seed(args.seed)
        device = resolve_device()
        if task.name == "beans":
            train_loader, val_loader, num_classes = make_beans_loaders(
                batch_size=args.batch_size,
                image_size=args.image_size,
                data_root=paths.datasets / "beans",
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
                num_workers=args.num_workers,
            )
        elif task.name == "trashnet":
            train_loader, val_loader, num_classes = make_trashnet_loaders(
                batch_size=args.batch_size,
                image_size=args.image_size,
                data_root=paths.datasets / "trashnet",
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
                num_workers=args.num_workers,
            )
        elif task.name == "flowers102":
            train_loader, val_loader, num_classes = make_flowers102_loaders(
                batch_size=args.batch_size,
                image_size=args.image_size,
                data_root=paths.datasets / "flowers102",
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
                num_workers=args.num_workers,
            )
        elif task.name == "food101":
            train_loader, val_loader, num_classes = make_food101_loaders(
                batch_size=args.batch_size,
                image_size=args.image_size,
                data_root=paths.datasets / "food101",
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
                num_workers=args.num_workers,
            )
        elif task.name == "oxford_pet":
            train_loader, val_loader, num_classes = make_oxford_pet_loaders(
                batch_size=args.batch_size,
                image_size=args.image_size,
                data_root=paths.datasets / "oxford_pet",
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
                num_workers=args.num_workers,
            )
        else:
            train_loader, val_loader, num_classes = make_cub200_loaders(
                batch_size=args.batch_size,
                image_size=args.image_size,
                data_root=paths.datasets / "cub200",
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
                num_workers=args.num_workers,
            )
        model = build_vision_model(args.model_name, num_classes, args.pretrained)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()
        checkpoint_file = checkpoint_path / "model.pt"
        log_path.write_text("", encoding="utf-8")
        append_text_log(log_path, f"task={task.name} dataset={task.dataset} model={args.model_name}")
        append_text_log(
            log_path,
            f"batch_size={args.batch_size} epochs={args.epochs} lr={args.lr} "
            f"weight_decay={args.weight_decay} pretrained={str(args.pretrained).lower()} patience={args.patience}",
        )
        artifacts = train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=args.epochs,
            device=device,
            checkpoint_path=checkpoint_file,
            patience=args.patience,
            progress_callback=lambda message: append_text_log(log_path, message),
        )
        append_json_log(
            log_path,
            {
                "task": task.name,
                "dataset": task.dataset,
                "metric": task.metric,
                "best_val_metric": artifacts.best_val_metric,
                "best_epoch": artifacts.best_epoch,
                "train_seconds": artifacts.train_seconds,
                "peak_memory_mb": artifacts.peak_memory_mb,
                "early_stopped": artifacts.early_stopped,
                "device": str(device),
                "model_name": args.model_name,
                "params_m": count_params_m(model),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        append_result_row(
            results_path,
            {
                "run_id": run_id,
                "task": task.name,
                "dataset": task.dataset,
                "modality": task.modality,
                "model_name": args.model_name,
                "params_m": f"{count_params_m(model):.3f}",
                "train_seconds": f"{artifacts.train_seconds:.3f}",
                "best_val_metric": f"{artifacts.best_val_metric:.6f}",
                "peak_memory_mb": f"{artifacts.peak_memory_mb:.3f}",
                "batch_size": str(args.batch_size),
                "seed": str(args.seed),
                "status": "completed",
                "notes": (
                    f"{task.name} baseline "
                    f"{'pretrained' if args.pretrained else 'random-init'} "
                    f"best_epoch={artifacts.best_epoch}"
                    f"{' early-stopped' if artifacts.early_stopped else ''}"
                ),
                "config_path": str(config_path),
                "checkpoint_path": str(artifacts.checkpoint_path),
                "log_path": str(log_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    print(f"Initialized vision run scaffold: {run_id}")
    print(f"Task: {task.problem} [{task.dataset}]")
    print(f"Storage root: {paths.root}")
    print(f"Run dir: {run_dir}")
    print(f"Results file: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autoresearch_av.results import append_result_row, initialize_results_file
from autoresearch_av.storage import create_run_dir, ensure_storage_layout, get_storage_paths, write_run_config
from autoresearch_av.tasks import AUDIO_TASKS, get_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or scaffold an audio autoresearch baseline.")
    parser.add_argument("--task", help="Audio task name")
    parser.add_argument("--root", help="Override AUTORESEARCH_AV_ROOT")
    parser.add_argument("--model-name", default="cnn_mel_small", help="Baseline model identifier")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Epoch count")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate")
    parser.add_argument("--clip-seconds", type=float, default=1.0, help="Clip duration")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience in epochs. Default: disabled")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional cap on train samples")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Optional cap on validation samples")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader workers")
    parser.add_argument("--dry-run", action="store_true", help="Create run scaffolding without training")
    parser.add_argument("--list-tasks", action="store_true", help="List available audio tasks")
    return parser.parse_args()


def build_audio_model(model_name: str, num_classes: int):
    from autoresearch_av.audio_models import SmallAudioCnn

    if model_name != "cnn_mel_small":
        raise RuntimeError(f"Only model_name=cnn_mel_small is implemented right now, got '{model_name}'.")
    return SmallAudioCnn(num_classes=num_classes)


def ensure_download(download_url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urlretrieve(download_url, destination)
    return destination


def extract_archive_once(archive_path: Path, extract_root: Path) -> None:
    extract_root.mkdir(parents=True, exist_ok=True)
    sentinel = extract_root / ".extracted"
    if sentinel.exists():
        return
    if archive_path.suffix == ".zip":
        with ZipFile(archive_path) as archive:
            archive.extractall(extract_root)
    else:
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(extract_root)
    sentinel.write_text("ok\n", encoding="utf-8")


def build_mel_frontend(sample_rate: int):
    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64)
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    return mel, amplitude_to_db


def load_wav_audio(audio_path: Path):
    import numpy as np
    import torch
    from scipy.io import wavfile

    current_sample_rate, waveform = wavfile.read(str(audio_path))
    waveform_array = np.asarray(waveform)
    if np.issubdtype(waveform_array.dtype, np.integer):
        info = np.iinfo(waveform_array.dtype)
        scale = float(max(abs(info.min), info.max))
        waveform_array = waveform_array.astype(np.float32) / scale
    else:
        waveform_array = waveform_array.astype(np.float32)

    waveform_tensor = torch.from_numpy(waveform_array)
    if waveform_tensor.ndim == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0)
    else:
        waveform_tensor = waveform_tensor.transpose(0, 1).contiguous()
    return waveform_tensor, int(current_sample_rate)


def load_wav_bytes(audio_bytes: bytes):
    import numpy as np
    import torch
    from scipy.io import wavfile

    current_sample_rate, waveform = wavfile.read(BytesIO(audio_bytes))
    waveform_array = np.asarray(waveform)
    if np.issubdtype(waveform_array.dtype, np.integer):
        info = np.iinfo(waveform_array.dtype)
        scale = float(max(abs(info.min), info.max))
        waveform_array = waveform_array.astype(np.float32) / scale
    else:
        waveform_array = waveform_array.astype(np.float32)

    waveform_tensor = torch.from_numpy(waveform_array)
    if waveform_tensor.ndim == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0)
    else:
        waveform_tensor = waveform_tensor.transpose(0, 1).contiguous()
    return waveform_tensor, int(current_sample_rate)


def normalize_waveform(*, waveform, current_sample_rate: int, sample_rate: int, max_samples: int):
    import torch
    import torchaudio

    if current_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, current_sample_rate, sample_rate)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.size(1) < max_samples:
        waveform = torch.nn.functional.pad(waveform, (0, max_samples - waveform.size(1)))
    else:
        waveform = waveform[:, :max_samples]
    return waveform


def make_speech_commands_loaders(
    *,
    batch_size: int,
    sample_rate: int,
    clip_seconds: float,
    data_root: Path,
    max_train_samples: int,
    max_val_samples: int,
    num_workers: int,
):
    try:
        import torch
        import torchaudio
        from torchaudio.datasets import SPEECHCOMMANDS
    except ImportError as exc:
        raise RuntimeError(
            "Audio training requires datasets and torchaudio. Install package dependencies from packages/autoresearch-av."
        ) from exc

    data_root.mkdir(parents=True, exist_ok=True)
    train_dataset = SPEECHCOMMANDS(root=str(data_root), download=True, subset="training")
    val_dataset = SPEECHCOMMANDS(root=str(data_root), download=True, subset="validation")

    def build_records(base_dataset):
        records: list[tuple[Path, int]] = []
        label_names_local = sorted({base_dataset.get_metadata(index)[2] for index in range(len(base_dataset))})
        label_to_id_local = {label: index for index, label in enumerate(label_names_local)}
        for index in range(len(base_dataset)):
            relative_path, _, label, _, _ = base_dataset.get_metadata(index)
            audio_path = Path(base_dataset._archive) / relative_path
            if audio_path.exists():
                records.append((audio_path, label_to_id_local[label]))
        return records, label_names_local

    train_records, train_label_names = build_records(train_dataset)
    val_records, val_label_names = build_records(val_dataset)
    label_names = sorted(set(train_label_names) | set(val_label_names))
    label_to_id = {label: index for index, label in enumerate(label_names)}

    def remap_records(records: list[tuple[Path, int]], names: list[str]):
        id_to_label = {index: label for index, label in enumerate(names)}
        return [(audio_path, label_to_id[id_to_label[label_id]]) for audio_path, label_id in records]

    train_records = remap_records(train_records, train_label_names)
    val_records = remap_records(val_records, val_label_names)
    if max_train_samples > 0:
        train_records = train_records[: min(max_train_samples, len(train_records))]
    if max_val_samples > 0:
        val_records = val_records[: min(max_val_samples, len(val_records))]

    max_samples = int(sample_rate * clip_seconds)
    mel, amplitude_to_db = build_mel_frontend(sample_rate)

    class SpeechCommandsTorchDataset(torch.utils.data.Dataset):
        def __init__(self, records: list[tuple[Path, int]]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int):
            audio_path, label = self.records[index]
            waveform, current_sample_rate = load_wav_audio(audio_path)
            waveform = normalize_waveform(
                waveform=waveform,
                current_sample_rate=current_sample_rate,
                sample_rate=sample_rate,
                max_samples=max_samples,
            )
            spec = amplitude_to_db(mel(waveform))
            return spec, label

    train_loader = torch.utils.data.DataLoader(
        SpeechCommandsTorchDataset(train_records),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        SpeechCommandsTorchDataset(val_records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, len(label_names)


def make_esc50_loaders(
    *,
    batch_size: int,
    sample_rate: int,
    clip_seconds: float,
    data_root: Path,
    max_train_samples: int,
    max_val_samples: int,
    num_workers: int,
):
    try:
        import pyarrow.ipc as ipc
        import torch
        import torchaudio
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Audio training requires datasets and torchaudio. Install package dependencies from packages/autoresearch-av."
        ) from exc

    load_dataset("renumics/esc50", split="train", cache_dir=str(data_root))
    arrow_files = sorted(data_root.rglob("esc50-train-*.arrow"))
    if not arrow_files:
        raise RuntimeError(f"Unable to locate cached ESC-50 arrow shards under {data_root}")

    train_records: list[tuple[bytes, int]] = []
    val_records: list[tuple[bytes, int]] = []
    for arrow_file in arrow_files:
        with ipc.open_stream(arrow_file.open("rb")) as reader:
            table = reader.read_all()
        for row in table.to_pylist():
            target_records = train_records if int(row["fold"]) != 5 else val_records
            target_records.append((row["audio"]["bytes"], int(row["label"])))

    if max_train_samples > 0:
        train_records = train_records[: min(max_train_samples, len(train_records))]
    if max_val_samples > 0:
        val_records = val_records[: min(max_val_samples, len(val_records))]

    max_samples = int(sample_rate * clip_seconds)
    mel, amplitude_to_db = build_mel_frontend(sample_rate)

    class Esc50TorchDataset(torch.utils.data.Dataset):
        def __init__(self, records: list[tuple[bytes, int]]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int):
            audio_bytes, label = self.records[index]
            waveform, current_sample_rate = load_wav_bytes(audio_bytes)
            waveform = normalize_waveform(
                waveform=waveform,
                current_sample_rate=current_sample_rate,
                sample_rate=sample_rate,
                max_samples=max_samples,
            )
            spec = amplitude_to_db(mel(waveform))
            return spec, label

    train_loader = torch.utils.data.DataLoader(
        Esc50TorchDataset(train_records),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        Esc50TorchDataset(val_records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, 50


def make_gtzan_loaders(
    *,
    batch_size: int,
    sample_rate: int,
    clip_seconds: float,
    data_root: Path,
    max_train_samples: int,
    max_val_samples: int,
    num_workers: int,
):
    try:
        from huggingface_hub import hf_hub_download
        import torch
        import torchaudio
        from torchaudio.datasets.gtzan import filtered_train, filtered_valid
    except ImportError as exc:
        raise RuntimeError(
            "Audio training requires torchaudio. Install package dependencies from packages/autoresearch-av."
        ) from exc

    data_root.mkdir(parents=True, exist_ok=True)
    archive_path = Path(
        hf_hub_download(
            repo_id="marsyas/gtzan",
            repo_type="dataset",
            filename="data/genres.tar.gz",
            local_dir=str(data_root),
        )
    )
    extract_archive_once(archive_path, data_root)

    max_samples = int(sample_rate * clip_seconds)
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64)
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    genre_to_id = {genre: index for index, genre in enumerate(genres)}

    def build_records(file_ids: list[str]):
        records: list[tuple[Path, int]] = []
        dataset_path = data_root / "genres"
        for file_id in file_ids:
            genre, _ = file_id.split(".")
            audio_path = dataset_path / genre / f"{file_id}.wav"
            if audio_path.exists():
                records.append((audio_path, genre_to_id[genre]))
        return records

    train_records = build_records(filtered_train)
    val_records = build_records(filtered_valid)

    if max_train_samples > 0:
        train_records = train_records[: min(max_train_samples, len(train_records))]
    if max_val_samples > 0:
        val_records = val_records[: min(max_val_samples, len(val_records))]

    class GtzanTorchDataset(torch.utils.data.Dataset):
        def __init__(self, records: list[tuple[Path, int]]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int):
            audio_path, label = self.records[index]
            waveform, current_sample_rate = load_wav_audio(audio_path)
            waveform = normalize_waveform(
                waveform=waveform,
                current_sample_rate=current_sample_rate,
                sample_rate=sample_rate,
                max_samples=max_samples,
            )
            spec = amplitude_to_db(mel(waveform))
            return spec, label

    train_loader = torch.utils.data.DataLoader(
        GtzanTorchDataset(train_records),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        GtzanTorchDataset(val_records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, len(genres)


def make_nsynth_loaders(
    *,
    batch_size: int,
    sample_rate: int,
    clip_seconds: float,
    data_root: Path,
    max_train_samples: int,
    max_val_samples: int,
    num_workers: int,
):
    try:
        import torch
        import torchaudio
    except ImportError as exc:
        raise RuntimeError(
            "Audio training requires torchaudio. Install package dependencies from packages/autoresearch-av."
        ) from exc

    downloads_root = data_root / "downloads"
    extracted_root = data_root / "extracted"
    split_specs = {
        "train": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz",
        "validation": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",
    }
    for split_name, download_url in split_specs.items():
        archive_path = ensure_download(download_url, downloads_root / f"{split_name}.tar.gz")
        extract_archive_once(archive_path, extracted_root / split_name)

    def build_examples(split_name: str):
        split_root = extracted_root / split_name
        examples_paths = sorted(split_root.rglob("examples.json"))
        if not examples_paths:
            raise RuntimeError(f"Unable to locate NSynth examples.json under {split_root}")
        examples_path = examples_paths[0]
        audio_root = examples_path.parent / "audio"
        with examples_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        records: list[tuple[Path, int]] = []
        for key, example in metadata.items():
            audio_path = audio_root / f"{key}.wav"
            if not audio_path.exists():
                continue
            records.append((audio_path, int(example["instrument_family"])))
        return records

    train_records = build_examples("train")
    val_records = build_examples("validation")

    if max_train_samples > 0:
        train_records = train_records[: min(max_train_samples, len(train_records))]
    if max_val_samples > 0:
        val_records = val_records[: min(max_val_samples, len(val_records))]

    max_samples = int(sample_rate * clip_seconds)
    mel, amplitude_to_db = build_mel_frontend(sample_rate)
    num_classes = max((label for _, label in [*train_records, *val_records]), default=-1) + 1

    class NsynthTorchDataset(torch.utils.data.Dataset):
        def __init__(self, records: list[tuple[Path, int]]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int):
            audio_path, label = self.records[index]
            waveform, current_sample_rate = load_wav_audio(audio_path)
            waveform = normalize_waveform(
                waveform=waveform,
                current_sample_rate=current_sample_rate,
                sample_rate=sample_rate,
                max_samples=max_samples,
            )
            spec = amplitude_to_db(mel(waveform))
            return spec, label

    train_loader = torch.utils.data.DataLoader(
        NsynthTorchDataset(train_records),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        NsynthTorchDataset(val_records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, num_classes


def make_ravdess_loaders(
    *,
    batch_size: int,
    sample_rate: int,
    clip_seconds: float,
    data_root: Path,
    max_train_samples: int,
    max_val_samples: int,
    num_workers: int,
):
    try:
        import torch
        import torchaudio
    except ImportError as exc:
        raise RuntimeError(
            "Audio training requires torchaudio. Install package dependencies from packages/autoresearch-av."
        ) from exc

    archive_path = ensure_download(
        "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip",
        data_root / "downloads" / "ravdess.zip",
    )
    extracted_root = data_root / "extracted"
    extract_archive_once(archive_path, extracted_root)

    emotion_id_by_code = {
        "01": 0,
        "02": 1,
        "03": 2,
        "04": 3,
        "05": 4,
        "06": 5,
        "07": 6,
        "08": 7,
    }

    train_records: list[tuple[Path, int]] = []
    val_records: list[tuple[Path, int]] = []
    for audio_path in sorted(extracted_root.rglob("*.wav")):
        if "Actor_" not in audio_path.parent.name:
            continue
        parts = audio_path.stem.split("-")
        if len(parts) < 7:
            continue
        label = emotion_id_by_code.get(parts[2])
        actor_id = int(parts[-1])
        if label is None:
            continue
        (train_records if actor_id <= 20 else val_records).append((audio_path, label))

    if max_train_samples > 0:
        train_records = train_records[: min(max_train_samples, len(train_records))]
    if max_val_samples > 0:
        val_records = val_records[: min(max_val_samples, len(val_records))]

    max_samples = int(sample_rate * clip_seconds)
    mel, amplitude_to_db = build_mel_frontend(sample_rate)

    class RavdessTorchDataset(torch.utils.data.Dataset):
        def __init__(self, records: list[tuple[Path, int]]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int):
            audio_path, label = self.records[index]
            waveform, current_sample_rate = load_wav_audio(audio_path)
            waveform = normalize_waveform(
                waveform=waveform,
                current_sample_rate=current_sample_rate,
                sample_rate=sample_rate,
                max_samples=max_samples,
            )
            spec = amplitude_to_db(mel(waveform))
            return spec, label

    train_loader = torch.utils.data.DataLoader(
        RavdessTorchDataset(train_records),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        RavdessTorchDataset(val_records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, 8


def main() -> int:
    args = parse_args()
    if args.list_tasks:
        for task_name, task in sorted(AUDIO_TASKS.items()):
            print(f"{task_name}: {task.problem} [{task.dataset}] metric={task.metric}")
        return 0

    if not args.task:
        print("--task is required unless --list-tasks is set", file=sys.stderr)
        return 2

    task = get_task(args.task)
    if task.modality != "audio":
        print(f"Task '{args.task}' is modality '{task.modality}', not audio", file=sys.stderr)
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
        "sample_rate": args.sample_rate,
        "clip_seconds": args.clip_seconds,
        "patience": args.patience,
        "dry_run": args.dry_run,
    }
    config_path = write_run_config(run_dir, config_payload)

    if args.dry_run:
        from autoresearch_av.training import save_json_log

        save_json_log(
            log_path,
            {
                "message": "Audio baseline scaffold created.",
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
                "notes": "Baseline audio scaffold created",
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
            raise RuntimeError("Audio training requires torch. Run `uv sync` in packages/autoresearch-av first.") from exc
        from autoresearch_av.training import (
            append_json_log,
            append_text_log,
            count_params_m,
            resolve_device,
            set_seed,
            train_classifier,
        )
        if task.name not in {"speech_commands", "esc50", "gtzan", "nsynth", "ravdess"}:
            print(
                "Real audio training is currently implemented only for tasks "
                "'speech_commands', 'esc50', 'gtzan', 'nsynth', and 'ravdess'; got "
                f"'{task.name}'.",
                file=sys.stderr,
            )
            return 2
        set_seed(args.seed)
        device = resolve_device()
        if task.name == "speech_commands":
            train_loader, val_loader, num_classes = make_speech_commands_loaders(
                batch_size=args.batch_size,
                sample_rate=args.sample_rate,
                clip_seconds=args.clip_seconds,
                data_root=paths.datasets / "speech_commands",
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
                num_workers=args.num_workers,
            )
        elif task.name == "esc50":
            train_loader, val_loader, num_classes = make_esc50_loaders(
                batch_size=args.batch_size,
                sample_rate=args.sample_rate,
                clip_seconds=args.clip_seconds,
                data_root=paths.datasets / "esc50",
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
                num_workers=args.num_workers,
            )
        elif task.name == "gtzan":
            train_loader, val_loader, num_classes = make_gtzan_loaders(
                batch_size=args.batch_size,
                sample_rate=args.sample_rate,
                clip_seconds=args.clip_seconds,
                data_root=paths.datasets / "gtzan",
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
                num_workers=args.num_workers,
            )
        elif task.name == "nsynth":
            train_loader, val_loader, num_classes = make_nsynth_loaders(
                batch_size=args.batch_size,
                sample_rate=args.sample_rate,
                clip_seconds=args.clip_seconds,
                data_root=paths.datasets / "nsynth",
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
                num_workers=args.num_workers,
            )
        else:
            train_loader, val_loader, num_classes = make_ravdess_loaders(
                batch_size=args.batch_size,
                sample_rate=args.sample_rate,
                clip_seconds=args.clip_seconds,
                data_root=paths.datasets / "ravdess",
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
                num_workers=args.num_workers,
            )
        model = build_audio_model(args.model_name, num_classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()
        checkpoint_file = checkpoint_path / "model.pt"
        log_path.write_text("", encoding="utf-8")
        append_text_log(log_path, f"task={task.name} dataset={task.dataset} model={args.model_name}")
        append_text_log(
            log_path,
            f"batch_size={args.batch_size} epochs={args.epochs} lr={args.lr} "
            f"weight_decay={args.weight_decay} patience={args.patience}",
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
                "notes": f"{task.name} baseline best_epoch={artifacts.best_epoch}{' early-stopped' if artifacts.early_stopped else ''}",
                "config_path": str(config_path),
                "checkpoint_path": str(artifacts.checkpoint_path),
                "log_path": str(log_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    print(f"Initialized audio run scaffold: {run_id}")
    print(f"Task: {task.problem} [{task.dataset}]")
    print(f"Storage root: {paths.root}")
    print(f"Run dir: {run_dir}")
    print(f"Results file: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

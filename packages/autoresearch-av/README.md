# @mariozechner/pi-autoresearch-av

Scaffold for an autoresearch-style vision/audio experiment loop.

This package does not try to force the current text `autoresearch` setup into a different modality. It provides a separate project shape for:

- shared storage under `/Volumes/Expansion/Data/autoresearch-av`
- a locked 11-task registry
- shared run/result bookkeeping
- one `train_vision.py` entrypoint
- one `train_audio.py` entrypoint

## Tasks

### Vision

- `food101`
- `beans`
- `trashnet`
- `cub200`
- `flowers102`
- `oxford_pet`

### Audio

- `esc50`
- `nsynth`
- `gtzan`
- `ravdess`
- `speech_commands`

## Storage

Default root:

```text
/Volumes/Expansion/Data/autoresearch-av
```

Override with:

```bash
export AUTORESEARCH_AV_ROOT=/custom/path
```

Directory layout:

```text
$AUTORESEARCH_AV_ROOT/
  datasets/
  cache/
  runs/
  results/
  models/
  logs/
  configs/
```

## Smoke Test

List tasks:

```bash
cd packages/autoresearch-av
python3 train_vision.py --list-tasks
python3 train_audio.py --list-tasks
```

Create dry-run baseline scaffolds:

```bash
cd packages/autoresearch-av
python3 train_vision.py --task beans --dry-run
python3 train_audio.py --task speech_commands --dry-run
```

Those commands create the storage tree, initialize `results.tsv` if needed, and write a per-run config bundle without training.

Run the first real baselines:

```bash
cd packages/autoresearch-av
python3 train_vision.py --task beans --epochs 3
python3 train_vision.py --task flowers102 --epochs 10 --lr 0.0003
python3 train_vision.py --task flowers102 --epochs 10 --lr 0.0003 --random-init
python3 train_audio.py --task speech_commands --epochs 3
```

Current implementation status:

- real training implemented for all registered tasks

Install Python dependencies before real training:

```bash
cd packages/autoresearch-av
uv sync
```

## Notes

- The first pass is deliberately conservative, but all 11 registered baselines now have concrete training paths.
- The recommended build order is `beans`, `trashnet`, `flowers102`, `speech_commands`, `esc50`.
- Shared encoders and autoresearch mutation loops should come only after single-task baselines are stable.
- Local macOS runs default to `--num-workers 0` to avoid Python multiprocessing pickling issues in these first dataset wrappers.
- Vision runs default to pretrained ImageNet initialization for supported models. Use `--random-init` when you want strict from-scratch baselines.
- `--patience <n>` enables early stopping after `n` non-improving epochs. It is off by default because some small datasets still improve late.

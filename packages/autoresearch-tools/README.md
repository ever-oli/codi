# @mariozechner/pi-autoresearch-tools

Pi package for running Karpathy-style autoresearch loops from Codi/Pi with a single `autoresearch` tool.

The tool is designed for repos like `karpathy/autoresearch` and forks:

- setup and validate `results.tsv`
- run bounded training experiments into `run.log`
- parse core metrics (`val_bpb`, `peak_vram_mb`, `training_seconds`, `total_seconds`)
- append normalized experiment rows to `results.tsv`
- inspect current best result

It also registers a slash command wrapper:

- `/autoresearch setup`
- `/autoresearch setup /path/to/repo`
- `/autoresearch run`
- `/autoresearch record <keep|discard|crash> [description]`
- `/autoresearch status`
- `/autoresearch av list`
- `/autoresearch av status`
- `/autoresearch av run vision beans --epochs 3`
- `/autoresearch av run vision flowers102 --epochs 10 --lr 0.0003`
- `/autoresearch av run vision flowers102 --epochs 10 --lr 0.0003 --random-init`
- `/autoresearch av run audio speech_commands --epochs 3`

## Install

```bash
pi install /absolute/path/to/packages/autoresearch-tools
```

If `CODI_AUTORESEARCH_ROOT` or `PI_AUTORESEARCH_ROOT` is set, `/autoresearch` uses that directory as its default working directory whenever `--cwd` is omitted. Relative `results.tsv` and `run.log` paths are resolved inside that root. The extension also remembers the last explicit repo path under that root, so follow-up `status` and `record` commands can omit `--cwd`.

## Smoke Test

```text
Use autoresearch with action "setup".
Use `/autoresearch setup /path/to/repo` to target a repo positionally.
Use autoresearch with action "run", runCommand "uv run train.py", timeoutSeconds 900.
Use autoresearch with action "record", status "keep", description "baseline run".
Use autoresearch with action "status".
/autoresearch setup
/autoresearch setup /path/to/repo
/autoresearch run --cmd "uv run train.py" --timeout 900
/autoresearch record keep baseline run
/autoresearch status
/autoresearch av list
/autoresearch av status
/autoresearch av run vision beans --epochs 3
/autoresearch av run vision flowers102 --epochs 10 --lr 0.0003
/autoresearch av run vision flowers102 --epochs 10 --lr 0.0003 --random-init
/autoresearch av run audio speech_commands --epochs 3
```

## AV Baselines

`/autoresearch av ...` dispatches into the standalone `packages/autoresearch-av` Python project.

Currently implemented:

- vision: `food101`, `beans`, `trashnet`, `cub200`, `flowers102`, `oxford_pet`
- audio: `esc50`, `nsynth`, `gtzan`, `ravdess`, `speech_commands`

Examples:

```text
/autoresearch av list
/autoresearch av status
/autoresearch av run vision beans --epochs 3
/autoresearch av run vision flowers102 --epochs 10 --lr 0.0003
/autoresearch av run vision flowers102 --epochs 10 --lr 0.0003 --random-init
/autoresearch av run vision trashnet --epochs 10 --lr 0.0003 --patience 3
/autoresearch av run audio speech_commands --epochs 3
/autoresearch av run audio ravdess --epochs 5 --patience 2
/autoresearch av run audio gtzan --epochs 3 --clip-seconds 3
```

The AV runner uses:

- package root: `packages/autoresearch-av`
- storage root: `AUTORESEARCH_AV_ROOT` if set, otherwise `/Volumes/Expansion/Data/autoresearch-av`

If `packages/autoresearch-av/.venv/bin/python` exists, the command uses that interpreter automatically. Otherwise it falls back to `python3`.

The current baseline scripts default to `--num-workers 0` for local macOS compatibility.

Vision runs default to ImageNet initialization for supported models such as `resnet18`. Use `--random-init` when you want strict scratch baselines, and `--patience <n>` when you want manual early stopping.

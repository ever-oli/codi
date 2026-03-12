## [Unreleased]

### Added

- Initial `autoresearch` extension package with actions for `setup`, `run`, `record`, and `status` to support durable Karpathy-style autoresearch experiment loops inside Pi/Codi.
- Added support for `CODI_AUTORESEARCH_ROOT` / `PI_AUTORESEARCH_ROOT` so autoresearch runs can default to a persistent external working directory when `--cwd` is omitted.
- Added positional `cwd` support for `/autoresearch setup` and `/autoresearch status`, and clearer run errors when the selected working directory does not contain `train.py`.
- Added last-repo tracking under the configured autoresearch root so follow-up `/autoresearch status` and `/autoresearch record` calls can reuse the most recent explicit repo path when `--cwd` is omitted.
- Added `/autoresearch av` subcommands that dispatch into `packages/autoresearch-av` so vision/audio baselines can be launched from Codi without dropping to a shell.

### Changed

- Updated `/autoresearch av run vision ...` to default to pretrained vision baselines, add `--random-init` for scratch ablations, and forward `--patience` for manual early stopping.
- Added `/autoresearch av status` and expanded `/autoresearch av` coverage to the full autoresearch-av task set.

# Changelog

## [Unreleased]

### Added

- Added initial vision/audio autoresearch scaffold with shared storage defaults, task registry, and baseline training entrypoints.

### Changed

- Vision baselines now default to pretrained ImageNet initialization for supported models, add `--random-init` for scratch ablations, and report best epoch details.
- Added optional early stopping to vision training with `--patience`, while keeping it disabled by default for late-improving small datasets.
- Implemented the remaining registered vision/audio task loaders so all 11 autoresearch-av tasks now have concrete training paths.

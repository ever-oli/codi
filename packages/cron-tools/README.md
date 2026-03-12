# @mariozechner/pi-cron-tools

Pi package that adds a durable `cron` tool for recurring interval-based tasks.

The tool stores jobs on disk and supports:

- `list`
- `add`
- `remove`
- `pause`
- `resume`
- `touch` (mark a job as run now)

## Install

```bash
pi install /absolute/path/to/packages/cron-tools
```

## Smoke Test

```text
Use cron with action "add", task "Check CI queue", everyMinutes 60.
Use cron with action "list".
Use cron with action "pause", id 1.
Use cron with action "resume", id 1.
Use cron with action "touch", id 1.
Use cron with action "remove", id 1.
```

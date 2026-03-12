# @mariozechner/pi-delegate-tools

Pi package that adds a `delegate` tool for running a bounded nested `pi` subprocess and returning the delegated result.

Use this when a task benefits from isolated context or explicit tool/model bounds.

## Install

```bash
pi install /absolute/path/to/packages/delegate-tools
```

## Smoke Test

```text
Use delegate with task "Summarize the current repo purpose in 5 bullets.", outputMode "summary".
Use delegate with task "List all package directories under ./packages.", tools ["ls","find"], timeoutSeconds 120.
```

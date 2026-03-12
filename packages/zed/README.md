# @mariozechner/pi-zed

Thin Zed integration wrapper for Pi built on top of [`pi-acp`](https://github.com/svkozak/pi-acp).

This package does not implement a native Pi ACP server. It provides:

- a stable `pi-zed` command that delegates to bundled `pi-acp`
- checked-in Zed config examples
- Pi-owned documentation for the current recommended Zed path

## Status

The recommended current Zed path is still upstream `pi-acp`, especially:

1. ACP Registry in Zed
2. `npx -y pi-acp`
3. global `pi-acp`

Use `@mariozechner/pi-zed` when you want a Pi-owned wrapper command instead of pointing Zed at `pi-acp` directly.

## Prerequisites

- Pi installed and available on `PATH`
- Zed or another ACP client

`pi-zed` launches `pi-acp`, which in turn launches `pi --mode rpc`.

## Install

```bash
npm install -g @mariozechner/pi-zed
```

## Zed setup

### Preferred: ACP Registry

If your ACP client supports the registry entry for `pi-acp`, prefer that path first.

Zed `settings.json`:

```json
{
  "agent_servers": {
    "pi-acp": {
      "type": "registry"
    }
  }
}
```

### Direct upstream `pi-acp` via `npx`

Zed `settings.json`:

```json
{
  "agent_servers": {
    "pi": {
      "type": "custom",
      "command": "npx",
      "args": ["-y", "pi-acp"],
      "env": {}
    }
  }
}
```

### Wrapper package via `pi-zed`

Zed `settings.json`:

```json
{
  "agent_servers": {
    "pi": {
      "type": "custom",
      "command": "pi-zed",
      "args": [],
      "env": {}
    }
  }
}
```

See the checked-in examples in `examples/`.

## Terminal auth

For interactive login/setup, run:

```bash
pi-zed --terminal-login
```

This forwards directly to `pi-acp --terminal-login`.

## Smoke test

```bash
pi-zed --help
```

Then in Zed:

1. Add the `pi-zed` custom server configuration.
2. Open the agent panel.
3. Start a session.
4. Confirm Pi connects and can answer a simple prompt.

## Files

- `examples/zed-settings-registry.json`
- `examples/zed-settings-npx.json`
- `examples/zed-settings-pi-zed.json`

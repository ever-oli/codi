# Zed Integration

The current recommended Zed path is [`pi-acp`](https://github.com/svkozak/pi-acp), not a future native Pi ACP server.

Pi currently has two layers for Zed:

1. Upstream `pi-acp`, which translates ACP to `pi --mode rpc`
2. `packages/zed`, a thin Pi-owned wrapper package around `pi-acp`

## Recommended current install order

1. ACP Registry entry for `pi-acp`
2. `npx -y pi-acp`
3. global `pi-acp`
4. global `@mariozechner/pi-zed`

## Why this is the current path

`pi-acp` already provides the ACP bridge, session mapping, terminal auth flow, and Zed-focused configuration story. That is enough to support Zed now without waiting for a native in-repo ACP server.

## What `packages/zed` should do

`packages/zed` should stay thin:

- provide a stable `pi-zed` wrapper command
- document the current Zed setup paths
- ship checked-in config examples
- avoid forking or reimplementing ACP behavior

## What remains deferred

Native in-repo ACP is still deferred. Revisit that only if one of these becomes true:

- `pi-acp` drifts from Pi behavior in ways that are hard to maintain
- Pi needs ACP features that are awkward over the current RPC bridge
- release control or support burden requires bringing the bridge in-repo

## Non-goals for the current package

- No Zed-specific workflow UI
- No native ACP protocol implementation
- No reimplementation of `pi-acp`

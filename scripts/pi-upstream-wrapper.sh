#!/usr/bin/env bash
set -euo pipefail

export PI_CODING_AGENT_DIR="${PI_UPSTREAM_AGENT_DIR:-$HOME/.pi-upstream/agent}"

mkdir -p "$PI_CODING_AGENT_DIR"

NPM_PREFIX="$(npm prefix -g)"
PI_GLOBAL_CLI=""
for candidate in \
  "$NPM_PREFIX/lib/node_modules/@mariozechner/pi-coding-agent/dist/cli.js" \
  /opt/zerobrew/prefix/Cellar/node/*/lib/node_modules/@mariozechner/pi-coding-agent/dist/cli.js \
  /opt/homebrew/lib/node_modules/@mariozechner/pi-coding-agent/dist/cli.js
do
  if [[ -f "$candidate" ]]; then
    PI_GLOBAL_CLI="$candidate"
    break
  fi
done

if [[ -z "$PI_GLOBAL_CLI" ]]; then
  echo "Upstream pi CLI not found in global npm locations." >&2
  echo "Install it with: npm install -g @mariozechner/pi-coding-agent" >&2
  exit 1
fi

exec node "$PI_GLOBAL_CLI" "$@"

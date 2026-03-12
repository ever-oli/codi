#!/usr/bin/env bash
set -euo pipefail

SCRIPT_SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SCRIPT_SOURCE" ]; do
  SCRIPT_DIR="$(cd -P -- "$(dirname -- "$SCRIPT_SOURCE")" && pwd)"
  SCRIPT_SOURCE="$(readlink -- "$SCRIPT_SOURCE")"
  [[ "$SCRIPT_SOURCE" != /* ]] && SCRIPT_SOURCE="$SCRIPT_DIR/$SCRIPT_SOURCE"
done
SCRIPT_DIR="$(cd -P -- "$(dirname -- "$SCRIPT_SOURCE")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCH_CWD="$PWD"
CLI_PATH="$REPO_DIR/packages/coding-agent/src/cli.ts"
TSX_BIN="$REPO_DIR/node_modules/.bin/tsx"
TSX_CLI="$REPO_DIR/node_modules/tsx/dist/cli.mjs"
NODE_BIN="$(command -v node)"
NODE_FLAGS=(--disable-warning=ExperimentalWarning)
AUTORESEARCH_ROOT_DEFAULT="/Volumes/Expansion/autoresearch"

# Check for --no-env flag
NO_ENV=false
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--no-env" ]]; then
    NO_ENV=true
  else
    ARGS+=("$arg")
  fi
done

if [[ "$NO_ENV" == "true" ]]; then
  unset ANTHROPIC_API_KEY
  unset ANTHROPIC_OAUTH_TOKEN
  unset OPENAI_API_KEY
  unset GEMINI_API_KEY
  unset GROQ_API_KEY
  unset CEREBRAS_API_KEY
  unset XAI_API_KEY
  unset OPENROUTER_API_KEY
  unset ZAI_API_KEY
  unset MISTRAL_API_KEY
  unset MINIMAX_API_KEY
  unset MINIMAX_CN_API_KEY
  unset AI_GATEWAY_API_KEY
  unset OPENCODE_API_KEY
  unset COPILOT_GITHUB_TOKEN
  unset GH_TOKEN
  unset GITHUB_TOKEN
  unset GOOGLE_APPLICATION_CREDENTIALS
  unset GOOGLE_CLOUD_PROJECT
  unset GCLOUD_PROJECT
  unset GOOGLE_CLOUD_LOCATION
  unset AWS_PROFILE
  unset AWS_ACCESS_KEY_ID
  unset AWS_SECRET_ACCESS_KEY
  unset AWS_SESSION_TOKEN
  unset AWS_REGION
  unset AWS_DEFAULT_REGION
  unset AWS_BEARER_TOKEN_BEDROCK
  unset AWS_CONTAINER_CREDENTIALS_RELATIVE_URI
  unset AWS_CONTAINER_CREDENTIALS_FULL_URI
  unset AWS_WEB_IDENTITY_TOKEN_FILE
  unset AZURE_OPENAI_API_KEY
  unset AZURE_OPENAI_BASE_URL
  unset AZURE_OPENAI_RESOURCE_NAME
  echo "Running without API keys..."
fi

cd "$REPO_DIR"
export PI_LAUNCH_CWD="$LAUNCH_CWD"
export CODI_AUTORESEARCH_ROOT="${CODI_AUTORESEARCH_ROOT:-${PI_AUTORESEARCH_ROOT:-$AUTORESEARCH_ROOT_DEFAULT}}"
export PI_AUTORESEARCH_ROOT="$CODI_AUTORESEARCH_ROOT"
export NODE_OPTIONS="${NODE_OPTIONS:+$NODE_OPTIONS }--disable-warning=ExperimentalWarning"
export NODE_NO_WARNINGS=1
mkdir -p "$CODI_AUTORESEARCH_ROOT"

if ((${#ARGS[@]} == 0)); then
  if [[ -f "$TSX_CLI" ]]; then
    exec "$NODE_BIN" "${NODE_FLAGS[@]}" "$TSX_CLI" "$CLI_PATH"
  fi
  if [[ -x "$TSX_BIN" ]]; then
    exec "$NODE_BIN" "${NODE_FLAGS[@]}" "$TSX_BIN" "$CLI_PATH"
  fi
  exec "$NODE_BIN" "${NODE_FLAGS[@]}" "$(command -v npx)" tsx "$CLI_PATH"
fi

if [[ -f "$TSX_CLI" ]]; then
  exec "$NODE_BIN" "${NODE_FLAGS[@]}" "$TSX_CLI" "$CLI_PATH" "${ARGS[@]}"
fi
if [[ -x "$TSX_BIN" ]]; then
  exec "$NODE_BIN" "${NODE_FLAGS[@]}" "$TSX_BIN" "$CLI_PATH" "${ARGS[@]}"
fi
exec "$NODE_BIN" "${NODE_FLAGS[@]}" "$(command -v npx)" tsx "$CLI_PATH" "${ARGS[@]}"

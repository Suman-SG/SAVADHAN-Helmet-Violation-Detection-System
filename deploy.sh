#!/usr/bin/env bash
set -euo pipefail

# Simple deploy helper for Fly.io
# Usage: ./deploy.sh [app-name]
# Requires: flyctl installed and you are logged in (flyctl auth login)

APP_NAME=${1:-helmet-system}

echo "Using app: $APP_NAME"

if ! command -v flyctl >/dev/null 2>&1; then
  echo "flyctl not found. Install from https://fly.io/docs/hands-on/install-flyctl/" >&2
  exit 1
fi

echo "Launching (or reusing) Fly app..."
flyctl apps create $APP_NAME || true

echo "Building and deploying via local Dockerfile..."
flyctl deploy --app $APP_NAME --config fly.toml --remote-only || flyctl deploy --app $APP_NAME --config fly.toml

echo "To set secrets (SMTP, ADMIN_PASSWORD, etc) run:"
echo "  flyctl secrets set ADMIN_PASSWORD=yourpass SEND_EMAIL=false SMTP_USER=... SMTP_PASSWORD=..."

echo "Deployment finished. Get URL with: flyctl info --app $APP_NAME"

#!/usr/bin/env bash
# bodyopt doclinks 拡張を VS Code にリンクする (開発用)。
# 実行後に VS Code で Reload Window (Ctrl+Shift+P) すると有効になる。
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="${HOME}/.vscode/extensions/bodyopt-doclinks"
mkdir -p "$(dirname "${dest}")"
ln -sfn "${repo}/extensions/doclinks" "${dest}"
echo "installed: ${dest} -> ${repo}/extensions/doclinks"
echo "VS Code で Reload Window を実行してください。"

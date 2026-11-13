#!/usr/bin/env bash
# Attach the LEGO-assets repository as a git submodule at asset/.
# Run once from the repository root after both repositories exist on GitHub:
#   bash tools/setup_asset_submodule.sh https://github.com/<org>/LEGO-assets.git
set -euo pipefail
URL="${1:?usage: $0 <LEGO-assets git URL>}"
cd "$(git rev-parse --show-toplevel)"
# A local symlink (asset -> ../LEGO_assets) is used during development; replace it.
if [ -L asset ]; then rm asset; fi
sed -i '/^\/asset$/d' .gitignore
git submodule add "$URL" asset
git add .gitignore .gitmodules asset
echo "asset/ is now a submodule pointing to $URL. Commit the change to pin the asset version."

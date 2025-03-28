#!/bin/bash

BASE_DIR="$HOME/.smartscan"
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

echo "Copying repository from '$SCRIPT_DIR' to '$BASE_DIR'..."
# cp -r "$SCRIPT_DIR" "$BASE_DIR" || { echo "Failed to copy repository."; exit 1; }
rsync -av --exclude='README.md' --exclude='venv/' --exclude='__pycache__/' "$SCRIPT_DIR/" "$BASE_DIR/" || { echo "Failed to copy repository."; exit 1; }

cd "$BASE_DIR" || exit 1

python3.10 -m venv venv

venv/bin/pip install -r requirements.txt

chmod 777 run.sh

echo "Copying run.sh to '/usr/local/bin/smartscan'..."
sudo cp run.sh /usr/local/bin/smartscan

echo "SmartScan successfully installed"

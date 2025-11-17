#!/bin/bash

BASE_DIR="$HOME/.smartscan"

setup(){
if [ ! -f "$BASE_DIR/smartscan.json" ]; then
  echo "Error: $BASE_DIR/smartscan.json not found. Please make sure the configuration file is present."
  exit 1
fi

TARGET_DIRS=($(jq -r '.target_dirs[]' "$BASE_DIR/smartscan.json"))

if [ ${#TARGET_DIRS[@]} -lt 1 ]; then
  echo "Error: Target directories (target_dirs) is empty."
  exit 1
fi

echo "Copying systemd files to the appropriate locations..."

SYSTEMD_DIR="$BASE_DIR/systemd"
if [ ! -d "$SYSTEMD_DIR" ]; then
  echo "Error: $SYSTEMD_DIR directory not found. Please make sure the systemd files are in place."
  exit 1
fi

mkdir -p $HOME/.config/systemd/user/

cp "$SYSTEMD_DIR/smartscan.timer" "$HOME/.config/systemd/user/"
cp "$SYSTEMD_DIR/smartscan.service" "$HOME/.config/systemd/user/"

echo "Reloading systemd daemon..."
systemctl --user daemon-reload

echo "Enabling and starting the smartscan service..."
systemctl --user enable --now smartscan.service

echo "SmartScan setup complete and running!"

}

enable(){
  systemctl --user enable --now smartscan.service
  echo "Service enabled successfully"
}

disable(){
  systemctl --user stop --now smartscan.service
  systemctl --user disable --now smartscan.service
  echo "Service disabled successfully"
}

case "$1" in
  setup) setup ;;
  enable) enable ;;
  disable) disable ;;
esac
#!/bin/bash

BASE_DIR="$HOME/.smartscan"

if [ ! -f "$BASE_DIR/smartscan.conf" ]; then
  echo "Error: $BASE_DIR/smartscan.conf not found. Please make sure the configuration file is present."
  exit 1
fi

if ! grep -q "TARGET_FILE" "$BASE_DIR/smartscan.conf" || ! grep -q "DESTINATION_FILE" "$BASE_DIR/smartscan.conf"; then
  echo "Error: Missing required values (TARGET_FILE or DESTINATION_FILE) in $BASE_DIR/smartscan.conf"
  exit 1
fi

TARGET_FILE=$(grep -oP '^TARGET_FILE=\K.+' "$BASE_DIR/smartscan.conf")
DESTINATION_FILE=$(grep -oP '^DESTINATION_FILE=\K.+' "$BASE_DIR/smartscan.conf")

if [ ! -f "$TARGET_FILE" ]; then
  echo "Error: TARGET_FILE ($TARGET_FILE) is not a valid file path."
  exit 1
fi

if [ ! -f "$DESTINATION_FILE" ]; then
  echo "Error: DESTINATION_FILE ($DESTINATION_FILE) is not a valid file path."
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

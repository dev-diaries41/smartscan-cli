#!/bin/bash

BASE_DIR="$HOME/.smartscan"

echo "Removing SmartScan from '$BASE_DIR'..."
rm -rf "$BASE_DIR" || { echo "Failed to remove $BASE_DIR"; exit 1; }

echo "Removing '/usr/local/bin/smartscan'..."
sudo rm -f /usr/local/bin/smartscan || { echo "Failed to remove /usr/local/bin/smartscan"; exit 1; }

echo "SmartScan successfully uninstalled"

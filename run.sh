#!/bin/bash

BASE_DIR="$HOME/.smartscan"

"$BASE_DIR/venv/bin/python" -m "$BASE_DIR/smartscan.main" "$@"

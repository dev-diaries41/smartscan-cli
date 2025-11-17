#!/bin/bash

BASE_DIR="$HOME/.smartscan"
cd "$BASE_DIR" || exit 1
PYTHONPATH="$BASE_DIR" "$BASE_DIR/venv/bin/python" -m smartscan.main "$@"

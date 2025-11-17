#!/bin/bash

BASE_DIR="$HOME/.smartscan"

PYTHONPATH="$BASE_DIR" "$BASE_DIR/venv/bin/python" -m smartscan.main "$@"

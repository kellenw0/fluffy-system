#!/bin/bash
# Keeps macOS awake while live.py runs.
# Usage: ./run.sh
cd "$(dirname "$0")"
caffeinate -i python live.py

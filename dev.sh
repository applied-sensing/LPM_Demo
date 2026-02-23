#!/usr/bin/env bash

if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt 2>/dev/null || true

echo "Environment ready."

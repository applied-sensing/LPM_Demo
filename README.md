# lpm-headphone

Configurable lumped-parameter modeling (small-signal) for headphones: Voltage -> Pressure.

## Install (editable)
python -m venv .venv
source .venv/Scripts/activate   # Git Bash on Windows
pip install -U pip
pip install -e ".[dev]"

## Run
lpm configs/headphone_min.json

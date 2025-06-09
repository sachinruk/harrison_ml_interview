#!/usr/bin/env bash
# This script sets up a Python virtual environment and installs the required packages.
set -euo pipefail

# Set up the environment for the project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

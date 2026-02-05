#!/bin/bash
# Activate the correct virtual environment and run the GUI simulation
source .venv/bin/activate
python run_sim_gui.py "$@"

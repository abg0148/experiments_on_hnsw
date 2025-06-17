# lib/paths.py
from pathlib import Path

# Anchor: root of the project
PROJECT_ROOT    =   Path(__file__).resolve().parent.parent

# Define standard paths
DATA_DIR        =   PROJECT_ROOT    /   "data"
RESULTS_DIR     =   PROJECT_ROOT    /   "results"
IMAGES_DIR      =   PROJECT_ROOT    /   "images"
EXPERIMENTS_DIR =   PROJECT_ROOT    /   "experiments"


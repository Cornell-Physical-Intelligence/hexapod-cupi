"""Portable repository lookup; optional exact standing-source input for published replay."""
from pathlib import Path
import os

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'robot/active_model.json').is_file()and(p/'pyproject.toml').is_file()),None)
if ROOT is None:raise RuntimeError('Run these CPU tests within the HEXAPOD repository')
STANDING=Path(os.environ.get('HEXAPOD_CANONICAL_STANDING_SOURCE',str(ROOT/'tmp/updated_native_standing_003'))).resolve()
TEST_DEPS=ROOT/'tmp/reference_residual_ppo_001/_deps'

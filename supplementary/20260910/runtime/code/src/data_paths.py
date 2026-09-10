"""Portable location of user-supplied lift3d datasets; no dataset is bundled."""
import os
from pathlib import Path

def sign_data_root():
    value = os.environ.get("SLP_SIGN_DATA_DIR")
    return Path(value).expanduser().resolve() if value else Path(__file__).resolve().parents[2] / "data"

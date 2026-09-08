#!/usr/bin/env python3
"""Download the indian_banking domain data from Hugging Face into ./data.

The benchmark code lives in this repository; the domain data (tasks, database, policy,
knowledge base, user-simulator guideline) is hosted as the dataset
``NPCI/tau-indian-banking``. Run this once after installing the package:

    python scripts/fetch_data.py            # writes data/tau2/...
    python scripts/fetch_data.py --force    # re-download even if data is present

Set HF_TOKEN in the environment if the dataset is private.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "NPCI/tau-indian-banking"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SENTINEL = DATA_DIR / "tau2" / "domains" / "indian_banking" / "tasks.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download even if data is already present")
    ap.add_argument("--revision", default=None, help="dataset revision (commit/tag); default: latest")
    args = ap.parse_args()

    if SENTINEL.exists() and not args.force:
        print(f"data already present at {DATA_DIR} (use --force to re-download)")
        return 0
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is required: pip install huggingface_hub", file=sys.stderr)
        return 1
    path = snapshot_download(
        REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(DATA_DIR),
        allow_patterns=["tau2/**"],
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"downloaded {REPO_ID} -> {path}")
    return 0 if SENTINEL.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())

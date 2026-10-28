"""Pinned upstream environment, assets and data preparation implementation."""

import os
import hashlib
import json
from pathlib import Path
import subprocess

REVISION = "6f95e133aa4b7ca6a784eac93725df3de4d41509"
URL = "https://github.com/Holiday-Robot/FlashDexRetarget.git"
PROJECT = Path(__file__).resolve().parents[1]


def source_root():
    root = Path(os.environ.get("FLASHDEX_ROOT", PROJECT / "third_party/FlashDexRetarget")).resolve()
    if not (root / "config/envs/flashdexretarget.yaml").is_file():
        raise FileNotFoundError(f"Missing {root}; run scripts/setup_source.py first")
    if (root / ".git").exists():
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if revision != REVISION:
            raise RuntimeError(f"Expected upstream {REVISION}, found {revision}")
    # sync-4090 excludes .git; verify the source/config/model descriptions also
    # in rsynced workspaces. Generated USD/SDF files remain outside this lock.
    lock = json.loads((PROJECT / "upstream-lock.json").read_text())
    if lock["revision"] != REVISION:
        raise RuntimeError("Upstream lock revision mismatch")
    for relative, digest in lock["files"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Upstream source differs from audited version: {relative}")
    return root

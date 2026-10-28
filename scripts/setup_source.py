"""Fetch the audited source revision, without installing/changing simulator packages."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _source import PROJECT, REVISION, URL, source_root

root = PROJECT / "third_party/FlashDexRetarget"
if not root.exists():
    subprocess.run(["git", "clone", URL, str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", "--detach", REVISION], check=True)
print(source_root())

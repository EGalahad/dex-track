"""Read the source lock without importing AA or initializing MDP components."""
from pathlib import Path
import runpy

_source = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'dex_track/source.py'))
PROJECT = _source['PROJECT']
REVISION = _source['REVISION']
URL = _source['URL']
source_root = _source['source_root']

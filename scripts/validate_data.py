import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dex_track.data import validate_motion

parser = argparse.ArgumentParser()
parser.add_argument("motion", type=Path)
parser.add_argument("--action-dim", type=int, default=36)
parser.add_argument("--without-sim-assets", action="store_true")
args = parser.parse_args()
print(json.dumps(validate_motion(args.motion, args.action_dim, not args.without_sim_assets), indent=2))

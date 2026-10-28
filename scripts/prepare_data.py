"""Run the pinned human-demo import and kinematic preparation using this interpreter."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _source import PROJECT, source_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["taco", "oakink2", "hot3d"], required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mano", type=Path, required=True)
    parser.add_argument("--robot", choices=["xhand", "sharpa"], default="xhand")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--import-only", action="store_true")
    parser.add_argument("--skip-import", action="store_true")
    parser.add_argument("--clip", help="Retarget one imported clip in a separate input view")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    root = source_root()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for side in ("LEFT", "RIGHT"):
        if not (args.mano / f"MANO_{side}.pkl").is_file():
            raise FileNotFoundError(f"Missing MANO_{side}.pkl in {args.mano}")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PROJECT / ".retarget-deps"), env.get("PYTHONPATH")]))
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    if not args.skip_import:
        subprocess.run([sys.executable, str(root / "src/retarget/import_human_demo.py"),
                        f"source={args.source}", f"raw={args.raw.resolve()}", f"dataset={output}",
                        f"mano_dir={args.mano.resolve()}", f"workers={args.workers}"], cwd=root, env=env, check=True)
    if args.import_only:
        return
    robot_dir = output / (args.robot if not args.clip else f"{args.robot}-{args.clip}")
    robot_dir.mkdir(exist_ok=True)
    for name in ("human_demo", "objects"):
        link = robot_dir / name
        if name == "human_demo" and args.clip:
            link.mkdir(exist_ok=True)
            clip = output / name / f"{args.clip}.npz"
            if not clip.is_file():
                raise FileNotFoundError(clip)
            link = link / clip.name
            if not link.exists():
                link.symlink_to(clip)
        elif not link.exists():
            link.symlink_to(output / name, target_is_directory=True)
    subprocess.run([sys.executable, str(root / "src/retarget/kinematic_retargeting.py"),
                    f"dataset={robot_dir}", f"robot={args.robot}", f"workers={args.workers}",
                    *args.overrides], cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()

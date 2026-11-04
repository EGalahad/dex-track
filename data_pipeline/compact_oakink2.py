"""Extract lossless MANO/object arrays; omit SMPL-X and cameras from training inputs."""
import argparse
import json
import os
from pathlib import Path
import pickle

import numpy as np


def compact(source, target):
    with source.open("rb") as stream:
        annotation = pickle.load(stream)
    frames = annotation["mocap_frame_id_list"]
    arrays = {"frame_ids": np.asarray(frames, dtype=np.int64), "fps": np.asarray(120.0)}
    for side in ("rh", "lh"):
        for field in ("pose_coeffs", "tsl", "betas"):
            key = f"{side}__{field}"
            arrays[key] = np.concatenate([
                annotation["raw_mano"][frame][key].detach().cpu().numpy() for frame in frames
            ], axis=0)
    object_ids = sorted(annotation["obj_transf"])
    arrays["object_ids"] = np.asarray(object_ids)
    arrays["object_pose"] = np.stack([
        np.stack([annotation["obj_transf"][obj][frame] for frame in frames])
        for obj in object_ids
    ], axis=1)
    for key, value in arrays.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite {key} in {source}")
    temporary = target.with_suffix(".npz.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(temporary, allow_pickle=False) as check:
        for key, value in arrays.items():
            if not np.array_equal(check[key], value):
                raise ValueError(f"Roundtrip mismatch: {source}: {key}")
    temporary.replace(target)
    return {"sequence": source.stem, "frames": len(frames), "objects": object_ids,
            "raw_bytes": source.stat().st_size, "compact_bytes": target.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("object_repair", "program"):
        source = (args.raw / name).resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        destination = args.output / name
        if not destination.exists():
            destination.symlink_to(source, target_is_directory=True)
    records = []
    paths = sorted((args.raw / "anno_preview").glob("*.pkl"))
    if args.limit:
        paths = paths[:args.limit]
    for source in paths:
        target = args.output / f"{source.stem}.npz"
        record_path = target.with_suffix(".json")
        if target.is_file() and record_path.is_file():
            record = json.loads(record_path.read_text())
        else:
            record = compact(source, target)
            record_path.write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        manifest = {"schema": "oakink2-mano-object-v1", "coordinates": "original OakInk2 y-up",
                    "rotation": "MANO wxyz quaternion, root plus 15 local joints",
                    "object_pose_axes": "frame, object, row, column",
                    "source_manifest": str((args.raw / "download-manifest.json").resolve()),
                    "collision_checks": "pending", "replay_checks": "pending",
                    "scope": "human MANO and object reference data; not yet retargeted",
                    "complete": len(records) == len(paths), "expected": len(paths), "sequences": records}
        temporary = args.output / "manifest.json.partial"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary, args.output / "manifest.json")
        print(f"Compacted {len(records)}/{len(paths)} {source.stem}: {record['compact_bytes']} bytes", flush=True)


if __name__ == "__main__":
    main()

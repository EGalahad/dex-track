"""Fetch training-clip JSON via HTTP ranges and pack MANO/object world trajectories."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
import shutil
from pathlib import Path
import tarfile
import time

import numpy as np
import requests


class RemoteTar(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "curl/8.5.0"
        self.start, self.buffer, self.received = 0, b"", 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (self.pos if whence == 1 else self.size if whence == 2 else 0)
        return self.pos

    def read(self, n=-1):
        n = min(self.size - self.pos, n if n >= 0 else self.size)
        if n <= 0:
            return b""
        if not (self.start <= self.pos and self.pos + n <= self.start + len(self.buffer)):
            end = min(self.size - 1, self.pos + max(n, 16384) - 1)
            for attempt in range(5):
                try:
                    with self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"},
                                          timeout=60, stream=True) as response:
                        if response.status_code == 429:
                            delay = min(300, max(1, int(response.headers.get("Retry-After", "60"))))
                            response.close()
                            time.sleep(delay)
                            continue
                        expected = f"bytes {self.pos}-{end}/{self.size}"
                        if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                            raise IOError(f"Server did not honor range: {response.status_code} {response.headers.get('Content-Range')}")
                        self.url = response.url  # Reuse signed CDN URL; do not resolve every range through Hub.
                        content = response.content
                    if len(content) != end - self.pos + 1:
                        raise IOError("Incomplete range")
                    self.start, self.buffer = self.pos, content
                    self.received += len(content)
                    break
                except (requests.RequestException, OSError):
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
            else:
                raise IOError("Range retries exhausted")
        result = self.buffer[self.pos - self.start:self.pos - self.start + n]
        self.pos += len(result)
        return result

    def close(self):
        self.session.close()
        super().close()


def pack(rows, shapes):
    frames = sorted(rows)
    if len(frames) != 150 or any(not {"hands", "objects", "info"} <= set(rows[f]) for f in frames):
        raise ValueError("Expected 150 complete annotation frames")
    objects, identity = [], {}
    for frame in frames:
        instances = {}
        for bop, group in rows[frame]["objects"].items():
            for obj in group:
                uid, model = str(obj["object_uid"]), int(obj["object_bop_id"])
                if model != int(bop) or uid in instances:
                    raise ValueError("Inconsistent model or duplicate UID")
                signature = (model, obj["object_name"])
                if uid in identity and identity[uid] != signature:
                    raise ValueError("UID changes model across frames")
                identity[uid], instances[uid] = signature, obj
        objects.append(instances)
    ids = sorted(identity)
    # Keep missing-hand/object validity explicitly; never interpolate over unavailable GT.
    hands = [(i, side, row["hands"][side]["mano_pose"])
             for i, row in enumerate(rows[f] for f in frames) for side in ("left", "right")
             if side in row["hands"] and row["hands"][side].get("mano_pose")]
    dims = {len(h[2]["thetas"]) for h in hands}
    if dims != {15}:
        raise ValueError(f"Inconsistent MANO pose dimensions: {dims}")
    n = len(frames)
    pose = np.zeros((n, 2, dims.pop()), dtype=np.float32)
    wrist = np.zeros((n, 2, 6), dtype=np.float32)
    valid = np.zeros((n, 2), dtype=bool)
    for i, side, hand in hands:
        j = int(side == "right")
        pose[i, j], wrist[i, j], valid[i, j] = hand["thetas"], hand["wrist_xform"], True
    obj_pose = np.zeros((n, len(ids), 7), dtype=np.float64)
    obj_valid = np.zeros((n, len(ids)), dtype=bool)
    timestamps = []
    for i, frame in enumerate(frames):
        row = rows[frame]
        timestamps.append(row["info"]["ref_timestamp_ns"])
        for j, oid in enumerate(ids):
            if oid in objects[i]:
                transform = objects[i][oid]["T_world_from_object"]
                obj_pose[i, j] = transform["quaternion_wxyz"] + transform["translation_xyz"]
                obj_valid[i, j] = True
    result = {"frame_ids": np.asarray(frames), "timestamp_ns": np.asarray(timestamps, dtype=np.int64),
              "mano_pose": pose, "mano_wrist_axis_angle_translation": wrist,
              "mano_betas": np.asarray(shapes["mano"], dtype=np.float32), "hand_valid": valid,
              "object_ids": np.asarray(ids),
              "object_bop_ids": np.asarray([identity[uid][0] for uid in ids], dtype=np.int32),
              "object_names": np.asarray([identity[uid][1] for uid in ids]),
              "object_mesh_paths": np.asarray([f"object_models/obj_{identity[uid][0]:06d}.glb" for uid in ids]), "object_quaternion_wxyz_translation": obj_pose,
              "object_valid": obj_valid, "fps": np.asarray(30.0)}
    for key, value in result.items():
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite {key}")
    if result["mano_betas"].shape != (10,):
        raise ValueError("Unexpected shared MANO beta shape")
    if obj_valid.any() and not np.allclose(np.linalg.norm(obj_pose[..., :4][obj_valid], axis=-1), 1, atol=1e-4):
        raise ValueError("Invalid object quaternion")
    if np.any(np.diff(result["timestamp_ns"]) <= 0):
        raise ValueError("Non-increasing timestamps")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--revision", help="Pin to an existing prepared dataset revision")
    parser.add_argument("--sim-annotations-only", action="store_true",
                        help="Cache only MANO, rigid-object poses/identities and frame metadata")
    parser.add_argument("--min-free-gb", type=float, default=2.0,
                        help="Stop new clip downloads below this free-space reserve")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    session = requests.Session()
    session.headers["User-Agent"] = "curl/8.5.0"
    repo = "bop-benchmark/hot3d"
    def get(url):
        for attempt in range(10):
            try:
                response = session.get(url, timeout=60)
                if response.status_code == 429:
                    delay = min(300, max(1, int(response.headers.get("Retry-After", "60"))))
                    response.close()
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt == 9:
                    raise
                time.sleep(min(30, 2 ** attempt))
        raise IOError(f"Metadata retries exhausted: {url}")
    revision = args.revision or get(f"{endpoint}/api/datasets/{repo}").json()["sha"]
    entries = []
    for split in ("train_quest3", "train_aria"):
        url = f"{endpoint}/api/datasets/{repo}/tree/{revision}/{split}?limit=1000"
        while url:
            response = get(url.replace("https://huggingface.co", endpoint))
            entries.extend(e for e in response.json() if e["path"].endswith(".tar"))
            url = response.links.get("next", {}).get("url")
    if args.limit:
        entries = entries[:args.limit]
    print(f"Preparing {len(entries)} training clips at {revision}", flush=True)
    def process(entry):
        path = Path(entry["path"])
        destination = args.output / path.parent / (path.stem + ".npz")
        destination.parent.mkdir(exist_ok=True)
        record_path = destination.with_suffix(".json")
        if destination.exists() and record_path.exists():
            saved = json.loads(record_path.read_text())
            if saved["revision"] == revision:
                return saved
        if shutil.disk_usage(args.output).free < args.min_free_gb * 1_000_000_000:
            raise RuntimeError("Free-space reserve reached; existing files preserved")
        cache = destination.with_suffix(".annotations.json")
        cached = json.loads(cache.read_text()) if cache.exists() else None
        rows, shapes = {}, None
        url = f"{endpoint}/datasets/{repo}/resolve/{revision}/{entry['path']}?download=true"
        if cached is not None and cached["revision"] == revision:
            rows, shapes, received = cached["rows"], cached["shapes"], cached["received_bytes"]
        else:
            with RemoteTar(url, entry["size"]) as remote:
                with tarfile.open(fileobj=remote, mode="r:") as archive:
                    for member in archive:
                        if not member.isfile():
                            continue
                        name = Path(member.name).name
                        if name == "__hand_shapes.json__":
                            shapes = json.load(archive.extractfile(member))
                        elif name.endswith((".hands.json", ".objects.json", ".info.json")):
                            frame, kind, _ = name.split(".")
                            annotation = json.load(archive.extractfile(member))
                            if args.sim_annotations_only:
                                if kind == "hands":
                                    annotation = {side: {"mano_pose": hand["mano_pose"]}
                                                  for side, hand in annotation.items()
                                                  if hand.get("mano_pose")}
                                elif kind == "objects":
                                    annotation = {bop: [{key: obj[key] for key in
                                                  ("object_uid", "object_bop_id", "object_name", "T_world_from_object")}
                                                  for obj in group] for bop, group in annotation.items()}
                                else:
                                    annotation = {key: annotation[key] for key in
                                                  ("ref_timestamp_ns", "sequence_id", "device")}
                            rows.setdefault(frame, {})[kind] = annotation
                            if kind == "info" and int(frame) % 50 == 0:
                                print(f"Reading {entry['path']} frame {frame}", flush=True)
                received = remote.received
            if args.sim_annotations_only and shapes is not None:
                shapes = {"mano": shapes["mano"]}
            temporary_cache = cache.with_suffix(".json.partial")
            temporary_cache.write_text(json.dumps({"revision": revision, "rows": rows, "shapes": shapes,
                                                   "received_bytes": received, "schema": "hot3d-sim-json-v1" if args.sim_annotations_only else "hot3d-original-selected-json-v1",
                                                   "source_path": entry["path"], "source_sha256": entry["lfs"]["oid"]}))
            temporary_cache.replace(cache)
        if shapes is None:
            raise ValueError("Missing hand shape")
        arrays = pack(rows, shapes)
        temp = destination.with_suffix(".npz.partial")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        with np.load(temp, allow_pickle=False) as loaded:
            if any(not np.array_equal(loaded[k], v) for k, v in arrays.items()):
                raise ValueError("NPZ roundtrip failed")
        temp.replace(destination)
        record = {"source": entry["path"], "revision": revision, "frames": len(rows),
                  "source_sha256": entry["lfs"]["oid"], "source_bytes": entry["size"],
                  "received_bytes": received, "prepared_bytes": destination.stat().st_size,
                  "integrity": "HTTP ranges, tar headers, JSON and NPZ validated; full tar hash not verified",
                  "coordinates": "original world coordinates; left/right axis order; meters",
                  "mano_pose_convention": "15 PCA coefficients; wrist axis-angle radians + translation meters; shared 10 betas",
                  "object_identity": "original object_uid with separate BOP ID and mesh path",
                  "schema": "hot3d-mano-rigid-world-v1",
                  "annotation_cache_scope": "MANO and rigid-object sim fields" if args.sim_annotations_only else "original selected JSON",
                  "replay_checks": "pending",
                  "source_sequence": rows[sorted(rows)[0]]["info"]["sequence_id"],
                  "device": rows[sorted(rows)[0]]["info"]["device"],
                  "collision_checks": "pending"}
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Prepared {entry['path']}: fetched {received:,}/{entry['size']:,} bytes", flush=True)
        return record
    records, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, entry): entry["path"] for entry in entries}
        for future in as_completed(futures):
            try:
                records.append(future.result())
            except Exception as error:
                errors.append({"source": futures[future], "error": str(error)})
                print(f"Failed {futures[future]}: {error}", flush=True)
            manifest = {"revision": revision, "expected": len(entries), "ready": len(records),
                        "complete": len(records) == len(entries), "errors": errors,
                        "split": "public training GT only", "files": records}
            temp = args.output / "preparation-manifest.json.partial"
            temp.write_text(json.dumps(manifest, indent=2) + "\n")
            temp.replace(args.output / "preparation-manifest.json")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

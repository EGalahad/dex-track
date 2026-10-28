"""Refine broadphase flags for selected preview stages using mesh vertices.

This is supporting evidence only: sampled vertex distances are not exact
triangle-triangle contact distances and cannot certify an object irrelevant.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-audit", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for record in json.loads(args.preview_audit.read_text()):
        with np.load(record["source_paths"]["compact_pose"], allow_pickle=False) as data:
            ids, frames, poses = data["object_ids"].tolist(), data["frame_ids"], data["object_pose"]
        start, end = record["frame_interval_half_open"]
        indices = np.flatnonzero((frames >= start) & (frames < end))[::120]
        vertices = {}
        for oid in record["safe_candidate_superset_object_ids"]:
            path = args.raw / "object_repair/align_ds" / oid / "model.obj"
            v = np.array([[float(x) for x in line.split()[1:4]] for line in path.read_text().splitlines() if line.startswith("v ")])
            vertices[oid] = v[::max(1, len(v)//10000)]
        pairs = []
        for candidate in record["candidate_objects"]:
            a = candidate["object_id"]
            for b in record["required_object_ids"]:
                distances = []
                for index in indices:
                    pa, pb = poses[index, ids.index(a)], poses[index, ids.index(b)]
                    va = vertices[a] @ pa[:3, :3].T + pa[:3, 3]
                    vb = vertices[b] @ pb[:3, :3].T + pb[:3, 3]
                    distances.append(float(cKDTree(va).query(vb, workers=1)[0].min()))
                pairs.append({"candidate_object_id": a, "required_object_id": b,
                    "minimum_sampled_vertex_distance_m": min(distances),
                    "fraction_samples_under_10mm": float(np.mean(np.asarray(distances) < .01)),
                    "frames": frames[indices].tolist(), "distance_m": distances})
        results.append({"task_number": record["task_number"], "stage_id": record["stage_id"], "pairs": pairs})
    args.output.write_text(json.dumps({"method": "1 Hz, up to approximately 10000 vertices per mesh, nearest vertex distance",
        "limitations": "A distance above threshold cannot rule out contact between triangle interiors or between samples. This report does not approve deletion of any object.",
        "tasks": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()

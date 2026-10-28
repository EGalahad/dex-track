"""Resume a server-side triangle-mesh narrowphase audit of broadphase flags.

Requires python-fcl. Uses exact BVH mesh distance at 10 Hz, not convex hulls or
vertex distances. A negative/near distance preserves a candidate. Consistently
separated sampled meshes remove only the broadphase evidence, never certify
semantic irrelevance or allow automatic changes to training inputs.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import time

import fcl
import numpy as np
import trimesh


def write_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def refine_sequence(task):
    raw, output, records, full_rate_borderline = task
    models = {}

    def model(oid):
        if oid not in models:
            mesh = trimesh.load(Path(raw) / "object_repair/align_ds" / oid / "model.obj", force="mesh", process=False)
            geometry = fcl.BVHModel()
            geometry.beginModel(len(mesh.vertices), len(mesh.faces))
            geometry.addSubModel(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32))
            geometry.endModel()
            models[oid] = fcl.CollisionObject(geometry)
        return models[oid]

    with np.load(Path(raw) / Path(records[0]["source_paths"]["compact_pose"]).name, allow_pickle=False) as data:
        ids, frames, poses = data["object_ids"].tolist(), data["frame_ids"], data["object_pose"]
    completed = []
    for record in records:
        target = Path(output) / "stages" / (record["stage_id"] + ".json")
        if target.exists():
            completed.append(record["stage_id"])
            continue
        result = copy.deepcopy(record)
        start, end = record["frame_interval_half_open"]
        indices = np.flatnonzero((frames >= start) & (frames < end))[::12]
        geometry_results = []
        retained_candidates = []
        for candidate in result["candidate_objects"]:
            a = candidate["object_id"]
            keep_evidence = []
            for evidence in candidate["evidence"]:
                if evidence["kind"] != "possible_contact_or_support_aabb":
                    keep_evidence.append(evidence)
                    continue
                retained_pairs = []
                for broadphase in evidence["detail"]:
                    b = broadphase["object_id"]
                    minimum = float("inf")
                    minimum_frame = None
                    tested = 0
                    error = None
                    full_rate_tested = 0
                    full_rate_clear = False
                    try:
                        ma, mb = model(a), model(b)
                        for index in indices:
                            pa, pb = poses[index, ids.index(a)], poses[index, ids.index(b)]
                            ma.setTransform(fcl.Transform(pa[:3, :3].copy(), pa[:3, 3].copy()))
                            mb.setTransform(fcl.Transform(pb[:3, :3].copy(), pb[:3, 3].copy()))
                            distance = fcl.distance(ma, mb, fcl.DistanceRequest(), fcl.DistanceResult())
                            tested += 1
                            if not np.isfinite(distance):
                                raise ValueError("nonfinite FCL distance")
                            if distance < minimum:
                                minimum, minimum_frame = float(distance), int(frames[index])
                            if minimum <= .015:
                                break
                        if full_rate_borderline and .015 < minimum <= .03:
                            full_indices = np.flatnonzero((frames >= start) & (frames < end))
                            for index in full_indices:
                                pa, pb = poses[index, ids.index(a)], poses[index, ids.index(b)]
                                ma.setTransform(fcl.Transform(pa[:3, :3].copy(), pa[:3, 3].copy()))
                                mb.setTransform(fcl.Transform(pb[:3, :3].copy(), pb[:3, 3].copy()))
                                distance = fcl.distance(ma, mb, fcl.DistanceRequest(), fcl.DistanceResult())
                                full_rate_tested += 1
                                if not np.isfinite(distance):
                                    raise ValueError("nonfinite FCL distance")
                                if distance < minimum:
                                    minimum, minimum_frame = float(distance), int(frames[index])
                                if minimum <= .015:
                                    break
                            full_rate_clear = full_rate_tested == len(full_indices) and minimum > .015
                    except Exception as exc:
                        error = repr(exc)
                    separated = error is None and ((tested == len(indices) and minimum > .03) or full_rate_clear)
                    pair = {"candidate_object_id": a, "required_object_id": b,
                        "min_sampled_triangle_distance_m": minimum if np.isfinite(minimum) else None,
                        "minimum_frame": minimum_frame, "tested_samples": tested,
                        "possible_samples": len(indices), "error": error,
                        "full_rate_samples_tested": full_rate_tested,
                        "classification": ("broadphase_only_full_recorded_rate" if full_rate_clear else "broadphase_only_at_10hz") if separated else "retain_near_or_unverified",
                        "early_exit_on_distance_at_most_15mm": tested < len(indices) and error is None}
                    geometry_results.append(pair)
                    if not separated:
                        retained_pairs.append(pair)
                if retained_pairs:
                    keep_evidence.append({"kind": "mesh_narrowphase_near_or_unverified", "confidence": "review", "detail": retained_pairs})
            if keep_evidence:
                candidate["evidence"] = keep_evidence
                retained_candidates.append(candidate)
        result["candidate_objects"] = retained_candidates
        result["narrowphase"] = {"method": "python-fcl BVH triangle mesh distance, 10Hz; early exit <=15mm; remove AABB-only evidence if all 10Hz samples >30mm, or optional borderline recheck at every recorded frame stays >15mm",
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pairs": geometry_results,
            "limitations": "Discrete samples can miss transient contacts. No semantic irrelevance is inferred. No training filter is enabled."}
        result["issues"] = [i for i in result["issues"] if i["kind"] != "unresolved_candidates"]
        if retained_candidates:
            result["issues"].append({"kind": "unresolved_candidates", "count": len(retained_candidates)})
        result["status"] = "ambiguous" if result["issues"] else "resolved_by_annotation_rules"
        result["safe_candidate_superset_object_ids"] = sorted(set(result["required_object_ids"]) | {c["object_id"] for c in retained_candidates})
        write_json(target, result)
        completed.append(record["stage_id"])
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--full-rate-borderline", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    grouped = defaultdict(list)
    for record in source["stages"]:
        grouped[record["sequence"]].append(record)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "stages").mkdir(exist_ok=True)
    tasks = [(str(args.raw), str(args.output), records, args.full_rate_borderline) for records in grouped.values()]
    done = 0
    write_json(args.output / "progress.json", {"status": "running", "pid": os.getpid(), "completed_stages": 0, "total_stages": len(source["stages"]), "time": time.time()})
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(refine_sequence, task) for task in tasks]
        for future in as_completed(futures):
            done += len(future.result())
            write_json(args.output / "progress.json", {"status": "running", "pid": os.getpid(), "completed_stages": done, "total_stages": len(source["stages"]), "time": time.time()})
            print(f"Narrowphase {done}/{len(source['stages'])}", flush=True)
    records = [json.loads((args.output / "stages" / (r["stage_id"] + ".json")).read_text()) for r in source["stages"]]
    summary = {"recordings": len(grouped), "stages": len(records), "human_verified_complete": 0,
        "status_counts": dict(Counter(r["status"] for r in records)),
        "issue_counts": dict(Counter(i["kind"] for r in records for i in r["issues"])),
        "narrowphase_pair_counts": dict(Counter(p["classification"] for r in records for p in r["narrowphase"]["pairs"])),
        "automatic_training_filter_allowed": False}
    write_json(args.output / "manifest.json", {"schema": "oakink2-stage-required-objects-narrowphase-v1", "summary": summary, "stages": records})
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "review-queue.json", [r for r in records if r["issues"]])
    write_json(args.output / "progress.json", {"status": "complete", "completed_stages": done, "total_stages": len(records), "time": time.time()})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

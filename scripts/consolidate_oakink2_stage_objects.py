"""Audit every OakInk2 program stage without changing training inputs.

Run on the data server. Text/annotation evidence establishes semantic candidates;
pose and bounding-box evidence raises review flags, never proves irrelevance.
"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np


# Physical mechanisms must retain their other annotated moving/fixed parts.
# These are NOT generic category closures (e.g. all bottles or all bread pieces).
MECHANISMS = {
    "O02@0019": "drawer and frame",
    "O02@0032": "tongs parts",
    "O02@0035": "scissors parts",
    "O02@0037": "stapler lever and base",
    "O02@0039": "microwave cavity, gate and lever",
    "O02@0044": "pencil sharpener base and handle",
    "O02@0045": "lamp base and head",
    "O02@0053": "laptop base and lid",
    "O02@0203": "test tube holder parts",
}
SYNONYMS = {
    "wine glass": "wineglass", "tooth paste": "toothpaste",
    "tooth brush": "toothbrush", "frying pan": "fryingpan",
    "alcohol lamp": "alcohol burner", "gamecontroller": "gamecontroler",
    "rack": "test tube rack", "holder": "test tube holder",
    "sharpener": "pencil sharpener", "porcelain cap": "teapot cap",
    "small green cup": "little green cup", "green little cup": "little green cup",
}
ACTION_REQUIRED = {
    "remove_lid": ["box body"], "put_on_lid": ["box body"],
    "plug_in_power_plug": ["power socket"], "remove_power_plug": ["power socket"],
    "heat_beaker": ["tripod", "asbestos mesh", "alcohol burner"],
    "heat_test_tube": ["alcohol burner"],
    "remove_from_test_tube_rack": ["test tube rack"],
    "place_on_test_tube_rack": ["test tube rack"],
    "place_test_tube_on_rack_with_holder": ["test tube rack", "test tube holder"],
    "remove_test_tube_from_rack_with_holder": ["test tube rack", "test tube holder"],
}


def normalize(value):
    value = value.lower().replace("-", " ")
    return " ".join(re.findall(r"[a-z0-9]+", value))


def aliases(name):
    names = {name}
    for suffix in (" body", " base", " lever", " cavity", " gate", " head", " handle", " lid"):
        if name.endswith(suffix):
            names.add(name[:-len(suffix)])
    # Caps do not match the whole container, otherwise pouring adds unused caps.
    if name == "toothpaste tube":
        names.add("toothpaste")
    for old, new in SYNONYMS.items():
        if new in names:
            names.add(old)
    return {normalize(n) for n in names}


def text_mentions(description, alias_ids):
    text = " " + normalize(description) + " "
    matches = []
    for phrase, ids in alias_ids.items():
        for m in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text):
            matches.append((m.start(), m.end(), phrase, sorted(ids)))
    # 'whiteboard eraser' must not also mean the separate whiteboard.
    return [m for m in matches if not any(a <= m[0] and b >= m[1] and b-a > m[1]-m[0]
                                         for a, b, _, _ in matches)]


def mesh_bounds(raw, object_id):
    path = raw / "object_repair/align_ds" / object_id / "model.obj"
    vertices = []
    if not path.is_file():
        return None
    for line in path.open():
        if line.startswith("v "):
            vertices.append([float(v) for v in line.split()[1:4]])
    v = np.asarray(vertices)
    if not len(v):
        return None
    lo, hi = v.min(0), v.max(0)
    return np.asarray([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--scene-inventory", type=Path, help="Optional legacy scene inventory; otherwise read compact metadata")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-selection", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = json.loads((args.raw / "object_repair/obj_desc.json").read_text())
    scenes = ({r["sequence"]: r["object_ids"] for r in json.loads(args.scene_inventory.read_text())["records"]["oakink2_scene_unique_from_raw_obj_list"]}
              if args.scene_inventory else {})
    alias_ids = defaultdict(set)
    for oid, desc in names.items():
        for phrase in aliases(desc["obj_name"]):
            alias_ids[phrase].add(oid)
    bounds = {oid: mesh_bounds(args.raw, oid) for oid in names}
    records = []
    for program_path in sorted((args.raw / "program/program_info").glob("*.json")):
        program = json.loads(program_path.read_text())
        desc_path = args.raw / "program/desc_info" / program_path.name
        descriptions = json.loads(desc_path.read_text())
        intervals = {key: [v for v in ast.literal_eval(key) if v is not None] for key in program}
        compact_path = args.compact / (program_path.stem + ".npz")
        with np.load(compact_path, allow_pickle=False) as data:
            frame_ids = data["frame_ids"]
            pose_ids = data["object_ids"].tolist()
            poses = data["object_pose"]
        scene_ids = set(scenes.get(program_path.stem, pose_ids))
        for index, (key, stage) in enumerate(program.items()):
            intervals_stage = intervals[key]
            start, end = min(v[0] for v in intervals_stage), max(v[1] for v in intervals_stage)
            description = descriptions.get(key, {})
            required, candidates, issues = defaultdict(list), defaultdict(list), []

            def add(oid, kind, detail, confidence="high"):
                target = required if confidence == "high" else candidates
                item = {"kind": kind, "detail": detail, "confidence": confidence}
                if item not in target[oid]:
                    target[oid].append(item)

            for field in ("obj_list", "obj_list_lh", "obj_list_rh"):
                for oid in stage.get(field) or []:
                    add(oid, "explicit_annotation", field)
                    if oid not in scene_ids:
                        issues.append({"kind": "annotated_object_absent_from_scene", "object_id": oid})
            explicit = set(required)
            if not explicit:
                issues.append({"kind": "no_explicit_objects"})
            if not description.get("seg_desc"):
                issues.append({"kind": "missing_stage_description"})
            matched = []
            for left, right, phrase, all_ids in text_mentions(description.get("seg_desc", ""), alias_ids):
                ids = sorted(scene_ids.intersection(all_ids))
                matched.append({"phrase": phrase, "scene_candidates": ids})
                if not ids:
                    issues.append({"kind": "text_object_absent_from_scene", "phrase": phrase})
                elif len(ids) == 1:
                    add(ids[0], "unique_text_reference", phrase)
                else:
                    # All mechanism parts are needed; other duplicate names remain ambiguous.
                    groups = {i.rsplit("@", 1)[0] for i in ids}
                    if len(groups) == 1 and next(iter(groups)) in MECHANISMS:
                        for oid in ids:
                            add(oid, "mechanism_text_reference", phrase)
                    elif set(ids).issubset(required):
                        for oid in ids:
                            add(oid, "text_reference_covered_by_explicit_objects", phrase)
                    else:
                        issues.append({"kind": "ambiguous_text_identity", "phrase": phrase, "object_ids": ids})
                        for oid in ids:
                            add(oid, "ambiguous_text_reference", phrase, "review")
            for category in ACTION_REQUIRED.get(stage["primitive"], []):
                ids = sorted(scene_ids.intersection(alias_ids.get(category, set())))
                if len(ids) == 1:
                    # A semantic dependency is inference, not an official object annotation.
                    add(ids[0], "primitive_dependency", {"primitive": stage["primitive"], "category": category}, "review")
                elif not ids:
                    issues.append({"kind": "primitive_dependency_missing", "category": category})
                else:
                    for oid in ids:
                        add(oid, "primitive_dependency_ambiguous", category, "review")
            # Simultaneous independently annotated actions need their objects in this union interval.
            overlaps = []
            for other_index, (other_key, other) in enumerate(program.items()):
                if other_key == key:
                    continue
                overlap = sum(max(0, min(end, b) - max(start, a)) for a, b in intervals[other_key])
                if overlap > 0:
                    overlaps.append(other_index)
                    for oid in set(other.get("obj_list") or []):
                        add(oid, "overlapping_stage_object", {"stage_index": other_index, "interval_key": other_key})
            for oid in list(required):
                prefix = oid.rsplit("@", 1)[0]
                if prefix in MECHANISMS:
                    for part in sorted(scene_ids):
                        if part.rsplit("@", 1)[0] == prefix:
                            add(part, "mechanism_dependency", {"object_id": oid, "mechanism": MECHANISMS[prefix]})
            indices = np.flatnonzero((frame_ids >= start) & (frame_ids < end))[::12]
            pose_summary = {}
            world_boxes = {}
            if not len(indices):
                issues.append({"kind": "no_pose_frames"})
            else:
                for j, oid in enumerate(pose_ids):
                    p = poses[indices, j]
                    t = p[:, :3, 3]
                    pose_summary[oid] = {"max_translation_from_first_m": round(float(np.linalg.norm(t-t[0], axis=1).max()), 5)}
                    if not np.isfinite(p).all():
                        issues.append({"kind": "nonfinite_pose", "object_id": oid})
                    if bounds.get(oid) is not None:
                        vertices = np.einsum("fij,kj->fki", p[:, :3, :3], bounds[oid]) + t[:, None]
                        world_boxes[oid] = (vertices.min(1), vertices.max(1))
                for oid, (lo, hi) in world_boxes.items():
                    if oid in required:
                        continue
                    proximity = []
                    for used in required:
                        if used not in world_boxes:
                            continue
                        ulo, uhi = world_boxes[used]
                        gap = np.maximum(np.maximum(lo-uhi, ulo-hi), 0)
                        distance = np.linalg.norm(gap, axis=1)
                        near = distance < .015
                        if near.mean() >= .1:
                            proximity.append({"object_id": used, "min_aabb_gap_m": round(float(distance.min()), 5), "fraction_within_15mm": round(float(near.mean()), 4)})
                    if proximity:
                        add(oid, "possible_contact_or_support_aabb", proximity, "review")
                    if pose_summary[oid]["max_translation_from_first_m"] > .05:
                        add(oid, "unassigned_moving_object", pose_summary[oid], "review")
            # Remove review status for already established identities, retaining all evidence.
            for oid in set(required).intersection(candidates):
                required[oid].extend(candidates.pop(oid))
            if stage["primitive"] == "shear_paper" and "stapler" in description.get("seg_desc", "").lower():
                issues.append({"kind": "primitive_description_conflict", "detail": "shear_paper description says stapler"})
            if candidates:
                issues.append({"kind": "unresolved_candidates", "count": len(candidates)})
            def objects(mapping):
                return [{"object_id": oid, "name": names.get(oid, {}).get("obj_name"), "evidence": evidence} for oid, evidence in sorted(mapping.items())]
            records.append({"sequence": program_path.stem, "stage_index": index, "stage_id": program_path.stem + f"__st{index:02d}",
                "interval_key": key, "frame_interval_half_open": [start, end], "fps": 120,
                "primitive": stage["primitive"], "description": description, "source_annotation": stage,
                "source_paths": {"program": str(program_path), "description": str(desc_path), "compact_pose": str(compact_path)},
                "source_program_sha256": hashlib.sha256(program_path.read_bytes()).hexdigest(),
                "scene_object_ids": sorted(scene_ids), "explicit_object_ids": sorted(explicit),
                "required_object_ids": sorted(required), "required_objects": objects(required),
                "candidate_objects": objects(candidates), "text_mentions": matched,
                "overlapping_stage_indices": overlaps, "pose_summary_10hz": pose_summary,
                "issues": issues, "status": "ambiguous" if issues else "resolved_by_annotation_rules",
                "verified_complete_by_human": False, "automatic_training_filter_allowed": False,
                "table_is_separate": True, "max8_known_required_pass": len(required) <= 8,
                "safe_candidate_superset_object_ids": sorted(set(required) | set(candidates)),
                "unselected_scene_objects_are_not_proven_irrelevant": True})
        print(f"Scanned {program_path.stem}: {len(program)} stages", flush=True)
    summary = {"recordings": len(scenes), "stages": len(records),
        "status_counts": dict(Counter(r["status"] for r in records)),
        "human_verified_complete": 0,
        "stages_with_additions_beyond_explicit": sum(set(r["required_object_ids"]) != set(r["explicit_object_ids"]) for r in records),
        "required_object_count_distribution": dict(sorted(Counter(len(r["required_object_ids"]) for r in records).items())),
        "issue_counts": dict(Counter(i["kind"] for r in records for i in r["issues"])),
        "missing_mesh_bounds": sorted(oid for oid in set().union(*map(set, scenes.values())) if bounds.get(oid) is None),
        "method": "explicit annotation union + longest object-name phrase resolution + mechanism dependencies + overlapping actions + 10Hz pose/AABB review flags",
        "limitations": ["Automatic semantic resolution is not human-verified ground truth.", "No candidate or unselected object is certified irrelevant.", "AABB overlap is broadphase proximity, NOT mesh contact; false positives expected.", "Pose sampling can miss brief contacts; no MANO mesh contact is calculated.", "Liquids, contents and untracked table are not discrete object IDs.", "No training files are modified; all records disallow automatic training filtering until reviewed."]}
    json_write(args.output / "manifest.json", {"schema": "oakink2-stage-required-objects-audit-v1", "summary": summary, "stages": records})
    json_write(args.output / "summary.json", summary)
    json_write(args.output / "review-queue.json", [r for r in records if r["issues"]])
    if args.preview_selection:
        selection = json.loads(args.preview_selection.read_text())
        preview = []
        for selected in selection:
            record = next(r for r in records if r["sequence"] == selected["sequence"] and r["stage_index"] == selected["source_stage"])
            preview.append({"task_number": selected["number"], **record})
        json_write(args.output / "preview-eight-audit.json", preview)
    with (args.output / "stages.jsonl").open("w") as stream:
        for r in records:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    (args.output / "README.md").write_text("# OakInk2 stage object audit\n\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n\n`required_object_ids` includes annotation-backed and rule-resolved identities. `candidate_objects` require review; absence from either list does not prove irrelevance. No row is approved as a complete physical scene. Retain table separately. The frame interval is the union of both hand intervals; simultaneous separately annotated actions are recorded.\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

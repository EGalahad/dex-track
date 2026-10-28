"""Resolve explicit hand-role references and retain complete rigid scenes.

The source manifest is immutable. Strict resolutions and conservative full-scene
candidates are separate outputs. Complete-scene retention resolves omission risk,
not semantic relevance, and does not grant physics-QA or training admission.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from label_oakink2_simulation_eligibility import label


# Only actions in which the named object is explicitly the manipulated item.
# Pouring/scooping destinations cannot be resolved from hand IDs alone.
HAND_ROLE_ACTIONS = {
    "hold", "rearrange", "place_onto", "shake_lab_container",
    "remove_from_test_tube_rack", "place_on_test_tube_rack",
}
RESOLVABLE_ISSUES = {"unresolved_candidates", "ambiguous_text_identity"}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def resolve_hand_roles(record):
    record = copy.deepcopy(record)
    changes = []
    if record["simulation_eligibility"]["status"] != "review":
        return record, changes
    if record["primitive"] not in HAND_ROLE_ACTIONS:
        return record, changes
    description = record["description"].get("seg_desc", "").lower().replace("-", " ")
    if re.search(r"\b(another|two|both|second|several|multiple)\b", description):
        return record, changes
    source = record["source_annotation"]
    hand_ids = set(source.get("obj_list_lh") or []) | set(source.get("obj_list_rh") or [])
    keep_issues = []
    for issue in record["issues"]:
        if issue["kind"] != "ambiguous_text_identity":
            keep_issues.append(issue)
            continue
        phrase = issue["phrase"]
        matches = set(issue["object_ids"]) & hand_ids
        occurrences = len(re.findall(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", description))
        if len(matches) != 1 or occurrences != 1:
            keep_issues.append(issue)
            continue
        selected = next(iter(matches))
        evidence = {"kind": "unique_explicit_hand_role", "confidence": "high",
            "detail": {"phrase": phrase, "selected_object_id": selected,
                       "hand_object_ids": sorted(hand_ids), "primitive": record["primitive"],
                       "reason": "Single text occurrence, no multiplicity language, exactly one matching explicit hand ID in a manipulated-item action."}}
        changes.append(evidence["detail"])
        for obj in record["required_objects"]:
            if obj["object_id"] == selected:
                obj["evidence"] = [e for e in obj["evidence"] if not (e["kind"] == "ambiguous_text_reference" and e["detail"] == phrase)]
                obj["evidence"].append(evidence)
        for obj in record["candidate_objects"]:
            obj["evidence"] = [e for e in obj["evidence"] if not (e["kind"] == "ambiguous_text_reference" and e["detail"] == phrase)]
    if changes:
        record["candidate_objects"] = [o for o in record["candidate_objects"] if o["evidence"]]
        record["issues"] = [i for i in keep_issues if i["kind"] != "unresolved_candidates"]
        if record["candidate_objects"]:
            record["issues"].append({"kind": "unresolved_candidates", "count": len(record["candidate_objects"])})
        record["safe_candidate_superset_object_ids"] = sorted(set(record["required_object_ids"]) | {o["object_id"] for o in record["candidate_objects"]})
        record["status"] = "ambiguous" if record["issues"] else "resolved_by_explicit_hand_role"
        record["simulation_eligibility"] = label(record)
    return record, changes


class PoseValidator:
    def __init__(self, raw):
        self.raw = raw
        self.cached_path = None

    def validate(self, record, object_ids):
        # Published metadata uses portable paths; the supplied reference root
        # takes precedence over paths recorded on the preparation machine.
        path = self.raw / Path(record["source_paths"]["compact_pose"]).name
        if self.cached_path != path:
            with np.load(path, allow_pickle=False) as data:
                self.ids = data["object_ids"].tolist()
                self.frames = data["frame_ids"]
                self.poses = data["object_pose"]
            self.cached_path = path
        start, end = record["frame_interval_half_open"]
        indices = np.flatnonzero((self.frames >= start) & (self.frames < end))
        problems = []
        expected = np.arange(start, end)
        if not np.array_equal(self.frames[indices], expected):
            problems.append("frame_ids_not_complete_half_open_stage_interval")
        if set(self.ids) != set(record["scene_object_ids"]):
            problems.append("compact_object_ids_disagree_with_recording_scene_inventory")
        for oid in object_ids:
            if oid not in self.ids:
                problems.append("missing_pose:" + oid)
                continue
            matrices = self.poses[indices, self.ids.index(oid)]
            if not np.isfinite(matrices).all():
                problems.append("nonfinite_pose:" + oid)
            if not np.allclose(matrices[:, 3, :], np.array([0, 0, 0, 1]), atol=1e-5):
                problems.append("invalid_homogeneous_pose:" + oid)
            mesh = self.raw / "object_repair/align_ds" / oid / "model.obj"
            if not mesh.is_file() or mesh.stat().st_size == 0:
                problems.append("missing_mesh:" + oid)
        return {"status": "pass" if not problems else "fail", "issues": problems,
            "frame_count": len(indices), "expected_frame_count": end-start,
            "object_ids": sorted(object_ids), "source": path,
            "scope": "Every original 120Hz frame; finite 4x4 pose and mesh presence; NOT collision/retarget QA"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_bytes = args.manifest.read_bytes()
    source = json.loads(source_bytes)
    provenance = {"source_manifest": str(args.manifest), "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_is_immutable": True, "human_verified_complete": False}
    names = json.loads((args.raw / "object_repair/obj_desc.json").read_text())
    validator = PoseValidator(args.raw)
    strict_records, strict_added, changed, scene_added, scene_rejected = [], [], [], [], []
    for original in source["stages"]:
        record, resolutions = resolve_hand_roles(original)
        record["audit_v2"] = {**provenance, "hand_role_resolutions": resolutions,
            "original_status": original["status"], "original_simulation_eligibility": original["simulation_eligibility"]["status"]}
        if resolutions:
            changed.append({"stage_id": record["stage_id"], "resolutions": resolutions,
                "remaining_issues": record["issues"]})
        if original["simulation_eligibility"]["status"] != "candidate_rigid_pose_tracking" and record["simulation_eligibility"]["status"] == "candidate_rigid_pose_tracking":
            verification = validator.validate(record, record["required_object_ids"])
            record["audit_v2"]["pose_validation"] = verification
            if verification["status"] == "pass":
                strict_added.append(record)
            else:
                record["issues"].append({"kind": "pose_validation_failed", "details": verification["issues"]})
                record["status"] = "ambiguous"
                record["simulation_eligibility"] = label(record)
        strict_records.append(record)
        # These independent alternatives deliberately start from the unchanged
        # baseline, so downstream users can identify overlap with strict additions.
        if original["simulation_eligibility"]["status"] != "review":
            continue
        review_kinds = {e["kind"] for e in original["simulation_eligibility"]["review_reasons"]}
        if review_kinds != {"object_set_not_resolved"}:
            continue
        blockers = []
        if any(i["kind"] not in RESOLVABLE_ISSUES for i in original["issues"]):
            blockers.append("hard_missing_id_or_target_or_annotation_conflict")
        scene_ids = original["scene_object_ids"]
        if not 1 <= len(scene_ids) <= 8:
            blockers.append("full_recorded_scene_not_within_1_to_8_objects")
        if not set(original["safe_candidate_superset_object_ids"]).issubset(scene_ids):
            blockers.append("candidate_or_required_id_absent_from_recorded_scene")
        if blockers:
            scene_rejected.append({"stage_id": original["stage_id"], "blockers": blockers})
            continue
        expanded = copy.deepcopy(original)
        expanded["semantic_relevance_audit"] = {"status": "unresolved_relevance_preserved_by_full_scene_retention",
            "original_required_object_ids": original["required_object_ids"],
            "original_candidate_objects": original["candidate_objects"], "original_issues": original["issues"],
            "claim": "Every recorded object is retained; no claim that every retained object is task-relevant or that this is a minimal scene."}
        expanded["required_object_ids"] = sorted(scene_ids)
        expanded["required_objects"] = [{"object_id": oid, "name": names[oid]["obj_name"],
            "evidence": [{"kind": "complete_recorded_scene_retention", "confidence": "complete_inventory_not_semantic_relevance",
                "detail": "All scene IDs retained to avoid omitting ambiguous participants, supports or incidental contacts."}]} for oid in sorted(scene_ids)]
        expanded["candidate_objects"] = []
        expanded["issues"] = []
        expanded["safe_candidate_superset_object_ids"] = sorted(scene_ids)
        expanded["status"] = "complete_recorded_scene_retained_relevance_uncertain"
        expanded["max8_known_required_pass"] = True
        expanded["simulation_eligibility"] = label(expanded)
        if expanded["simulation_eligibility"]["status"] != "candidate_rigid_pose_tracking":
            scene_rejected.append({"stage_id": original["stage_id"], "blockers": ["expanded_scene_fails_rigid_capability_rules"],
                "simulation_eligibility": expanded["simulation_eligibility"]})
            continue
        verification = validator.validate(expanded, scene_ids)
        if verification["status"] != "pass":
            scene_rejected.append({"stage_id": original["stage_id"], "blockers": ["pose_validation_failed"], "pose_validation": verification})
            continue
        expanded["audit_v2"] = {**provenance, "selection_mode": "all_recorded_scene_objects",
            "pose_validation": verification, "original_stage_issues_preserved_in": "semantic_relevance_audit"}
        expanded["simulation_eligibility"]["selection_mode"] = "all_recorded_scene_objects"
        expanded["simulation_eligibility"]["physics_qa_status"] = "pending"
        scene_added.append(expanded)
    strict_ids = {r["stage_id"] for r in strict_added}
    scene_ids = {r["stage_id"] for r in scene_added}
    new_preparation = {r["stage_id"]: r for r in scene_added}
    new_preparation.update({r["stage_id"]: r for r in strict_added})
    new_preparation = sorted(new_preparation.values(), key=lambda r: r["stage_id"])
    summary = {"source": provenance, "recordings": len({r["sequence"] for r in strict_records}), "stages": len(strict_records),
        "baseline_candidate_count": source["summary"]["simulation_eligibility_counts"]["candidate_rigid_pose_tracking"],
        "strict_new_candidate_count": len(strict_added), "strict_changed_stage_count": len(changed),
        "strict_simulation_eligibility_counts": dict(Counter(r["simulation_eligibility"]["status"] for r in strict_records)),
        "complete_scene_candidate_count": len(scene_added), "overlap_strict_and_scene_candidate_count": len(strict_ids & scene_ids),
        "unique_new_preparation_candidates": len(strict_ids | scene_ids),
        "combined_baseline_and_new_candidate_count": source["summary"]["simulation_eligibility_counts"]["candidate_rigid_pose_tracking"] + len(new_preparation),
        "deduplication": "Prefer explicit-hand-role strict resolution over full-scene retention when both identify the same stage.",
        "complete_scene_object_count_distribution": dict(sorted(Counter(len(r["required_object_ids"]) for r in scene_added).items())),
        "full_scene_rejected_reasons": dict(Counter(b for r in scene_rejected for b in r["blockers"])),
        "remaining_strict_review_reasons": dict(Counter(e["kind"] for r in strict_records if r["simulation_eligibility"]["status"] == "review" for e in r["simulation_eligibility"]["review_reasons"])),
        "physics_qa": "pending for all new candidates", "training_data_modified": False}
    strict_manifest = {"schema": "oakink2-stage-objects-strict-audit-v2", "summary": summary, "stages": strict_records}
    write_json(args.output / "manifest.json", strict_manifest)
    write_json(args.output / "strict-new-candidates.json", {"schema": strict_manifest["schema"], "summary": summary, "stages": strict_added})
    write_json(args.output / "complete-scene-candidates.json", {"schema": "oakink2-complete-recorded-scene-candidates-v1", "summary": summary, "stages": scene_added})
    write_json(args.output / "new-preparation-candidates.json", {"schema": "oakink2-new-preparation-candidates-v2", "summary": summary, "stages": new_preparation})
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "hand-role-resolutions.json", changed)
    write_json(args.output / "complete-scene-rejected.json", scene_rejected)
    write_json(args.output / "review-queue.json", [r for r in strict_records if r["simulation_eligibility"]["status"] == "review"])
    write_json(args.output / "remaining-after-scene-preparation.json", [r for r in strict_records if r["simulation_eligibility"]["status"] == "review" and r["stage_id"] not in scene_ids])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

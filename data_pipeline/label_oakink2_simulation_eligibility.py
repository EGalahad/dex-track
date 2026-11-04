"""Label stage feasibility for today's independent-rigid-object simulator.

These labels describe current model capabilities, not dataset validity. All
stages remain eligible for kinematic reference visualization.
"""
import argparse
from collections import Counter
import json
from pathlib import Path


DEFORMATION_ACTIONS = {
    "cut": "cutting/fracture changes object geometry",
    "sharpen_pencil": "material removal changes pencil geometry",
    "shear_paper": "cutting/deforming paper",
    "staple_paper_together": "paper penetration and permanent fastening",
    "squeeze_tooth_paste": "deformable tube and paste flow",
    "spread": "deposition/spreading of deformable material",
    "wipe": "sponge deformation and surface cleaning",
    "open_book": "flexible pages and binding",
    "close_book": "flexible pages and binding",
}
MECHANISM_ACTIONS = {
    "screw", "unscrew", "insert_lightbulb", "remove_lightbulb",
    "insert_pencil", "remove_pencil", "pull_out_drawer", "push_in_drawer",
    "open_gate", "close_gate", "trigger_lever", "open_laptop_lid",
    "close_laptop_lid", "press_button", "use_keyboard", "use_mouse",
    "use_gamecontroller", "flip_open_tooth_paste_cap", "flip_close_tooth_paste_cap",
}
MECHANISM_GROUPS = {
    "O02@0019": "drawer and frame require their sliding mechanism",
    "O02@0032": "tongs require articulation/compliance",
    "O02@0035": "scissors require articulation",
    "O02@0037": "stapler requires articulation and spring/latch",
    "O02@0039": "microwave door/lever require hinges/latches",
    "O02@0044": "sharpener requires coupled mechanism",
    "O02@0045": "lamp base/head mechanism is not represented",
    "O02@0053": "laptop base/lid hinge is not represented",
    "O02@0203": "test tube holder requires articulation/compliance",
}
FLOW_ACTIONS = {"pour", "pour_in_lab", "scoop", "scrape", "stir", "stir_experiment_substances", "shake_lab_container"}
NONMECHANICAL_ACTIONS = {"heat_beaker", "heat_test_tube", "ignite_alcohol_lamp", "put_off_alcohol_lamp"}
FIT_REVIEW_ACTIONS = {"cap", "uncap", "cap_the_pen", "remove_the_pen_cap", "insert_usb", "remove_usb", "plug_in_power_plug", "remove_power_plug", "assemble"}
SOFT_ASSET_GROUPS = {
    "O02@0056": "paper", "O02@0060": "book/pages", "O02@0100": "sponge",
    "O02@0003": "toothpaste tube/cap assembly", "O02@0081": "flower",
    "O02@0094": "bread", "O93001": "donut",
}


def label(record):
    action = record["primitive"]
    reasons = []
    exclusions = []
    if action in DEFORMATION_ACTIONS:
        exclusions.append({"kind": "deformation_or_material_change", "evidence": DEFORMATION_ACTIONS[action]})
    if action in MECHANISM_ACTIONS:
        exclusions.append({"kind": "articulation_thread_or_latch", "evidence": f"primitive={action}; independent free rigid objects do not implement the mechanism"})
    for oid in record["required_object_ids"]:
        group = oid.rsplit("@", 1)[0]
        if group in MECHANISM_GROUPS:
            exclusions.append({"kind": "required_articulated_asset", "object_id": oid, "evidence": MECHANISM_GROUPS[group]})
        if group in SOFT_ASSET_GROUPS or oid in SOFT_ASSET_GROUPS:
            reasons.append({"kind": "soft_asset_rigid_approximation_not_approved", "object_id": oid,
                "evidence": "Even if this action only moves the asset, a rigid approximation must not be assumed acceptable; deformation-dependent actions are excluded separately."})
    if action in FIT_REVIEW_ACTIONS:
        reasons.append({"kind": "fit_or_latch_model_uncertain", "evidence": f"primitive={action}; review mating geometry, friction, interference and any omitted latch/compliance"})
    if action in NONMECHANICAL_ACTIONS:
        exclusions.append({"kind": "thermal_or_combustion_goal_unsupported", "evidence": f"primitive={action}; no heat, flame, ignition or extinguishing model"})
    for candidate in record["candidate_objects"]:
        group = candidate["object_id"].rsplit("@", 1)[0]
        if group in MECHANISM_GROUPS:
            reasons.append({"kind": "candidate_articulated_asset", "object_id": candidate["object_id"]})
    if record["issues"]:
        reasons.append({"kind": "object_set_not_resolved", "evidence": sorted({i["kind"] for i in record["issues"]})})
    if len(record["required_object_ids"]) > 8:
        exclusions.append({"kind": "required_object_count_exceeds_8"})
    if len(record["safe_candidate_superset_object_ids"]) > 8:
        reasons.append({"kind": "candidate_object_count_exceeds_8"})
    status = "exclude" if exclusions else "review" if reasons else "candidate_rigid_pose_tracking"
    goal = {"supported": "hand and rigid-object reference pose tracking only",
        "task_success_claim_allowed": False,
        "fluid_or_contents_success": "not_applicable",
        "physical_validation": "collision replay and learned rollout still required"}
    if action in FLOW_ACTIONS:
        goal["fluid_or_contents_success"] = "unsupported: no fluid, granular contents, transfer amount, mixing or filling simulation"
        goal["supported"] = "rigid containers/tools and hand pose tracking only; fluid/material task success is not modeled"
    if action in {"write_on_paper", "write_on_whiteboard", "brush_whiteboard"}:
        goal["unmodeled_surface_effect"] = "ink deposition/removal and written content are not modeled; only rigid tool/board pose tracking"
    return {"status": status, "visualization": "kinematic_replay_allowed",
        "current_backend_object_model": "independent RigidObject/freejoint bodies; no articulated object constraints",
        "excluded_reasons": exclusions, "review_reasons": reasons, "goal_scope": goal,
        "training_admission": False,
        "training_admission_reason": "candidate is not approval: object completeness, asset collision checks and environment support must also pass"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for record in data["stages"]:
        record["simulation_eligibility"] = label(record)
    counts = Counter(r["simulation_eligibility"]["status"] for r in data["stages"])
    data["summary"]["simulation_eligibility_counts"] = dict(counts)
    data["summary"]["simulation_eligibility_exclusion_reasons"] = dict(Counter(e["kind"] for r in data["stages"] for e in r["simulation_eligibility"]["excluded_reasons"]))
    data["schema"] = "oakink2-stage-required-objects-and-simulation-eligibility-v1"
    for name, value in (("manifest.json", data), ("summary.json", data["summary"]),
                        ("review-queue.json", [r for r in data["stages"] if r["issues"] or r["simulation_eligibility"]["status"] == "review"]),
                        ("eligibility.json", [{"stage_id": r["stage_id"], "required_object_ids": r["required_object_ids"], "simulation_eligibility": r["simulation_eligibility"]} for r in data["stages"]])):
        temporary = args.output / (name + ".partial")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output / name)
    print(json.dumps(data["summary"], indent=2))


if __name__ == "__main__":
    main()

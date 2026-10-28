"""Validate a reviewed object-removal plan and export a fresh stage patch.

This writes metadata only. Prepare the exported stages in a NEW output directory,
then rerun hand association, reset retargeting and full-scene physical QA. Original
datasets, meshes, human references and training packages are never modified.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def pair(row):
    return row['dataset'], row['clip'], row['object_id']


def build_patch(plan_path):
    plan = json.loads(plan_path.read_text())
    if plan['schema'] != 'oakink2_object_removal_plan_v1':
        raise ValueError('Unsupported object audit schema')
    checked, records, entries = {}, {}, {}

    def verify(path, expected):
        path = str(Path(path).resolve())
        actual = checked.setdefault(path, None)
        if actual is None:
            actual = checked[path] = sha256(path)
        if actual != expected:
            raise ValueError(f'Audit input changed: {path}')

    for path, expected in plan['source_audits'].items():
        verify(path, expected)
    for entry in plan['entries']:
        key = pair(entry)
        if key in entries:
            raise ValueError(f'Duplicate audit entry: {key}')
        if (entry['action'] != 'remove_from_new_package_only'
                or entry['conflicting_protection_evidence']):
            raise ValueError(f'Entry is not approved by the audit: {key}')
        entries[key] = entry
        for source in entry['source_inputs'].values():
            verify(source['path'], source['sha256'])
        for evidence in entry['evidence']:
            verify(evidence['audit_path'], evidence['audit_sha256'])
            verify(evidence['source_record'], evidence['source_record_sha256'])
        for geometry in entry['geometry_provenance']:
            verify(Path(geometry['requested_object_directory']) / 'visual.obj',
                   geometry['requested_visual_sha256'])
            verify(Path(geometry['geometry_directory']) / 'visual.obj',
                   geometry['reference_visual_sha256'])
            for path, expected in geometry['convex_sha256'].items():
                verify(path, expected)
        source = entry['source_inputs']['clip_record']
        key = entry['dataset'], entry['clip']
        record = json.loads(Path(source['path']).read_text())
        if record['clip'] != entry['clip'] or record['dataset'] != 'oakink2':
            raise ValueError(f'Wrong source identity: {key}')
        if key in records and records[key] != record:
            raise ValueError(f'Conflicting clip records: {key}')
        records[key] = record

    proposed = set(entries)
    protected = {pair(row) for row in plan['uncertain_do_not_delete']}
    if proposed & protected:
        raise ValueError('Uncertain objects cannot be removed')
    for group in plan['atomic_groups']:
        members = {pair(row) for row in group['members']}
        selected = members & proposed
        if selected and (group['blocked_by_conflict'] or selected != members):
            raise ValueError(f'Partial or blocked background group: {group["group_id"]}')

    plan_digest = sha256(plan_path)
    stages, covered, identities = [], set(), set()
    for clip_plan in plan['clip_plans']:
        key = clip_plan['dataset'], clip_plan['clip']
        record = records[key]
        stage = deepcopy(record['source_stage'])
        if stage['stage_id'] in identities:
            raise ValueError(f'Duplicate stage identity: {stage["stage_id"]}')
        identities.add(stage['stage_id'])
        if stage['simulation_eligibility']['status'] != 'candidate_rigid_pose_tracking':
            raise ValueError(f'Audit cannot promote an excluded/review stage: {key}')
        original = set(stage['required_object_ids'])
        remove = set(clip_plan['remove_object_ids'])
        retained = set(clip_plan['retained_object_ids'])
        selected = {(*key, oid) for oid in remove}
        if (original != set(clip_plan['original_object_ids']) or not remove
                or not remove <= original or retained != original - remove
                or not 1 <= len(retained) <= 8 or not selected <= proposed):
            raise ValueError(f'Object set mismatch: {key}')
        for item in selected:
            if entries[item]['source_inputs']['clip_record']['sha256'] != clip_plan['source_record_sha256']:
                raise ValueError(f'Clip record hash mismatch: {key}')
        covered.update(selected)
        stage['object_selection_audit'] = dict(
            plan_sha256=plan_digest, source_dataset=key[0],
            original_source_stage=deepcopy(stage),
            source_record_sha256=clip_plan['source_record_sha256'],
            removed_object_ids=sorted(remove), retained_object_ids=sorted(retained),
            rebuild_hand_association=True, rebuild_reset_retarget=True,
            preserve_whole_recording_table_fit_context=True)
        stage['required_object_ids'] = sorted(retained)
        stages.append(stage)
    if covered != proposed:
        raise ValueError('Clip plans do not cover exactly the audited removals')
    return dict(
        schema='dex-track-oakink2-audited-stage-patch-v1', dataset='oakink2',
        scope='affected stages only; overlay rebuilt clips onto the full corpus',
        plan_path=str(plan_path.resolve()), plan_sha256=plan_digest,
        checked_files=len(checked), removal_count=len(proposed),
        affected_clips=len(stages), uncertain_objects_preserved=len(protected),
        removed_assigned_objects=sum(bool(e['current_assigned_object_removed']) for e in entries.values()),
        source_dataset_counts=dict(Counter(key[0] for key in records)),
        training_admission=False, requires_fresh_output=True,
        pending=['regenerate human references and hand association',
                 'reset-only retarget and scene pack', 'full-scene physical QA'],
        stages=stages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path,
                        help='New stage manifest path. Omit for read-only validation.')
    args = parser.parse_args()
    result = build_patch(args.plan)
    if args.output:
        # Exclusive create prevents replacing an input manifest or previous view.
        with args.output.open('x') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'stages'}, indent=2))


if __name__ == '__main__':
    main()

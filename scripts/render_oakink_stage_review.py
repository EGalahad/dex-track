"""Render original compact OakInk2 MANO/objects for stage-object review.

All recorded object meshes and poses are shown at original scale. Two fixed
views share the same source frames. No retargeting or physics is applied.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
import pyrender
from scipy.spatial.transform import Rotation
import torch
import trimesh

from render_episode_pair import camera_pose, UPSTREAM
from retarget.human_demo.sources.base import Hand, _forward, _mano, YUP_TO_ZUP
from retarget.human_demo.sources.oakink2 import Source


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compact_hand(raw, prefix, indices):
    quat = raw[f'{prefix}__pose_coeffs'][indices].astype(np.float64)
    rotations = Rotation.from_quat(quat.reshape(-1, 4)[:, [1, 2, 3, 0]])
    aa = rotations.as_rotvec().reshape(len(indices), 16, 3)
    return Hand(aa[:, 0], aa[:, 1:].reshape(len(indices), 45),
                raw[f'{prefix}__betas'][indices].astype(np.float64),
                wrist=raw[f'{prefix}__tsl'][indices].astype(np.float64))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('compact', 'mano', 'corpus', 'selection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--only', type=int, help='One-based review number')
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    selection = json.loads(args.selection.read_text())
    source = Source(args.compact, OmegaConf.load(UPSTREAM / 'config/retarget/source/oakink2.yaml'))
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    renderer = pyrender.OffscreenRenderer(800, 640)
    results = []
    for number, row in enumerate(selection, 1):
        if args.only and number != args.only:
            continue
        folder = args.output / f'review-{number:02d}'
        folder.mkdir(exist_ok=True)
        record_path = args.corpus / row['dataset'] / 'clip-records' / (row['clip'] + '.json')
        record = json.loads(record_path.read_text())
        stage = record['source_stage']
        compact_path = args.compact / (stage['sequence'] + '.npz')
        with np.load(compact_path, allow_pickle=False) as archive:
            raw = dict(archive)
        start, end = stage['frame_interval_half_open']
        indices = np.flatnonzero((raw['frame_ids'] >= start) & (raw['frame_ids'] < end))[::4]
        if float(raw['fps']) != 120. or not len(indices):
            raise ValueError('Expected a nonempty 120 Hz source stage')
        hands, models, wrists = {}, {}, {}
        for side, prefix in (('right', 'rh'), ('left', 'lh')):
            hand = compact_hand(raw, prefix, indices)
            model = _mano(str(args.mano), side, tuple(sorted(Source.mano.items())))
            joints, vertices = _forward(model, hand)
            hands[side] = vertices @ YUP_TO_ZUP.T
            wrists[side] = joints[:, 0] @ YUP_TO_ZUP.T
            models[side] = model
        # A single translation makes the source coordinates easier to view; it
        # changes no relative motion or interaction geometry.
        shift = -(wrists['right'][0] + wrists['left'][0]) / 2
        hands = {side: vertices + shift for side, vertices in hands.items()}
        meshes, poses, bounds = {}, {}, []
        targets = set(row['review_target_ids'])
        names = {r.get('object_id', r.get('id')): r['name'] for r in stage.get('required_objects', [])}
        names.update(row['object_names'])
        for j, value in enumerate(raw['object_ids']):
            oid = str(value)
            meshes[oid] = source.mesh(oid).copy()
            pose = raw['object_pose'][indices, j].astype(np.float64).copy()
            pose[:, :3, :3] = YUP_TO_ZUP @ pose[:, :3, :3]
            pose[:, :3, 3] = pose[:, :3, 3] @ YUP_TO_ZUP.T + shift
            poses[oid] = pose
            corners = trimesh.bounds.corners(meshes[oid].bounds)
            bounds.append((np.einsum('tij,kj->tki', pose[:, :3, :3], corners) + pose[:, None, :3, 3]).reshape(-1, 3))
        points = np.concatenate(bounds + [v.reshape(-1, 3) for v in hands.values()])
        center = (points.min(0) + points.max(0)) / 2
        extent = max(.45, np.linalg.norm(points.max(0) - points.min(0)))
        target_points = np.concatenate([poses[oid][:, :3, 3] for oid in targets])
        target_center = (target_points.min(0) + target_points.max(0)) / 2
        target_extent = max(.28, np.linalg.norm(target_points.max(0) - target_points.min(0)) + .20)
        views = [camera_pose(center + np.array([.75, -1.05, .80]) * extent, center),
                 camera_pose(target_center + np.array([-.80, -.85, .70]) * target_extent, target_center)]
        scene = pyrender.Scene(bg_color=[.94, .95, .97, 1.], ambient_light=[.65, .65, .65])
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4, znear=.01)
        camera_node = scene.add(camera, pose=views[0])
        light = scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=views[0])
        nodes = {}
        for oid, mesh in meshes.items():
            color = ([.96, .24, .13, 1.] if oid in targets else
                     [.20, .58, .75, 1.] if oid in row['object_ids'] else [.63, .69, .72, 1.])
            nodes[oid] = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False,
                material=pyrender.MetallicRoughnessMaterial(baseColorFactor=color, roughnessFactor=.8)))
        hand_nodes = []
        video = folder / 'source-scene.mp4'
        with imageio.get_writer(video, fps=30, codec='libx264', quality=8, macro_block_size=1) as writer:
            for k, source_index in enumerate(indices):
                for oid, node in nodes.items():
                    scene.set_pose(node, poses[oid][k])
                for node in hand_nodes:
                    scene.remove_node(node)
                hand_nodes = []
                for side in ('right', 'left'):
                    mesh = trimesh.Trimesh(hands[side][k], models[side].faces, process=False)
                    hand_nodes.append(scene.add(pyrender.Mesh.from_trimesh(mesh,
                        material=pyrender.MetallicRoughnessMaterial(
                            baseColorFactor=[.80, .57, .44, 1.], roughnessFactor=.9))))
                panels = []
                for view in views:
                    scene.set_pose(camera_node, view)
                    scene.set_pose(light, view)
                    panels.append(renderer.render(scene)[0])
                image = Image.fromarray(np.concatenate(panels, axis=1))
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 1600, 84), fill=(24, 32, 43))
                time_s = (int(raw['frame_ids'][source_index]) - start) / 120.
                draw.text((14, 7), f'Review {number:02d} | Original MANO + ALL {len(nodes)} recorded objects | {time_s:.2f} / {(end-start)/120:.2f}s', font=font, fill='white')
                draw.text((14, 32), stage['description'].get('seg_desc', ''), font=font, fill='white')
                labels = ', '.join(names.get(oid, oid) for oid in sorted(targets))
                draw.text((14, 57), 'RED: review targets: ' + labels + ' | BLUE: group partner | GRAY: other scene objects', font=font, fill=(255, 190, 162))
                draw.text((14, 610), 'Full scene | Source meshes at scale 1; no physics / no retarget', font=font, fill=(25, 35, 45))
                draw.text((814, 610), 'Review objects: close view from the opposite side', font=font, fill=(25, 35, 45))
                writer.append_data(np.asarray(image))
                if k in (0, len(indices)//2, len(indices)-1):
                    image.save(folder / f'frame-{k:04d}.png')
                if k % 150 == 0:
                    print(f'Review {number}: {k}/{len(indices)}', flush=True)
        manifest = dict(number=number, clip=row['clip'], description=stage['description'],
            targets=row['review_target_ids'], group_objects=row['object_ids'], object_names=names,
            all_recorded_objects=list(meshes), source_fps=120., video_fps=30.,
            frames=len(indices), duration_s=(end-start)/120., source_frame_interval=[start,end],
            source_compact_sha256=digest(compact_path), source_record_sha256=digest(record_path),
            source_mesh_scale=1., kinematic_replay=True, physics_checked=False,
            video=str(video), camera_poses=[v.tolist() for v in views])
        (folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        results.append(manifest)
        (args.output / 'progress.json').write_text(json.dumps(dict(completed=len(results), expected=len(selection)), indent=2))
    renderer.delete()
    (args.output / 'manifest.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

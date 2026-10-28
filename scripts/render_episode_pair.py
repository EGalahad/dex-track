"""Render matching MANO and retargeted XHand kinematic replays; no policy inference."""
import argparse
import ast
import json
import os
from pathlib import Path
import pickle
import sys

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import pyrender
import trimesh

PROJECT = Path(__file__).resolve().parents[1]
UPSTREAM = PROJECT / 'third_party/FlashDexRetarget'
sys.path.insert(0, str(UPSTREAM / 'src'))
from retarget.human_demo.sources.base import _forward, _mano, YUP_TO_ZUP
from retarget.human_demo.sources.oakink2 import Source


def camera_pose(eye, target):
    z = np.asarray(eye) - target
    z /= np.linalg.norm(z)
    x = np.cross([0., 0., 1.], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    transform = np.eye(4)
    transform[:3, :3] = np.stack([x, y, z], axis=1)
    transform[:3, 3] = eye
    return transform


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--mano', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--clip', help='Exact retargeted clip stem')
    p.add_argument('--title', default='')
    p.add_argument('--context-object', action='append', default=[])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    clip_path = (args.dataset / 'retargeted' / f'{args.clip}.npz' if args.clip
                 else next((args.dataset / 'retargeted').glob('*.npz')))
    clip = dict(np.load(clip_path))
    human = dict(np.load(args.dataset / 'human_demo' / clip_path.name))
    sequence, stage = clip_path.stem.rsplit('__st', 1)
    T = len(clip['qpos'])
    offsets = range(len(human['mano_right_wrist_pos']) - T + 1)
    errors = [max(np.max(np.abs(human[f'mano_{s}_wrist_pos'][o:o+T] - clip[f'mano_{s}_wrist_pos']))
                  for s in ['right', 'left']) for o in offsets]
    offset = int(np.argmin(errors))
    if errors[offset] > 1e-5:
        raise ValueError(f'Retarget timeline mismatch: {errors[offset]}')
    cache = args.output / 'mano_mesh_cache.npz'
    hands = dict(np.load(cache)) if cache.exists() else {}
    cache_valid = bool(hands) and all('context_' + oid in hands for oid in args.context_object)
    if not cache_valid:
        print('Loading source MANO parameters', flush=True)
        with (args.raw / 'anno_preview' / f'{sequence}.pkl').open('rb') as f:
            raw = pickle.load(f)
        program = json.loads((args.raw / 'program/program_info' / f'{sequence}.json').read_text())
        interval = ast.literal_eval(list(program)[int(stage)])
        spans = [s for s in interval if s is not None]
        start, end = min(s[0] for s in spans), max(s[1] for s in spans)
        frames = [f for f in raw['mocap_frame_id_list'] if start <= f < end][::2]
        if len(frames) != len(human['mano_right_wrist_pos']):
            raise ValueError('Unexpected source sampling rate')
        hands = {}
        for side, prefix in [('right', 'rh'), ('left', 'lh')]:
            hand = Source._hand(raw, frames, prefix)
            model = _mano(str(args.mano), side, tuple(sorted(Source.mano.items())))
            joints, vertices = _forward(model, hand)
            joints, vertices = joints @ YUP_TO_ZUP.T, vertices @ YUP_TO_ZUP.T
            shift = human[f'mano_{side}_wrist_pos'][0] - joints[0, 0]
            residual = np.max(np.abs(joints[:, 0] + shift - human[f'mano_{side}_wrist_pos']))
            if residual > 1e-5:
                raise ValueError(f'MANO alignment failed {side}: {residual}')
            hands[side] = (vertices + shift)[offset:offset+T].astype(np.float32)
            hands[side + '_faces'] = model.faces
        for oid in args.context_object:
            poses = np.stack([raw['obj_transf'][oid][f] for f in frames]).copy()
            poses[:,:3,:3] = YUP_TO_ZUP @ poses[:,:3,:3]
            poses[:,:3,3] = poses[:,:3,3] @ YUP_TO_ZUP.T + shift
            hands['context_' + oid] = poses[offset:offset+T]
        np.savez_compressed(cache, **hands)
    model = mujoco.MjModel.from_xml_path(str(UPSTREAM / 'assets/robot/xhand/bimanual.xml'))
    data = mujoco.MjData(model)
    renderer = pyrender.OffscreenRenderer(1280, 720)
    object_meshes = {}
    object_poses = {s: clip[f'obj_{s}_pose'] for s in ['right', 'left']}
    object_ids = {s: str(clip[f'obj_{s}_id']) for s in ['right', 'left']}
    for oid in args.context_object:
        object_ids[oid] = oid
        object_poses[oid] = hands['context_' + oid]
    bounds = [clip[f'mano_{s}_wrist_pos'] for s in ['right', 'left']]
    for side, oid in object_ids.items():
        mesh = trimesh.load(args.dataset / 'objects' / oid / 'visual.obj', force='mesh', process=False)
        mesh.apply_scale(float(clip.get(f'obj_{side}_scale', 1.)))
        object_meshes[side] = mesh
        corners = trimesh.bounds.corners(mesh.bounds)
        poses = object_poses[side][::15]
        bounds.append((np.einsum('tij,kj->tki', poses[:,:3,:3], corners) + poses[:,None,:3,3]).reshape(-1,3))
    points = np.concatenate(bounds)
    centre = (points.min(0) + points.max(0)) / 2
    centre[2] = max(.14, centre[2] - .03)
    extent = max(.7, np.linalg.norm(points.max(0) - points.min(0)))
    view = camera_pose(centre + np.array([.75, -1.05, .85]) * extent, centre)
    obj_colours = {'right': [.92, .52, .15, 1.], 'left': [.23, .60, .72, 1.]}
    frame_indices = list(range(0, T, 2))
    for mode in ['human', 'robot']:
        scene = pyrender.Scene(bg_color=[.94, .95, .97, 1.], ambient_light=[.5, .5, .5])
        scene.add(pyrender.PerspectiveCamera(yfov=np.pi/4), pose=view)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=view)
        table = trimesh.creation.box(extents=[3., 3., .02])
        table.apply_translation([centre[0], centre[1], -.025])
        scene.add(pyrender.Mesh.from_trimesh(table, material=pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[.78,.81,.84,1.], roughnessFactor=1.0)))
        object_nodes = {}
        seen_objects = set()
        for side, oid in object_ids.items():
            if oid in seen_objects:
                continue
            seen_objects.add(oid)
            mesh = object_meshes[side]
            object_nodes[side] = scene.add(pyrender.Mesh.from_trimesh(mesh, material=pyrender.MetallicRoughnessMaterial(
                baseColorFactor=obj_colours.get(side, [.6,.67,.47,1.]), roughnessFactor=.75)))
        robot_nodes = {}
        if mode == 'robot':
            for gid in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
                if 'collision' in name or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                mid = model.geom_dataid[gid]
                va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
                mesh = trimesh.Trimesh(model.mesh_vert[va:va+vn], model.mesh_face[fa:fa+fn], process=False)
                robot_nodes[gid] = scene.add(pyrender.Mesh.from_trimesh(mesh, material=pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[.72,.75,.79,1.], metallicFactor=.15, roughnessFactor=.65)))
        hand_nodes = []
        writer = imageio.get_writer(args.output / f'{mode}_object.mp4', fps=30, codec='libx264', quality=8,
                                    macro_block_size=1)
        for count, frame in enumerate(frame_indices):
            for side, node in object_nodes.items():
                scene.set_pose(node, object_poses[side][frame])
            if mode == 'human':
                for node in hand_nodes:
                    scene.remove_node(node)
                hand_nodes = []
                for side in ['right', 'left']:
                    mesh = trimesh.Trimesh(hands[side][frame], hands[side+'_faces'], process=False)
                    hand_nodes.append(scene.add(pyrender.Mesh.from_trimesh(mesh, material=pyrender.MetallicRoughnessMaterial(
                        baseColorFactor=[.79,.56,.43,1.], roughnessFactor=.9))))
            else:
                data.qpos[:model.nq] = clip['qpos'][frame]
                mujoco.mj_forward(model, data)
                for gid, node in robot_nodes.items():
                    transform = np.eye(4)
                    transform[:3,:3] = data.geom_xmat[gid].reshape(3,3)
                    transform[:3,3] = data.geom_xpos[gid]
                    scene.set_pose(node, transform)
            colour, _ = renderer.render(scene)
            image = Image.fromarray(colour)
            draw = ImageDraw.Draw(image)
            draw.rectangle([0,0,1280,52], fill=(28,35,46))
            draw.text((20,10), f'{"MANO HUMAN HANDS" if mode == "human" else "RETARGETED XHAND"} + OBJECTS  |  {frame/60:.2f} / {T/60:.2f} s', fill='white')
            draw.text((20,29), args.title or 'Kinematic reference replay - not a policy rollout; retarget is used only for reset.', fill=(205,216,230))
            writer.append_data(np.asarray(image))
            if count in [0, len(frame_indices)//3, 2*len(frame_indices)//3, len(frame_indices)-1]:
                image.save(args.output / f'{mode}_{frame:04d}.png')
            if count % 150 == 0:
                print(f'{mode}: {count}/{len(frame_indices)}', flush=True)
        writer.close()
    renderer.delete()
    (args.output / 'render_manifest.json').write_text(json.dumps({
        'clip': clip_path.stem, 'frames_60hz': T, 'video_fps': 30, 'duration_s': T/60,
        'title': args.title, 'objects': {s: str(clip[f'obj_{s}_id']) for s in ['right', 'left']},
        'context_objects': args.context_object,
        'retarget_source_offset': offset, 'alignment_max_error_m': float(errors[offset]),
        'camera_pose': view.tolist(), 'kinematic_replay': True, 'physics_stability_checked': False}, indent=2))


if __name__ == '__main__':
    main()

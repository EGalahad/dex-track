#!/usr/bin/env python3
"""Viser reference-motion viewer for dex-track NPZ packs.

Loads one clip from a portable motion folder / index.json, kinematics the
pinned bimanual XHand MJCF, and draws scene objects. GUI checkboxes toggle
visual vs collision geoms for both hands and objects.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np
import trimesh
import viser

PROJECT = Path(__file__).resolve().parents[1]


def _quat_apply(quat_wxyz: np.ndarray, points: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    qvec = np.array([x, y, z], dtype=np.float64)
    uv = np.cross(qvec, points)
    return points + 2.0 * (w * uv + np.cross(qvec, uv))


def _transform_mesh(mesh: trimesh.Trimesh, pos: np.ndarray, quat_wxyz: np.ndarray,
                    scale: float = 1.0) -> trimesh.Trimesh:
    out = mesh.copy()
    verts = np.asarray(out.vertices, dtype=np.float64) * float(scale)
    out.vertices = _quat_apply(np.asarray(quat_wxyz, dtype=np.float64), verts) + np.asarray(pos, dtype=np.float64)
    return out


def _load_trimesh(path: Path) -> trimesh.Trimesh | None:
    if not path.is_file():
        return None
    loaded = trimesh.load(path, force='mesh', process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            return None
        loaded = trimesh.util.concatenate(geoms)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) == 0:
        return None
    return loaded


def _mujoco_mesh(model: mujoco.MjModel, mesh_id: int) -> trimesh.Trimesh:
    vadr = int(model.mesh_vertadr[mesh_id])
    vn = int(model.mesh_vertnum[mesh_id])
    fadr = int(model.mesh_faceadr[mesh_id])
    fn = int(model.mesh_facenum[mesh_id])
    return trimesh.Trimesh(
        vertices=model.mesh_vert[vadr:vadr + vn].copy(),
        faces=model.mesh_face[fadr:fadr + fn].copy(),
        process=False,
    )


def _robot_model() -> mujoco.MjModel:
    import sys
    sys.path.insert(0, str(PROJECT))
    from dex_track.source import source_root
    return mujoco.MjSpec.from_file(str(source_root() / 'assets/robot/xhand/bimanual.xml')).compile()


def _list_motions(dataset: Path) -> list[dict]:
    import sys
    sys.path.insert(0, str(PROJECT))
    from dex_track.motion_storage import resolve_motion_path
    path = resolve_motion_path(dataset)
    if path.name == 'metadata.json':
        return [{'index': 0, 'name': path.parent.name, 'path': str(path), 'frames': None}]
    import json
    manifest = json.loads(path.read_text())
    rows = []
    for i, row in enumerate(manifest['motions']):
        rows.append({
            'index': i,
            'name': row.get('name') or Path(row['path']).parent.name,
            'path': str((path.parent / row['path']).resolve()),
            'frames': row.get('frames'),
        })
    return rows


def _resolve_motion(rows: list[dict], query: str) -> dict | None:
    q = query.strip()
    if not q:
        return None
    by_index = {int(r['index']): r for r in rows}
    if q.isdigit():
        return by_index.get(int(q))
    matches = [r for r in rows if q == r['name'] or q in r['name'] or q in r['path']]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f'Multiple matches for {q!r}; be more specific or use an index.')
    return None


def _ask_motion(rows: list[dict], query: str | None) -> dict:
    if query is not None:
        selected = _resolve_motion(rows, query)
        if selected is not None:
            return selected
        if query.strip().isdigit():
            raise SystemExit(f'Motion index {query} out of range 0..{max(r["index"] for r in rows)}')
        if not any(query.strip() in r['name'] or query.strip() in r['path'] for r in rows):
            raise SystemExit(f'No motion matching {query!r}')
    print(f'{len(rows)} motion(s):')
    width = len(str(max(r['index'] for r in rows)))
    for row in rows:
        frames = '' if row['frames'] is None else f'  ({row["frames"]} frames)'
        print(f'  [{row["index"]:{width}d}] {row["name"]}{frames}')
    while True:
        choice = input('Select motion index (or unique name substring): ').strip()
        selected = _resolve_motion(rows, choice)
        if selected is not None:
            return selected
        print('Invalid selection; try again.')


def _load_clip(metadata_path: Path):
    import sys
    sys.path.insert(0, str(PROJECT))
    from dex_track.motion_storage import load_motion_data
    raw = load_motion_data(metadata_path)
    joint_pos = np.asarray(raw['joint_pos'], dtype=np.float64)
    if joint_pos.ndim != 2:
        raise ValueError(f'Expected joint_pos [T,nq], got {joint_pos.shape}')
    fps = float(raw['meta']['fps'])
    scene_ids = list(raw['motion_scene_object_ids'][0])
    scene_scales = [float(x) for x in raw['motion_scene_object_scales'][0]]
    pos = np.asarray(raw['scene_object_pos'], dtype=np.float64)
    quat = np.asarray(raw['scene_object_quat'], dtype=np.float64)
    dataset_root = metadata_path.parent
    # Motion folders symlink objects/ → ../../objects; prefer local then pack root.
    if (dataset_root / 'objects').is_dir() and (dataset_root / 'index.json').is_file():
        asset_root = dataset_root
    else:
        asset_root = dataset_root.parent.parent if (dataset_root.parent.parent / 'index.json').is_file() else dataset_root
    return {
        'name': raw['motion_filename'][0],
        'fps': fps,
        'joint_pos': joint_pos,
        'scene_ids': scene_ids,
        'scene_scales': scene_scales,
        'scene_pos': pos,
        'scene_quat': quat,
        'asset_root': Path(asset_root),
        'motion_dir': metadata_path.parent,
    }


def _paint(mesh: trimesh.Trimesh, rgba: Iterable[float]) -> trimesh.Trimesh:
    out = mesh.copy()
    color = (np.asarray(rgba, dtype=np.float64) * 255.0).clip(0, 255).astype(np.uint8)
    out.visual.face_colors = np.tile(color, (len(out.faces), 1))
    return out


def _add_geom_meshes(server: viser.ViserServer, model: mujoco.MjModel) -> dict[str, list]:
    """Upload hand meshes once under body frames; return visibility handles."""
    visual_handles: list = []
    collision_handles: list = []
    body_frames: dict[int, object] = {}
    for body_id in range(model.nbody):
        name = model.body(body_id).name or f'body_{body_id}'
        body_frames[body_id] = server.scene.add_frame(
            f'/robot/{name}', show_axes=False, axes_length=0.03, axes_radius=0.001)

    for geom_id in range(model.ngeom):
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        name = model.geom(geom_id).name or f'geom_{geom_id}'
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        mesh = _transform_mesh(
            _mujoco_mesh(model, mesh_id),
            model.geom_pos[geom_id],
            model.geom_quat[geom_id],
        )
        is_collision = name.startswith('collision') or int(model.geom_group[geom_id]) == 0
        if is_collision:
            mesh = _paint(mesh, (0.95, 0.35, 0.1, 0.35))
            path = f'/robot/{model.body(body_id).name}/collision/{name}'
            handle = server.scene.add_mesh_trimesh(path, mesh)
            collision_handles.append(handle)
        else:
            rgba = model.geom_rgba[geom_id].copy()
            rgba[3] = max(float(rgba[3]), 0.85)
            mesh = _paint(mesh, rgba)
            path = f'/robot/{model.body(body_id).name}/visual/{name}'
            handle = server.scene.add_mesh_trimesh(path, mesh)
            visual_handles.append(handle)
    return {
        'body_frames': body_frames,
        'visual': visual_handles,
        'collision': collision_handles,
    }


def _add_objects(server: viser.ViserServer, clip: dict) -> dict[str, list]:
    visual_handles: list = []
    collision_handles: list = []
    frames = []
    for slot, (oid, scale) in enumerate(zip(clip['scene_ids'], clip['scene_scales'])):
        root = clip['motion_dir'] / 'objects' / oid
        if not root.is_dir():
            root = clip['asset_root'] / 'objects' / oid
        frame = server.scene.add_frame(
            f'/objects/{slot}_{oid}', show_axes=False, axes_length=0.05, axes_radius=0.001)
        frames.append(frame)
        visual = _load_trimesh(root / 'visual.obj')
        added_visual = False
        added_collision = False
        if visual is not None:
            visual = visual.copy()
            visual.apply_scale(scale)
            visual = _paint(visual, (0.75, 0.78, 0.82, 0.95))
            visual_handles.append(server.scene.add_mesh_trimesh(
                f'/objects/{slot}_{oid}/visual', visual))
            added_visual = True
        convex_dir = root / 'convex'
        if convex_dir.is_dir():
            parts = []
            for path in sorted(convex_dir.glob('*.obj')):
                part = _load_trimesh(path)
                if part is not None:
                    parts.append(part)
            if parts:
                collision = trimesh.util.concatenate(parts)
                collision.apply_scale(scale)
                collision = _paint(collision, (0.2, 0.85, 0.35, 0.3))
                collision_handles.append(server.scene.add_mesh_trimesh(
                    f'/objects/{slot}_{oid}/collision', collision))
                added_collision = True
        if not added_visual and not added_collision:
            print(f'[visualize_motion] missing meshes for object {oid} under {root}', flush=True)
    return {'frames': frames, 'visual': visual_handles, 'collision': collision_handles}


def _set_visible(handles: list, visible: bool) -> None:
    for handle in handles:
        handle.visible = bool(visible)


def _update_robot(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray, body_frames: dict) -> None:
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    for body_id, frame in body_frames.items():
        frame.position = data.xpos[body_id].astype(np.float32)
        frame.wxyz = data.xquat[body_id].astype(np.float32)


def _update_objects(clip: dict, frame_idx: int, frames: list) -> None:
    for slot, frame in enumerate(frames):
        frame.position = clip['scene_pos'][frame_idx, slot].astype(np.float32)
        frame.wxyz = clip['scene_quat'][frame_idx, slot].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path,
                        help='NPZ pack root, index.json, or a single motion metadata.json')
    parser.add_argument('--motion', default=None,
                        help='Motion index or name substring; prompted if omitted')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--start-frame', type=int, default=0)
    args = parser.parse_args()

    rows = _list_motions(args.dataset)
    if not rows:
        raise SystemExit(f'No motions in {args.dataset}')
    selected = _ask_motion(rows, args.motion)
    clip = _load_clip(Path(selected['path']))
    n_frames = int(clip['joint_pos'].shape[0])
    print(f"Loaded {clip['name']}: {n_frames} frames @ {clip['fps']:g} Hz, "
          f"{len(clip['scene_ids'])} scene object(s)", flush=True)

    model = _robot_model()
    if clip['joint_pos'].shape[1] != model.nq:
        raise SystemExit(
            f'joint_pos width {clip["joint_pos"].shape[1]} != model.nq {model.nq}')
    data = mujoco.MjData(model)

    server = viser.ViserServer(host=args.host, port=args.port, label='dex-track-motion')
    server.scene.add_grid('/grid', width=2.0, height=2.0, cell_size=0.1)
    robot = _add_geom_meshes(server, model)
    objects = _add_objects(server, clip)

    with server.gui.add_folder('Motion'):
        play = server.gui.add_checkbox('Play', True)
        frame_slider = server.gui.add_slider('Frame', min=0, max=max(n_frames - 1, 0),
                                            step=1, initial_value=int(np.clip(args.start_frame, 0, n_frames - 1)))
        speed = server.gui.add_slider('Speed', min=0.1, max=2.0, step=0.1, initial_value=1.0)
        info = server.gui.add_markdown(f"**{clip['name']}**")
    with server.gui.add_folder('Geometry'):
        show_visual = server.gui.add_checkbox('Show visual geoms', True)
        show_collision = server.gui.add_checkbox('Show collision geoms', False)

    @show_visual.on_update
    def _(_) -> None:
        _set_visible(robot['visual'] + objects['visual'], show_visual.value)

    @show_collision.on_update
    def _(_) -> None:
        _set_visible(robot['collision'] + objects['collision'], show_collision.value)

    _set_visible(robot['visual'] + objects['visual'], True)
    _set_visible(robot['collision'] + objects['collision'], False)

    print(f'Viser: http://localhost:{args.port}  (bind {args.host})', flush=True)
    last = time.perf_counter()
    accum = 0.0
    try:
        while True:
            now = time.perf_counter()
            dt = now - last
            last = now
            if play.value and n_frames > 1:
                accum += dt * float(speed.value) * clip['fps']
                steps = int(accum)
                if steps:
                    accum -= steps
                    frame_slider.value = (int(frame_slider.value) + steps) % n_frames
            frame = int(frame_slider.value)
            _update_robot(model, data, clip['joint_pos'][frame], robot['body_frames'])
            _update_objects(clip, frame, objects['frames'])
            info.content = (f"**{clip['name']}**  \n"
                            f"frame `{frame}` / `{n_frames - 1}`  ·  "
                            f"{frame / clip['fps']:.2f}s")
            time.sleep(0.01)
    except KeyboardInterrupt:
        print('Stopped.', flush=True)
    finally:
        server.stop()


if __name__ == '__main__':
    main()

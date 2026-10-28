"""Replay recorded scene poses in mjlab Viser, without stepping physics or policy."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectory', required=True)
    p.add_argument('--motion-file', required=True, help='Local motion file used to resolve matching scene assets')
    p.add_argument('--speed', type=float, default=1., choices=[i/10 for i in range(2,11)])
    p.add_argument('--test-frames', type=int, default=0, help='Validate N frames and exit')
    p.add_argument('--ready-file')
    a=p.parse_args()
    recording=torch.load(a.trajectory, map_location='cpu', weights_only=False)
    if recording.get('format') != 'aa_scene_trajectory_v1':
        raise ValueError('Requires scene_trajectory.pt; the old metrics-only trajectory.pt cannot replay a scene')
    if recording['quaternion_order']!='wxyz':raise ValueError('Unsupported quaternion order')
    cfg=OmegaConf.create(recording['config'])
    cfg.backend='mjlab';cfg.device='cuda:0';cfg.headless=False
    cfg.app={'headless':False,'enable_cameras':False};cfg.wandb.mode='disabled'
    cfg.task.motion_file=str(Path(a.motion_file).resolve())
    cfg.task.command.motion_file=cfg.task.motion_file
    cfg.task.num_envs=recording['num_envs'];cfg.task.robot.usd_path=None
    cfg.task.enable_cameras=False
    import runpy
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'dex_track/data.py'))['configure_scene_slots'](cfg)

    aa.init(cfg, auto_rank=False)
    import dex_track
    from dex_track.assets import body_name
    from dex_track.tasks.reference.math import matrix_from_quat
    from active_adaptation.helpers import _ensure_backend_env_imported
    from active_adaptation.envs.env_base import _EnvBase
    _ensure_backend_env_imported('mjlab')
    env=_EnvBase.registry['MjlabBackendEnv'](cfg.task,cfg.device,False)
    viewer=env.sim.viewer
    server=viewer._server
    scene=viewer.scene
    count=len(recording['episode_step']); n=recording['num_envs']
    if set(env.scene.entities)-{'terrain'}!=set(recording['entities'])-{'terrain'}:
        raise ValueError(f'Recorded {list(recording["entities"])} != replay {list(env.scene.entities)}')
    # Preserve static world/terrain geometry from the AA scene; overwrite all
    # dynamic link transforms directly with recorded source-backend world poses.
    positions=np.broadcast_to(env.sim.mj_data.xpos,(n,env.sim.mj_model.nbody,3)).copy()
    rotations=np.broadcast_to(env.sim.mj_data.xmat.reshape(-1,3,3),(n,env.sim.mj_model.nbody,3,3)).copy()
    mapping={}
    for name, src in recording['entities'].items():
        entity=env.scene.entities[name]
        if len(entity.body_names)==1 and len(src['body_names'])==1:
            indices=[0]  # Isaac baseLink and MuJoCo object are the same rigid frame.
        else:
            source_names=src['body_names']
            target_names=[body_name(b, recording['backend']) for b in entity.body_names]
            missing=set(target_names)-set(source_names)
            # mjlab adds an invisible mocap wrapper around this fixed-base asset.
            ignored = [i for i,b in enumerate(target_names) if b=='mocap_base' and b in missing]
            body_ids=entity.indexing.body_ids.cpu().numpy()
            if any(env.sim.mj_model.body_geomnum[body_ids[i]] for i in ignored):
                raise ValueError('Unexpected visible geometry on mocap wrapper')
            missing.discard('mocap_base')
            if missing:raise ValueError(f'{name}: missing recorded link names {sorted(missing)}')
            selected=[i for i in range(len(target_names)) if i not in ignored]
            indices=[source_names.index(target_names[i]) for i in selected]
            mapping[name]=(body_ids[selected],indices)
            continue
        mapping[name]=(entity.indexing.body_ids.cpu().numpy(),indices)
    for state in recording['entities'].values():
        if not torch.isfinite(state['position']).all() or not torch.isfinite(state['quaternion']).all():
            raise ValueError('Recording contains non-finite poses')
    with server.gui.add_folder('Recorded scene replay'):
        server.gui.add_markdown('**Recorded poses only — physics and policy are paused.**')
        playing=server.gui.add_checkbox('Play', initial_value=not bool(a.test_frames))
        loop=server.gui.add_checkbox('Loop', initial_value=True)
        speed=server.gui.add_dropdown('Speed', options=tuple(f'{i/10:.1f}x' for i in range(2,11)), initial_value=f'{a.speed:.1f}x')
        frame=server.gui.add_slider('Frame',min=0,max=max(1,count-1),step=1,initial_value=0)
        previous=server.gui.add_button('Previous frame');following=server.gui.add_button('Next frame')
        status=server.gui.add_markdown('')
    @previous.on_click
    def back(_):
        playing.value=False;frame.value=max(0,frame.value-1)
    @following.on_click
    def forward(_):
        playing.value=False;frame.value=min(count-1,frame.value+1)
    @server.on_client_connect
    def camera(event):
        event.client.camera.position=tuple(cfg.task.viewer.eye)
        event.client.camera.look_at=tuple(cfg.task.viewer.lookat)
    def show(index):
        for name,(body_ids,indices) in mapping.items():
            state=recording['entities'][name]
            positions[:,body_ids]=state['position'][index,:,indices].numpy()
            rotations[:,body_ids]=matrix_from_quat(state['quaternion'][index,:,indices]).numpy()
        with server.atomic():
            scene.update_from_arrays(positions,rotations)
            episode=recording['episode_id'][index].flatten().tolist()
            done=recording['done'][index].flatten().tolist()
            status.content=f'Frame **{index}/{count-1}** · {index*recording["dt"]:.2f} s · episode {episode} · done {done}'
        # No mj_forward/step: no cross-backend dynamics or FK drift is introduced.
    try:
        show(0)
        ready=dict(format=recording['format'],frames=count,dt=recording['dt'],
            entities=list(mapping),speed_options=list(speed.options),physics_stepped=False,
            body_count=sum(len(ids) for ids,_ in mapping.values()))
        print(json.dumps(ready),flush=True)
        if a.ready_file:Path(a.ready_file).write_text(json.dumps(ready,indent=2))
        if a.test_frames:
            for i in np.linspace(0,count-1,min(count,a.test_frames),dtype=int):show(int(i))
            print('Scene replay validation passed',flush=True)
            return
        last_frame=0;last_speed=speed.value;last_play=playing.value
        deadline=time.monotonic()+recording['dt']/float(speed.value[:-1])
        while True:
            now=time.monotonic()
            interval=recording['dt']/float(speed.value[:-1])
            index=min(frame.value,count-1)
            if index!=last_frame or speed.value!=last_speed or playing.value!=last_play:
                deadline=now+interval
            elif playing.value and now>=deadline:
                increment=max(1,int((now-deadline)/interval)+1)
                index+=increment
                if index>=count:
                    if loop.value:index%=count
                    else:index=count-1;playing.value=False
                frame.value=index;deadline+=increment*interval
            if index!=last_frame:show(index)
            last_frame=index;last_speed=speed.value;last_play=playing.value
            time.sleep(.002)
    finally:
        env.close()

if __name__=='__main__':main()

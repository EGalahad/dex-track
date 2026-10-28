"""Exercise the AA backend and verify retarget is used only at reset."""
import argparse
from pathlib import Path
import torch
from omegaconf import OmegaConf
import active_adaptation as aa

p = argparse.ArgumentParser()
p.add_argument('--backend', choices=('mjlab','isaaclab'), default='mjlab')
p.add_argument('--motion-file', required=True)
p.add_argument('--usd-path')
p.add_argument('--device', default='cuda:0')
a = p.parse_args()
config = OmegaConf.create({'backend':a.backend, 'device':a.device, 'headless':True,
    'app':{'headless':True,'enable_cameras':False}, 'wandb':{'mode':'disabled','project':'aa-dex-track'}})
aa.init(config, auto_rank=False)
import dex_track
from active_adaptation.helpers import _ensure_backend_env_imported
from active_adaptation.envs.env_base import _EnvBase
_ensure_backend_env_imported(aa.get_backend())
backend_class = 'MjlabBackendEnv' if a.backend=='mjlab' else 'IsaacBackendEnv'
cfg=OmegaConf.load(Path(__file__).resolve().parents[1]/'cfg/task/tracking.yaml')
cfg.motion_file=str(Path(a.motion_file).resolve())
cfg.robot.usd_path=a.usd_path
cfg.num_envs=2
cfg.command.sampling='start'
full=OmegaConf.create({'task':cfg})
from dex_track.data import configure_scene_slots
configure_scene_slots(full)
OmegaConf.resolve(full)
e=_EnvBase.registry[backend_class](full.task, a.device, True)
td=e.reset()
for _ in range(8):
    td['wrist']=e.action_spec['wrist'].zero()
    td['finger']=e.action_spec['finger'].zero()
    td=e.step(td)['next']
    assert all(td[name].isfinite().all() for name in e.observation_groups)
c=e.command_manager
obs={name: group.compute_dense().clone() for name, group in e.observation_groups.items()}
rewards={n:r._compute().clone() for n,r in e.reward_groups['tracking'].funcs.items()}
terminated=torch.zeros(e.num_envs,1,dtype=torch.bool,device=e.device)
terminations={name:term.compute(terminated).clone() for name,term in e.termination_funcs.items()}
old_pos, old_vel=c.dataset.reset.joint_pos.clone(), c.dataset.reset.joint_vel.clone()
c.dataset.reset.joint_pos.fill_(float('nan'))
c.dataset.reset.joint_vel.fill_(float('nan'))
for name, value in obs.items():
    torch.testing.assert_close(e.observation_groups[name].compute_dense(), value)
for n,r in e.reward_groups['tracking'].funcs.items():
    torch.testing.assert_close(r._compute(),rewards[n])
for name,term in e.termination_funcs.items():
    torch.testing.assert_close(term.compute(terminated),terminations[name])
c.dataset.reset.joint_pos.copy_(old_pos)
c.dataset.reset.joint_vel.copy_(old_vel)
print({'backend':a.backend, 'env':type(e).__name__, 'actions':{'wrist':12,'finger':24},
       'obs_dims':{name: value.shape[-1] for name, value in obs.items()}, 'steps':8, 'retarget_reset_only':True}, flush=True)
e.close()

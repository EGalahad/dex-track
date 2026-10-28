"""Robot and controller state; no scene or reference geometry."""
import torch
from .common import TrackObservation
from active_adaptation.envs.utils import find_joints
from ..reference.math import matrix_from_quat


class wrist_pose(TrackObservation, namespace='dex_track'):
    def compute(self):
        c = self.command_manager
        p = c.robot_wrist_pos_w-self.env.scene.env_origins[:,None]
        r = matrix_from_quat(c.robot_wrist_quat_w)[..., :2].flatten(-2)
        return torch.cat((p,r),-1).flatten(1)


class wrist_velocity(TrackObservation, namespace='dex_track'):
    def compute(self):
        c = self.command_manager
        return torch.cat((c.robot_wrist_lin_vel_w,c.robot_wrist_ang_vel_w),-1).flatten(1)


class joint_pos(TrackObservation, namespace='dex_track'):
    def _initialize(self, env):
        super()._initialize(env)
        self.asset = env.scene['robot']
        self.ids, _ = find_joints(self.asset, '.*')

    def compute(self):
        return self.asset.data.joint_pos[:,self.ids]


class joint_vel(joint_pos, namespace='dex_track'):
    def compute(self):
        return self.asset.data.joint_vel[:,self.ids]


class previous_action(TrackObservation, namespace='dex_track'):
    def compute(self):
        return torch.cat([self.env.input_managers[n].action_buf for n in ('wrist','finger')],-1)


class wrist_controller_state(TrackObservation, namespace='dex_track'):
    def compute(self):
        return self.env.input_managers['wrist'].controller_state()


class voc_active(TrackObservation, namespace='dex_track'):
    def compute(self):
        return self.command_manager.voc.indicator()

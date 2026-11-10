"""Two independent floating XHands exposed as a named bimanual MDP asset.

The simulator owns two Entity instances. This facade only maps named tensors
and writes; it never advances physics or refreshes derived state on reads.
"""
from types import SimpleNamespace
import torch
from active_adaptation.utils.string import resolve_matching_names
from dex_track.tasks.transforms import rotate

class HandPairData:
    def __init__(self, pair): self.pair = pair
    def __getattr__(self, name):
        values = [getattr(e.data, name) for e in self.pair.hands]
        if name == 'default_root_state': return values[0]
        if isinstance(values[0], torch.Tensor): return torch.cat(values, dim=1)
        raise AttributeError(name)

class HandPair:
    is_fixed_base = False
    is_free_hand_pair = True
    def __init__(self, right, left):
        self.hands = (right,left)
        # Construct mjlab's lazy zero-copy Torch views before compiling controls.
        # Keep reading the live model arrays so randomized COM offsets stay current.
        for hand in self.hands:
            _ = hand.data.model.body_ipos, hand.data.data.ctrl
        self.joint_names = tuple(right.joint_names)+tuple(left.joint_names)
        self.body_names = tuple(right.body_names)+tuple(left.body_names)
        self.num_joints = len(self.joint_names); self.num_bodies = len(self.body_names)
        from dex_track.assets import JOINT_NAMES_SIMULATION
        self.cfg = SimpleNamespace(joint_names_simulation=[n for n in JOINT_NAMES_SIMULATION if n in self.joint_names], body_names_simulation=self.body_names)
        self.data = HandPairData(self)
    def find_joints(self, names, preserve_order=False):
        return resolve_matching_names(names,self.joint_names,preserve_order=preserve_order)
    def find_bodies(self, names, preserve_order=False):
        return resolve_matching_names(names,self.body_names,preserve_order=preserve_order)
    def _split(self, ids, kind):
        lengths = [len(getattr(e,kind+'_names')) for e in self.hands]
        ids = list(range(sum(lengths))) if ids is None or isinstance(ids,slice) else list(ids)
        start=0
        for e,n in zip(self.hands,lengths):
            columns=[i for i,x in enumerate(ids) if start<=x<start+n]
            local=[ids[i]-start for i in columns]
            if local: yield e,columns,local
            start+=n
    def set_joint_position_target(self, target, joint_ids=None, env_ids=None):
        for e,cols,ids in self._split(joint_ids,'joint'):
            e.set_joint_position_target(target[:,cols],joint_ids=ids,env_ids=env_ids)
    def set_joint_velocity_target(self, target, joint_ids=None, env_ids=None):
        for e,cols,ids in self._split(joint_ids,'joint'):
            e.set_joint_velocity_target(target[:,cols],joint_ids=ids,env_ids=env_ids)
    def write_joint_state_to_sim(self,pos,vel,joint_ids=None,env_ids=None):
        for e,cols,ids in self._split(joint_ids,'joint'):
            e.write_joint_state_to_sim(pos[:,cols],vel[:,cols],joint_ids=ids,env_ids=env_ids)
    def write_external_wrench_to_sim(self,force,torque,body_ids=None,env_ids=None):
        for e,cols,ids in self._split(body_ids,'body'):
            e.write_external_wrench_to_sim(force[:,cols],torque[:,cols],body_ids=ids,env_ids=env_ids)
    def direct_wrist_state(self):
        rows=[]
        for e in self.hands:
            q=e.data.data.qpos[:,e.indexing.free_joint_q_adr]
            v=e.data.data.qvel[:,e.indexing.free_joint_v_adr]
            rows.append((q[:,:3],q[:,3:],v[:,:3],rotate(q[:,3:],v[:,3:])))
        return tuple(torch.stack([r[i] for r in rows],1) for i in range(4))
    def direct_wrist_com(self):
        p,q,_,_=self.direct_wrist_state()
        offsets=torch.stack([e.data.model.body_ipos[:, e.indexing.root_body_id] for e in self.hands], dim=1)
        return p+rotate(q,offsets.expand_as(p))
    def write_wrist_states(self,p,q,v,omega,env_ids):
        for i,e in enumerate(self.hands):
            # Entity root-state API uses world link linear/angular velocity.
            e.write_root_state_to_sim(torch.cat((p[:,i],q[:,i],v[:,i],omega[:,i]),-1),env_ids=env_ids)

class InstallHandPair:
    def _initialize(self,robot,env):
        pair=HandPair(robot,env.scene.entities['left_hand'])
        env.scene._entity_overrides={'robot':pair}
        env.robot=pair

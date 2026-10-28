"""Dataset-sized rigid-object slots; unused slots never enter tracking statistics."""
import torch
import trimesh
from .reference.math import matrix_from_quat


class SceneObjects:
    def __init__(self, command, catalog):
        self.c = command
        self.env = command.env
        self.device = command.device
        self.catalog = catalog
        self.count = self.catalog['slots']
        self.names = [f'object_{i}' for i in range(self.count)]
        self.mask_table = self.catalog['mask'].to(self.device)
        self.hand_slots = self.catalog['hand_slots'].to(self.device)
        self.state = command.dataset.objects
        self.groups = [torch.tensor(g,device=self.device) for g in self.catalog['groups']]
        self.env_groups = torch.arange(command.num_envs,device=self.device)%len(self.groups)
        self.points = torch.zeros(len(self.mask_table),self.count,128,3,device=self.device)
        cache = {}
        for i,dirs in enumerate(self.catalog['directories']):
            for slot,directory in enumerate(dirs):
                if directory is None: continue
                scale = self.catalog['scales'][i][slot]
                key = (directory,scale)
                if key not in cache:
                    mesh = trimesh.load(self.catalog['path'].parent/directory/'visual.obj',force='mesh',process=False)
                    p,_ = trimesh.sample.sample_surface(mesh,128,seed=0)
                    p = torch.as_tensor(p*scale,device=self.device,dtype=torch.float32)
                    cache[key] = p
                self.points[i,slot] = cache[key]*(.058/cache[key].norm(dim=-1).max().clamp_min(1e-6))

    @property
    def mask(self): return self.mask_table[self.c.motion_ids]

    def sample_ids(self, env_ids):
        result = torch.empty_like(env_ids)
        for group, clips in enumerate(self.groups):
            selected = self.env_groups[env_ids] == group
            n = int(selected.sum())
            if n: result[selected] = clips[torch.randint(len(clips),(n,),device=self.device)]
        return result

    def ref(self, field):
        value = getattr(self.state, f'{field}_w')[self.c.frame_indices]
        if field=='pos': value = value+self.env.scene.env_origins[:,None]
        return value

    def sim(self, field):
        attrs={'pos':'root_link_pos_w','quat':'root_link_quat_w','lin_vel':'root_link_lin_vel_w','ang_vel':'root_link_ang_vel_w'}
        return torch.stack([getattr(self.env.scene[name].data,attrs[field]) for name in self.names],1)

    def hand_field(self, side, attr):
        attr = attr.replace('body_link_', 'body_').replace('body_com_', 'body_')
        field = {'body_pos_w':'pos','body_quat_w':'quat','body_lin_vel_w':'lin_vel','body_ang_vel_w':'ang_vel'}[attr]
        slots = self.hand_slots[self.c.motion_ids,('right','left').index(side)]
        values = self.sim(field)
        return values[torch.arange(len(slots),device=self.device),slots]

    def reset(self, env_ids):
        values = torch.cat([self.ref(k) for k in ('pos','quat','lin_vel','ang_vel')],-1)[env_ids]
        parked = torch.zeros_like(values)
        parked[...,:3] = self.env.scene.env_origins[env_ids,None]
        parked[...,2] -= 10
        parked[...,3] = 1
        values = torch.where(self.mask[env_ids,:,None],values,parked)
        for slot,name in enumerate(self.names):
            self.env.scene[name].write_root_state_to_sim(values[:,slot],env_ids=env_ids)

    def tracking_error(self):
        points = self.points[self.c.motion_ids]
        dr = matrix_from_quat(self.ref('quat'))-matrix_from_quat(self.sim('quat'))
        dt = self.ref('pos')-self.sim('pos')
        error = (dt[:,:,None]+torch.einsum('bsij,bspj->bspi',dr,points)).norm(dim=-1).topk(3).values.mean(-1)
        return error

    def reward(self, sigma=.1):
        value = torch.exp(-self.tracking_error()/sigma)
        return torch.where(self.mask,value,0.).sum(-1)/self.mask.sum(-1)

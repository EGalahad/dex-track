"""Associated collision surfaces expressed in each current wrist frame."""
import torch
import trimesh
from .common import TrackObservation
from dex_track.assets import body_name
from dex_track.tasks.reference.collision_geometry import collision_mesh
from ..reference.hand import ALLLINK_BODY_NAMES_BY_SIDE
from ..reference.math import quat_apply, quat_apply_inverse
from ..reference.object_points import farthest_point_sample


def associated_surface_points(command, n_points):
    cache = getattr(command, '_collision_points', None)
    if cache is None:
        cache = command._collision_points = {}
    if n_points not in cache:
        scene = command.scene_objects
        pools = {}; sides = []
        for side in command.hand_sides:
            si = ('right','left').index(side)
            clips = []
            for i, dirs in enumerate(scene.catalog['directories']):
                slot = int(scene.catalog['hand_slots'][i,si])
                path = scene.catalog['path'].parent/dirs[slot]/'convex'
                scale = scene.catalog['scales'][i][slot]
                key = (str(path),scale)
                if key not in pools:
                    mesh = collision_mesh(path,scale)
                    points,_ = trimesh.sample.sample_surface(mesh,max(2048,n_points),seed=0)
                    points = torch.as_tensor(points,dtype=torch.float32)
                    pools[key] = points[farthest_point_sample(points,n_points)]
                clips.append(pools[key])
            sides.append(torch.stack(clips))
        cache[n_points] = torch.stack(sides,1).to(command.device)
    local = cache[n_points][command.motion_ids]
    world = quat_apply(command.object_quat_w[:,:,None].expand(-1,-1,n_points,-1),local)+command.object_pos_w[:,:,None]
    return quat_apply_inverse(command.robot_wrist_quat_w[:,:,None].expand(-1,-1,n_points,-1),
                              world-command.robot_wrist_pos_w[:,:,None])


def nearest_surface_points(links, points):
    """Per-link [nearest surface xyz, unsigned Euclidean distance]."""
    distances = torch.cdist(links,points,compute_mode='donot_use_mm_for_euclid_dist')
    ids = distances.argmin(-1)
    closest = points.gather(-2,ids[...,None].expand(*ids.shape,3))
    distance = (closest-links).norm(dim=-1,keepdim=True)
    return torch.cat((closest,distance),-1)


class object_surface_points_wrist(TrackObservation, namespace='dex_track'):
    def __init__(self,n_points=128):
        super().__init__()
        if n_points < 1: raise ValueError('n_points must be positive')
        self.n_points = n_points

    def compute(self):
        return associated_surface_points(self.command_manager,self.n_points).flatten(1)


class link_nearest_surface_wrist(object_surface_points_wrist, namespace='dex_track'):
    def _initialize(self,env):
        super()._initialize(env)
        self.asset = env.scene['robot']
        self.body_ids = [[self.asset.body_names.index(body_name(n,env.backend))
                          for n in ALLLINK_BODY_NAMES_BY_SIDE[side]]
                         for side in self.command_manager.hand_sides]

    def compute(self):
        c = self.command_manager
        links = self.asset.data.body_link_pos_w[:,self.body_ids]
        links = quat_apply_inverse(c.robot_wrist_quat_w[:,:,None].expand(-1,-1,links.shape[2],-1),
                                   links-c.robot_wrist_pos_w[:,:,None])
        return nearest_surface_points(links,associated_surface_points(c,self.n_points)).flatten(1)

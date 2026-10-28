"""Asset factories registered with AA; no project-owned simulation environment."""
from functools import lru_cache
from pathlib import Path
import yaml
from active_adaptation.registry import Registry
from dex_track.source import source_root

# Stable observation contract: preserve the joint order of existing Isaac policies.
# AA find_joints maps this named order to each backend at runtime.
JOINT_NAMES_SIMULATION = [
    'L_forearm_pos_x_joint',
    'R_forearm_pos_x_joint',
    'L_forearm_pos_y_joint',
    'R_forearm_pos_y_joint',
    'L_forearm_pos_z_joint',
    'R_forearm_pos_z_joint',
    'L_forearm_rot_z_joint',
    'R_forearm_rot_z_joint',
    'L_forearm_rot_x_joint',
    'R_forearm_rot_x_joint',
    'L_forearm_rot_y_joint',
    'R_forearm_rot_y_joint',
    'left_hand_index_bend_joint',
    'left_hand_mid_joint1',
    'left_hand_pinky_joint1',
    'left_hand_ring_joint1',
    'left_hand_thumb_bend_joint',
    'right_hand_index_bend_joint',
    'right_hand_mid_joint1',
    'right_hand_pinky_joint1',
    'right_hand_ring_joint1',
    'right_hand_thumb_bend_joint',
    'left_hand_index_joint1',
    'left_hand_mid_joint2',
    'left_hand_pinky_joint2',
    'left_hand_ring_joint2',
    'left_hand_thumb_rota_joint1',
    'right_hand_index_joint1',
    'right_hand_mid_joint2',
    'right_hand_pinky_joint2',
    'right_hand_ring_joint2',
    'right_hand_thumb_rota_joint1',
    'left_hand_index_joint2',
    'left_hand_thumb_rota_joint2',
    'right_hand_index_joint2',
    'right_hand_thumb_rota_joint2',
]


@lru_cache(None)
def description():
    return yaml.safe_load((source_root()/'config/envs/robot/xhand_bimanual.yaml').read_text())['robot']


@lru_cache(None)
def robot_model():
    import mujoco
    return mujoco.MjModel.from_xml_path(str(source_root()/'assets/robot/xhand/bimanual.xml'))


def xhand(backend, usd_path=None):
    from active_adaptation.assets.asset_cfg import AssetCfg, ActuatorCfg, InitialStateCfg
    cfg = AssetCfg(
        mjcf_path=source_root()/'assets/robot/xhand/bimanual.xml',
        usd_path=usd_path or '', init_state=InitialStateCfg(),
        actuators={
            'wrist': ActuatorCfg(joint_names_expr='[RL]_forearm_.*', effort_limit=100., velocity_limit=1000.,
                                stiffness=0., damping=0., friction=0., armature=0.),
            'fingers': ActuatorCfg(joint_names_expr='(right|left)_hand_.*', effort_limit=100., velocity_limit=1000.,
                                  stiffness=20., damping=1.5, friction=0., armature=.1),
        },
        self_collisions=True,
        joint_names_simulation=JOINT_NAMES_SIMULATION,
    )
    result = cfg.to_asset_spec(backend)
    if backend == 'isaaclab':
        if not usd_path or not Path(usd_path).is_file():
            raise FileNotFoundError('Set task.robot.usd_path to the converted XHand USD')
        result.config.spawn.rigid_props.disable_gravity = True
        result.config.actuators["fingers"].damping = 2.5
        result.config.spawn.collision_props.contact_offset = .002
        result.config.soft_joint_pos_limit_factor = 1.
    elif backend == 'mjlab':
        from mjlab.utils.spec_config import CollisionCfg
        result.config.articulation.soft_joint_pos_limit_factor = 1.
        spec = result.config.spec_fn()
        for actuator in list(spec.actuators):
            spec.delete(actuator)
        result.config.articulation.actuators = result.config.articulation.actuators[1:]
        for body in spec.bodies:
            body.gravcomp = 1.
        for joint in spec.joints:
            if 'forearm' in joint.name:
                joint.damping[:] = 0.
                joint.armature = 0.
                joint.frictionloss = 0.
        result.config.collisions = (CollisionCfg(
            geom_names_expr=('collision_hand_.*',), priority=0, contype=1, conaffinity=1,
            condim=3, friction=(.5, .005, .0001), disable_other_geoms=True),)
    else:
        raise ValueError(f'Unsupported backend: {backend}')
    return result


Registry.instance().register('asset', 'dex_track.xhand', xhand)


def body_name(name, backend):
    # The pinned MJCF-to-URDF converter renames only the two palm bodies.
    if backend == 'isaaclab':
        return {'R_forearm_rot_y_link':'right_hand_link',
                'L_forearm_rot_y_link':'left_hand_link'}.get(name, name)
    return name



@lru_cache(None)
def object_inertial(directory):
    """Match the pinned USD converter: density 800 and mass clamped to 0.05–0.5 kg."""
    import trimesh
    root = Path(directory)
    solid = trimesh.load(root/'visual.obj', force='mesh')
    if not solid.is_watertight or solid.volume <= 0:
        solid = trimesh.util.concatenate([
            trimesh.load(p, force='mesh') for p in sorted((root/'convex').glob('*.obj'))])
    solid.density = 800.
    raw_mass = float(solid.mass)
    if raw_mass <= 0:
        raise ValueError(f'Invalid object mass: {root}: {raw_mass}')
    mass = min(max(raw_mass, .05), .5)
    return mass, solid.center_mass, solid.moment_inertia*(mass/raw_mass)


def _spawn_isaac_padding(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Match the converted objects' rigid-body path: <object>/baseLink."""
    import isaaclab.sim as sim
    from isaacsim.core.utils import prims
    root = prims.create_prim(prim_path, 'Xform', translation=translation, orientation=orientation)
    sim.spawn_cuboid(f'{prim_path}/baseLink', cfg, **kwargs)
    return root


def scene_object(backend, motion_file, slot):
    """One physical object slot, with per-environment mesh layouts owned by AA."""
    from dex_track.data import load_scene_catalog
    catalog = load_scene_catalog(motion_file)
    variants = [layout[slot] for layout in catalog['layouts']]
    base = Path(motion_file).resolve().parent
    if backend == 'isaaclab':
        from isaaclab.assets import RigidObjectCfg
        import isaaclab.sim as sim
        from isaaclab.sim.utils import clone
        spawns = []
        for directory, scale in variants:
            if directory is None:
                spawns.append(sim.CuboidCfg(func=clone(_spawn_isaac_padding), size=(.001,.001,.001),
                    rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=True,kinematic_enabled=True),
                    collision_props=sim.CollisionPropertiesCfg(collision_enabled=False),
                    mass_props=sim.MassPropertiesCfg(mass=.001),visible=False))
            else:
                path = base/directory/'.isaac_usd/object.usd'
                if not path.is_file(): raise FileNotFoundError(path)
                spawns.append(sim.UsdFileCfg(usd_path=str(path),scale=(scale,)*3,
                    rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=False)))
        spawn = spawns[0] if len(spawns)==1 else sim.MultiAssetSpawnerCfg(assets_cfg=spawns,random_choice=False)
        return RigidObjectCfg(prim_path='',spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,-10.)))
    if backend == 'mjlab':
        import mujoco
        from mjlab.entity import EntityCfg
        from mjlab.entity.variants import VariantEntityCfg
        def builder(directory, scale):
            def build():
                spec=mujoco.MjSpec()
                body=spec.worldbody.add_body(name='object')
                body.add_freejoint(name='free')
                if directory is None:
                    import trimesh
                    mesh=trimesh.creation.box(extents=(.001,)*3)
                    spec.add_mesh(name='padding',uservert=mesh.vertices.flatten(),userface=mesh.faces.flatten())
                    body.add_geom(name='padding',type=mujoco.mjtGeom.mjGEOM_MESH,meshname='padding',
                        mass=.001,contype=0,conaffinity=0,rgba=(0.,0.,0.,0.))
                    # VariantEntityCfg requires one inertia representation across
                    # real and padding bodies. Preserve the tiny box's inertia.
                    body.explicitinertial = True
                    body.mass = .001
                    body.ipos = [0., 0., 0.]
                    box_inertia = .001 * (.001**2 + .001**2) / 12
                    body.fullinertia = [box_inertia]*3 + [0., 0., 0.]
                else:
                    parts=sorted((base/directory/'convex').glob('*.obj'))
                    if not parts:raise FileNotFoundError(base/directory/'convex')
                    mass, com, inertia = object_inertial(str(base/directory))
                    body.explicitinertial = True
                    body.mass = mass
                    body.ipos = com*scale
                    inertia = inertia*scale**2
                    body.fullinertia = [inertia[0,0], inertia[1,1], inertia[2,2],
                                        inertia[0,1], inertia[0,2], inertia[1,2]]
                    for j,path in enumerate(parts):
                        name=f'part_{j}'
                        spec.add_mesh(name=name,file=str(path),scale=[scale]*3)
                        body.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_MESH,meshname=name,
                            density=800.,friction=[.5,.005,.0001],condim=3)
                return spec
            return build
        init=EntityCfg.InitialStateCfg(pos=(0.,0.,-10.))
        if len(variants)==1:return EntityCfg(spec_fn=builder(*variants[0]),init_state=init)
        return VariantEntityCfg(variants={f'layout_{i}':builder(*v) for i,v in enumerate(variants)},
            assignment=lambda n:[i%len(variants) for i in range(n)],init_state=init)
    raise ValueError(backend)


Registry.instance().register('asset','dex_track.scene_object',scene_object)

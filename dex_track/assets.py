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
    spec = mujoco.MjSpec.from_file(str(source_root()/'assets/robot/xhand/bimanual.xml'))
    _remove_palm_sites(spec)
    return spec.compile()


def _remove_palm_sites(spec):
    """Palm alignment is a mathematical transform, never a simulator site."""
    for site in list(spec.sites):
        if 'palm' in site.name.lower():
            spec.delete(site)


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
    # MultiAssetSpawner replaces child collision_props with its top-level
    # offsets. Preserve the absent object's no-collision contract explicitly.
    cfg = cfg.replace(collision_props=sim.CollisionPropertiesCfg(collision_enabled=False))
    sim.spawn_cuboid(f'{prim_path}/baseLink', cfg, **kwargs)
    return root


def _spawn_isaac_support(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """USD spawners need an explicit physics-material binding on this Isaac version."""
    import isaaclab.sim as sim
    root = sim.spawn_from_usd(prim_path, cfg, translation=translation,
                              orientation=orientation, **kwargs)
    material = f'{prim_path}/supportMaterial'
    sim.spawn_rigid_body_material(material, sim.RigidBodyMaterialCfg(
        static_friction=.5, dynamic_friction=.5, restitution=0.))
    sim.bind_physics_material(prim_path, material)
    return root


def scene_object(backend, motion_file, slot, collision_variant='simplified'):
    """One physical object slot, with per-environment mesh layouts owned by AA."""
    from dex_track.data import load_scene_catalog
    catalog = load_scene_catalog(motion_file)
    variants = [layout[slot] for layout in catalog['layouts']]
    base = Path(motion_file).resolve().parent
    return _rigid_variants(backend, base, variants, staging_position=(3. * (slot + 1), 0., 10.), collision_variant=collision_variant)


def scene_support(backend, motion_file):
    """Fixed scene geometry from per-motion support poses; no tracking slots."""
    from dex_track.data import load_scene_catalog
    from dex_track.supports import support_asset_directory
    catalog = load_scene_catalog(motion_file)
    rows = catalog['layout_supports']
    # mjlab's variant realization currently shares geom friction across worlds.
    # Do not silently misrepresent per-motion friction in either backend.
    if any(s['friction'] != .5 for row in rows for s in row):
        raise ValueError('Scene support assets currently require friction=0.5')
    variants = [(support_asset_directory(row), 1.) for row in rows]
    return _rigid_variants(backend, Path(motion_file).resolve().parent, variants, fixed=True)


def _rigid_variants(backend, base, variants, fixed=False, staging_position=(0., 0., 10.), collision_variant='simplified'):
    if collision_variant not in ('original', 'simplified'):
        raise ValueError(collision_variant)
    if backend != 'mjlab' and collision_variant != 'original':
        raise NotImplementedError('Simplified collision currently requires mjlab; do not silently use original USD')
    # Keep uninitialized objects apart during the template mj_forward.
    # Command reset overwrites these absolute free-root poses before physics.
    position = (0., 0., 0.) if fixed else staging_position
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
                    rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=fixed, kinematic_enabled=fixed),
                    **({'func': clone(_spawn_isaac_support),
                        'collision_props': sim.CollisionPropertiesCfg(
                          contact_offset=.002, rest_offset=0.)} if fixed else {})))
        spawn = spawns[0] if len(spawns)==1 else sim.MultiAssetSpawnerCfg(assets_cfg=spawns,random_choice=False)
        if fixed:
            spawn.collision_props = sim.CollisionPropertiesCfg(contact_offset=.002, rest_offset=0.)
        return RigidObjectCfg(prim_path='',spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=position))
    if backend == 'mjlab':
        import mujoco
        from mjlab.entity import EntityCfg
        from mjlab.entity.variants import VariantEntityCfg
        def builder(directory, scale):
            def build():
                spec=mujoco.MjSpec()
                body=spec.worldbody.add_body(name='object', pos=position)
                if not fixed:
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
                        collision_path = simplified_object_collision(path) if collision_variant == 'simplified' else path
                        spec.add_mesh(name=name,file=str(collision_path),scale=[scale]*3)
                        body.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_MESH,meshname=name,
                            density=800.,friction=[.5,.005,.0001],condim=3)
                if fixed:
                    # Missing variant mesh slots are excluded by the backend
                    # before broadphase; no distant collision placeholders.
                    body.mocap = True
                    # Explicit fixed-body inertia also makes empty padding variants
                    # compatible with real meshes during the authoring merge.
                    body.mass = 1.
                    body.ipos = [0.,0.,0.]
                    body.fullinertia = [1.,1.,1.,0.,0.,0.]
                return spec
            return build
        init=EntityCfg.InitialStateCfg(pos=position)
        if len(variants)==1:return EntityCfg(spec_fn=builder(*variants[0]),init_state=init)
        variant_cls = VariantEntityCfg
        if fixed:
            from active_adaptation.envs.backends.mjlab.fixed_variants import FixedVariantEntityCfg
            variant_cls = FixedVariantEntityCfg
        return variant_cls(variants={f'layout_{i}':builder(*v) for i,v in enumerate(variants)},
            assignment=lambda n:[i%len(variants) for i in range(n)],init_state=init)
    raise ValueError(backend)


Registry.instance().register('asset','dex_track.scene_object',scene_object)
Registry.instance().register('asset','dex_track.scene_support',scene_support)


def xhand_free(backend, side='right', finger_damping=2.5, usd_path=None, palm_position_root=None, collision_variant='simplified'):
    """Independent wrist-root entity with pinned upstream inertial parameters."""
    if backend != 'mjlab':
        raise NotImplementedError('Integrated free-hand pair currently requires mjlab; Isaac assets are separately audited')
    import mujoco
    from active_adaptation.assets.asset_cfg import AssetCfg, ActuatorCfg, InitialStateCfg
    from mjlab.utils.spec_config import CollisionCfg
    cfg=AssetCfg(mjcf_path=Path(__file__).resolve().parents[1]/'assets/github_xhand'/f'{side}.xml',
        usd_path='', init_state=InitialStateCfg(),
        # mode='explicit' maps to mjlab BuiltinPdActuatorCfg: native <position> (kp) + <velocity> (kd) pair, so the
        # PD tracks both the position target and the velocity target (kd*(v_target - v)); effort is a sum-clamp.
        # 'implicit' (BuiltinPositionActuator) would silently ignore velocity targets.
        actuators={'fingers':ActuatorCfg(joint_names_expr=f'{side}_hand_.*',mode='explicit',effort_limit=100.,
            velocity_limit=1000.,stiffness=20.,damping=finger_damping,friction=0.,armature=.01)},
        self_collisions=True)
    result=cfg.to_asset_spec(backend)
    spec=result.config.spec_fn()
    _remove_palm_sites(spec)
    if collision_variant not in ('original', 'simplified'):
        raise ValueError(collision_variant)
    if collision_variant == 'simplified':
        import json
        root = Path(__file__).resolve().parents[1]/'assets/collision_simplified'
        mapping = json.loads((root/'explicit-mesh-map.json').read_text())['hands']
        for geom in spec.geoms:
            if not geom.name.startswith('collision_hand_'):
                continue
            original = geom.meshname
            name = 'simplified_'+original
            if spec.mesh(name) is None:
                spec.add_mesh(name=name, file=str(root/mapping[original]))
            geom.meshname = name
    if palm_position_root is not None:
        import numpy as np
        position = np.asarray(palm_position_root[side], dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('palm_position_root must contain finite xyz for each hand')
        # Validated metadata only: command applies semantic alignment, no sites.
    for a in list(spec.actuators):spec.delete(a)
    # The XML <default><position kp kv forcerange/> would leak into every actuator mjlab adds later: the BuiltinPd
    # <velocity> element would inherit biasprm[1]=-kp (an extra, target-free stiffness toward q=0). Neutralize it.
    act_default=spec.default.actuator
    act_default.gainprm[:]=0.;act_default.gainprm[0]=1.;act_default.biasprm[:]=0.
    act_default.forcelimited=mujoco.mjtLimited.mjLIMITED_AUTO;act_default.forcerange[:]=0.
    for b in spec.bodies:b.gravcomp=1.
    result.config.articulation.soft_joint_pos_limit_factor=1.
    result.config.collisions=(CollisionCfg(geom_names_expr=('collision_hand_.*',),priority=0,
        contype=1,conaffinity=1,condim=3,friction=(.5,.005,.0001),disable_other_geoms=True),)
    if side=='right':
        from dex_track.free_hands import InstallHandPair
        result.wrapper=InstallHandPair()
    return result

Registry.instance().register('asset','dex_track.xhand_free',xhand_free)


def simplified_object_collision(path):
    # Preserve original visual/inertial assets and each independent convex piece.
    root = Path(__file__).resolve().parents[1]/'assets/collision_simplified'
    if '/objects/' not in str(path):
        raise ValueError(f'Object collision path lacks stable object identity: {path}')
    target = root/'objects'/str(path).split('/objects/', 1)[1]
    if not target.is_file():
        raise FileNotFoundError(f'Missing verified simplified collision: {target}')
    return target

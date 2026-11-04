import os
"""Independent source-contract check before expensive rest catchup."""
from pathlib import Path
import os,json,time,hashlib
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,trimesh
from scipy.spatial import cKDTree
T=Path(os.environ['DEX_CLEANUP_WORK']).resolve();V=T/'artifacts/parking-v1';O=T/'staging';S=Path(os.environ['DEX_DATASET']);idx=json.loads((O/'index.json').read_text());trees={};max_error=0.;native_frames=0;active=parked=0;hashes={}
for mi,e in enumerate(idx['motions']):
 p=(O/e['path']).parent;src=(S/e['path']).parent;meta=json.loads((p/'metadata.json').read_text());contact=dict(np.load(p/'contact.npz'));old=dict(np.load(src/'contact.npz'));mano=dict(np.load(p/'mano.npz'));scene=dict(np.load(p/'scene.npz'));r=json.loads((V/'motions'/f'{mi:04d}.json').read_text());native_frames+=len(mano['mano_right_joints'])
 for side in ['right','left']:
  key='contact_contact_pos_full_'+side;tip=mano['mano_'+side+'_joints'][:,[19,3,7,15,11]].astype(float);world=np.einsum('tij,tkj->tki',scene['obj_'+side+'_rotmat'].astype(float),contact[key].astype(float))+scene['obj_'+side+'_pos'][:,None];distance=np.linalg.norm(tip-world,axis=-1);error=float(abs(distance-contact['tips_distance_'+side]).max());max_error=max(max_error,error);assert np.allclose(distance,contact['tips_distance_'+side],atol=2e-6),(mi,side,'local-to-world distance')
  assert np.array_equal(contact['contact_contact_'+side],old['contact_contact_'+side])
  if side not in r['parking_world_delta_m']:
   active+=1;assert np.array_equal(contact[key],old[key]);assert np.array_equal(contact['tips_distance_'+side],old['tips_distance_'+side]);continue
  parked+=1;assert not contact['contact_contact_'+side].any();slot=int(scene['motion_object_slot_'+side][0]);oid=Path(meta[side+'_object_mesh_dirs'][slot]).name;scale=float(meta['motion_scene_object_scales'][0][meta['motion_scene_object_ids'][0].index(oid)]);tree_key=(oid,scale)
  if tree_key not in trees:
   verts=np.asarray(trimesh.load(S/'objects'/oid/'visual.obj',force='mesh',process=False,skip_materials=True).vertices,dtype=float)*scale;trees[tree_key]=(verts,cKDTree(verts))
  verts,tree=trees[tree_key];local=np.einsum('tji,tkj->tki',scene['obj_'+side+'_rotmat'].astype(float),tip-scene['obj_'+side+'_pos'][:,None]);nearest_distance,indices=tree.query(local.reshape(-1,3));assert np.allclose(nearest_distance.reshape(distance.shape),contact['tips_distance_'+side],atol=2e-6);assert np.allclose(verts[indices].reshape(contact[key].shape),contact[key],atol=2e-7)
 hashes[str((p/'contact.npz').relative_to(O))]=hashlib.sha256((p/'contact.npz').read_bytes()).hexdigest()
result=dict(status='passed',revision='object-local-contact-v2',motions=704,native_frames=native_frames,active_sides_exact_original_local_points_and_distances=active,parked_sides_recomputed_and_nearest_vertex_verified=parked,all_flags_exact_original=True,max_local_to_world_tip_distance_error_m=max_error,source_contract='object-local nearest scaled original visual vertex; original build_motion_pt.py and world conversion consumer',supersedes='incorrect world-translation contact validation',contact_files_sha256=hashes,physical_admission=False,published=False,heartbeat_unix=time.time());(V/'contact-local-validation.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='contact_files_sha256'},flush=True)

import os
"""Source contract: contact positions are object-local vertices, not world coordinates."""
from pathlib import Path
code=Path(__file__).with_name('parking_process.py').read_text();exec(code.split('OUT.mkdir(exist_ok=True)')[0])
from scipy.spatial import cKDTree
names=sorted(f'{finger}_{part}' for finger in ['thumb','index','middle','ring','pinky'] for part in ['proximal','intermediate','distal','tip']);tips=[names.index(f'{finger}_tip') for finger in ['thumb','index','middle','ring','pinky']];trees={};reports=[]
for mi,e in enumerate(idx['motions']):
 path=OUT/e['path'];meta=json.loads(path.read_text());version='object-local-contact-v2'
 if meta['meta'].get('contact_semantics_revision')==version:reports.append(dict(index=mi,already_corrected=True,sides=meta['meta']['contact_semantics_provenance']['sides']));continue
 source=dict(np.load((S/e['path']).parent/'contact.npz'));out={k:v.copy() for k,v in source.items()};mano=dict(np.load(path.parent/'mano.npz'));scene=dict(np.load(path.parent/'scene.npz'));r=json.loads((V/'motions'/f'{mi:04d}.json').read_text());provenance={}
 for side in ['right','left']:
  if side not in r['parking_world_delta_m']:
   assert np.array_equal(out['contact_contact_pos_full_'+side],source['contact_contact_pos_full_'+side]);provenance[side]=dict(kind='source_object_local_points_distances_flags_preserved_exactly');continue
  slot=int(scene['motion_object_slot_'+side][0]);oid=Path(meta[side+'_object_mesh_dirs'][slot]).name;scale=float(meta['motion_scene_object_scales'][0][meta['motion_scene_object_ids'][0].index(oid)]);key=(oid,scale)
  if key not in trees:
   verts=np.asarray(trimesh.load(S/'objects'/oid/'visual.obj',force='mesh',process=False,skip_materials=True).vertices,dtype=float)*scale;trees[key]=(verts,cKDTree(verts))
  verts,tree=trees[key];local=np.einsum('tji,tkj->tki',scene['obj_'+side+'_rotmat'].astype(float),mano['mano_'+side+'_joints'][:,tips].astype(float)-scene['obj_'+side+'_pos'][:,None].astype(float));distance,nearest=tree.query(local.reshape(-1,3));out['tips_distance_'+side]=distance.reshape(local.shape[:2]).astype(np.float32);out['contact_contact_pos_full_'+side]=verts[nearest].reshape(local.shape).astype(np.float32)
  assert np.count_nonzero(source['contact_contact_'+side])==0 and distance.min()>.01,(mi,side,'parking must remain noncontact')
  provenance[side]=dict(kind='recomputed_nearest_scaled_original_visual_vertex_object_local',minimum_parked_tip_distance_m=float(distance.min()),flags_source_zero_preserved=True,object_id=oid,scale=scale,tip_joint_columns=tips)
 contact=path.parent/'contact.npz';tmp=contact.with_name('contact.repair.partial.npz');np.savez_compressed(tmp,**out);tmp.replace(contact)
 meta['meta']['contact_semantics_revision']=version;meta['meta']['contact_position_coordinate_frame']='object_local';meta['meta']['contact_semantics_provenance']=dict(source_contract='FlashDexRetarget/src/retarget/build_motion_pt.py:27..39 tip_contacts; nearest visual vertex in object frame',sides=provenance,supersedes='incorrect world translation of object-local contact positions and stale parked tip distances');meta['meta']['inactive_hand_parking']['tips_distance_semantics']='recomputed for parked hand; source unchanged for active hand';save(path,meta);reports.append(dict(index=mi,sides=provenance))
save(V/'contact-local-repair-manifest.json',reports);save(V/'contact-local-repair-summary.json',dict(motions=len(reports),revision='object-local-contact-v2',active_contact_points_exact_source=True,parked_tips_distance_and_local_points_recomputed=True,flags_zero_preserved=True,published=False,physical_admission=False,heartbeat_unix=time.time()));print('object-local contact repair704 done',flush=True)

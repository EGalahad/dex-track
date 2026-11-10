"""Convert a v2 portable corpus to independently verified pose-only v3; never in-place."""
import argparse,copy,hashlib,json,shutil,tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
SCHEMA='dex-track-motion-folders-v3';REVISION='2026-11-10-pose-only-v3'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def convert(source,destination,calibration):
 source=Path(source).resolve();destination=Path(destination).resolve()
 if source==destination or destination.exists():raise FileExistsError(destination)
 index=json.loads((source/'index.json').read_text());assert index['schema']=='dex-track-motion-folders-v2'
 metas=[json.loads((source/r['path']).read_text()) for r in index['motions']]
 common={k:copy.deepcopy(v) for k,v in metas[0]['meta'].items() if all(m['meta'].get(k)==v for m in metas)}
 defaults={'meta':common,**{k:metas[0][k] for k in ['mano_right_joint_names','mano_left_joint_names']}}
 for m in metas:
  for k in ['robot','mano_right_joint_names','mano_left_joint_names']:assert m[k]==metas[0][k],k
 contract=json.loads((source/'dataset.json').read_text());contract.update(schema=SCHEMA,revision=REVISION,motion_metadata_defaults=defaults,host='local-prepromotion-candidate',promotion_status='pending_receipt_approval')
 contract['robot'].update(metas[0]['robot']);contract['robot'].pop('velocity',None);contract['robot']['velocities']='derived after policy-rate pose resampling'
 fit=json.loads(Path(calibration).read_text())
 if fit['provenance']['source_index_sha256'] != digest(source/'index.json'):
  raise ValueError('Palm calibration was fitted to a different source corpus')
 for side in ['right','left']:
  p=np.asarray(fit['palm_in_root'][side]['position_xyz']);q=np.asarray(fit['palm_in_root'][side]['quaternion_wxyz'])
  if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(p).all() or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-5:
   raise ValueError('Invalid palm calibration')
 contract['robot']['palm_in_root']=fit['palm_in_root'];contract['robot']['palm_alignment_provenance']=fit['provenance'];contract['robot']['palm_alignment_residuals']=fit['residuals']
 stage=Path(tempfile.mkdtemp(prefix='.'+destination.name+'-',dir=destination.parent));before={};after={};verified=0;max_qdiff=0.;kept_checks=[]
 try:
  for asset in ['objects','supports','collision_simplified']:
   if (source/asset).exists():shutil.copytree(source/asset,stage/asset,ignore=shutil.ignore_patterns('.sdf_cache','.isaac_usd','__pycache__'))
  for row,meta in zip(index['motions'],metas):
   src=(source/row['path']).parent;out=(stage/row['path']).parent;out.mkdir(parents=True);arrays={}
   for name in ['robot','mano','objects']:
    before[name]=before.get(name,0)+(src/(name+'.npz')).stat().st_size
    with np.load(src/(name+'.npz'),allow_pickle=False) as f:arrays[name]={k:f[k].copy() for k in f.files}
   obj=arrays['objects'];ids=meta['motion_scene_object_ids'][0]
   for side in ['right','left']:
    oldslot=int(obj[f'motion_object_slot_{side}'][0]);oid=Path(meta[f'{side}_object_mesh_dirs'][oldslot]).name;slot=ids.index(oid)
    assert meta['motion_segments'][0]['hand_objects'][side]==oid
    assert np.isclose(meta[f'{side}_object_mesh_scales'][oldslot],meta['motion_scene_object_scales'][0][slot])
    np.testing.assert_array_equal(obj[f'obj_{side}_pos'],obj['scene_object_pos'][:,slot])
    rot=Rotation.from_quat(obj['scene_object_quat'][:,slot],scalar_first=True).as_matrix();delta=float(np.max(np.abs(rot-obj[f'obj_{side}_rotmat'])));max_qdiff=max(max_qdiff,delta);assert delta<2e-6
    obj[f'motion_object_slot_{side}']=np.full_like(obj[f'motion_object_slot_{side}'],slot)
   keep={'robot':{'qpos'},'mano':{f'mano_{s}_{k}' for s in ['right','left'] for k in ['wrist_pos','wrist_rot','joints']},'objects':{'scene_object_pos','scene_object_quat','motion_object_slot_right','motion_object_slot_left','support_disks','motion_num_frames','length_starts'}}
   deleted=set()
   for name,a in arrays.items():
    deleted.update(set(a)-keep[name]);saved={k:v for k,v in a.items() if k in keep[name]}
    np.savez_compressed(out/(name+'.npz'),**saved)
    with np.load(out/(name+'.npz'),allow_pickle=False) as reread:
     assert set(reread.files)==set(saved)
     for k,v in saved.items():
      assert v.dtype==reread[k].dtype and v.shape==reread[k].shape and v.tobytes()==reread[k].tobytes();verified+=1
    # All retained poses are compared against original archives, not reconstructed values.
    with np.load(src/(name+'.npz'),allow_pickle=False) as original:
     for k,v in saved.items():
      if not k.startswith('motion_object_slot_'):assert original[k].tobytes()==v.tobytes(),k
    after[name]=after.get(name,0)+(out/(name+'.npz')).stat().st_size
   def clean(v):
    if isinstance(v,dict):
     return {k:clean(x) for k,x in v.items() if not (isinstance(x,dict) and x.get('__array__') in deleted)}
    if isinstance(v,list):return [clean(x) for x in v]
    return v
   m=clean(meta);m['schema']=SCHEMA;m['dataset_file']='../../dataset.json';m['meta']={k:v for k,v in m['meta'].items() if k not in common}
   for k in ['robot','mano_right_joint_names','mano_left_joint_names','right_object_mesh_dirs','left_object_mesh_dirs','right_object_mesh_scales','left_object_mesh_scales']:m.pop(k,None)
   before['metadata']=before.get('metadata',0)+(src/'metadata.json').stat().st_size
   (out/'metadata.json').write_text(json.dumps(m,indent=2)+'\n');after['metadata']=after.get('metadata',0)+(out/'metadata.json').stat().st_size
   for asset in ['objects','supports','collision_simplified']:
    if (stage/asset).exists():(out/asset).symlink_to(Path('../..')/asset,target_is_directory=True)
   if (src/'topology.npz').exists():shutil.copy2(src/'topology.npz',out/'topology.npz')
  index['schema']=SCHEMA;index['revision']=REVISION;(stage/'index.json').write_text(json.dumps(index,indent=2)+'\n');contract['index_sha256']=digest(stage/'index.json');(stage/'dataset.json').write_text(json.dumps(contract,indent=2)+'\n')
  # Copy only portable asset files; no new geometry and no generated caches.
  asset_count=0
  for folder in ['objects','supports','collision_simplified']:
   for p in (stage/folder).rglob('*'):
    if p.is_file():assert digest(p)==digest(source/p.relative_to(stage));asset_count+=1
  receipt={'schema':SCHEMA,'revision':REVISION,'source':str(source),'source_index_sha256':digest(source/'index.json'),'segments':len(metas),'frames':sum(r['frames'] for r in index['motions']),'retained_pose_arrays_bit_identical':True,'npz_roundtrip_arrays':verified,'deleted_associated_positions_exact':True,'associated_rotation_matrix_max_abs_diff':max_qdiff,'hand_scene_slots_verified':True,'preserved_asset_files':asset_count,'storage_before_bytes':before,'storage_after_bytes':after,'global_meta_keys_moved':sorted(common),'palm_calibration':fit}
  (stage/'conversion-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
  (stage/'checksums.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(stage).as_posix()}\n' for p in sorted(stage.rglob('*')) if p.is_file() and not p.is_symlink() and 'checksums.sha256'!=p.name))
  stage.rename(destination);return receipt
 except BaseException:
  shutil.rmtree(stage);raise
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--destination',required=True,type=Path);p.add_argument('--palm-calibration',required=True,type=Path);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();r=convert(a.source,a.destination,a.palm_calibration);a.receipt.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='palm_calibration'},indent=2))

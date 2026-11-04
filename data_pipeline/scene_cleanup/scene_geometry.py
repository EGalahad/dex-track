"""Scene construction shared by geometry and rest admission. No renderer."""
def build(meta,indices):
 tree=handtree();root=tree.getroot();asset=tree.find('asset');world=tree.find('worldbody');E.SubElement(world,'geom',name='ground',type='plane',size='4 4 .01',rgba='.85 .88 .91 1',group='0');E.SubElement(world,'light',pos='1 -1 3',dir='-1 1 -3',diffuse='.8 .8 .8')
 for j in indices:
  oid=old['motion_scene_object_ids'][0][j];sc=str(old['motion_scene_object_scales'][0][j]);body=E.SubElement(world,'body',name=f'object_{j}',mocap='true');path=S/'objects'/oid/'visual.obj';name=f'visual_object{j}';E.SubElement(asset,'mesh',name=name,file=str(path),scale=' '.join([sc]*3));E.SubElement(body,'geom',name=name,type='mesh',mesh=name,group='1',contype='0',conaffinity='0',rgba='.72 .52 .26 1')
  for k,p in enumerate(sorted((D/'assets/collision_simplified/objects'/oid/'convex').glob('*.obj'))):
   name=f'object{j}_piece{k}';E.SubElement(asset,'mesh',name=name,file=str(p),scale=' '.join([sc]*3));E.SubElement(body,'geom',name=name,type='mesh',mesh=name,group='3',contype='1',conaffinity='1',rgba='.72 .52 .26 1')
 for j,s in enumerate(meta['motion_scene_supports'][0]):
  g=s['geometry'];size=np.array(g['size'])/2 if g['type']=='box' else [g['radius'],g['height']/2];body=E.SubElement(world,'body',name=f'support_{j}',pos=' '.join(map(str,s['position'])),quat=' '.join(map(str,s['quaternion_wxyz'])));E.SubElement(body,'geom',name=f'support_{j}_geom',type=g['type'],size=' '.join(map(str,size)),rgba='.2 .6 .7 1',group='0',contype='1',conaffinity='1')
 m=mj.MjModel.from_xml_string(E.tostring(root,encoding='unicode'));m.vis.global_.offwidth=640;m.vis.global_.offheight=480;m.light_castshadow[:]=0;m.vis.headlight.active=1;m.vis.headlight.ambient[:]=.4;m.vis.headlight.diffuse[:]=.7;return m,mj.MjData(m)

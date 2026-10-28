"""Render actual source meshes and CoACD hulls, without changing assets."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from pathlib import Path
import colorsys
import json
import numpy as np
import pyrender
import trimesh
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parents[2]/'data/flash-dex/preview-eight/objects'
OUT=ROOT/'outputs/eight-task-preview/convex-comparison'
OUT.mkdir(parents=True,exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',23)
small=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',17)
renderer=pyrender.OffscreenRenderer(600,520)
canvas=Image.new('RGB',(1800,1160),'#edf1f6')
manifest=[]
for row,(oid,label) in enumerate([('C14001','酒杯'),('S10021','马克杯')]):
    mesh=trimesh.load(DATA/oid/'visual.obj',force='mesh',process=False)
    parts=[trimesh.load(p,force='mesh',process=False) for p in sorted((DATA/oid/'convex').glob('*.obj'),key=lambda p:int(p.stem))]
    center=mesh.bounds.mean(0);size=max(mesh.extents)
    for col,mode in enumerate(['原始三角网格','凸分解 · 保持原位置','凸块展开示意 · 非仿真状态']):
        scene=pyrender.Scene(bg_color=[.94,.96,.98,1],ambient_light=[.5,.5,.5])
        # Object meshes are in their canonical asset frame; same view in all panels.
        eye=np.array([1.15,-1.55,1.0])*1.6
        z=eye/np.linalg.norm(eye);x=np.cross([0,0,1.],z);x/=np.linalg.norm(x);y=np.cross(z,x)
        view=np.eye(4);view[:3,:3]=np.stack([x,y,z],axis=1);view[:3,3]=eye
        scene.add(pyrender.PerspectiveCamera(yfov=np.pi/5),pose=view)
        scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2.5),pose=view)
        for i,m in enumerate([mesh] if col==0 else parts):
            m=m.copy();m.vertices=(m.vertices-center)/size
            if col==2:
                direction=m.centroid.copy();m.vertices+=direction*.65
            color=[.58,.67,.75,1] if col==0 else [*colorsys.hsv_to_rgb((i*.61803398875)%1,.62,.86),1]
            scene.add(pyrender.Mesh.from_trimesh(m,smooth=False,material=pyrender.MetallicRoughnessMaterial(baseColorFactor=color,roughnessFactor=.8)))
        pixels,_=renderer.render(scene)
        panel=Image.fromarray(pixels);draw=ImageDraw.Draw(panel)
        draw.text((20,16),mode,font=font,fill='#172638')
        caption=f'{label} · {len(mesh.faces):,} 个三角面' if col==0 else f'{label} · {len(parts)} 个凸块'
        draw.text((20,50),caption,font=small,fill='#45566b')
        panel.save(OUT/f'{oid}-{col}.png');canvas.paste(panel,(col*600,row*570))
    manifest.append({'object_id':oid,'name':label,'source_triangles':len(mesh.faces),'convex_parts':len(parts)})
draw=ImageDraw.Draw(canvas);draw.text((25,1120),'同一物体、同一相机；颜色只区分凸块。展开图仅用于看清组成，不改变实际 collision。',font=small,fill='#36465b')
canvas.save(OUT/'comparison.png');(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));renderer.delete()
print(OUT/'comparison.png')

"""Export adaptive per-convex-piece collision meshes without changing visuals.

The support-function test samples 4102 directions; it is not a Hausdorff bound.
Output is a separate mesh. Preserve all source body inertials and collision filters.
"""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def hull_vertices(model, mesh_id=0):
    address=int(model.mesh_graphadr[mesh_id]);count=int(model.mesh_graph[address])
    ids=model.mesh_graph[address+2+count:address+2+2*count].astype(int)
    start=int(model.mesh_vertadr[mesh_id]);size=int(model.mesh_vertnum[mesh_id])
    vertices=np.asarray(model.mesh_vert[start:start+size],dtype=float)
    rotation=Rotation.from_quat(model.mesh_quat[mesh_id][[1,2,3,0]]).as_matrix()
    return (vertices@rotation.T+model.mesh_pos[mesh_id])[ids]


def simplify(source, destination, max_physical_scale=1.):
    spec=mujoco.MjSpec();spec.add_mesh(name='part',file=str(source.resolve()))
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH,meshname='part')
    original=hull_vertices(spec.compile())
    k=np.arange(4096);z=1-2*(k+.5)/4096;angle=k*np.pi*(3-np.sqrt(5))
    directions=np.concatenate([np.stack((np.sqrt(1-z*z)*np.cos(angle),
      np.sqrt(1-z*z)*np.sin(angle),z),-1),np.eye(3),-np.eye(3)])
    def support(vertices):
        return np.concatenate([(vertices@chunk.T).max(0) for chunk in np.array_split(directions,32)])
    original_support=support(original);tests=[];chosen=original;budget=-1
    for candidate_budget in (64,128,256):
        candidate=spec.copy()
        for mesh in candidate.meshes:mesh.maxhullvert=candidate_budget
        vertices=hull_vertices(candidate.compile())
        gap=float((original_support-support(vertices)).max()*max_physical_scale)
        ratio=float(ConvexHull(vertices).volume/ConvexHull(original).volume)
        tests.append({'budget':candidate_budget,'sampled_support_gap_m':gap,'volume_ratio':ratio})
        if budget==-1 and gap<=.001 and ratio>=.98:
            chosen,budget=vertices,candidate_budget
    destination.parent.mkdir(parents=True,exist_ok=True)
    trimesh.convex.convex_hull(chosen).export(destination)
    return {'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'output_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
      'budget':budget,'original_fallback':budget==-1,'tests':tests,
      'directions':4102,'max_physical_scale':max_physical_scale,
      'physical_admission':False,'preserve_mass_com_inertia_visual_filters':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-physical-scale',type=float,default=1.)
    a=parser.parse_args()
    if a.source.resolve()==a.output.resolve():raise ValueError('Never overwrite source collision meshes')
    if a.max_physical_scale<=0:raise ValueError('Scale must be positive')
    report=simplify(a.source,a.output,a.max_physical_scale)
    a.output.with_suffix(a.output.suffix+'.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

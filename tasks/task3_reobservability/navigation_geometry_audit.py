"""Nominal standing-camera feasibility diagnostic, never counted as a rollout."""
import argparse,json,math
from pathlib import Path
import numpy as np
from api import scene_catalogue
from benchmark import build_scene_terrain
from experiment import target_anchor
from go2_piper_camera import TARGET_RADIUS_M
from scenes import ROBOT_BASE_HEIGHT_M,CAMERA_OFFSET_XYZ

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();records=[]
for name,scene in scene_catalogue().items():
 terrain=build_scene_terrain(scene);anchor=target_anchor(terrain,scene);centre=anchor.copy();centre[2]-=TARGET_RADIUS_M
 # Candidate base position, heading towards marker, nominal upright body on local surface.
 # Cover the full base disk capable of putting a forward-offset camera within 1m of sphere.
 candidates=[]
 for radius in np.linspace(.05,1+TARGET_RADIUS_M+CAMERA_OFFSET_XYZ[0],60):
  for yaw in np.linspace(-math.pi,math.pi,72,endpoint=False):
   xy=centre[:2]-radius*np.array([math.cos(yaw),math.sin(yaw)])
   if not (terrain.x[0]+.3<=xy[0]<=terrain.x[-1]-.3 and terrain.y[0]+.3<=xy[1]<=terrain.y[-1]-.3):continue
   z=terrain.height(float(xy[0]),float(xy[1]))+ROBOT_BASE_HEIGHT_M+CAMERA_OFFSET_XYZ[2]
   horizontal=abs(radius-CAMERA_OFFSET_XYZ[0]);vertical=abs(float(centre[2])-z)
   centre_range=math.hypot(horizontal,vertical);surface_lower_bound=max(0,centre_range-TARGET_RADIUS_M)
   elevation=math.atan2(vertical,max(horizontal,1e-9));angular_radius=math.asin(min(1,TARGET_RADIUS_M/max(centre_range,1e-9)))
   candidates.append((surface_lower_bound,elevation-angular_radius,xy.tolist(),z))
 closest=min(candidates,key=lambda x:x[0]) if candidates else None
 accepted=[c for c in candidates if c[0]<1 and c[1]<math.radians(87/2)]
 records.append({'scene':name,'target_centre_xyz':centre.tolist(),'candidate_count':len(candidates),'nominal_candidate_distance_lower_bound_min_m':closest[0] if closest else None,'candidate_min':closest,'nominal_near_and_vertical_fov_candidate_count':len(accepted),'status':'nominal_pose_candidates_exist' if accepted else 'no_nominal_near_camera_candidates_on_sampled_ground','limits':'Analytic static standing-camera sample. No physics reachability, no collision/path check, no RGB occlusion check. Candidate existence does not establish solvability. Empty sample is a diagnostic under nominal height, not a universal impossibility proof.'})
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(records,indent=2))
for r in records:print(r['scene'],r['status'],round(r['nominal_candidate_distance_lower_bound_min_m'],3) if r['candidate_count'] else None,flush=True)

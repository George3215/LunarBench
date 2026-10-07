"""Positive and negative renderer checks; evaluator segmentation never feeds a policy."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tasks/task3_reobservability'))
import api
import numpy as np
from scipy.ndimage import binary_dilation
from camera_reacquisition_audit import mask_rgb
from camera_reacquisition_audit import Camera
checks=[]
for name in ['stairs_up','hist001_left','basin']:
 scene=api.scene_catalogue()[name];env=api.MuJoCoEnvironment(scene);cam=Camera(env,red_min=50)
 try:
  for label,xy in [('history',scene.history_xy or scene.approach_xy),('nominal_loss',scene.lost_xy)]:
   env.reset(xy=xy,replay_approach=False);s=cam.read(env.model,env.data);ev=cam.segmentation_check()
   cam.renderer.enable_segmentation_rendering()
   cam.renderer.update_scene(cam.render_data,camera='task3_go2_piper_front_camera')
   seg=cam.renderer.render().copy();cam.renderer.disable_segmentation_rendering()
   target=seg[...,0]==env.model.geom('task3_target_geom').id
   outside=int((mask_rgb(cam.rgb,50)&~binary_dilation(target,iterations=1)).sum())
   checks.append({'scene':name,'pose_type':label,'red_pixels':s['pixels'],'red_outside_one_pixel_target_support':outside,**ev})
 finally:cam.close();env.close()
assert any(x['red_pixels']>=3 for x in checks),'Need at least one detector-positive validation frame'
print(checks,flush=True)
assert all(x['red_outside_one_pixel_target_support']==0 for x in checks),'Red response outside target silhouette and one-pixel rendering boundary'
assert all(x['target_segmentation_pixels']>0 for x in checks if x['red_pixels']>=3),'Detector-positive frame has no target geometry'
p=Path(__file__).with_name('renderer_checks.json');p.write_text(json.dumps({'checks':checks,'red_min':50,'scope':'six static frames; finite identity spot check with one-pixel raster-boundary allowance, not global detector accuracy'},indent=2),encoding='utf8');print(p,checks)

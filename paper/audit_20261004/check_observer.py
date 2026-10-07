from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,'tasks/task3_reobservability')
import torch;torch.set_num_threads(1);torch.set_grad_enabled(False)
import api,episode as ep
from benchmark import rollout_with_visibility
from camera_reacquisition_audit import Camera
scene=api.scene_catalogue()['basin'];env=api.MuJoCoEnvironment(scene,action_table=ep.protocol_action_table());cam=Camera(env)
try:
 env.reset(replay_approach=False);saved=ep.snapshot(env)
 ep.restore(env,saved)
 r0=rollout_with_visibility(env.model,env.data,env.terrain,env.controller,env.action_table['forward'],env.target);q0=env.data.qpos.copy()
 ep.restore(env,saved)
 r1=rollout_with_visibility(env.model,env.data,env.terrain,env.controller,env.action_table['forward'],env.target,sample_observer=cam.read);q1=env.data.qpos.copy()
 report={'comparison':'both paths explicitly restored before execution','max_qpos_difference':float(abs(q1-q0).max()),'path_cost_difference_m':abs(r0['motion_cost_m']-r1['motion_cost_m']),'same_geometry_curve':[x['visible'] for x in r0['curve']]==[x['visible'] for x in r1['curve']]}
 Path('paper/audit_20261004/observer_check_corrected.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(report);assert report['max_qpos_difference']<1e-6
finally:cam.close();env.close()

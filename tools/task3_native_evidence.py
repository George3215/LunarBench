"""Generate a self-contained answer viewer and an offline point-to-motion check."""
import base64
import html
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'paper/vlm_goal_navigation/native_diagnostics_20261019'


def main():
    out=[]
    for scene in ['barrier_a0','pit_a0','window_a0']:
        pred=json.loads((ROOT/'runs/robobrain25_8b_nv'/f'{scene}_pointing/result.json').read_text())
        xy=list(map(float,re.findall(r'\d+',pred['raw'])))
        u,v=xy[0]/1000*640,xy[1]/1000*480
        fx=480/(2*math.tan(math.radians(87)/2))
        bearing=math.degrees(math.atan2(-(u-320),fx))
        # Actual archived fixed-range camera encoding, not relative-depth input.
        q=REPO/'paper/vlm_goal_navigation/action_only_transfer_v3_20261018/runs/qwen3vl8b/rgbd/event4_odom_plan/trials'/scene/'observations/00000/input_depth.png'
        saved=ROOT/'assets'/f'{scene}_encoded_camera_depth.png'
        if not saved.exists():saved.write_bytes(q.read_bytes())
        d=np.array(Image.open(saved).convert('L'),dtype=float)
        x=int(u);y=int(v)
        depth=float(np.median(10*(1-d[max(0,y-1):y+2,max(0,x-1):x+2]/255)))
        intent='TURN_LEFT' if bearing>8 else 'TURN_RIGHT' if bearing < -8 else 'FORWARD'
        out.append(dict(scene=scene,point=xy,approx_bearing_deg=bearing,
            approx_encoded_axial_depth_m=depth,candidate_motor_intent=intent,
            depth_source=str(saved.relative_to(REPO)),
            caveat='Offline coordinate-to-command check only; ignores route traversability and occluded-target planning; no action executed.'))
    (ROOT/'POINT_TO_MOTION_CHECK.json').write_text(json.dumps(dict(
        scope='offline mapping, not navigation or a learned controller',
        convention='image x right; positive bearing left; 8 degree deadband',checks=out),indent=2)+'\n')
    scores=json.loads((ROOT/'scores.json').read_text())
    inputs=json.loads((ROOT/'inputs.json').read_text())
    body=['<!doctype html><meta charset="utf-8"><title>TASK3 native diagnostic evidence</title>',
        '<style>body{font:16px system-ui;max-width:1200px;margin:auto;background:#17191d;color:#eee}article{border:1px solid #666;margin:20px 0;padding:14px}img{width:320px}pre{white-space:pre-wrap}summary{cursor:pointer}.ok{color:#8edc94}.fail{color:#ffb09c}</style>',
        '<h1>TASK3 原生格式诊断</h1><p>105 次推理；三场景九张 RGB。此页为自包含证据浏览器，不是导航成功率。逐题展开原始回答，可检查模型是否编造三维数值。</p>']
    for row in inputs:
        body.append('<article><h2>'+html.escape(row['id'])+'</h2>')
        for p in row['images']:
            b=base64.b64encode((REPO/p).read_bytes()).decode()
            body.append('<img src="data:image/png;base64,'+b+'">')
        body.append('<pre>'+html.escape(row['question'])+'</pre>')
        for s in [s for s in scores if s['id']==row['id']]:
            escaped=html.escape(s['raw_final_output']).replace('\n','&#10;').replace('\r','&#13;')
            body.append('<details><summary class="'+('ok' if s['correct'] else 'fail')+'">'+html.escape(s['model'])+(' ✓' if s['correct'] else ' ×')+'</summary><pre>'+escaped+'</pre></details>')
        body.append('</article>')
    (ROOT/'evidence.html').write_text('\n'.join(body)+'\n')


if __name__=='__main__':main()

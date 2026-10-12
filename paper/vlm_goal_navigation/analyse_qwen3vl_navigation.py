"""Analyse recorded physical state without exposing it to the VLM policy."""
import argparse, collections, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def analyse(root):
 reports=[]; episodes=sorted((root/'episodes').iterdir())
 fig,axes=plt.subplots(6,4,figsize=(18,22));fig2,axes2=plt.subplots(6,4,figsize=(18,22))
 for ax,ax2,folder in zip(axes.flat,axes2.flat,episodes):
  states=[json.loads(x) for x in (folder/'state.jsonl').read_text().splitlines() if x.strip()]
  decisions=[json.loads(x) for x in (folder/'decisions.jsonl').read_text().splitlines() if x.strip()]
  if not states:continue
  result_path=folder/'result.json';result=json.loads(result_path.read_text()) if result_path.exists() else None
  xy=np.array([s['qpos'][:2] for s in states]);t=np.array([s['sim_time_s'] for s in states]);dist=np.array([s['target_distance_xy_m_evaluator'] for s in states]);u=np.array([s['upright'] for s in states])
  speeds=np.linalg.norm(np.array([s['qvel'][:2] for s in states]),axis=1)
  actions=[d['action'] for d in decisions];turns=sum(a in ('TURN_LEFT','TURN_RIGHT') for a in actions)
  switches=sum(a!=b and a in ('TURN_LEFT','TURN_RIGHT') and b in ('TURN_LEFT','TURN_RIGHT') for a,b in zip(actions,actions[1:]))
  step_end={s['step']:s for s in states};no_progress=[]
  for d in decisions:
   k=d['step']
   if d['action']=='FORWARD' and k in step_end and k-1 in step_end:
    delta=np.linalg.norm(np.array(step_end[k]['qpos'][:2])-np.array(step_end[k-1]['qpos'][:2]))
    if delta<.05:no_progress.append(k)
  report={'scene':folder.name,'completed':result is not None,'sample_count':len(states),'steps_observed':len(decisions),'start_distance_m':float(dist[0]),'final_distance_m':float(dist[-1]),'min_distance_m':float(dist.min()),'distance_reduction_m':float(dist[0]-dist[-1]),'mean_base_speed_m_s':float(speeds.mean()),'max_base_speed_m_s':float(speeds.max()),'min_upright':float(u.min()),'contact_count_min':min(s['ncon'] for s in states),'contact_count_max':max(s['ncon'] for s in states),'max_abs_actuator_force':max(max(abs(v) for v in s['actuator_force']) for s in states),'turn_fraction':turns/max(1,len(actions)),'adjacent_opposite_turns':switches,'forward_steps_displacement_below_5cm':no_progress,'termination':result['termination'] if result else 'running'}
  reports.append(report)
  init=json.loads((folder/'initialization.json').read_text());target=init['target_xyz_evaluator'];ax.plot(xy[:,0],xy[:,1],lw=1);ax.scatter(*xy[0],marker='o',label='start');ax.scatter(*xy[-1],marker='x',label='end');ax.scatter(*target[:2],marker='*',c='red',label='target');ax.add_patch(plt.Circle(target[:2],1,fill=False,color='red',alpha=.3));ax.set_title(f"{folder.name}: {report['termination']} ({len(actions)} actions)");ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
  ax2.plot(t,dist,label='target distance (m)');ax2.plot(t,u,label='upright');ax2.axhline(1,color='red',ls='--',lw=.5);ax2.set_title(folder.name);ax2.grid(alpha=.2)
 for i in range(len(episodes),24):axes.flat[i].set_visible(False);axes2.flat[i].set_visible(False)
 axes.flat[0].legend(fontsize=8);axes2.flat[0].legend(fontsize=8)
 fig.tight_layout();fig.savefig(root/'physical_trajectories.png',dpi=130);plt.close(fig)
 fig2.tight_layout();fig2.savefig(root/'distance_upright.png',dpi=130);plt.close(fig2)
 (root/'physical_analysis.json').write_text(json.dumps(reports,indent=2))
 text=['# 机器人实际物理状态分析','','数据来源：state.jsonl 的实际 MuJoCo qpos/qvel/actuator_force/contact；10 Hz 采样，终止状态额外保存。','本报告不表示实体机械狗运行。推理时仿真暂停。','','|场景|状态|步数|起始→最终距离 m|最低 upright|转向比例|左右反复切换|前进位移<5cm次数|','|---|---|---:|---|---:|---:|---:|---:|']
 for r in reports:text.append(f"|{r['scene']}|{r['termination']}|{r['steps_observed']}|{r['start_distance_m']:.2f} → {r['final_distance_m']:.2f}|{r['min_upright']:.3f}|{r['turn_fraction']:.1%}|{r['adjacent_opposite_turns']}|{len(r['forward_steps_displacement_below_5cm'])}|")
 text+=['','判读：高转向比例和相邻反向转向次数描述搜索/振荡；前进动作净位移低于5cm描述原地受阻，不单独证明碰撞原因。upright 为机身局部 z 轴与世界 z 轴的点积，低于0.25终止。','地形和目标坐标仅用于评估及本报告；这些数据没有提供给模型。']
 (root/'PHYSICAL_ANALYSIS.md').write_text('\n'.join(text))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();analyse(a.root.resolve())

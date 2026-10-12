"""Static physical trajectories and visible-depth curves for paired trials."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot(root):
    manifest=json.loads((root/'scene_manifest.json').read_text())
    fig,axes=plt.subplots(6,4,figsize=(18,22))
    fig2,axes2=plt.subplots(6,4,figsize=(18,22))
    colours={'rgb':'#1762a3','rgbd':'#d46b10'}
    for i,entry in enumerate(manifest):
        scene=entry['scene'];name=scene['name'];ax=axes.flat[i];ax2=axes2.flat[i]
        wall=scene['terrain_params']['wall_half_width_m']
        ax.add_patch(Rectangle((0,-wall),.25,2*wall,color='#666',alpha=.6))
        ax.axvspan(-3.5,-.8,color='#bbb',alpha=.15)
        ax.scatter(*scene['target_xy'],color='red',marker='*',s=70,label='target XY (evaluator)')
        statuses=[]
        for mode in ('rgb','rgbd'):
            folder=root/mode/'trials'/name
            path=folder/'state.jsonl'
            if not path.exists():continue
            states=[]
            for line in path.read_text().splitlines():
                try:states.append(json.loads(line))
                except json.JSONDecodeError:continue
            if not states:continue
            gate=json.loads((folder/'approach_gate.json').read_text()) if (folder/'approach_gate.json').exists() else None
            result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else None
            xyz=np.array([s['qpos'][:3] for s in states]);ax.plot(xyz[:,0],xyz[:,1],color=colours[mode],label=mode,lw=1.2)
            ax.scatter(*xyz[0,:2],color=colours[mode],marker='o',s=15)
            ax.scatter(*xyz[-1,:2],color=colours[mode],marker='*' if result and result['success'] else 'x',s=40)
            times=np.array([s['sim_time_s'] for s in states]);distances=np.array([s['evaluation']['median_target_surface_range_m'] if s['evaluation']['median_target_surface_range_m'] is not None else np.nan for s in states])
            ax2.plot(times,distances,color=colours[mode],label=mode,lw=1.1)
            if gate:ax2.axvline(gate['prefix_actions'],color='#aaa',ls=':',lw=.8)
            if result and result['model_stopped']:
                d=result['final_evaluator']['median_target_surface_range_m']
                if d is not None:ax2.scatter(times[-1],d,color=colours[mode],marker='x',s=45)
            statuses.append(f"{mode}: {result['termination'] if result else 'running'}")
        ax.set_title(name+'\n'+' | '.join(statuses),fontsize=8)
        ax.grid(alpha=.2);ax.set_xlim(-5.2,5.5);ax.set_ylim(-3.8,3.8);ax.set_aspect('equal')
        ax2.set_title(name,fontsize=9);ax2.axhline(1,color='red',ls='--',lw=.8);ax2.grid(alpha=.2)
        ax2.set_xlabel('actual simulation time (s)',fontsize=7);ax2.set_ylabel('visible surface range (m)',fontsize=7)
    axes.flat[0].legend(fontsize=7);axes2.flat[0].legend(fontsize=7)
    fig.suptitle('Actual physical trajectories: RGB vs RGB-D (common approach included)',fontsize=14)
    fig2.suptitle('Camera depth surface range: gaps mean target not visible; dashed line = 1 m; x = model STOP',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.97));fig2.tight_layout(rect=(0,0,1,.97))
    fig.savefig(root/'paired_physical_trajectories.png',dpi=120);fig2.savefig(root/'paired_visible_depth.png',dpi=120)
    plt.close(fig);plt.close(fig2)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();plot(a.root.resolve())

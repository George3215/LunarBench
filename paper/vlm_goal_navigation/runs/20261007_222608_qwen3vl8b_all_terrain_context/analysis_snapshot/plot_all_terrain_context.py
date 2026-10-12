"""Physical paths and visible depth curves; six paired conditions per scene."""
import json,os,sys,time
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/task3_matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot(root):
    setups=json.loads((root/'scene_setup.json').read_text());dest=root/'physical_plots';dest.mkdir(exist_ok=True)
    cache=json.loads((dest/'cache.json').read_text()) if (dest/'cache.json').exists() else {};changed=0
    colours=['#1967b3','#db7313','#33a56b','#c14379','#8760c0','#735136']
    for setup in setups:
        name=setup['scene_name'];folders=sorted(root.glob(f'*/*/trials/{name}'));paths=[p/'state.jsonl' for p in folders if (p/'state.jsonl').exists()]
        signature=[(str(p),p.stat().st_size) for p in paths]
        if not paths or cache.get(name)==[list(v) for v in signature]:continue
        fig,axes=plt.subplots(1,3,figsize=(16,4.5));ax,depth,pose=axes
        ax.scatter(*setup['adjusted_scene']['target_xy'],c='red',marker='*',s=90,label='Target (evaluation only)')
        wall=setup['added_occluder']
        if wall:ax.add_patch(Rectangle((wall['x'],-wall['half_y']),wall['width'],2*wall['half_y'],color='gray',alpha=.7))
        spec=setup['terrain_spec']
        if 'wall_x_m' in spec:ax.axvline(spec['wall_x_m'],color='gray',ls=':',label='Original window wall')
        for i,folder in enumerate(folders):
            path=folder/'state.jsonl'
            if not path.exists():continue
            states=[]
            with path.open() as stream:
                for line in stream:
                    try:states.append(json.loads(line))
                    except json.JSONDecodeError:pass
            if not states:continue
            label='/'.join(folder.relative_to(root).parts[:2]);colour=colours[i%6]
            xy=np.array([s['qpos'][:2] for s in states]);times=np.array([s['sim_time_s'] for s in states])
            ranges=np.array([s['evaluation']['median_target_surface_range_m'] if s['evaluation']['median_target_surface_range_m'] is not None else np.nan for s in states])
            ax.plot(xy[:,0],xy[:,1],color=colour,label=label,lw=1.3);ax.scatter(*xy[0],color=colour,s=20);ax.scatter(*xy[-1],color=colour,marker='x')
            depth.plot(times,ranges,color=colour,label=label,lw=1.2);pose.plot(times,[s['evaluation']['upright'] for s in states],color=colour,lw=1.2)
        ax.set_title(name+' actual base path');ax.set_xlabel('World x (m)');ax.set_ylabel('World y (m)');ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=6)
        depth.set_title('Visible target surface range; gaps = invisible');depth.set_xlabel('Simulation time (s)');depth.set_ylabel('Camera depth 3D range (m)');depth.axhline(1,color='red',ls='--',lw=.8)
        pose.set_title('Physical upright');pose.set_xlabel('Simulation time (s)');pose.set_ylabel('Base up projection');pose.axhline(.25,color='red',ls='--',lw=.8)
        for a in axes:a.grid(alpha=.2)
        fig.tight_layout();fig.savefig(dest/(name+'.png'),dpi=120);plt.close(fig);cache[name]=signature;changed+=1
    (dest/'cache.json').write_text(json.dumps(cache));print('Updated physical plots',changed,flush=True)
    import html
    cards=['<!doctype html><meta charset="utf-8"><title>Task3 物理记录</title><style>body{font:16px sans-serif;margin:24px}img{max-width:100%}</style><h1>六条件真实模拟器路径、深度与姿态</h1><p>圆点是初始位置，叉号是当前或终止位置；红星是仅供评估的目标世界位置。距离曲线缺口表示目标不可见，红虚线为 1 m。以下不是实体机械狗数据。</p>']
    for p in sorted(dest.glob('*.png')):
        cards.append('<h2>'+html.escape(p.stem)+'</h2><img loading="lazy" src="physical_plots/'+html.escape(p.name)+'">')
    (root/'physical_state_browser.html').write_text(''.join(cards))
if __name__=='__main__':
    root=Path(sys.argv[1]).resolve();watch='--watch' in sys.argv
    while True:
        if (root/'scene_setup.json').exists():plot(root)
        status=json.loads((root/'RUN_STATUS.json').read_text()) if (root/'RUN_STATUS.json').exists() else {}
        if not watch or status.get('status') in ('complete','audit_failed','interrupted') or (root/'ERROR.json').exists():break
        time.sleep(45)

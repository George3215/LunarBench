"""Summarize only completed VLM rollouts; no outcome imputation."""
from pathlib import Path
import json, argparse, collections, numpy as np

def summarize(root):
 rows=[]
 for f in root.glob('episodes/*/*/*/result.json'):
  d=json.loads(f.read_text(encoding='utf8'));d['result_file']=str(f);rows.append(d)
 groups=collections.defaultdict(list)
 for r in rows:groups[(r['model'],r['mode'])].append(r)
 summary=[]
 for (model,mode),rr in sorted(groups.items()):
  choices=collections.Counter(s['action'] for r in rr for s in r['steps'])
  latency=[s['model_s'] for r in rr for s in r['steps']]
  summary.append({'model':model,'mode':mode,'completed_episodes':len(rr),'reacquired_count':sum(r['reacquired'] for r in rr),
   'reacquired_rate':float(np.mean([r['reacquired'] for r in rr])),
   'arrival_count':sum(r['arrival_stop_near_marker'] for r in rr),'mean_decision_s':float(np.mean(latency)) if latency else None,
   'terminations':dict(collections.Counter(r['termination'] for r in rr)),'actions':dict(choices),
   'base_scenes':{'n':sum(not r['scene'].startswith('hist') for r in rr),'reacquired':sum(r['reacquired'] for r in rr if not r['scene'].startswith('hist'))},
   'window_worlds':{'n':sum(r['scene'].startswith('hist') for r in rr),'reacquired':sum(r['reacquired'] for r in rr if r['scene'].startswith('hist'))}})
 out={'completed_episodes':len(rows),'summary':summary,'episode_files':[r['result_file'] for r in rows]}
 (root/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf8')
 lines=['# Local VLM benchmark results','','Only completed episodes are included. Read protocol and limitations before comparison.','',
 '| Model | Input condition | Episodes | Reacquired | Stop near marker | Mean decision s |','|---|---|---:|---:|---:|---:|']
 for r in summary:
  lines.append(f"| {r['model']} | {r['mode']} | {r['completed_episodes']} | {r['reacquired_count']} ({r['reacquired_rate']:.3f}) | {r['arrival_count']} | {r['mean_decision_s']:.2f} |")
 lines+=['','Reacquisition uses actual camera frames during executed motion. Arrival is supplemental (STOP + visible target + distance/depth thresholds), not validated full navigation success. These are simulation runs, one model seed, a designed scene suite, and quantized local models; they do not establish global SOTA or universal task difficulty.']
 (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 print(json.dumps(out,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();summarize(a.directory)

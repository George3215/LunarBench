"""Detailed paired diagnostics for completed runs, including input accounting."""
from pathlib import Path
import argparse,json,collections,numpy as np
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory
rows=[]
for f in root.glob('episodes/*/*/*/result.json'):
 r=json.loads(f.read_text());path=f.parent;timeline=[x for step in r['steps'] for x in step.get('curve',[])]
 # Exploratory spatial proximity: separate from detector-defined reobservation and STOP.
 near=any(x['range_xy_evaluator']<=1.0 and x['upright']>=.25 and x['in_bounds'] for x in timeline)
 decisions=[]
 for dp in sorted(path.glob('decision_*.json')):
  d=json.loads(dp.read_text());q=d.get('parsed');resp=d.get('response',{});thought=d.get('deliberation_response') or {}
  decisions.append({'image_count':len(d['image_files']),'source_frames':len(d.get('source_image_files',d['image_files'])),
   'service_s':(resp.get('total_duration',0)+thought.get('total_duration',0))/1e9,
   'eval_tokens':resp.get('eval_count',0)+thought.get('eval_count',0),
   'claims_target_visible':None if q is None else q.get('target_visible'),'action':None if q is None else q.get('action')})
 rows.append({'scene':r['scene'],'mode':r['mode'],'model':r['model'],'reacquired':r['reacquired'],
  'within_1m_ever_exploratory':near,'stop_near':r['arrival_stop_near_marker'],'termination':r['termination'],
  'min_range_xy_m':min((x['range_xy_evaluator'] for x in timeline),default=None),'path_m':r['path_m'],
  'rsr_by_step':{str(n):any(x.get('reacquired_so_far',False) for x in r['steps'][:n]) for n in [1,3,5,10]},
  'decisions':decisions})
summary={}
for key in sorted({(r['model'],r['mode']) for r in rows}):
 rr=[r for r in rows if (r['model'],r['mode'])==key]
 dd=[d for r in rr for d in r['decisions']]
 summary['/'.join(key)]={'n':len(rr),'reacquired':sum(r['reacquired'] for r in rr),'within_1m_exploratory':sum(r['within_1m_ever_exploratory'] for r in rr),'stop_near':sum(r['stop_near'] for r in rr),
  'model_service_mean_s':float(np.mean([d['service_s'] for d in dd])) if dd else None,
  'first_step_false_visible_claims':sum(bool(r['decisions'] and r['decisions'][0]['claims_target_visible']) for r in rr),
  'terminations':dict(collections.Counter(r['termination'] for r in rr))}
(root/'detailed_analysis.json').write_text(json.dumps({'summary':summary,'episodes':rows,
 'limits':'Proximity <=1m is exploratory planar-distance evidence, not a validated same-surface/obstacle-free arrival criterion. Source frame count and composed image count differ in labeled-sheet conditions.'},indent=2),encoding='utf8')
print(json.dumps(summary,indent=2))

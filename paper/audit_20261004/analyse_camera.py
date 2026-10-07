"""Camera-only one-step policy comparison on identical executed branch tables.
Features exclude ground-truth pose, target coordinates, geometry labels and outcomes.
"""
import json,sys,math
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from threadpoolctl import threadpool_limits
from reanalyse_legacy import cluster_ci
OUT=Path(__file__).parent
ACTIONS=['forward','back','left','right','turn_left','turn_right']
SEEDS=[7,19,37]

def analyse():
 d=json.loads((OUT/'camera_rollouts.json').read_text());assert len(d['records'])==len(d['protocol']['scenes'])==24,'Full acquisition must finish before analysis'
 records=d['records'];valid=[r for r in records if r['eligible']];groups=np.array([r['group'] for r in valid]);N=len(valid)
 Y=np.array([[b['success'] for b in r['branches']] for r in valid],int)
 image=np.array([r['loss_sensor']['depth_grid']+r['loss_sensor']['rgb_mean'] for r in valid])
 memory=np.array([r['loss_memory']['centroid']+[r['loss_memory']['depth'],r['loss_time']-r['loss_memory']['time']] for r in valid])
 decisions={a:Y[:,i].astype(float) for i,a in enumerate(ACTIONS)}
 decisions['uniform_random_exact']=Y.mean(axis=1);decisions['oracle_branch_bound']=Y.max(axis=1)
 # Image coordinate x grows rightwards. This heuristic intentionally uses no world coordinates.
 picks=np.array([2 if m[0]<.5 else 3 for m in memory]);decisions['last_image_bearing']=Y[np.arange(N),picks].astype(float)
 aucs={};predictions={};folds=[]
 def expand(features):
  return np.hstack([np.repeat(features,6,axis=0),np.tile(np.eye(6),(len(features),1))])
 with threadpool_limits(limits=1):
  for condition in ['RGBD+a','memory+a','RGBD+memory+a','RGBD+shuffled_memory+a']:
   allp=[];alls=[]
   for seed in SEEDS:
    P=np.zeros((N,6));rng=np.random.default_rng(seed)
    for g in np.unique(groups):
     te=groups==g;tr=~te
     if condition=='RGBD+a':ftr,fte=image[tr],image[te]
     elif condition=='memory+a':ftr,fte=memory[tr],memory[te]
     else:
      mt=memory[tr].copy()
      if condition=='RGBD+shuffled_memory+a':mt=mt[rng.permutation(len(mt))]
      ftr=np.hstack([image[tr],mt]);fte=np.hstack([image[te],memory[te]])
     model=RandomForestClassifier(n_estimators=200,max_depth=4,min_samples_leaf=2,random_state=seed,n_jobs=1)
     model.fit(expand(ftr),Y[tr].ravel())
     if len(model.classes_)==1:P[te]=float(model.classes_[0])
     else:P[te]=model.predict_proba(expand(fte))[:,list(model.classes_).index(1)].reshape(-1,6)
     if seed==7 and condition=='RGBD+a':folds.append({'group':str(g),'test_scenes':[valid[i]['scene'] for i in np.where(te)[0]],'train_worlds':int(tr.sum())})
    # fixed action order for exact ties, identical across every method
    success=Y[np.arange(N),P.argmax(axis=1)]
    alls.append(success);allp.append(P)
   decisions[condition]=np.array(alls).mean(axis=0)
   aucs[condition]={'auc_by_seed':[roc_auc_score(Y.ravel(),p.ravel()) for p in allp],
     'ap_by_seed':[average_precision_score(Y.ravel(),p.ravel()) for p in allp],
     'brier_by_seed':[brier_score_loss(Y.ravel(),p.ravel()) for p in allp],
     'rsr_by_seed':[float(s.mean()) for s in alls],
     'selected_actions_by_seed':[[ACTIONS[i] for i in p.argmax(axis=1)] for p in allp]}
   predictions[condition]=np.array(allp)
 summary={k:{'rsr1':float(v.mean()),'group_bootstrap_ci95':cluster_ci(v,groups),'per_world':v.tolist()} for k,v in decisions.items()}
 pairs={k:{'difference':float((v-decisions['uniform_random_exact']).mean()),'ci95':cluster_ci(v-decisions['uniform_random_exact'],groups)} for k,v in decisions.items() if k!='uniform_random_exact'}
 samples=[s for r in valid for b in r['branches'] for s in b['curve']]
 confusion={f'G{g}C{c}':sum(bool(s['visible'])==bool(g) and bool(s['sensor']['visible'])==bool(c) for s in samples) for g in [0,1] for c in [0,1]}
 lost_g=sum(min(r['approach_curve'],key=lambda s:abs(s['t_s']-r['loss_time']))['visible'] for r in valid)
 threshold={str(t):{'random_rsr':float(np.mean([[b['threshold_success'][str(t)] for b in r['branches']] for r in valid])), 'branch_oracle_rsr':float(np.mean([any(b['threshold_success'][str(t)] for b in r['branches']) for r in valid]))} for t in [1,3,10]}
 result={'protocol':'3-s single-step, camera success >=3 consecutive detections; all methods same six branches; LOGO by paired construction',
  'candidate_worlds':24,'eligible_worlds':N,'eligible_groups':len(set(groups)),'excluded':[r['scene'] for r in records if not r['eligible']],
  'eligible_scenes':[r['scene'] for r in valid],'groups':groups.tolist(),'summary':summary,'paired_vs_uniform_random':pairs,'predictors':aucs,'folds':folds,
  'diagnostics':{'sample_confusion':confusion,'sample_agreement':float(np.mean([s['visible']==s['sensor']['visible'] for s in samples])), 'loss_states_still_geometry_visible':int(lost_g), 'nominal_states_with_camera_target':sum(r['nominal_start']['visible'] for r in records)},
  'threshold_sensitivity_fixed_loss_states':threshold,
  'three_action_subset':{'actions':ACTIONS[:1]+ACTIONS[2:4],'random_rsr':float(Y[:,[0,2,3]].mean()),'oracle_rsr':float(Y[:,[0,2,3]].max(axis=1).mean())},
  'paired_key_comparisons':{k:{'difference':float(np.mean(decisions[a]-decisions[b])),'ci95':cluster_ci(decisions[a]-decisions[b],groups)} for k,a,b in [('RGBD_vs_forward','RGBD+a','forward'),('memory_added','RGBD+memory+a','RGBD+a'),('memory_vs_shuffled','RGBD+memory+a','RGBD+shuffled_memory+a')]},
  'learning':{'algorithm':'RandomForestClassifier','n_estimators':200,'max_depth':4,'min_samples_leaf':2,'seeds':SEEDS,'tuning':'none; fixed before full collection completed','shuffled_control':'training memory permuted by world within training fold; test memory unchanged'},
  'limits':['designed scenes, not random population sample','one frozen loss per world','single-step, not RSR@10','colored target marker, injected simulation camera','threshold sensitivity conditional on primary loss states','prediction-trained policies selected offline from exhaustive executed counterfactual branches']}
 (OUT/'camera_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf8')
 np.savez_compressed(OUT/'camera_oof_predictions.npz',Y=Y,groups=groups,**predictions)
 print(json.dumps({k:v for k,v in result.items() if k in ['eligible_worlds','eligible_groups','excluded','summary','diagnostics','paired_vs_uniform_random']},indent=2))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=OUT);args=p.parse_args();OUT=args.directory;analyse()

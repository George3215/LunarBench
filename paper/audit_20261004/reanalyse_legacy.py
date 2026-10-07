"""Recompute evidence at its actual sampling unit; preserve legacy artefacts."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import sys,json,hashlib,collections
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tasks/task3_reobservability'))
from consequence_model import feature_matrix,build_training_matrix
BASE=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent;GEN=BASE/'tasks/task3_reobservability/generated'

def cluster_ci(values,groups,seed=20261004,n=10000):
    groups=np.asarray(groups);v=np.asarray(values);names=np.unique(groups);rng=np.random.default_rng(seed)
    sums=np.array([v[groups==g].sum() for g in names]);counts=np.array([(groups==g).sum() for g in names]);draw=rng.integers(len(names),size=(n,len(names)))
    boot=sums[draw].sum(axis=1)/counts[draw].sum(axis=1)
    return [float(x) for x in np.quantile(boot,[.025,.975])]

def main():
    d=json.loads((GEN/'consequence_dataset.json').read_text());rows=d['rows'];ids=d['action_ids']
    legacy=json.loads((GEN/'consequence_model.json').read_text());rnd=json.loads((GEN/'random_baseline.json').read_text())
    common=rnd['terrains'];pv=legacy['planner']['per_terrain']
    groups=np.array([r['terrain'].rsplit('_',1)[0] if r['terrain'].startswith('hist') else r['terrain'] for r in rows]);terrains=np.array([r['terrain'] for r in rows]);y=np.array([r['reobserved'] for r in rows],dtype=int)
    state_keys=[(r['terrain'],r['state_id']) for r in rows];unique_states=list(dict.fromkeys(state_keys));idxs=[np.array([i for i,k in enumerate(state_keys) if k==key]) for key in unique_states]
    report={'dataset':{'rows':len(rows),'states':len(unique_states),'scenes':len(set(terrains)),'independent_construction_groups':len(set(groups))},
        'historical_comparison':{'random_n_scenes':len(common),'learned_n_scenes':len(pv),'learned_success_on_common_scenes':sum(pv[n]['termination']=='success' for n in common),'learned_rsr_common':np.mean([pv[n]['termination']=='success' for n in common]),'random_rsr_common':rnd['mean_over_seeds']['rsr@10'],'note':'Matched-subset descriptive comparison of historical artifacts, not a fresh rerun; paired seed data absent'},'experiments':{}}
    frame=np.load(GEN/'vision_frames.npz');rgb=frame['rgb'].astype(np.int16);mask=(rgb[:,:,:,0]>120)&(rgb[:,:,:,0]>1.5*rgb[:,:,:,1])&(rgb[:,:,:,0]>1.5*rgb[:,:,:,2]);pixels=mask.sum(axis=(1,2));report['legacy_image_screen']={'frames':len(pixels),'frames_with_3_red_pixels':int((pixels>=3).sum()),'note':'RGB threshold on archived 160x120 frames; not segmentation or eligibility proof'}
    matrices={'O+M':feature_matrix(rows,'O+M',8),'O+M+a':build_training_matrix(rows,'O+M+a',ids,history=8),'a':np.array([[float(r['action']==a) for a in ids] for r in rows])}
    allpred={}
    with threadpool_limits(limits=1):
      for split,gr in [('scene',terrains),('construction_pair',groups)]:
       for condition,X in matrices.items():
        pred=np.zeros(len(rows));folds=[]
        for g in np.unique(gr):
          te=gr==g;tr=~te
          clf=HistGradientBoostingClassifier(max_depth=3,max_iter=250,learning_rate=.08,random_state=7)
          clf.fit(X[tr],y[tr]);pred[te]=clf.predict_proba(X[te])[:,1]
          folds.append({'group':str(g),'train_rows':int(tr.sum()),'test_rows':int(te.sum())})
        success=[];sg=[];choice=[]
        for key,ii in zip(unique_states,idxs):
          # Fixed action-ID order breaks exact ties. No outcome-based tie breaking.
          best=ii[np.argmax(pred[ii])];success.append(int(y[best]));sg.append(groups[best]);choice.append(rows[best]['action'])
        key=split+'/'+condition;allpred[key]=pred
        result={'roc_auc':roc_auc_score(y,pred),'average_precision':average_precision_score(y,pred),'brier':brier_score_loss(y,pred),'offline_action_success':float(np.mean(success)),'cluster_ci':cluster_ci(success,sg),'n_states':len(success),'n_groups':len(set(sg)),'folds':folds,'action_counts':dict(collections.Counter(choice))}
        report['experiments'][key]=result;print(key,{k:v for k,v in result.items() if k not in ('folds','action_counts')},flush=True)
    report['offline_random_exact_success']=float(np.mean([y[ii].mean() for ii in idxs]))
    report['offline_oracle_success']=float(np.mean([y[ii].max() for ii in idxs]))
    report['pair_twin_leakage_folds']=int(sum(t.startswith('hist') for t in set(terrains)))
    np.savez_compressed(OUT/'legacy_oof_predictions.npz',y=y,groups=groups,**allpred)
    report['input_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [GEN/'consequence_dataset.json',GEN/'consequence_model.json',GEN/'random_baseline.json',GEN/'vision_frames.npz']}
    (OUT/'legacy_reanalysis.json').write_text(json.dumps(report,indent=2),encoding='utf8');print('wrote legacy_reanalysis.json',flush=True)
if __name__=='__main__':main()

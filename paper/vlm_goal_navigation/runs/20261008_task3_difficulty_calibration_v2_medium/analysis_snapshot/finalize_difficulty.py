"""Final metrics and empirical difficulty assignment, strictly after a full suite."""
import json,math,statistics,collections,sys
from pathlib import Path

def read(p):return json.loads(p.read_text())
def rows(p):
    with p.open() as f:
        for line in f:
            yield json.loads(line)
def mean(a):return statistics.mean(a) if a else None
def median(a):return statistics.median(a) if a else None

def finalize(root):
    protocol=read(root/'protocol.json');analysis=read(root/'analysis.json');audit=read(root/'audit.json');setups=read(root/'scene_setup.json');expected=protocol['trial_count']
    if not analysis['all_complete'] or not audit['pass'] or analysis['completed']!=expected:
        raise RuntimeError('Full suite and passing audit required; partial rates cannot calibrate terrain')
    results=[];classifications=[];reference={}
    for setup in setups:
        folder=Path(setup['evidence_path']);previous=None;refpath=0.;refstart=setup['approach_end_sim_time_s'];recovered=None
        for s in rows(folder/'state.jsonl'):
            if s['sim_time_s']<refstart-.000001:continue
            xy=s['qpos'][:2]
            if previous:refpath+=math.dist(xy,previous)
            previous=xy
            if s['step']>100 and s['evaluation']['target_visible_rgb_pixels']>=20 and recovered is None:recovered=s['step']-100
        reference[setup['scene_name']]={'sampled_reference_path_m':refpath,'reference_actions':len(setup['reference_actions']),'reference_recovery_actions':recovered}
    bykey={f"{s['mode']}/{s['horizon']}/{s['scene']}":s for s in analysis['trials']}
    for folder in sorted(root.glob('*/*/trials/*')):
        result=read(folder/'result.json');gate=read(folder/'approach_gate.json');key='/'.join(folder.relative_to(root).parts[:2])+'/'+folder.name;summary=bykey[key]
        previous=None;path=0.;first_recover=None;start_time=gate['lost']['sim_time_s'];end_time=result['final_evaluator']['sim_time_s'];obstacle_contacts=0
        metadata=read(folder/'model_metadata.json')
        obstacle_ids={int(k) for k,v in metadata['geom_names'].items() if v=='navigation_added_occluder' or v.endswith(('_wall','_wall_low','_wall_high','_sill'))}
        for state in rows(folder/'state.jsonl'):
            if state['sim_time_s']<start_time-.000001:continue
            xy=state['qpos'][:2]
            if state['step']>gate['prefix_actions']:
                obstacle_contacts+=any(c['geom1'] in obstacle_ids or c['geom2'] in obstacle_ids for c in state['contacts'])
            if previous:path+=math.dist(previous,xy)
            previous=xy
            if state['step']>gate['prefix_actions'] and state['evaluation']['target_visible_rgb_pixels']>=20 and first_recover is None:first_recover=state['sim_time_s']-start_time
        ref=reference[result['scene']]['sampled_reference_path_m'];spl=(ref/max(ref,path,1e-9)) if result['success'] else 0.
        results.append({**result,'key':key,'sampled_autonomous_path_m':path,'sampled_reference_path_m':ref,'reference_path_efficiency':spl,
            'first_recovery_sim_s':first_recover,'autonomous_sim_s':end_time-start_time,
            'wall_contact_samples':obstacle_contacts,'sampled_actuator_force_peak':summary['sampled_actuator_force_peak'],
            'stalled_forward_actions':summary['stalled_forward_actions'],'false_visible_claims':summary['false_visible_claims'],
            'unsupported_evidence_claims':summary['unsupported_evidence_claims'],'max_context_tokens':summary['max_context_tokens'],'max_input_frames':summary['max_input_frames']})
        summary['wall_contact_samples']=obstacle_contacts
        summary['obstacle_contact_scope']='autonomous phase, 10Hz sampled contacts with explicit obstacles; normal support floor excluded'
    if len(results)!=expected:raise RuntimeError('Result count mismatch')
    def metrics(items):
        n=len(items);success=[r for r in items if r['success']];nonresource=[r for r in items if not r['resource_limit']]
        return {'trial_count':n,'success_count':len(success),'success_rate':len(success)/n if n else None,
            'resource_limit_count':n-len(nonresource),'success_rate_excluding_resource_limits':sum(r['success'] for r in nonresource)/len(nonresource) if nonresource else None,
            'success_by_action_budget':{str(b):sum(r['success'] and r['action_steps']<=b for r in items)/n for b in (1000,2000,3000)} if n else {},
            'recovery_rate':sum(r['first_recovery_sim_s'] is not None for r in items)/n if n else None,
            'median_recovery_sim_s':median([r['first_recovery_sim_s'] for r in items if r['first_recovery_sim_s'] is not None]),
            'mean_reference_path_efficiency':mean([r['reference_path_efficiency'] for r in items]),
            'median_success_actions':median([r['autonomous_actions'] for r in success]),'median_success_sim_s':median([r['autonomous_sim_s'] for r in success]),
            'median_success_path_m':median([r['sampled_autonomous_path_m'] for r in success]),
            'collision_trial_rate':sum(r['wall_contact_samples']>0 for r in items)/n if n else None,
            'fallen_trial_rate':sum(r['termination']=='fallen' for r in items)/n if n else None,
            'out_of_bounds_rate':sum(r['termination']=='out_of_bounds' for r in items)/n if n else None,
            'premature_stop_rate':sum(r['termination']=='premature_stop' for r in items)/n if n else None,
            'budget_exhausted_rate':sum(r['termination']=='budget_exhausted' for r in items)/n if n else None,
            'stalled_forward_actions':sum(r['stalled_forward_actions'] for r in items),
            'false_visible_claims':sum(r['false_visible_claims'] for r in items),'unsupported_evidence_claims':sum(r['unsupported_evidence_claims'] for r in items),
            'median_inference_latency_s':median([r['total_inference_s']/r['inference_count'] for r in items if r['inference_count']]),
            'max_context_tokens':max((r['max_context_tokens'] for r in items),default=0),'max_input_frames':max((r['max_input_frames'] for r in items),default=0),
            'terminations':dict(collections.Counter(r['termination'] for r in items))}
    conditions=[{'mode':m,'horizon':h,**metrics([r for r in results if r['mode']==m and r['horizon']==h])} for m in ('rgb','rgbd') for h in ('current','first_current','all_history')]
    indexed={(r['scene'],r['mode'],r['horizon']):r for r in results};comparisons=[]
    specifications=[(f'{m}: {b} vs {a}',(m,a),(m,b)) for m in ('rgb','rgbd') for a,b in (('current','first_current'),('current','all_history'),('first_current','all_history'))]
    specifications += [(f'{h}: rgbd vs rgb',('rgb',h),('rgbd',h)) for h in ('current','first_current','all_history')]
    for label,(ma,ha),(mb,hb) in specifications:
        pairs=[(indexed[(s['scene_name'],ma,ha)],indexed[(s['scene_name'],mb,hb)]) for s in setups]
        eligible=[(a,b) for a,b in pairs if not a['resource_limit'] and not b['resource_limit']]
        comparisons.append({'comparison':label,'all_paired_scenes':len(pairs),'non_resource_pairs':len(eligible),
            'second_only_success':sum(not a['success'] and b['success'] for a,b in eligible),
            'first_only_success':sum(a['success'] and not b['success'] for a,b in eligible),
            'paired_success_gain_percentage_points':100*mean([int(b['success'])-int(a['success']) for a,b in eligible]) if eligible else None,
            'statistical_claim':'descriptive paired comparison; no causal internal-reasoning claim or independent-sample confidence interval'})
    for setup in setups:
        scene=setup['scene_name'];scene_results=[r for r in results if r['scene']==scene];ok=sum(r['success'] for r in scene_results);resource=sum(r['resource_limit'] for r in scene_results)
        first=setup['initial'];range_m=setup['initial_depth']['red_region_median_surface_range_m'];pixels=first['target_visible_rgb_pixels'];ref=reference[scene]
        # Transparent geometric strata; empirical labels do not diagnose VLM internals.
        perception_score=int(range_m>3)+int(range_m>5)+int(pixels<200)+int(pixels<80)
        route_score=int(ref['reference_actions']>12)+int(ref['reference_actions']>24)+int((ref['reference_recovery_actions'] or 0)>8)
        score=perception_score+route_score
        geometry='easy' if score<=1 else 'medium' if score<=3 else 'hard'
        observed='resource_unresolved' if resource else 'easy_for_this_baseline' if ok>=4 else 'medium_for_this_baseline' if ok else 'hard_for_this_baseline'
        geom=setup['loss'].get('blocking_geom','') or ''
        mechanism='added_thin_wall' if setup['added_occluder'] and setup['added_occluder'].get('width',1)<.3 else 'added_box' if setup['added_occluder'] else 'window_structure' if setup['source_scene']['terrain_profile']=='window_wall' else 'original_nonwindow_wall' if 'wall' in geom else 'native_relief'
        classifications.append({'scene':scene,'terrain_family':'window_wall' if setup['source_scene']['terrain_profile']=='window_wall' else setup['source_scene']['terrain_profile'],
            'geometry_difficulty':geometry,'geometry_score':score,'perception_score':perception_score,'route_score':route_score,
            'empirical_difficulty':observed,'successes_in_six_conditions':ok,'resource_limits':resource,
            'initial_camera_depth_range_m':range_m,'initial_visible_pixels':pixels,**ref,
            'loss_mechanism':mechanism,
            'terminations':dict(collections.Counter(r['termination'] for r in scene_results))})
    family_lookup={c['scene']:c['terrain_family'] for c in classifications};family_groups=collections.defaultdict(list)
    for r in results:family_groups[family_lookup[r['scene']]].append(r)
    families=[{'family':family,**metrics(items)} for family,items in sorted(family_groups.items())]
    overall=metrics(results);family_balanced_rate=mean([g['success_rate_excluding_resource_limits'] for g in families if g['success_rate_excluding_resource_limits'] is not None])
    # The thresholds use pooled six-condition SR on completed non-resource trials.
    calibration=overall['success_rate_excluding_resource_limits'];resource_fraction=overall['resource_limit_count']/expected
    direction='lower' if calibration is not None and calibration<.05 else 'higher' if calibration is not None and calibration>.4 else 'keep'
    decision={'complete_suite_required':True,'rate_definition':'successful trials / completed non-resource trials, all six conditions pooled; family-balanced rate reported separately',
        'calibration_rate':calibration,'family_balanced_rate':family_balanced_rate,'threshold_low_strict':.05,'threshold_high_strict':.4,
        'resource_fraction':resource_fraction,'resource_review_required':resource_fraction>.1,'direction':direction,'basis':'full suite, not interim observations',
        'scene_design_revision_required':True,'next_design':'native low relief preferred, seeded cubic blocks fallback, walls only in window structures',
        'maximum_adaptive_revisions':2,'no_unbounded_retries':True,'max_actions_each_trial':3000}
    value={'all_complete':True,'overall':overall,'six_conditions':conditions,'terrain_families':families,
        'loss_mechanism_scene_counts':dict(collections.Counter(c['loss_mechanism'] for c in classifications)),
        'paired_context_and_depth_comparisons':comparisons,
        'family_balanced_success_rate':mean([g['success_rate'] for g in families]),'calibration_decision':decision,
        'path_efficiency_definition':'success * reference 10Hz sampled physical route / max(reference route, autonomous 10Hz sampled physical route); reference is feasible route, not shortest path, so this is not shortest-path SPL',
        'collision_definition':'10Hz sampled contact with added cube/wall or original barrier/window wall/sill; ordinary floor support excluded; contact is not impact severity',
        'comparison_scope':'descriptive, one deterministic trial per scenario and condition; revisions calibrate task design rather than form an independent held-out benchmark'}
    (root/'evaluation_metrics.json').write_text(json.dumps(value,ensure_ascii=False,indent=2));(root/'difficulty_classification.json').write_text(json.dumps({'completed_before_revision':True,'scenes':classifications,'decision':decision},ensure_ascii=False,indent=2))
    temporary=root/'analysis.json.final.tmp';temporary.write_text(json.dumps(analysis,ensure_ascii=False,indent=2));temporary.replace(root/'analysis.json')
    lines=['# 完整 Task3 实验评价','',f"完成 {expected}/{expected}；通过记录审计。成功 {overall['success_count']}/{expected}，整体任务成功率 {100*overall['success_rate']:.2f}%。",'', '|传感器|上下文|成功/总数|成功率|恢复视野率|资源终止|','|---|---|---:|---:|---:|---:|']
    lines += [f"|{c['mode']}|{c['horizon']}|{c['success_count']}/{c['trial_count']}|{100*c['success_rate']:.2f}%|{100*c['recovery_rate']:.2f}%|{c['resource_limit_count']}|" for c in conditions]
    lines += ['',f"按15类地形等权的成功率：{100*value['family_balanced_success_rate']:.2f}%。窗口变体不视为独立地形。",'', '完整指标见 evaluation_metrics.json，逐场景难度与阈值决策见 difficulty_classification.json。', '',f"难度调整决策：{direction}；用于阈值比较的完成且未资源终止 trial 成功率：{100*calibration:.2f}%。" if calibration is not None else '资源条件未解决，不能仅据资源终止改变地形。', '', '几何难度与本基线经验难度分别记录；经验失败不等同于已证明的地形复杂性或模型内部原因。', '', '参考路径效率使用可行参考路径，不是最短路径SPL；模型策略说明是自述。MuJoCo物理状态不是实体机械狗记录。']
    (root/'FINAL_REPORT.md').write_text('\n'.join(lines))
    from analyse_task_quality import build
    build(root,value)
    return value
if __name__=='__main__':
    value=finalize(Path(sys.argv[1]).resolve());print(json.dumps(value['calibration_decision'],ensure_ascii=False))

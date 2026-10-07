"""Summarize exact original six-action selections and verify archived scoring."""
from pathlib import Path
import collections, hashlib, json, statistics

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'paper/vlm_goal_navigation/runs/20261004_qwen35'
OUT = RUN / 'original_six_action'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
raw_path = ROOT / 'paper/audit_20261004/strict_camera/camera_rollouts.json'
records = {r['scene']: r for r in json.loads(raw_path.read_text())['records'] if r['eligible']}
rows = []
checks = collections.Counter()
for f in sorted(OUT.glob('*/*/*/result.json')):
    r = json.loads(f.read_text()); d = json.loads((f.parent/'decision.json').read_text())
    branch = next((b for b in records[r['scene']]['branches'] if b['action']==r['action']), None)
    assert bool(branch and branch['success']) == r['success'], str(f)
    if branch:
        assert hashlib.sha256(json.dumps(branch,sort_keys=True).encode()).hexdigest() == r['source_branch_sha256']
    for p, expected in zip(d['image_files'], d['image_sha256'], strict=True):
        assert sha(Path(p)) == expected, p
        checks['input_image_hashes_verified'] += 1
    protocol = json.loads((OUT/('protocol_'+r['model'].replace(':','_')+'.json')).read_text())
    assert protocol['branch_table_sha256'] == sha(raw_path)
    assert r['action'] == (d['parsed'] or {}).get('action')
    checks['results_and_selected_branches_verified'] += 1
    r['source'] = str(f.relative_to(ROOT)); rows.append(r)
summary = []
for model,mode in sorted({(r['model'],r['mode']) for r in rows}):
    rr = [r for r in rows if (r['model'],r['mode'])==(model,mode)]
    rec = {'model':model,'mode':mode,'n':len(rr),'success':sum(r['success'] for r in rr),
        'invalid':sum(not r['valid_output'] for r in rr),'actions':dict(collections.Counter(r['action'] for r in rr)),
        'median_decision_s':statistics.median(r['model_s'] for r in rr),'mean_decision_s':statistics.mean(r['model_s'] for r in rr)}
    for label,window in [('base',False),('window',True)]:
        subset = [r for r in rr if r['scene'].startswith('hist')==window]
        rec[label] = {'n':len(subset),'success':sum(r['success'] for r in subset)}
    # All methods are scored against exactly the same starting states.
    paired = collections.Counter()
    for r in rr:
        fw = next(b['success'] for b in records[r['scene']]['branches'] if b['action']=='forward')
        paired['both_success' if fw and r['success'] else 'vlm_only' if r['success'] else 'forward_only' if fw else 'both_fail'] += 1
    rec['paired_vs_forward'] = dict(paired); summary.append(rec)
baseline = {a:sum(next(b['success'] for b in r['branches'] if b['action']==a) for r in records.values()) for a in ['forward','back','left','right','turn_left','turn_right']}
result = {'n_worlds':len(records),'n_construction_groups':len({r['group'] for r in records.values()}),
    'summary':summary,'fixed_action_success_counts':baseline,
    'uniform_random_expected_success_rate':sum(baseline.values())/(6*len(records)),
    'oracle_any_action_success_count':sum(any(b['success'] for b in r['branches']) for r in records.values()),
    'checks':dict(checks),'rows':rows,
    'limits':'Offline selection from previously physically executed branches. One nominal model seed. Descriptive counts only; worlds sharing construction groups are not independent. No new online navigation claim.'}
(OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
lines = ['# Exact original six-action VLM test','',result['limits'],'',
    '| Model | Mode | Success | Base worlds | Window worlds | Invalid | Median decision s |',
    '|---|---|---:|---:|---:|---:|---:|']
for r in summary:
    lines.append(f"| {r['model']} | {r['mode']} | {r['success']}/{r['n']} | {r['base']['success']}/{r['base']['n']} | {r['window']['success']}/{r['window']['n']} | {r['invalid']} | {r['median_decision_s']:.2f} |")
lines += ['',f"Fixed forward: {baseline['forward']}/{len(records)}. Uniform random expectation: {result['uniform_random_expected_success_rate']:.4f}.",
    '', 'See summary.json for action distributions, paired outcomes, raw record paths and integrity verification.']
(OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

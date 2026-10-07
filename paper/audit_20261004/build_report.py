"""Generate manuscript tables and scientific figures from audited JSON only."""
from pathlib import Path
import json,hashlib,csv,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
OUT=Path(__file__).parent
DATA=OUT/'strict_camera'

def f(x):return f'{x:.3f}'
def ci(x):return '['+', '.join(f(t) for t in x)+']'
def main():
 d=json.loads((DATA/'camera_analysis.json').read_text());legacy=json.loads((OUT/'legacy_reanalysis.json').read_text());raw=json.loads((DATA/'camera_rollouts.json').read_text())
 sm=d['summary'];dg=d['diagnostics'];co=dg['sample_confusion'];key=d['paired_key_comparisons']
 names=[('uniform_random_exact','Uniform random (exact)'),('forward','Fixed forward'),('back','Fixed backward'),('last_image_bearing','Last-bearing strafe'),('RGBD+a','Image summaries + action'),('memory+a','Last detection + action'),('RGBD+memory+a','Image + memory + action'),('RGBD+shuffled_memory+a','Shuffled training memory'),('oracle_branch_bound','Executed-branch upper bound')]
 lines=['# 补充实验结果（2026-10-04）','','数据源：`strict_camera/camera_rollouts.json` → `strict_camera/camera_analysis.json`，全部表项由 `build_report.py` 生成。','',f"24 个候选世界中 {d['eligible_worlds']} 个满足相机定义的先看见再丢失条件；{d['eligible_groups']} 个构造组；{d['eligible_worlds']*6} 个恢复分支。{len(d['excluded'])} 个前置条件未通过世界："+', '.join(d['excluded'])+'。','','| 方法 | 相机单步成功率 | 构造组 bootstrap 95% 区间 |','|---|---:|---|']
 for k,n in names:lines.append(f"| {n} | {f(sm[k]['rsr1'])} | {ci(sm[k]['group_bootstrap_ci95'])} |")
 lines+=['','学习行平均三个模型 seed；同一物理数据不是三次实验。分支上界是特权诊断；全成功导致退化区间 [1,1]，不能解释为总体确定性。','',f"当前图像评分器相对固定前进：差值 {f(key['RGBD_vs_forward']['difference'])}，区间 {ci(key['RGBD_vs_forward']['ci95'])}，需结合区间与受限场景解释。",f"增加最后检测记忆：差值 {f(key['memory_added']['difference'])}，区间 {ci(key['memory_added']['ci95'])}；不由此推广到任意历史表示的必要性。",'',f"几何/图像检测：{sum(co.values())} 个配对采样帧，一致率 {dg['sample_agreement']:.4f}；混淆计数 {co}。{dg['loss_states_still_geometry_visible']}/{d['eligible_worlds']} 个相机 loss 状态仍被几何判为可见。这些帧具有时间相关性，不作独立样本显著性检验。",'',f"仅前进/左移/右移：随机期望 {f(d['three_action_subset']['random_rsr'])}，分支上界 {f(d['three_action_subset']['oracle_rsr'])}；完整六动作上界 {f(sm['oracle_branch_bound']['rsr1'])}。",'','| 恢复检测像素阈值（固定主 loss 状态） | 随机期望 | 分支上界 |','|---|---:|---:|']
 for t,r in d['threshold_sensitivity_fixed_loss_states'].items():lines.append(f"| {t} | {f(r['random_rsr'])} | {f(r['branch_oracle_rsr'])} |")
 lines+=['','**结论边界**：仿真注入相机、静态红色球、有效世界数见上表、单步离线分支评价。不是开放世界检测、RSR@10、多步在线部署或实机验证；未证明记忆必要性或普适学习优势。','', '历史代理结果：共同八场景上 Learned=0.625、Random=0.625；成对构造组留出时 O+M+a 的 AUC=0.9301。该结果仍含真值记忆和几何标签，只作归档诊断。']
 (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 with (OUT/'camera_results.csv').open('w',newline='',encoding='utf8') as ff:
  w=csv.writer(ff);w.writerow(['method','rsr1','group_ci_low','group_ci_high']);w.writerows([[k,sm[k]['rsr1'],*sm[k]['group_bootstrap_ci95']] for k,n in names])
 tex=(OUT/'manuscript_template.tex').read_text(encoding='utf8')
 repl={'N':str(d['eligible_worlds']),'GROUPS':str(d['eligible_groups']),'BRANCHES':str(d['eligible_worlds']*6),'AGREE':f"{dg['sample_agreement']*100:.2f}",'RANDOM':f(sm['uniform_random_exact']['rsr1']),'FORWARD':f(sm['forward']['rsr1']),'IMAGE':f(sm['RGBD+a']['rsr1']),'MEMORY':f(sm['RGBD+memory+a']['rsr1']),'SHUFFLED':f(sm['RGBD+shuffled_memory+a']['rsr1']),'FORWARD_DIFF':f(key['RGBD_vs_forward']['difference']),'FORWARD_CI':ci(key['RGBD_vs_forward']['ci95']),'MEMORY_DIFF':f(key['memory_added']['difference']),'MEMORY_CI':ci(key['memory_added']['ci95']),'EXCLUDED':', '.join('\\texttt{'+n.replace('_','\\_')+'}' for n in d['excluded']),'FRAMES':str(sum(co.values())),'LOSSG':str(dg['loss_states_still_geometry_visible']),'ORACLE':f(sm['oracle_branch_bound']['rsr1']),'THREE_ORACLE':f(d['three_action_subset']['oracle_rsr']),'THREE_RANDOM':f(d['three_action_subset']['random_rsr']),'THRESHOLD1':f(d['threshold_sensitivity_fixed_loss_states']['1']['random_rsr']),'THRESHOLD10':f(d['threshold_sensitivity_fixed_loss_states']['10']['random_rsr']),'ORACLE10':f(d['threshold_sensitivity_fixed_loss_states']['10']['branch_oracle_rsr'])}
 repl['FORWARD_INTERPRETATION']=('The descriptive interval excludes zero on this designed suite, but it does not establish a general deployment advantage.' if key['RGBD_vs_forward']['ci95'][0]>0 else 'The descriptive interval does not establish an advantage over this fixed action on the small designed suite.')
 repl['MEMORY_INTERPRETATION']=('The descriptive interval supports a positive memory-added difference within this designed suite.' if key['memory_added']['ci95'][0]>0 else 'The descriptive interval does not establish a positive benefit of this memory representation.')
 repl.update({k:str(v) for k,v in co.items()})
 table_names=[('uniform_random_exact','Uniform random'),('forward','Fixed forward'),('last_image_bearing','Last-bearing strafe'),('RGBD+a','Image + action'),('memory+a','Memory + action'),('RGBD+memory+a','Image + memory + action'),('RGBD+shuffled_memory+a','Shuffled training memory'),('oracle_branch_bound','Branch upper bound')]
 repl['CAMERA_TABLE']='\n'.join(n+' & '+f(sm[k]['rsr1'])+' & '+ci(sm[k]['group_bootstrap_ci95'])+'\\\\' for k,n in table_names)
 for k,v in repl.items():tex=tex.replace('@'+k+'@',v)
 assert not re.search(r'@[A-Z_0-9]+@',tex)
 (OUT/'manuscript.tex').write_text(tex,encoding='utf8')
 from render_markdown import render
 render(tex,OUT)
 # Professional static figures: fixed axes and readable type; numbers from JSON.
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
 fig,axes=plt.subplots(1,3,figsize=(13.6,4.5),layout='constrained')
 chosen=[('uniform_random_exact','Uniform random'),('forward','Fixed forward'),('RGBD+a','Image + action'),('RGBD+memory+a','Image + memory'),('oracle_branch_bound','Branch upper bound')]
 ax=axes[0];ys=np.arange(len(chosen));means=np.array([sm[k]['rsr1'] for k,n in chosen]);intervals=np.array([sm[k]['group_bootstrap_ci95'] for k,n in chosen]);ax.barh(ys,means,color=['#9b9da4','#586477','#176e88','#df9134','#bbc2c7']);ax.errorbar(means,ys,xerr=np.vstack([means-intervals[:,0],intervals[:,1]-means]),fmt='none',ecolor='#253746',capsize=3);ax.set_yticks(ys,[n for k,n in chosen]);ax.invert_yaxis();ax.set_xlim(0,1.07);ax.set_xlabel('Camera one-step success');ax.set_title('(a) Same loss states and actions',loc='left',weight='bold');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
 ax=axes[1];mat=np.array([[co['G0C0'],co['G0C1']],[co['G1C0'],co['G1C1']]]);ax.imshow(mat,cmap='Blues');
 for (i,j),v in np.ndenumerate(mat):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>mat.max()/2 else '#203344',fontsize=15)
 ax.set_xticks([0,1],['Not detected','Detected']);ax.set_yticks([0,1],['Geometry hidden','Geometry visible']);ax.set_xlabel('RGB detector');ax.set_title('(b) Paired frame labels',loc='left',weight='bold')
 ax=axes[2];ths=[1,3,10];rand=[d['threshold_sensitivity_fixed_loss_states'][str(t)]['random_rsr'] for t in ths];ora=[d['threshold_sensitivity_fixed_loss_states'][str(t)]['branch_oracle_rsr'] for t in ths];ax.plot(ths,rand,'o-',color='#176e88',label='Uniform random');ax.plot(ths,ora,'s--',color='#df9134',label='Branch upper bound');ax.set_xticks(ths);ax.set_ylim(0,1.05);ax.set_xlabel('Required red pixels');ax.set_ylabel('One-step success');ax.legend(frameon=False,loc='lower left');ax.set_title('(c) Fixed loss-state sensitivity',loc='left',weight='bold');ax.grid(alpha=.15)
 fig.savefig(OUT/'audit_results.png',dpi=200);fig.savefig(OUT/'audit_results.pdf');plt.close(fig)
 # Actual captured frames; annotated titles only, no altered image content.
 by={r['scene']:r for r in raw['records']};selected=['basin','hist001_left','hist001_right'];fig,axes=plt.subplots(3,3,figsize=(10.4,7.9),layout='constrained')
 for i,name in enumerate(selected):
  r=by[name];action=next(b['action'] for b in r['branches'] if b['success'])
  for j,(suffix,label) in enumerate([('seen','Previously detected'),('lost','Loss trigger'),(action+'_visible','Recovery branch: '+action)]):
   axes[i,j].imshow(Image.open(DATA/'frames'/f'{name}_{suffix}.png'));axes[i,j].axis('off');axes[i,j].set_title(name+' | '+label,fontsize=9)
 fig.savefig(OUT/'camera_evidence.png',dpi=180);fig.savefig(OUT/'camera_evidence.pdf');plt.close(fig)
 manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.py','.tex','.md','.csv','.npz'] and p.name!='deliverable_manifest.json'}
 (OUT/'deliverable_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
 print('Generated RESULTS.md, manuscript.tex, CSV, two PNG/PDF figures and manifest.')
if __name__=='__main__':main()

"""Scientific summary and unannotated camera evidence from saved runs."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
r=ROOT/'paper/vlm_goal_navigation/runs/20261004_qwen35'
out=ROOT/'paper/vlm_goal_navigation/figures';out.mkdir(exist_ok=True)
d=json.loads((r/'original_six_action/summary.json').read_text())
modes=['current_direct','current_reasoned','actions_reasoned','images_reasoned','images_actions_reasoned']
labels=['Current / direct','Current / reasoning','+ action history','+ image history','+ both histories']
fig,ax=plt.subplots(figsize=(9.2,4.8))
for model,shift,color in [('qwen3.5:4b',-.18,'#24779e'),('qwen3.5:9b',.18,'#d97736')]:
 vals={x['mode']:x for x in d['summary'] if x['model']==model}
 for i,mode in enumerate(modes):
  if mode not in vals:continue
  s=vals[mode];y=s['success']/s['n'];ax.bar(i+shift,y,.33,color=color,label=model if i==0 else None)
  ax.text(i+shift,y+.022,f"{s['success']}/{s['n']}",ha='center',fontsize=9)
ax.axhline(d['fixed_action_success_counts']['forward']/d['n_worlds'],color='black',ls='--',label='Fixed forward (17/23)')
ax.set_xticks(range(5),labels);ax.tick_params(axis='x',labelsize=9);ax.set_ylim(0,1.05)
ax.set_ylabel('Camera reacquisition success');ax.set_title('Original six-action task: offline selection on physically executed branches')
ax.legend(loc='upper right',fontsize=8);ax.spines[['top','right']].set_visible(False)
fig.text(.5,.015,'23 designed worlds / 15 construction groups; one nominal seed; missing 9B bars were not tested.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'original_six_action.png',dpi=180);fig.savefig(out/'original_six_action.pdf');plt.close(fig)
# Camera frames, untouched except for display labels placed outside the image.
ep=r/'closed_loop_v2/episodes/qwen3.5_4b/current_reasoned/stairs_up'
paths=[ROOT/'paper/audit_20261004/strict_camera/frames/stairs_up_seen.png',r/'prefixes/stairs_up/loss.png',ep/'frame_02.png',ep/'final.png']
captions=['Earlier: target seen','Start: target lost','After 2 forward actions','Later: out of bounds']
sheet=Image.new('RGB',(1280,295),'white');draw=ImageDraw.Draw(sheet)
try:font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
except OSError:font=ImageFont.load_default()
for i,(path,label) in enumerate(zip(paths,captions)):
 sheet.paste(Image.open(path).convert('RGB').resize((320,240)),(i*320,35));draw.text((i*320+8,8),label,font=font,fill='black')
draw.text((8,278),'4B current + reasoning: camera reacquisition at 4.502 s; later approach failed. Rendered simulation RGB.',fill='black')
sheet.save(out/'camera_reacquisition_then_failure.png')
print(out)

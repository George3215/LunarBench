"""Readable Markdown companion to the standalone LaTeX source (no PDF claim)."""
import re,json
from pathlib import Path

def render(tex,out):
 data=json.loads((out/'strict_camera/camera_analysis.json').read_text());s=data['summary'];co=data['diagnostics']['sample_confusion']
 body=tex.split('\\begin{document}',1)[1].rsplit('\\end{document}',1)[0]
 body=body.replace('\\maketitle','# Camera-Defined Target Reacquisition on Legged Robots\n\n## Observation, Memory, and Motion\n\nAnonymous research draft · 4 October 2026')
 body=body.replace('\\begin{abstract}','## Abstract\n').replace('\\end{abstract}','')
 def table(m):
  t=m.group(0)
  if 'tab:legacy' in t:
   return '\n**Table 1. Archived proxy labels; not camera-control results.**\n\n| Split | Inputs | AUC | AP |\n|---|---|---:|---:|\n| Scene | O+M | 0.854 | 0.600 |\n| Scene | O+M+a | 0.934 | 0.748 |\n| Construction group | O+M | 0.850 | 0.596 |\n| Construction group | O+M+a | 0.930 | 0.743 |\n| Construction group | a | 0.448 | 0.139 |\n'
  if 'tab:confusion' in t:return f"\n**Table 2. Paired geometry/camera frame labels.**\n\n| | Camera absent | Camera detected |\n|---|---:|---:|\n| Geometry hidden | {co['G0C0']} | {co['G0C1']} |\n| Geometry visible | {co['G1C0']} | {co['G1C1']} |\n"
  names=[('uniform_random_exact','Uniform random'),('forward','Fixed forward'),('last_image_bearing','Last-bearing strafe'),('RGBD+a','Image + action'),('memory+a','Memory + action'),('RGBD+memory+a','Image + memory + action'),('RGBD+shuffled_memory+a','Shuffled training memory'),('oracle_branch_bound','Branch upper bound')]
  lines=['\n**Table 3. Camera-defined one-step recovery.**\n','| Method | RSR1 | Group bootstrap 95% interval |','|---|---:|---|']
  for k,n in names:
   r=s[k];lo,hi=r['group_bootstrap_ci95'];lines.append(f"| {n} | {r['rsr1']:.3f} | [{lo:.3f}, {hi:.3f}] |")
  return '\n'.join(lines)+'\n'
 body=re.sub(r'\\begin\{table\}.*?\\end\{table\}',table,body,flags=re.S)
 body=re.sub(r'\\section\*?\{([^}]+)\}',r'\n## \1\n',body)
 body=re.sub(r'\\subsection\{([^}]+)\}',r'\n### \1\n',body)
 body=body.replace('\\begin{equation}','\n$$\n').replace('\\end{equation}','\n$$\n')
 refs={'bajcsy':1,'pomdp':2,'radmard':3,'radmard2018':4,'oanbv':5,'womap':6}
 body=re.sub(r'\\cite\{([^}]+)\}',lambda m:'['+', '.join(str(refs[x]) for x in m[1].split(','))+']',body)
 body=re.sub(r'\\ref\{([^}]+)\}',lambda m:str({'tab:legacy':1,'tab:confusion':2,'tab:camera':3}[m[1]]),body)
 body=body.replace('\\begin{thebibliography}{9}','\n## References\n').replace('\\end{thebibliography}','')
 body=re.sub(r'\\bibitem\{([^}]+)\}',lambda m:'\n'+str(refs[m[1]])+'. ',body)
 body=re.sub(r'\\href\{([^}]+)\}\{([^}]+)\}',r'[\2](\1)',body)
 body=re.sub(r'\\url\{([^}]+)\}',r'[\1](\1)',body)
 body=re.sub(r'\\textbf\{([^}]+)\}',r'**\1**',body)
 body=re.sub(r'\\emph\{([^}]+)\}',r'*\1*',body)
 body=re.sub(r'\\texttt\{([^}]+)\}',r'`\1`',body)
 body=body.replace('\\noindent','').replace('\\%','%').replace('\\_','_').replace('~',' ').replace('``','“').replace("''",'”').replace('---','—').replace('--','–')
 # Restore standard Markdown table separator syntax after typographic dash replacement.
 body=re.sub(r'\|—', '|---',body)
 body=re.sub(r'\n{3,}','\n\n',body)
 (out/'manuscript.md').write_text(body.strip()+'\n',encoding='utf8')
if __name__=='__main__':
 out=Path(__file__).parent;render((out/'manuscript.tex').read_text(encoding='utf8'),out)

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).parent;r=json.loads((p/'report.json').read_text());mus=[0,.3,.6,1.]
sets={'Alternating tripod':['lf+lr+rm','lm+rf+rr'],'Opposite-pair four supports':['lf+lr+rf+rr','lm+lr+rf+rm','lf+lm+rm+rr'],'Any five supports':[c['name'] for c in r['cases'] if c['count']==5],'All six supports':['lf+lm+lr+rf+rm+rr']}
summary={}
fig,ax=plt.subplots(figsize=(8.2,4.8),layout='constrained')
for label,names in sets.items():
    cases=[c for c in r['cases'] if c['name'] in names];low=[];high=[]
    for mu in mus:
        rows=[next(x for x in c['rows'] if x['mu']==mu) for c in cases]
        low.append(max(x['outer_lower_bound']['minimax_motor_torque_nm'] for x in rows))
        high.append(max(x['inner_feasible_witness']['minimax_motor_torque_nm'] for x in rows))
    summary[label]=dict(support_patterns=names,mu=mus,worst_pattern_ideal_lower_bound_nm=low,worst_pattern_ideal_feasible_upper_bound_nm=high)
    line,=ax.plot(mus,high,'o-',label=label);ax.fill_between(mus,low,high,color=line.get_color(),alpha=.25)
ax.axhline(1.6,ls='--',color='black',label='Current 1.6 N·m contract')
ax.set(xlabel='Assumed Coulomb friction coefficient (unmeasured)',ylabel='Minimum possible peak joint torque (N·m)',title='Nominal C stance: optimistic static load allocation\nExact link gravity; ≥1 N at each declared supporting foot')
ax.legend(fontsize=8);fig.savefig(p/'static_headroom.png',dpi=170)
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

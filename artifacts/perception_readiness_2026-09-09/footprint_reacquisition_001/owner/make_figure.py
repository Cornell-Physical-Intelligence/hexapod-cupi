"""Static prototype evidence figure, with no extrapolated robot motion."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'report.json').read_text());cases={c['assumptions']['id']:c for c in r['cases']};base=cases['continuous_six_camera'];rows=base['rows'];t=np.array([x['query_time_s'] for x in rows])
mt=np.array([x['time_s'] for x in base['all_control_map_rows']])
fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,gridspec_kw={'height_ratios':[1,1,1.1]})
fig.suptitle('Fresh map coverage does not guarantee a usable landing footprint',x=.07,ha='left',fontsize=17,fontweight='bold')
axes[0].plot(mt,[100*x['ever_observed_fraction'] for x in base['all_control_map_rows']],label='Ever observed',color='#8c939d',lw=2)
axes[0].plot(mt,[100*x['retained_usable_fraction'] for x in base['all_control_map_rows']],label='Usable now · continuous capture',color='#238b74',lw=2)
axes[0].plot(mt,[100*x['retained_usable_fraction'] for x in cases['outage_8p0_to_8p6']['all_control_map_rows']],label='Usable now · 0.6 s capture outage',color='#d46345',lw=1.5)
axes[0].set_ylabel('Map cells (%)');axes[0].set_ylim(0,52);axes[0].legend(frameon=False,ncol=3,fontsize=9,loc='upper left')
axes[1].plot(t,[100*x['usable_now_use_cells']/x['required_cells_use'] for x in rows],label='Usable portion of the\noriginal planned footprint',color='#3479b8',lw=1.5)
axes[1].scatter(t,[100*x['complete_usable_eligible_now'] for x in rows],label='Entire footprint fresh now',s=6,color='#1d6557',alpha=.55)
axes[1].plot(t,[100*x['lease_covers_requested_use_time'] for x in rows],label='Entire footprint valid\nthrough planned use',color='#b6465f',lw=2)
axes[1].set_ylabel('Footprint coverage (%)');axes[1].set_ylim(-6,115);axes[1].legend(frameon=False,ncol=1,fontsize=9,loc='center left',bbox_to_anchor=(1.005,.5))
axes[2].plot(t,[x['remaining_planned_horizon_s'] for x in rows],color='#3479b8',label='Remaining time to original\nplanned landing',lw=2)
axes[2].axhline(.25,color='#b6465f',ls='--',label='Capture-age lease · 0.25 s')
axes[2].axhline(.21,color='#a67720',ls=':',label='Lease left on receipt · 0.21 s')
axes[2].set_ylim(-.05,2.15);axes[2].set_ylabel('Time remaining (s)');axes[2].set_xlabel('Actual004 recorded control time (s)');axes[2].legend(frameon=False,fontsize=9,loc='center left',bbox_to_anchor=(1.005,.5))
for event in base['summary']['events']:
    axes[2].text(event['first_query_time_s']+.06,1.63,event['foot_id'].upper(),fontweight='bold',color='#315069')
for ax in axes:
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_xlim(4,12.55)
    ax.axvspan(8.04,8.64,color='#d46345',alpha=.08)
fig.text(.07,.033,'Synthetic flat acquisition • hypothetical six D405 cameras • actual C meshes and poses • 15 mm proxy footprint + registration uncertainty',fontsize=9,color='#454d57')
fig.text(.07,.011,'No future observations, terrain traversal, physical abort path, hardware qualification, or actor integration. Source009 stop took 5.54 s to reference quiet.',fontsize=9,color='#454d57')
fig.subplots_adjust(left=.08,right=.74,top=.90,bottom=.11,hspace=.24)
fig.savefig(ROOT/'footprint_lease.png',dpi=150,facecolor='white');plt.close(fig)

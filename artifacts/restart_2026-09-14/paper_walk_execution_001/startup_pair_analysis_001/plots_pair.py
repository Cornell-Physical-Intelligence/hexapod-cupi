"""Render maintained numeric plot functions with explicit startup-prefix labels."""
from pathlib import Path
import datetime
import hashlib
import json
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import numpy as np
import analyze_pair as pair


def main():
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    pair.pin(Path(__file__).resolve())
    pair.pin(Path(pair.__file__).resolve())
    analyzer=pair.module('startup_plot_analyzer',pair.ROOT/'experiments/paper_walk/analyze.py')
    original_save=Figure.savefig
    outputs={}
    for arm in ('cold','neutral4'):
        result,controls,native,_=pair.load_arm(analyzer,arm)
        output=pair.HERE/arm
        output.mkdir(exist_ok=False)
        prefix=result['scripted_prefix_controls']/50
        title=('COLD: BC controls 0–20 s (1,000 controls); no scripted prefix' if not prefix else
               'NEUTRAL4: scripted zero actions 0–4 s (200 controls); BC 4–24 s (1,000 controls)')
        def decorated_save(fig,filename,*args,**kwargs):
            fig.suptitle(title+'\nComplete native record; both forward screens fail planar tracking',fontsize=10)
            if prefix:
                for ax in fig.axes:
                    time_axis=any('Time' in sibling.get_xlabel() for sibling in ax.get_shared_x_axes().get_siblings(ax))
                    if time_axis:
                        ax.axvspan(0,prefix,color='tab:blue',alpha=.13,zorder=5)
                        ax.axvline(prefix,color='tab:blue',ls=':',lw=1.2,zorder=6)
            return original_save(fig,filename,*args,**kwargs)
        Figure.savefig=decorated_save
        try: outputs[arm]=analyzer.plots(controls,native,output,0)
        finally: Figure.savefig=original_save
        assert outputs[arm]['status']=='written' and len(outputs[arm]['files'])==3
        outputs[arm]['sha256']={name:pair.sha(output/name) for name in outputs[arm]['files']}
        outputs[arm]['prefix_scope']='The blue time interval, where present, is actual scripted zero-action control. Spectrum and torque bars use the complete record, including the prefix. Numeric policy-only summaries are in RESULT.json.'
    assert all(pair.sha(pair.ROOT/p)==h for p,h in pair.PINS.items())
    receipt={'started_utc':started,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'runtime':sys.executable,'matplotlib':matplotlib.__version__,'numpy':np.__version__,
        'scope':'Maintained analyze.plots, unchanged numeric records and plot calculations; artifact-only title/prefix annotation. No dependencies installed.',
        'input_sha256':pair.PINS,'outputs':outputs,'inputs_unchanged':True}
    with (pair.HERE/'PLOTS.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps({'complete':True,'outputs':outputs}))


if __name__=='__main__':main()

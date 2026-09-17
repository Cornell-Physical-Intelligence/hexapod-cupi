"""Plot the complete measured forward-speed traces from both native replays."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
trials=[("Original reference",ROOT/"artifacts/trajectory_optimizer_20260917/replay_001/standing/evaluation/control_trace.npz","#b75d35"),("Improved reference",ROOT/"artifacts/ppo_reference_comparison_20260917/replay_pack_001/replay_001/standing/evaluation/control_trace.npz","#176b87")]
fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,sharey=True,layout="constrained")
for ax,(title,path,color) in zip(axes,trials):
    with np.load(path,allow_pickle=False) as d:
        velocity=d["velocity_navigation_mps"][:,0,0].copy()
    assert len(velocity)==1000 and np.isfinite(velocity).all()
    t=(np.arange(len(velocity))+1)*.02
    ax.plot(t,velocity,color=color,lw=1.3,label="Measured forward speed")
    ax.axhline(.05,color="#333333",ls="--",lw=1,label="Command: 0.05 m/s")
    ax.axvspan(0,2,color="#dddddd",alpha=.5,label="Startup interval")
    ax.set(title=title,ylabel="Forward speed (m/s)",xlim=(0,20),ylim=(-.025,.125))
    ax.grid(alpha=.2)
axes[0].legend(loc="upper right",ncol=3,fontsize=8)
axes[-1].set_xlabel("Simulation time (s)")
fig.suptitle("Measured forward speed in Isaac Sim",fontsize=15)
fig.savefig(Path(__file__).with_name("forward_speed.png"),dpi=180)
fig.savefig(Path(__file__).with_name("forward_speed.svg"))

"""Plot the complete forward trials and decode the saved PPO recording."""
from pathlib import Path
import hashlib
import json

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
A = ROOT/'artifacts/ppo_reference_comparison_20260917'
trials = [
    ('Original reference (no RL)', ROOT/'artifacts/trajectory_optimizer_20260917/replay_001/standing/evaluation', '#b75d35'),
    ('Improved reference (no RL)', A/'replay_pack_001/replay_001/standing/evaluation', '#176b87'),
    ('Vanilla PPO (1,200 updates)', A/'vanilla_evaluate_001/run/standing/evaluation_00', '#7255a0'),
]
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True, layout='constrained')
extrema = []
for ax, (label, base, color) in zip(axes, trials):
    with np.load(base/'control_trace.npz', allow_pickle=False) as data:
        velocity = data['velocity_navigation_mps'][:, 0, 0].copy()
        time = data['time_s'].copy()
    assert len(velocity) == 1000 and np.isfinite(velocity).all()
    extrema.extend((float(velocity.min()), float(velocity.max())))
    ax.plot(time, velocity, color=color, lw=1.2, label='Measured forward speed')
    ax.axhline(.05, color='#333333', ls='--', lw=1, label='Command: 0.05 m/s')
    ax.axvspan(0, 2, color='#dddddd', alpha=.5, label='Startup interval')
    ax.set(title=label, ylabel='Forward speed (m/s)', xlim=(0, 20))
    ax.grid(alpha=.2)
axes[0].set_ylim(min(extrema)-.01, max(extrema)+.01)
axes[0].legend(loc='upper right', ncol=3, fontsize=8)
axes[-1].set_xlabel('Simulation time (s)')
fig.suptitle('Same forward command, robot and motor-target limiter', fontsize=14)
for extension in ('png', 'svg'):
    path = HERE/f'forward_comparison.{extension}'
    assert not path.exists()
    fig.savefig(path, dpi=180)
plt.close(fig)

video = trials[-1][1]/'rollout.mp4'
selected = (0, 50, 100, 150, 250, 350, 450, 499)
frames = {}
count = 0
with imageio.get_reader(video, format='ffmpeg') as reader:
    metadata = reader.get_meta_data()
    for index, frame in enumerate(reader):
        assert frame.ndim == 3 and frame.shape[-1] == 3
        if index in selected:
            frames[index] = frame.copy()
        count += 1
assert count == 500 and metadata['fps'] == 25.
fig, axes = plt.subplots(2, 4, figsize=(14, 5), layout='constrained')
for ax, index in zip(axes.flat, selected):
    ax.imshow(frames[index])
    ax.set_title(f'{(index+1)/25:.2f} s', fontsize=10)
    ax.axis('off')
fig.suptitle('Vanilla PPO forward trial: sampled native frames', fontsize=14)
path = HERE/'forward_frames.jpg'
assert not path.exists()
fig.savefig(path, dpi=150)
plt.close(fig)
record = {'video_sha256': hashlib.sha256(video.read_bytes()).hexdigest(),
    'frames_decoded': count, 'fps': metadata['fps'], 'sampled_frames': list(selected),
    'frame_shape': list(frames[0].shape), 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'scope': 'Complete decode plus sampled frames. This does not supply human gait acceptance.'}
with (HERE/'FORWARD_VIDEO_DECODE.json').open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
print(json.dumps(record))

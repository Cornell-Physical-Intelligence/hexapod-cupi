"""Orthographic CAD/collider sections and support-height comparison."""
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

from build_candidate import OUT, MESH, read_stl


def section(triangles, axis, offset):
    segments = []
    keep = [a for a in range(3) if a != axis]
    for tri in triangles:
        points = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = tri[i], tri[j]
            da, db = a[axis]-offset, b[axis]-offset
            if abs(da) < 1e-12:
                points.append(a)
            if da*db < 0:
                points.append(a + da/(da-db)*(b-a))
        if len(points) >= 2:
            points = np.unique(points, axis=0)
            if len(points) >= 2:
                pairs = [(np.linalg.norm(a-b), a, b) for i, a in enumerate(points) for b in points[i+1:]]
                _, a, b = max(pairs, key=lambda p: p[0])
                segments.append(np.array([a[keep], b[keep]])*1000)
    return segments


report = json.loads((OUT/'geometry_report.json').read_text())
source = read_stl(MESH)
candidate = read_stl(OUT/report['candidate']['path'])
centers = np.array(report['placements'][0]['existing_sphere_centers_in_mesh_m'])
radii = report['placements'][0]['existing_sphere_radii_m']
colors = {'CAD': '#1762a6', 'spheres': '#b84468', 'candidate': '#d87700'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig = plt.figure(figsize=(13, 8), facecolor='white')
grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=.47, wspace=.25)
axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
for ax, axis, offset, label in [(axes[0], 1, .001, 'Longitudinal section: CAD y = 1 mm'),
                                (axes[1], 2, 0., 'Orthogonal section: CAD z = 0 mm')]:
    keep = [a for a in range(3) if a != axis]
    for center, radius in zip(centers, radii):
        projected_sq = radius**2-(center[axis]-offset)**2
        if projected_sq > 0:
            ax.add_patch(Circle(center[keep]*1000, math.sqrt(projected_sq)*1000,
                                facecolor=colors['spheres'], edgecolor=colors['spheres'], alpha=.10))
            ax.add_patch(Circle(center[keep]*1000, math.sqrt(projected_sq)*1000,
                                fill=False, edgecolor=colors['spheres'], linewidth=1.6, linestyle=':'))
    ax.add_collection(LineCollection(section(source, axis, offset), colors=colors['CAD'], linewidths=2.4))
    ax.add_collection(LineCollection(section(candidate, axis, offset), colors=colors['candidate'], linewidths=1.4, linestyles='--'))
    ax.set_xlim(187, 258); ax.set_ylim(-22, 22); ax.set_aspect('equal')
    ax.set_xlabel('Original CAD x (mm)')
    ax.set_ylabel(f'Original CAD {"xyz"[keep[1]]} (mm)')
    ax.set_title(label, loc='left', fontsize=11, fontweight='bold')
    ax.grid(alpha=.16)
handles = [Line2D([0], [0], color=colors['CAD'], lw=2.4, label='Exact triangulated CAD section'),
           Line2D([0], [0], color=colors['spheres'], lw=1.6, ls=':', label='Current two-sphere sections'),
           Line2D([0], [0], color=colors['candidate'], lw=1.6, ls='--', label='64-vertex convex candidate')]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .918), ncol=3, frameon=False, fontsize=10)
ax = fig.add_subplot(grid[1, 0])
sweep = report['lf_plane_normal_sweep']
angles = [x['tilt_from_nominal_degrees'] for x in sweep]
for key, color, label in [('candidate', colors['candidate'], 'Candidate'), ('sphere', colors['spheres'], 'Current spheres')]:
    lo = np.array([x[f'{key}_bottom_error_min_m'] for x in sweep])*1000
    hi = np.array([x[f'{key}_bottom_error_max_m'] for x in sweep])*1000
    ax.fill_between(angles, lo, hi, alpha=.15, color=color)
    ax.plot(angles, hi, color=color, label=label+' (min–max over azimuth)')
    ax.plot(angles, lo, color=color, lw=1)
ax.axhline(0, color='#6a6a6a', lw=.7)
ax.set_xlabel('Plane-normal tilt from LF nominal stance (degrees)')
ax.set_ylabel('Collider bottom − CAD bottom (mm)')
ax.set_title('Contact-height error across 72 azimuths per tilt', loc='left', fontsize=11, fontweight='bold')
ax.legend(frameon=False, fontsize=8, loc='lower left'); ax.grid(alpha=.16)
ax = fig.add_subplot(grid[1, 1]); ax.axis('off')
gap = report['candidate']['maximum_cad_surface_to_candidate_distance_m']*1000
old = report['existing_two_spheres']['maximum_outside_distance_m']*1000
fill = report['concavity_fill']['sampled_maximum_candidate_surface_to_cad_surface_distance_m']*1000
summary = (f'{gap:.3f} mm maximum candidate undercoverage\n'
           f'{old:.3f} mm maximum current-sphere undercoverage\n\n'
           '64 vertices / 124 triangles • original CAD coordinates\n'
           'Bounds cover the entire triangulated CAD surface.\n\n'
           f'{fill:.3f} mm sampled cavity overfill: see CAD sections.\n'
           'Collision volume must not replace the mass model.\n\n'
           'UNQUALIFIED RIGID ENVELOPE PROPOSAL\n'
           'Not installed in the robot simulation.\n'
           'Silicone compliance and contact behavior unmeasured.')
ax.text(0, .98, summary, va='top', fontsize=10.5, linespacing=1.45)
fig.suptitle('Silicone foot: CAD geometry and collision-envelope comparison', x=.07, ha='left', y=.985, fontsize=17, fontweight='bold')
fig.text(.07, .945, 'True planar sections, equal axis scales, millimeter dimensions. No mesh recentering or rescaling.', color='#555', fontsize=10)
fig.text(.07, .025, 'Positive bottom error: collider sits above the CAD surface. Support-height plots do not reveal the two-sphere waist.', color='#555', fontsize=9)
fig.subplots_adjust(top=.84, bottom=.10, left=.075, right=.96)
fig.savefig(OUT/'pad_comparison.png', dpi=180, facecolor='white')
fig.savefig(OUT/'pad_comparison.pdf', facecolor='white', metadata={'CreationDate': None, 'ModDate': None})

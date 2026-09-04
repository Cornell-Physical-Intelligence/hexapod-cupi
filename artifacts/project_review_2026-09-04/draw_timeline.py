"""Generate the repository's static, shareable program timeline; stdlib only."""
from html import escape
from pathlib import Path

OUT = Path(__file__).with_name("TIMELINE.svg")
W, H = 1200, 1000
X, DX = 245, 150
parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">
<title id="title">Hexapod autonomy: parallel development timeline</title>
<desc id="desc">Six software weeks after resumption, with a simulation prototype at week four and a transfer candidate at week six. Mechanical and electrical work prepare a single leg, order parts after it passes, and assemble the robot at an unknown date H. Hardware transfer and field trials then take approximately two to three weeks, conditional on the software and physical gates.</desc>
<style>
text {{ font-family: Arial, Helvetica, sans-serif; fill: #172b36; font-size: 15px; }}
.title {{ font-size: 26px; font-weight: 600; }}
.section {{ font-size: 19px; font-weight: 600; }}
.muted {{ fill: #435965; }}
.bartext {{ font-size: 14px; }}
.grid {{ stroke: #d1dce1; stroke-width: 1; }}
.milestone {{ stroke: #526876; stroke-width: 1.3; stroke-dasharray: 4 4; }}
</style>
<rect width="1200" height="1000" fill="#ffffff"/>
''']


def text(x, y, label, cls="", anchor="start"):
    parts.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(label)}</text>')


def line(x1, y1, x2, y2, cls="grid"):
    parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}"/>')


def bar(y, start, end, label, color="#d8e9f5", width=DX):
    x = X + start * width + 3
    w = (end - start) * width - 6
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="36" rx="3" fill="{color}"/>')
    text(x + w / 2, y + 23, label, "bartext", "middle")


text(30, 43, "Hexapod autonomy: develop in parallel", "title")
text(30, 73, "Planning targets • 4 September 2026 • No runs queued; weeks begin after development resumes", "muted")
text(30, 112, "Software and simulation", "section")
for i in range(7):
    line(X + i * DX, 152, X + i * DX, 495)
for i in range(6):
    text(X + (i + .5) * DX, 140, f"Week {i+1}", anchor="middle")

lanes = [
    ("Model / dynamics", [(0, .6, "A0", "#f3dfc6"), (.6, 6, "Measure → update dynamics → screen affected policies", "#eee9dc")]),
    ("Learned locomotion", [(.6, 2, "L0: flat omni", "#d8e9f5"), (2, 3, "L1: terrain", "#d8e9f5"), (3, 4.5, "L2: perceptive", "#d8e9f5"), (4.5, 6, "V3: screen", "#e1e8f2")]),
    ("Vision / LiDAR / IMU", [(0, 2, "P0: bags + fusion benchmark", "#d6ebe4"), (2, 3, "P1: visibility", "#d6ebe4"), (3, 5, "P2: sensor-to-map pipeline", "#d6ebe4"), (5, 6, "V3: faults", "#e1e8f2")]),
    ("Coverage navigation", [(0, 1, "N0: polygon", "#dce9d2"), (1, 3, "N1: policy-in-loop tracking", "#dce9d2"), (3, 4, "N2: replan", "#dce9d2"), (4, 6, "V3: held-out missions", "#e1e8f2")]),
    ("Runtime / embedded", [(0, 2, "R0: contracts + motor emulator", "#e7ddf1"), (2, 4.5, "R1: deployment parity + Jetson", "#e7ddf1"), (4.5, 6, "V3: transfer package", "#e1e8f2")]),
    ("Integration / verification", [(0, 1, "V0: scenarios", "#e1e8f2"), (1, 3, "V1: daily integrated tests", "#e1e8f2"), (3, 4, "V2: prototype", "#c7d7ed"), (4, 6, "V3: qualification matrix", "#c7d7ed")]),
    ("Geo Data interface", [(0, .6, "D0", "#f3dfc6"), (1, 3, "D1: generic adapter + records", "#f3dfc6"), (3, 6, "Payload / coverage checks in integrated missions", "#eee9dc")]),
]
for i, (label, bars) in enumerate(lanes):
    y = 163 + i * 47
    text(30, y + 23, label)
    for a, b, label, color in bars:
        bar(y, a, b, label, color)

for week in (2, 3, 4, 6):
    line(X + week * DX, 151, X + week * DX, 500, "milestone")
for week, label in [(2, "G1: motion"), (3, "G2: terrain"), (4, "G3: sim prototype"), (6, "G4: transfer candidate")]:
    text(X + week * DX - 5, 521, label, anchor="end")
text(30, 549, "A0 / G0: first 3 days, correct USD + reset + frames before training. D0: payload contract in first 3 days.", "muted")

text(30, 597, "Mechanical and electrical: availability sets the hardware clock", "section")
boxes = [(30, 260, "Single-leg rig + load tests"), (330, 240, "Pass → order remaining parts"), (610, 255, "Delivery + fit + assembly"), (905, 255, "H: ready for powered tests")]
for x, w, label in boxes:
    parts.append(f'<rect x="{x}" y="619" width="{w}" height="44" rx="3" fill="#f2ece1"/>')
    text(x + w / 2, 647, label, "bartext", "middle")
for x in (305, 585, 880):
    text(x, 647, "→", anchor="middle")
text(30, 690, "H is unknown. Weight / strength / fit and CAN / power testing run in parallel; failed tests trigger revision.", "muted")

text(30, 737, "Physical transfer, after H and prerequisite simulation gates", "section")
hdx = 300
for i in range(4):
    line(X + i * hdx, 776, X + i * hdx, 925)
    text(X + i * hdx, 766, "H" if i == 0 else f"H+{i} week" + ("s" if i > 1 else ""), anchor="middle")
text(30, 814, "Runtime + controls")
bar(791, 0, 1, "R2 / G5: mapping → restrained → free", "#e7ddf1", hdx)
text(30, 861, "Integrated missions")
bar(838, 1, 2, "H1: indoor polygon coverage", "#dce9d2", hdx)
bar(838, 2, 3, "H2 / G6: bounded field survey", "#dce9d2", hdx)
text(30, 908, "Mechanical + electrical")
bar(885, 0, 3, "Loaded endurance, power / CAN faults, thermal limits and model feedback", "#f3dfc6", hdx)
text(30, 959, "Targets are conditional. Real calibration, hardware availability and failed gates can extend the schedule.", "muted")
parts.append("</svg>\n")
OUT.write_text("\n".join(parts))
print(OUT)

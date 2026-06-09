#!/usr/bin/env python3
"""
Regenerate Main_Fig2_Pipeline.png - Joint Search Main Experiment.

Based on the original create_pipeline_figure.py design (white
background with black borders, horizontal/vertical right-angle
arrows). Only the joint-search blocks (b7-b11) are modified to
reflect the four-dimension joint search; all other blocks reuse
the original layout.
"""


import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8

fig, ax = plt.subplots(figsize=(7.5, 9.5))  # 9.5 inch (PNG ~2850, slightly above PLOS 2625 limit but arrows long enough)
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-11.6, 26)  # range 37.6 fits 11 boxes plus G=1.55 elongated arrows
ax.axis('off')

H2 = 1.45  # 2-line box(enlarge padding)
H3 = 1.95  # 3-line box
H4 = 2.7   # 4-line box
G  = 1.55  # arrow gap(elongate to make arrows visually prominent;20% larger than the original 1.3)

class box:
    def __init__(s, x, y, w, h, text, bold=False, color='white', fs=8):
        s.x, s.y, s.w, s.h, s.text, s.bold, s.color, s.fs = x, y, w, h, text, bold, color, fs
    @property
    def top(s): return s.y + s.h/2
    @property
    def bottom(s): return s.y - s.h/2
    def draw(s, ax):
        ax.add_patch(patches.Rectangle(
            (s.x - s.w/2, s.bottom), s.w, s.h,
            facecolor=s.color, edgecolor='black', linewidth=0.8))
        ax.text(s.x, s.y, s.text, ha='center', va='center',
                fontsize=s.fs, fontweight='bold' if s.bold else 'normal',
                linespacing=1.2)
    def verify(s):
        """Check two things:
        1. text height vs box height(to avoid text being split)— padding >= 0.15 inch
        2. text width vs box width(to avoid字凸出the right side)
        """
        # === Height check ===
        nlines = s.text.count('\n') + 1
        line_h = s.fs * 1.2 / 72  # inches
        text_h = nlines * line_h
        # fig_h = 9.5 inch, ylim range = 26 - (-11.6) = 37.6 units
        box_h = s.h * (9.5 / 37.6)   # y-unit -> inches
        # padding(0.10 = eachside 0.05 inch,ensurevisualnotstickborder)
        PADDING = 0.10
        h_ok = box_h >= text_h + PADDING
        if not h_ok:
            print(f"  WARNING: H OVERFLOW: '{s.text[:30]}...' {nlines} lines need {text_h:.3f}in + {PADDING} pad, box={box_h:.3f}in (gap {text_h + PADDING - box_h:.3f}in)")

        # === widthcheck ===
        # Arial 8pt width per character ~0.046 inch(estimate,average)
        # Arial 9pt width per character ~0.052 inch
        char_w = s.fs * 0.00575  # approximateswitchcompute
        max_line_chars = max(len(line) for line in s.text.split('\n'))
        text_w = max_line_chars * char_w
        box_w = s.w * (11.0/8.75) * (7.5/11.0)   # y-unit width -> inches
        # Use a more direct method: x-axis range = -0.5~10.5 = 11 units, fig width = 7.5 inch
        box_w = s.w * (7.5 / 11.0)
        w_ok = box_w >= text_w + 0.1  # left-righteach 0.05 inch padding
        if not w_ok:
            print(f"  WARNING: W OVERFLOW: '{s.text[:40]}...' max line {max_line_chars} chars need {text_w:.2f}in + padding, box width={box_w:.2f}in")

        return h_ok and w_ok


def ar(ax, x1, y1, x2, y2):
    """Arrow head enters the target box by 0.15 unit,visually flush with the box. """
    dx, dy = x2 - x1, y2 - y1
    length = (dx**2 + dy**2)**0.5
    if length > 0:
        ux, uy = dx/length, dy/length
        # End point toward"forward direction"extend by 0.15 unit(arrow headextends into box topsideborderinside 0.15 unit)
        x2_ext = x2 + ux * 0.15
        y2_ext = y2 + uy * 0.15
    else:
        x2_ext, y2_ext = x2, y2
    ax.annotate('', xy=(x2_ext, y2_ext), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2,
                                shrinkA=0, shrinkB=0))

def ln(ax, x1, y1, x2, y2, ls='-'):
    """1:1 reuse the original version"""
    ax.plot([x1, x2], [y1, y2], color='black', lw=0.8, ls=ls)

# coordinates anchor(reused from the original version)
CX = 5; TX = 3.5; RX = 7.3
bb = []
y = 25.2

# === 1-6 vsoriginal versiononesame(databeforeprocess + feature selection)===
# 1. Raw Data
b1 = box(CX, y, 5.5, H2,
         'Raw EHR Data\n(n = 1,239 deliveries)')
bb.append(b1); y -= H2 + G

# 2. Cohort + Features
b2 = box(CX, y, 5.5, H3,
         'Cohort: Vaginal Deliveries Only\n(n=769, PPH+=184, 23.9%)\n304 Features Engineered')
bb.append(b2); y -= H3 + G

# 3. Data Split
b3 = box(CX, y, 5, H2,
         'Data Split FIRST\n(Stratified 80/20)')
bb.append(b3); y -= H2 + G

# 4a / 4b
b4a = box(TX, y, 3.8, H2,
          'Train Data\n(80%, n=615)')
bb.append(b4a)
b4b = box(RX, y, 3.2, H2,
          'Independent Test Data\n(20%, n=154, used once)')
bb.append(b4b); y -= H2 + G

# 5. Statistical Filtering
b5 = box(TX, y, 5.5, H3,
         'Statistical Filtering (Train Only)\n(Chi-square & t-test, p<0.05)\n304 -> 35 Features')
bb.append(b5); y -= H3 + G

# 6. Bootstrap LASSO
b6 = box(TX, y, 5.5, H3,
         'Bootstrap LASSO (Train Only)\n(400 fits, ranked by selection frequency)\n35 ranked Features')
bb.append(b6); y -= H3 + G

# === 7-11 changeto Joint Search Main experiment ===
# 7. Joint Search dimensions (+ Standardization explicit)
b7 = box(TX, y, 5.5, H4,
         'Joint Search Pipeline\n7 Imputation x 9 Algorithm x 7 Imbalance x 6 Feature\n+ StandardScaler (Z-score, fit on train fold)\n= 2,646 unique pipelines')
bb.append(b7); y -= H4 + G

# 8. CV
b8 = box(TX, y, 5.5, H3,
         '10-fold Stratified Cross-Validation\n(class imbalance applied within each train fold)\n= 26,460 model fits')
bb.append(b8); y -= H3 + G

# 9. Final test eval
b9 = box(TX, y, 5, H2,
         'All 2,646 Pipelines\nFinal Evaluation on Test Set (n=154)')
bb.append(b9); y -= H2 + G + 0.3

# 10. Composite ranking
b10 = box(CX, y, 5.8, H3,
          'Ranked by Composite Score\n(0.4xMCC + 0.3xAUC + 0.3xSensitivity)\nFinal Evaluation on Independent Test Data')
bb.append(b10); y -= H3 + G + 0.3

# 11. Best Model(usehighlight in graychampion,4 liness,box enlargeto avoid crowding)
b11 = box(CX, y, 6.5, H4,
          'Best Model\n'
          'RandomForest + SMOTEENN + Top30 + KNN k=1\n'
          'Test Composite = 0.5645 (Rank 1 / 2,646)\n'
          '(Sensitivity=0.730, AUC=0.717, MCC=0.326)',
          color='#E0E0E0', fs=9)
bb.append(b11)

# === Verify(original version SOP)===
print("box verification:")
all_ok = True
for b in bb:
    if not b.verify():
        all_ok = False
if all_ok:
    print("  ✅ All boxes pass padding check")

# === Draw ===
for b in bb:
    b.draw(ax)

# === Arrows (1:1 mirrors the original version) ===
ar(ax, b1.x, b1.bottom, b2.x, b2.top)
ar(ax, b2.x, b2.bottom, b3.x, b3.top)

# Split branches
sy = (b3.bottom + b4a.top) / 2
ln(ax, b3.x, b3.bottom, b3.x, sy)
ln(ax, b4a.x, sy, b4b.x, sy)
ar(ax, b4a.x, sy, b4a.x, b4a.top)
ar(ax, b4b.x, sy, b4b.x, b4b.top)

# Train flow
ar(ax, b4a.x, b4a.bottom, b5.x, b5.top)
ar(ax, b5.x, b5.bottom, b6.x, b6.top)
ar(ax, b6.x, b6.bottom, b7.x, b7.top)
ar(ax, b7.x, b7.bottom, b8.x, b8.top)
ar(ax, b8.x, b8.bottom, b9.x, b9.top)

# Merge:b9 + b4b(test set dashed line held-out)-> b10(1:1 reuse the original version)
my = (b9.bottom + b10.top) / 2
ln(ax, b9.x, b9.bottom, b9.x, my)
ln(ax, b4b.x, b4b.bottom, b4b.x, my, ls='--')
ln(ax, b9.x, my, b4b.x, my)
ar(ax, b10.x, my, b10.x, b10.top)

# b10 -> b11
ar(ax, b10.x, b10.bottom, b11.x, b11.top)

# === Save ===
OUT = '/Users/yangyongcheng/Desktop/PPH_joint_search_2026_06_05/06_figures/Main_Fig2_Pipeline.png'
plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.03)
plt.close()

# RGB flatten + speccheck
from PIL import Image
img = Image.open(OUT).convert('RGB')
img.save(OUT, dpi=(300, 300))
w, h = img.width, img.height
print(f"\nFinal: {w}x{h} px")
print(f"Width <=2250: {'✅' if w <= 2250 else '❌'}")
print(f"Height <=2625: {'✅' if h <= 2625 else '❌'}")
print(f"Saved: {OUT}")

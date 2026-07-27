#!/usr/bin/env python3
"""
Fig2.png — Joint Search 主實驗版本 (2026-06-22 v2 改版)
=====================================================================
白底黑框 + 水平/垂直直角箭頭。對齊 PLOS Digital Health 規範:
- 圖檔內無 figure number / title / caption / footnote / 解釋性文字
  (caption 一律放正文,圖內只有 flowchart box 文字 = 視覺數據)
- Arial 8pt (rcParams) / 9pt (b11 best model)
- 300 DPI / Width ≤2250 px / Height ≤2625 px / RGB

2026-06-22 v2 改動 (用戶反饋三點):
1. Joint Search box 每維度加括號代表例 (reviewer 看得懂 4 個維度在搜什麼)
2. Train 流向明示:b4a 標 "→ all subsequent steps use training set only";
   各 train box 維持 "(Train set)" 標記
3. Test 路徑明示:b4b 標明未參與訓練/選特徵 + 虛線標 final evaluation and model ranking
   + b9 標 "trained on full training set (n = 615), evaluated on test set (n = 154)"
   (原版 test 只有孤零零 n=154,現在 train/test 對稱清楚)

2026-07-26 誠實化改動 (與正文段 55／108 的 selection-on-test 揭露對齊,兩處):
- b4b 第 3 行: "not used in model development" → "not used for training or feature selection"
- 虛線標籤: "used once for final evaluation" → "used for final evaluation and model ranking"
  原因:最終模型是全 2,646 個 pipeline 依 test composite 排名第 1 選出的(見 b10/b11 的
  "Rank 1 / 2,646"),舊寫法與同圖下方及正文自相矛盾。姊妹圖 S3_fig 由
  generate_internal_flowchart.py 同步改同兩處,兩張圖必須一起重生。
4. 修正存檔路徑舊檔名 Main_Fig2_Pipeline.png → Fig2.png (6/11 PLOS DH 改名)
5. 壓回 PLOS 高度上限:figsize 9.5→8.7 inch (2610px ≤ 2625),用 ylim 容納維度展開

舊檔名沿用紀錄: 06_figures/Fig2.png (PLOS DH 規範,無底線描述符)
"""
import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8

FIG_H = 9.7   # inch (給足空間讓箭頭間距正常;末段保底縮放確保最終 ≤2625 px)
fig, ax = plt.subplots(figsize=(7.5, FIG_H))
ax.set_xlim(-0.5, 10.5)
ax.axis('off')

# === 2026-07-26 新增:真實尺度與文字實測寬度(供 Box.verify 用) ===
# 🚨 1 x-unit 的真實英吋數必須從「軸的實際寬度」量,不能用 figsize/xrange。
#    figsize 是 7.5 in,但 subplots 預設邊界讓軸只佔 ~5.812 in → 1 unit = 0.5284 in(不是 0.6818)。
#    舊 verify 用 7.5/11 高估框寬 29%,導致 b4b 真的爆框卻檢查通過。
fig.canvas.draw()
_RENDERER = fig.canvas.get_renderer()
UNIT_IN = (ax.get_window_extent(renderer=_RENDERER).width / fig.dpi) / 11.0

def _text_w_in(s, fs):
    """用 renderer 實測單行文字寬度(英吋)。Arial 是比例字型,不可用字元數 × 固定字寬估算。"""
    t = fig.text(0, 0, s, fontsize=fs)
    w = t.get_window_extent(renderer=_RENDERER).width / fig.dpi
    t.remove()
    return w

# 🔑 關鍵:IN_PER_UNIT 是「固定常數」,不從 ylim 反算 → box 高度永遠固定,不隨 ylim 浮動。
#    (之前 bug:調低 YLO → YRANGE 變大 → IN_PER_UNIT 變小 → box 變高 → 又掉出 ylim,無限追)
IN_PER_UNIT = 0.232      # 每 y-unit 對應的 inch (固定。設小 → YRANGE 大 → 畫布容得下全部內容)
# ylim 的 YRANGE 必須鎖死 = FIG_H / IN_PER_UNIT,box 渲染 inch 高度才等於 box_h() 算的值
YRANGE = FIG_H / IN_PER_UNIT   # 34.52 units (固定)
# 由「行數 × 行高 + padding」用 inch 精算反推 box 高度 (y-units),數學保證不爆框
PAD_IN = 0.20            # 上下總 padding (inch),每邊 0.10
def box_h(nlines, fs=8):
    line_h = fs * 1.2 / 72          # inch
    need_in = nlines * line_h + PAD_IN
    return need_in / IN_PER_UNIT     # → y-units

H2 = box_h(2)   # 2-line box
H3 = box_h(3)   # 3-line box
H4 = box_h(4, fs=9)   # 4-line box (best model 用 9pt)
H4b = box_h(4)  # 4-line box @8pt (b4b test set:誠實化後第 3-4 行拆兩行,見下)
H5 = box_h(5)   # 5-line box (joint search 維度展開)
G  = 1.05       # arrow gap (箭頭舒適間距)


class Box:
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
        """高度 + 寬度雙檢查 (避免字裂開 / 凸出右邊)。"""
        nlines = s.text.count('\n') + 1
        line_h = s.fs * 1.2 / 72  # inches
        text_h = nlines * line_h
        box_h = s.h * (FIG_H / YRANGE)   # y-unit -> inches
        PADDING = 0.10
        h_ok = box_h >= text_h + PADDING
        if not h_ok:
            print(f"  ⚠️ H OVERFLOW: '{s.text[:34]}...' {nlines} lines need {text_h:.3f}in + {PADDING} pad, box={box_h:.3f}in (差 {text_h + PADDING - box_h:.3f}in)")
        # 寬度
        # 🚨 2026-07-26 修兩個 bug(這兩個 bug 合起來放過了 b4b 的實際爆框):
        #   bug 1 尺度錯:原用 s.w * (7.5/11.0) 當框寬,但 7.5 是 figsize 不是軸寬。
        #        bbox_inches='tight' 後實際軸寬只有 ~5.812 in → 1 unit = 0.5284 in,不是 0.6818。
        #        原式高估框寬 29%,等於檢查形同虛設。
        #   bug 2 字寬用字元數估算:等寬假設對 Arial(比例字型)不成立,'i' 與 'W' 差 3 倍以上。
        #   對策:改用 renderer 實測真實像素寬(與姊妹腳本 generate_internal_flowchart.py 同一做法)。
        text_w = max(_text_w_in(line, s.fs) for line in s.text.split('\n'))
        box_w = s.w * UNIT_IN
        w_ok = box_w >= text_w + 0.10
        if not w_ok:
            print(f"  ⚠️ W OVERFLOW: '{s.text[:40]}...' 最長行實測 {text_w:.3f}in + 0.10 pad > box {box_w:.3f}in (差 {text_w + 0.10 - box_w:.3f}in)")
        return h_ok and w_ok


def ar(ax, x1, y1, x2, y2):
    """箭頭頭部進入目標 box 0.15 unit,視覺貼框無縫。"""
    dx, dy = x2 - x1, y2 - y1
    length = (dx**2 + dy**2)**0.5
    if length > 0:
        ux, uy = dx/length, dy/length
        x2_ext = x2 + ux * 0.15
        y2_ext = y2 + uy * 0.15
    else:
        x2_ext, y2_ext = x2, y2
    ax.annotate('', xy=(x2_ext, y2_ext), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2,
                                shrinkA=0, shrinkB=0))

def ln(ax, x1, y1, x2, y2, ls='-'):
    ax.plot([x1, x2], [y1, y2], color='black', lw=0.8, ls=ls)


# 座標 anchor
CX = 5; TX = 3.5; RX = 7.7
bb = []
y = 25.0

# === 1-4 資料前處理 + split ===
b1 = Box(CX, y, 5.5, H2,
         'Raw EHR data\n(n = 1,239 deliveries)')
bb.append(b1); y -= H2 + G

b2 = Box(CX, y, 5.5, H3,
         'Vaginal delivery cohort\n(n = 769, PPH+ = 184, 23.9%)\n304 candidate features')
bb.append(b2); y -= H3 + G

b3 = Box(CX, y, 5, H2,
         'Stratified 80/20 split\n(performed before feature selection)')
bb.append(b3); y -= H2 + G

# 4a Train / 4b Test (兩邊文字標籤明示後續流向)
b4a = Box(TX, y, 4.2, H3,
          'Training set\n(80%, n = 615)\nused for all steps below')
bb.append(b4a)
# 2026-07-26 誠實化:原第 3 行寫 'not used in model development' 與 b10/b11(依 test composite
# 排名、Rank 1 / 2,646)及正文段 55／108 的 selection-on-test 揭露互相矛盾 → 改為只主張真正成立的
# 「沒參與訓練與特徵選擇」。
# 🚨 必須拆成兩行:實際軸寬只有 5.812 in(bbox_inches='tight' 後),1 unit = 0.5284 in,
#    所以 3.6-unit 框真實寬度僅 1.902 in,而單行 'not used for training or feature selection'
#    要 1.962 in → 會凸出左右框線(2026-07-26 首次重生時目視抓到)。
#    拆兩行後最長行 'not used for training' 僅 0.986 in,餘裕充足。框高改用 H4b(4 行 @8pt)。
#    ⚠️ 加寬到 4.2 unit 不可行:左緣會撞到 b4a 右緣(皆為 5.6)。
# 🚨 2026-07-27 修既有版面 bug:b4a 是 3 行(H3)、b4b 是 4 行(H4b),原本兩者共用同一個中心 y,
#    導致較高的 b4b 頂緣高於 b4a 頂緣。而分支橫線 sy 是用 b4a.top 算的
#    → 橫線直接切過 b4b 上緣,且 ar(b4b.x, sy → b4b.top) 變成反向往上的箭頭,視覺很亂。
#    正解:讓兩個分支框「頂緣對齊」(CONSORT 分支慣例) → b4b 中心下移 (H3 - H4b)/2。
#    這樣 b4a.top == b4b.top,橫線落在 b3 與兩框之間的空隙,兩支箭頭都乾淨向下。
b4b = Box(RX, y + (H3 - H4b) / 2, 3.6, H4b,
          'Independent test set\n(20%, n = 154)\nnot used for training\nor feature selection')
bb.append(b4b)
assert abs(b4a.top - b4b.top) < 1e-9, f"分支框頂緣未對齊:b4a.top={b4a.top} b4b.top={b4b.top}"
y -= H3 + G

# === 5-8 Train-only feature selection + joint search ===
b5 = Box(TX, y, 6.0, H3,
         'Univariate statistical filtering (training set only)\n(chi-square & t-test, p < 0.05)\n304 → 35 features')
bb.append(b5); y -= H3 + G

b6 = Box(TX, y, 6.0, H3,
         'Bootstrap LASSO (training set only)\n(400 fits, ranked by selection frequency)\n35 ranked features')
bb.append(b6); y -= H3 + G

# 7. Joint Search 維度展開 (5 行,每維度括號代表例)
b7 = Box(TX, y, 6.0, H5,
         'Joint search = 2,646 unique pipelines\n'
         '7 imputation (KNN k=1–10, MICE, median)\n'
         '× 9 algorithm (Random Forest, XGBoost, SVM, …)\n'
         '× 7 imbalance (SMOTEENN, SMOTE, ADASYN, …)\n'
         '× 6 feature subset + StandardScaler')
bb.append(b7); y -= H5 + G

# 8. CV (train fold 內)
# 🚨 2026-07-27 修既有版面 bug:本框是 4 行文字,原本卻套 3 行高的 H3
#    → Box.verify 一直在報 H OVERFLOW 0.033in,實際渲染就是最後一行
#    「= 26,460 model fits」壓在下框線上。改用 H4b(4 行 @8pt)即解決。
b8 = Box(TX, y, 6.0, H4b,
         '10-fold stratified cross-validation\n(imputation & scaling fit on the full training set;\nimbalance resampling within each train fold)\n= 26,460 model fits')
bb.append(b8); y -= H4b + G

# 9. Final test eval (明示 train→test 銜接)
# 2026-07-27 誠實化補漏:原寫 'evaluated once on the independent test set'。
# 「once」暗示「測試集只被用過一次」,但實際是 2,646 個 pipeline 全部在 test 上評估後
# 才依 composite 挑出冠軍(見 b10)。2026-07-26 誠實化只改了虛線標籤(見下方 b4b 標籤),
# 漏改此處,導致與姊妹圖 S3_fig 打架(S3 的 b_f1 已是 'Evaluated on the independent test set')。
# 移除 once 後與 S3 用字一致,且行寬變短不影響版面。
b9 = Box(TX, y, 6.0, H3,
         'All 2,646 pipelines\nre-trained on full training set (n = 615),\nevaluated on the independent test set (n = 154)')
bb.append(b9); y -= H3 + G + 0.25

# === 10-11 ranking + best model (中央) ===
b10 = Box(CX, y, 6.0, H3,
          'Ranked by composite score\n(0.4×MCC + 0.3×AUC + 0.3×Sensitivity)\non the independent test set')
bb.append(b10); y -= H3 + G + 0.25

b11 = Box(CX, y, 6.6, H4,
          'Final model\n'
          'Random Forest + SMOTEENN + Top30 + KNN k=1\n'
          'Test Composite = 0.5645 (Rank 1 / 2,646)\n'
          '(Sensitivity = 0.730, AUC = 0.717, MCC = 0.326)',
          color='#E0E0E0', fs=9)
bb.append(b11)

# === Verify ===
print("Box verification:")
all_ok = True
for b in bb:
    if not b.verify():
        all_ok = False
if all_ok:
    print("  ✅ All boxes pass padding check")

# 設 ylim:範圍鎖死 = YRANGE (固定),頂部貼 b1 上框留 0.6,底部自然 = YHI - YRANGE。
# 因 YRANGE 固定,box 渲染 inch 高度 = box_h() 算的值 → 字不爆框;只要 lowest 在範圍內就不裁。
YHI = b1.top + 0.6
YLO = YHI - YRANGE
ax.set_ylim(YLO, YHI)
lowest = b11.bottom
margin = lowest - YLO
print(f"YRANGE={YRANGE:.2f} | ylim=({YLO:.2f}, {YHI:.2f}) | lowest box bottom={lowest:.2f}")
print(f"底部餘量 = {margin:.2f} units {'✅ 不裁切' if margin >= 0.3 else '❌ 餘量不足,內容太高需精簡'}")
print(f"IN_PER_UNIT={IN_PER_UNIT} | H2={H2:.2f} H3={H3:.2f} H4={H4:.2f} H5={H5:.2f}")

# === Draw ===
for b in bb:
    b.draw(ax)

# 底部 / 頂部留白:畫實際橫跨整寬的白色細線 (看不見) 強制撐開繪圖範圍,
# 確保 bbox_inches='tight' 一定含完整上下框 + 白邊 (2026-06-22 修,加大到 1.2 unit)
ax.plot([-0.5, 10.5], [b11.bottom - 1.2, b11.bottom - 1.2], color='white', lw=0.1)
ax.plot([-0.5, 10.5], [b1.top + 0.6, b1.top + 0.6], color='white', lw=0.1)

# === Arrows ===
ar(ax, b1.x, b1.bottom, b2.x, b2.top)
ar(ax, b2.x, b2.bottom, b3.x, b3.top)

# Split 分支
sy = (b3.bottom + b4a.top) / 2
ln(ax, b3.x, b3.bottom, b3.x, sy)
ln(ax, b4a.x, sy, b4b.x, sy)
ar(ax, b4a.x, sy, b4a.x, b4a.top)
ar(ax, b4b.x, sy, b4b.x, b4b.top)

# Train 流線
ar(ax, b4a.x, b4a.bottom, b5.x, b5.top)
ar(ax, b5.x, b5.bottom, b6.x, b6.top)
ar(ax, b6.x, b6.bottom, b7.x, b7.top)
ar(ax, b7.x, b7.bottom, b8.x, b8.top)
ar(ax, b8.x, b8.bottom, b9.x, b9.top)

# Merge: b9 + b4b (test set 虛線 held-out, used once) → b10
my = (b9.bottom + b10.top) / 2
ln(ax, b9.x, b9.bottom, b9.x, my)
ln(ax, b4b.x, b4b.bottom, b4b.x, my, ls='--')
ln(ax, b9.x, my, b4b.x, my)
ar(ax, b10.x, my, b10.x, b10.top)
# 虛線中段標籤
# PLOS DH: "Use only Arial, Times, or Symbol font in 8-12 point" → 下限 8pt,與 S3 姊妹圖一致
# 2026-07-26 誠實化:原寫 'used once for final evaluation' 隱含「最後才拿已定案模型驗一次」,
# 實際是 2,646 個 pipeline 都在 test 上評估、再從中挑冠軍 → 改為明講 model ranking。
# 維持兩行(最長行 1.176 in < 右側可用 1.807 in);併成一行會是 2.124 in,會凸出右邊界。
ax.text(b4b.x + 0.15, (b4b.bottom + my) / 2, 'used for final evaluation\nand model ranking',
        ha='left', va='center', fontsize=8, style='italic', color='black')

# b10 → b11
ar(ax, b10.x, b10.bottom, b11.x, b11.top)

# === Save ===
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, '06_figures', 'Fig2.png')
plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.08)
plt.close()

from PIL import Image
img = Image.open(OUT).convert('RGB')
# 保底:若像素超 PLOS 上限 (寬 2250 / 高 2625),等比例縮到合規 (字相對比例不變,不會爆框)
MAXW, MAXH = 2250, 2625
scale = min(MAXW / img.width, MAXH / img.height, 1.0)
if scale < 1.0:
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    print(f"等比例縮放 {scale:.3f} 以守住 PLOS 上限")
img.save(OUT, dpi=(300, 300))
w, h = img.width, img.height
print(f"\nFinal: {w}x{h} px")
print(f"Width ≤2250: {'✅' if w <= 2250 else '❌'}")
print(f"Height ≤2625: {'✅' if h <= 2625 else '❌'}")
print(f"Saved: {OUT}")

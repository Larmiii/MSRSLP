"""CSL-Daily 版 6 张深度可视化论文图 (输出 fig1_csl-fig6_csl 到 figures/paper/csl/)。

数据源：
  poses_3d : data/csl/csl_daily_lift3d.dev.pt                              — GT
  sign     : results/csl_{baseline,M1,M1M2}/slp_pickles/csl_daily.dev      — 生成
  tokens   : checkpoints/csl/tokens/vq_{baseline,M1M2}/dev_tokens.pt

CSL vs PHIX 差异:
  - 码本对称: baseline 4096, M1+M2 六子流均 K=512 (合计 3072)
  - 偏移:    [0, 512, 1024, 1536, 2048, 2560]
  - SLP pickle 命名: csl_daily.dev 而非 dev.pickle
  - 第二语言: 中文 char-level (BT BLEU 数值天然偏低)
"""

from __future__ import annotations
import gzip
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Chinese font
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 11
mpl.rcParams['savefig.dpi'] = 400
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['lines.antialiased'] = True
mpl.rcParams['text.antialiased'] = True

# 仓库根目录（从脚本位置推导，可移植）
ROOT = Path(__file__).resolve().parents[2]
# 图输出目录：默认写到仓库内 figures/paper/csl；生成论文图时用环境变量 SLP_FIG_OUT 覆盖。
OUT  = Path(os.environ.get('SLP_FIG_OUT', str(ROOT / 'figures' / 'paper'))) / 'csl'
OUT.mkdir(parents=True, exist_ok=True)

# CSL 178-kpt 真实布局（经实验诊断，与 PHIX 不同）
IDX_BODY     = list(range(0, 8))      # 8 kpt body 世界坐标
IDX_FACE_AUX = list(range(8, 50))     # 42 kpt face mesh 辅助 (归一化 0-1，与 50-135 重复)
IDX_FACE     = list(range(50, 136))   # 86 kpt face mesh (归一化 0-1，需重锚)
IDX_LH_WORLD = list(range(136, 157))  # 21 kpt 左手 (世界坐标，可直接渲染)
IDX_RH_WORLD = list(range(157, 178))  # 21 kpt 右手 (世界坐标)

# 颜色（统一论文配色）
C_GT    = '#000000'
C_BASE  = '#d62728'
C_M1    = '#1f77b4'
C_MSR   = '#2ca02c'
C_BODY  = '#3d5a80'
C_HAND  = '#ee6c4d'
C_FACE  = '#98c1d9'

# Body bones (8-kpt 子集 L/R shoulder/elbow/wrist + L/R hip)
BODY_BONES = [(0,1),(1,2),(3,4),(4,5),(0,3),(0,6),(3,7),(6,7)]
HAND_BONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

DPI = 400


# ============================================================
# 数据加载
# ============================================================

def load_gt():
    return torch.load(ROOT / 'data/csl/csl_daily_lift3d.dev.pt',
                      map_location='cpu', weights_only=False)

def load_gen(tag: str):
    with gzip.open(ROOT / f'results/csl_{tag}/slp_pickles/csl_daily.dev', 'rb') as f:
        lst = pickle.load(f)
    return {s['name']: s['sign'].numpy().reshape(s['sign'].shape[0], 178, 3) for s in lst}

def load_gt_arr(gt):
    return {k: v['poses_3d'].numpy() for k, v in gt.items()}

def load_tokens(tag: str):
    return torch.load(ROOT / f'checkpoints/csl/tokens/vq_{tag}/dev_tokens.pt',
                      map_location='cpu', weights_only=False)


# ============================================================
# 工具：单帧 stick figure
# ============================================================

def draw_stick(ax, p178, color, alpha=1.0, lw=2.0, body_s=42, hand_s=14,
                face_s=4, draw_face=True):
    """画单帧 stick figure: body 骨架 + 双手（世界坐标 8-49） + 面部（重锚到肩上）。

    PHIX 178-kpt 真实布局：
      - 0-7   body (世界坐标)
      - 8-49  双手 (世界坐标，直接渲染) ← 这是真正的手
      - 50-135 face mesh (局部坐标，需重锚)
      - 136-177 局部手部（与 8-49 重复，忽略）
    """
    p = p178.copy()

    # ---- 重锚面部到肩上方 ----
    sh_x = float((p[0, 0] + p[3, 0]) / 2)
    sh_y = float((p[0, 1] + p[3, 1]) / 2)
    shoulder_w = float(abs(p[3, 0] - p[0, 0]))

    face = p[IDX_FACE, :2].copy()
    face_cx = float(face[:, 0].mean()); face_cy = float(face[:, 1].mean())
    face_w  = float(face[:, 0].max() - face[:, 0].min())
    target_face_w = shoulder_w * 0.55
    scale = target_face_w / max(face_w, 1e-6)
    face = (face - np.array([face_cx, face_cy])) * scale
    head_cy = sh_y - shoulder_w * 0.50
    face[:, 0] += sh_x
    face[:, 1] += head_cy

    if draw_face:
        # 面部点云：小点，让脸部细节（眼/嘴/轮廓）可见但不喧宾夺主
        ax.scatter(face[:, 0], face[:, 1], c=color, s=face_s, alpha=alpha*0.75,
                   zorder=3, edgecolors='none')

    # 颈部连线：面部底端到肩部中点
    ax.plot([sh_x, sh_x], [face[:, 1].max(), sh_y], color=color,
            linewidth=lw*0.9, alpha=alpha*0.85, zorder=2)

    # ---- body 骨架 ----
    x, y = p[:, 0], p[:, 1]
    for a, b in BODY_BONES:
        ax.plot([x[a], x[b]], [y[a], y[b]], color=color, linewidth=lw, alpha=alpha, zorder=3)
    ax.scatter(x[IDX_BODY], y[IDX_BODY], c=color, s=body_s, alpha=alpha, zorder=5,
               edgecolors='white', linewidths=0.8)

    # ---- 双手（直接用世界坐标 8-49，不需重锚也不画 bones，
    #       因 PHIX hand 拓扑不是标准 MediaPipe 21-kpt 顺序，连线会乱）----
    hand_pts = p[IDX_LH_WORLD + IDX_RH_WORLD, :2]
    ax.scatter(hand_pts[:, 0], hand_pts[:, 1], c=color, s=hand_s, alpha=alpha*0.95,
                zorder=5, edgecolors='white', linewidths=0.4)
    # 手腕到 body wrist 连线（用 hand cluster center 近似手腕位置）
    for hand_idx_range, body_wrist_i in [(IDX_LH_WORLD, 2), (IDX_RH_WORLD, 5)]:
        hc_x = float(p[hand_idx_range, 0].mean())
        hc_y = float(p[hand_idx_range, 1].mean())
        ax.plot([p[body_wrist_i, 0], hc_x], [p[body_wrist_i, 1], hc_y],
                color=color, linewidth=lw*0.5, alpha=alpha*0.5, zorder=2, linestyle=':')


# ============================================================
# 1. 骨架序列网格（最关键的图）
# ============================================================

def fig_skeleton_grid(gt_arr, gens):
    """单个样本，GT/Base/Ours 三行 × 6 帧（更大更清晰，面部用 86 点）。"""
    # CSL 按 Ours/Baseline 手部方差比挑选: O/B=5.91, GT=0.022 Base=0.005 Ours=0.028
    name = 'S002370_P0007_T00'
    N_FRAMES = 6

    fig = plt.figure(figsize=(20, 10), facecolor='white')
    outer = fig.add_gridspec(1, 1, left=0.05, right=0.99, top=0.97, bottom=0.03)

    inner = outer[0].subgridspec(3, N_FRAMES, wspace=0.04, hspace=0.10)
    gt = gt_arr[name]
    T_gt = gt.shape[0]
    seqs = [
        ('真实序列',         gt,                     C_GT),
        ('基线',             gens['baseline'][name], C_BASE),
        ('本文方法 (M1+M2)', gens['M1M2'][name],     C_MSR),
    ]
    row_tags = ['(a)', '(b)', '(c)']

    # 坐标范围：根据 body 关键点 + 肩宽估算头部和双手空间
    body_pts = np.concatenate([s[1][:, IDX_BODY, :2].reshape(-1, 2) for s in seqs], axis=0)
    shoulders_x = np.concatenate([s[1][:, [0, 3], 0].reshape(-1, 2) for s in seqs], axis=0)
    sh_w = float(np.abs(shoulders_x[:, 1] - shoulders_x[:, 0]).mean())
    x_min = float(body_pts[:, 0].min()) - sh_w * 1.0
    x_max = float(body_pts[:, 0].max()) + sh_w * 1.0
    y_min = float(body_pts[:, 1].min()) - sh_w * 1.2   # 头部空间（含面部点云）
    y_max = float(body_pts[:, 1].max()) + sh_w * 0.3

    row_bgs = ['#fafafa', '#fdf3f3', '#f1f8f0']
    for r_i, (lbl, seq, col) in enumerate(seqs):
        T = seq.shape[0]
        idx = np.linspace(0, T-1, N_FRAMES).astype(int)
        for c_i, t in enumerate(idx):
            ax = fig.add_subplot(inner[r_i, c_i])
            ax.set_facecolor(row_bgs[r_i])
            draw_stick(ax, seq[t], col, lw=1.6, body_s=45, hand_s=12, face_s=3)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_max, y_min)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect('equal')
            for sp in ax.spines.values():
                sp.set_edgecolor('#bbbbbb'); sp.set_linewidth(0.6)
            if c_i == 0:
                ax.set_ylabel(f'{row_tags[r_i]} {lbl}', fontsize=15, fontweight='bold',
                              color=col, labelpad=10)
            if r_i == 0:
                pct = (t / max(T-1, 1)) * 100
                ax.set_title(f't = {pct:.0f}%', fontsize=11, pad=6, color='#333333')

    out = OUT / 'fig1_csl.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [OK] {out.name}')


# ============================================================
# 2. 手腕 3D + 2D 轨迹
# ============================================================

def fig_trajectory_3d(gt_arr, gens):
    name = 'S002370_P0007_T00'
    gt   = gt_arr[name]
    base = gens['baseline'][name]
    msr  = gens['M1M2'][name]
    T = min(gt.shape[0], base.shape[0], msr.shape[0])
    gt, base, msr = gt[:T], base[:T], msr[:T]

    RH_WRIST = 157  # 右手腕（右手起点）

    fig = plt.figure(figsize=(18, 6.2), facecolor='white')
    gs = fig.add_gridspec(1, 3, wspace=0.32, width_ratios=[1.0, 1.0, 1.35],
                          left=0.06, right=0.98, top=0.96, bottom=0.13)

    series = [
        ('真实序列',         gt[:, RH_WRIST],   C_GT,   '-',  2.4),
        ('基线',             base[:, RH_WRIST], C_BASE, '--', 1.8),
        ('本文方法 (M1+M2)', msr[:, RH_WRIST],  C_MSR,  '-',  2.0),
    ]

    def _start_end(ax, p, c, is3d=False, zproj=None):
        if is3d:
            ax.scatter(p[0, 0], p[0, 2], -p[0, 1], c=c, marker='o', s=90,
                       edgecolors='white', linewidths=1.4, zorder=10)
            ax.scatter(p[-1, 0], p[-1, 2], -p[-1, 1], c=c, marker='s', s=90,
                       edgecolors='white', linewidths=1.4, zorder=10)
        else:
            xx, yy = p[:, 0], p[:, zproj]
            ax.scatter(xx[0], yy[0], c=c, marker='o', s=80,
                       edgecolors='white', linewidths=1.4, zorder=10)
            ax.scatter(xx[-1], yy[-1], c=c, marker='s', s=80,
                       edgecolors='white', linewidths=1.4, zorder=10)

    # x-y 投影
    ax1 = fig.add_subplot(gs[0], facecolor='#fafafa')
    for lbl, p, c, ls, lw in series:
        ax1.plot(p[:, 0], p[:, 1], color=c, ls=ls, linewidth=lw, label=lbl, alpha=0.92)
        _start_end(ax1, p, c, zproj=1)
    ax1.invert_yaxis()
    ax1.set_xlabel('x（左右）', fontsize=11); ax1.set_ylabel('y（上下）', fontsize=11)
    ax1.set_title('(a) x–y 投影', fontsize=12, pad=8)
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.95)
    ax1.grid(alpha=0.3)

    # x-z 投影
    ax2 = fig.add_subplot(gs[1], facecolor='#fafafa')
    for lbl, p, c, ls, lw in series:
        ax2.plot(p[:, 0], p[:, 2], color=c, ls=ls, linewidth=lw, alpha=0.92)
        _start_end(ax2, p, c, zproj=2)
    ax2.set_xlabel('x（左右）', fontsize=11); ax2.set_ylabel('z（前后）', fontsize=11)
    ax2.set_title('(b) x–z 投影', fontsize=12, pad=8)
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.grid(alpha=0.3)

    # 3D（更宽 + 拉开 tick 间距 + 缩短 label）
    ax3 = fig.add_subplot(gs[2], projection='3d', facecolor='white')
    for lbl, p, c, ls, lw in series:
        ax3.plot(p[:, 0], p[:, 2], -p[:, 1], color=c, ls=ls, linewidth=lw, alpha=0.92)
        _start_end(ax3, p, c, is3d=True)
    ax3.set_xlabel('x', fontsize=11, labelpad=8)
    ax3.set_ylabel('z', fontsize=11, labelpad=8)
    ax3.set_zlabel('-y', fontsize=11, labelpad=8)
    ax3.set_title('(c) 3D 视角', fontsize=12, pad=10)
    ax3.view_init(elev=20, azim=-55)
    # 简化 tick: 各轴 3 个刻度即可，避免数字挤一起
    for axis_ in ('x', 'y', 'z'):
        getattr(ax3, f'{axis_}axis').set_major_locator(plt.MaxNLocator(3))
        ax3.tick_params(axis=axis_, labelsize=8, pad=2)
    ax3.grid(True, alpha=0.3)

    out = OUT / 'fig3_csl.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK] {out.name}')


# ============================================================
# 3. 速度曲线（motion collapse 量化）
# ============================================================

def fig_velocity_profile(gt_arr, gens):
    # CSL: 选两个对比强的样本
    names = [
        'S002370_P0007_T00',
        'S001656_P0000_T00',
    ]
    fig, axes = plt.subplots(2, 2, figsize=(17, 9), sharex=False)

    for col_i, name in enumerate(names):
        gt   = gt_arr[name]
        base = gens['baseline'][name]
        msr  = gens['M1M2'][name]
        T = min(gt.shape[0], base.shape[0], msr.shape[0])
        gt, base, msr = gt[:T], base[:T], msr[:T]

        def vel(seq, idx):
            # |Δp| per frame, 平均到选定 kpts
            d = np.linalg.norm(np.diff(seq[:, idx], axis=0), axis=2)  # (T-1, K)
            return d.mean(axis=1)

        # 全身平均
        ax_top = axes[0, col_i]
        for lbl, seq, c, ls in [('真实序列', gt, C_GT, '-'),
                                ('基线', base, C_BASE, '--'),
                                ('本文方法 (M1+M2)', msr, C_MSR, '-')]:
            v = vel(seq, IDX_BODY + IDX_LH_WORLD + IDX_RH_WORLD)
            ax_top.plot(np.arange(len(v)), v, color=c, ls=ls, linewidth=1.6, label=lbl, alpha=0.9)
        ax_top.set_xlabel('帧索引 t', fontsize=10); ax_top.set_ylabel('平均关节速度 |Δp| / 帧', fontsize=10)
        ax_top.set_title(f'({chr(97+col_i)}) 样本 {col_i+1} · 全身', fontsize=11)
        ax_top.grid(alpha=0.25)
        ax_top.legend(fontsize=9, loc='upper left', frameon=True)

        # 仅双手
        ax_bot = axes[1, col_i]
        for lbl, seq, c, ls in [('真实序列', gt, C_GT, '-'),
                                ('基线', base, C_BASE, '--'),
                                ('本文方法 (M1+M2)', msr, C_MSR, '-')]:
            v = vel(seq, IDX_LH_WORLD + IDX_RH_WORLD)
            ax_bot.plot(np.arange(len(v)), v, color=c, ls=ls, linewidth=1.6, label=lbl, alpha=0.9)
            # 平均速度横线
            ax_bot.axhline(v.mean(), color=c, ls=':', linewidth=0.7, alpha=0.5)
        ax_bot.set_xlabel('帧索引 t'); ax_bot.set_ylabel('双手平均关节速度')
        ax_bot.set_title(f'({chr(99+col_i)}) 样本 {col_i+1} · 仅双手 42 kpt', fontsize=11)
        ax_bot.grid(alpha=0.25)
        ax_bot.legend(fontsize=9, loc='upper left', frameon=True)

    fig.tight_layout()
    out = OUT / 'fig2_csl.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK] {out.name}')


# ============================================================
# 4. token 色带（collapse 可视化）
# ============================================================

def fig_token_bands(toks_base, toks_msr):
    name = 'S002370_P0007_T00'

    base = toks_base[name]['tokens']
    msr_flat = toks_msr[name]['tokens']
    T_tok = len(msr_flat) // 6
    msr = msr_flat[:T_tok*6].reshape(T_tok, 6)

    sub_labels = ['身体-基础', '身体-残差', '手部-基础', '手部-残差', '面部-基础', '面部-残差']
    # CSL: 对称 6×512
    sub_K = [512, 512, 512, 512, 512, 512]
    sub_off = [0, 512, 1024, 1536, 2048, 2560]
    K_base = 4096

    # baseline 1 行 + MSR 6 行
    fig = plt.figure(figsize=(17, 9), facecolor='white')
    outer = fig.add_gridspec(2, 1, height_ratios=[1.4, 6.5], hspace=0.30,
                              left=0.14, right=0.97, top=0.96, bottom=0.06)

    # ----- baseline 区块（单行）-----
    ax_b = fig.add_subplot(outer[0])
    # ----- MSR 六子流 -----
    gs_m = outer[1].subgridspec(6, 1, hspace=0.30)
    ax_msr = [fig.add_subplot(gs_m[i]) for i in range(6)]

    def stripe(ax, ids, K, max_T):
        cmap = plt.get_cmap('turbo')
        for t, idv in enumerate(ids):
            color = cmap((idv % K) / K)
            ax.axvspan(t, t+1, color=color, alpha=0.95, lw=0)
        ax.set_xlim(0, max_T)
        ax.set_ylim(0, 1)
        ax.set_yticks([]); ax.set_xticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#888'); sp.set_linewidth(0.5)

    max_T = max(len(base), T_tok)
    stripe(ax_b, base, K_base, max_T)
    uniq_b = len(np.unique(base))
    ax_b.set_ylabel(f'基线\n(单码本)\n|U|={uniq_b}/{K_base}',
                    fontsize=11, rotation=0, ha='right', va='center', labelpad=12,
                    fontweight='bold', color=C_BASE)
    # 小节标识（精简）
    ax_b.set_title('(a) 基线 单码本 (K=4096)', fontsize=11, color='#444',
                    loc='left', pad=8)
    ax_msr[0].set_title('(b) M1+M2 六子流', fontsize=11, color='#444',
                         loc='left', pad=8)

    palette_text = ['#3d5a80','#3d5a80','#ee6c4d','#ee6c4d','#98c1d9','#98c1d9']
    for i in range(6):
        local_ids = msr[:, i] - sub_off[i]
        stripe(ax_msr[i], local_ids, sub_K[i], T_tok)
        uniq = len(np.unique(local_ids))
        ax_msr[i].set_ylabel(f'{sub_labels[i]}\n|U|={uniq}/{sub_K[i]}',
                              fontsize=10, rotation=0, ha='right', va='center',
                              labelpad=12, color=palette_text[i], fontweight='bold')

    # 底部 x 轴刻度
    ax_msr[-1].set_xticks(np.linspace(0, T_tok, 6).astype(int))
    ax_msr[-1].set_xlabel('时间步 t（token 单位，下采样后）', fontsize=11)
    ax_msr[-1].tick_params(axis='x', labelsize=10)

    out = OUT / 'fig4_csl.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [OK] {out.name}')


# ============================================================
# 5. 码本使用热力图 + 排序衰减
# ============================================================

def fig_codebook_heat(toks_base, toks_msr):
    sub_labels = ['身体-基础', '身体-残差', '手部-基础', '手部-残差', '面部-基础', '面部-残差']
    # CSL: 对称 6×512
    sub_K = [512, 512, 512, 512, 512, 512]
    sub_off = [0, 512, 1024, 1536, 2048, 2560]

    # 收集所有 dev 样本的频次
    base_counts = {}
    for k, v in toks_base.items():
        for tid in v['tokens']:
            base_counts[int(tid)] = base_counts.get(int(tid), 0) + 1
    K_base = 4096

    msr_counts = [np.zeros(K, dtype=np.int64) for K in sub_K]
    for k, v in toks_msr.items():
        arr = v['tokens']
        T_tok = len(arr) // 6
        m = arr[:T_tok*6].reshape(T_tok, 6)
        for s in range(6):
            for tid in m[:, s]:
                local = int(tid) - sub_off[s]
                if 0 <= local < sub_K[s]:
                    msr_counts[s][local] += 1

    fig = plt.figure(figsize=(17, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.45, wspace=0.25)

    # 左上：M1+M2 六子码本频次热力图（每行 sort 后归一化）
    ax1 = fig.add_subplot(gs[:, 0])
    # 由于 K 不同（32/512/128），统一可视化为 100 bin
    HM = np.zeros((6, 100))
    for s in range(6):
        freq = msr_counts[s].astype(float)
        order = np.argsort(-freq)
        sorted_freq = freq[order]
        # downsample/upsample 到 100
        if sub_K[s] >= 100:
            step = sub_K[s] // 100
            sampled = sorted_freq[:step*100].reshape(100, step).mean(axis=1)
        else:
            sampled = np.interp(np.linspace(0, sub_K[s]-1, 100),
                                np.arange(sub_K[s]), sorted_freq)
        if sampled.max() > 0:
            sampled = sampled / sampled.max()
        HM[s] = sampled

    im = ax1.imshow(HM, aspect='auto', cmap='viridis', interpolation='nearest')
    ax1.set_yticks(range(6))
    ax1.set_yticklabels([f'{sub_labels[i]}\nK={sub_K[i]}' for i in range(6)], fontsize=10)
    ax1.set_xlabel('code rank（按频次降序，归一化到 100 bin）', fontsize=11)
    ax1.set_title('(a) M1+M2 子码本使用热力图', fontsize=12)
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('使用频次（行内归一化）', fontsize=10)

    # 右上：baseline 单码本排序衰减
    ax2 = fig.add_subplot(gs[0, 1])
    base_freq = np.zeros(K_base, dtype=np.int64)
    for tid, cnt in base_counts.items():
        if 0 <= tid < K_base:
            base_freq[tid] = cnt
    sorted_b = np.sort(base_freq)[::-1]
    nonzero = (sorted_b > 0).sum()
    ax2.plot(np.arange(K_base), sorted_b, color=C_BASE, linewidth=1.5)
    ax2.fill_between(np.arange(K_base), sorted_b, alpha=0.25, color=C_BASE)
    ax2.set_yscale('log')
    ax2.set_xlim(0, K_base)
    ax2.set_xlabel('code rank'); ax2.set_ylabel('使用频次 (log)')
    ax2.set_title(f'(b) 基线 单码本（K={K_base}, 实际使用 {nonzero}）',
                  fontsize=11)
    ax2.grid(alpha=0.3)

    # 右下：六子码本排序衰减对比
    ax3 = fig.add_subplot(gs[1, 1])
    palette = ['#4c72b0', '#94c1e8', '#dd8452', '#ffb084', '#55a868', '#a3d4a4']
    for s in range(6):
        freq = msr_counts[s].astype(float)
        sorted_f = np.sort(freq)[::-1]
        nonzero = (sorted_f > 0).sum()
        ax3.plot(np.arange(len(sorted_f)) / sub_K[s],
                 sorted_f / max(sorted_f[0], 1),
                 color=palette[s], linewidth=1.6,
                 label=f'{sub_labels[s]} ({nonzero}/{sub_K[s]})')
    ax3.set_xlabel('归一化 rank (0~1)'); ax3.set_ylabel('归一化频次')
    ax3.set_title('(c) M1+M2 六子码本排序衰减对比', fontsize=11)
    ax3.legend(fontsize=8, loc='upper right', ncol=2)
    ax3.grid(alpha=0.3)
    ax3.set_xlim(0, 1)

    out = OUT / 'fig5_csl.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK] {out.name}')


# ============================================================
# 6. 误差链路 ladder（GT → VQ ceiling → trans → BT）
# ============================================================

def fig_error_chain():
    """CSL dev BLEU-4 (SLRTP-canonical, char-tokenized).
       中文 char-level BLEU 数值范围比德语 word-level 低许多。"""

    # CSL Trans 数据: 来自 ABLATION_SUMMARY.md CSL DEV (M2 也补上)
    # CSL VQ ceiling: 实测 (GT→VQ encode→decode→BT, SLRTP-canonical, backTranslation_CSL_model)
    #   见 results/csl_vqceil_{baseline,M1,M2,M1M2}/csl_vqceil_*_dev.json
    TRANS   = {'baseline': 0.91, 'M1': 1.88, 'M2': 0.94, 'M1+M2': 3.35}
    VQ_CEIL = {'baseline': 4.21, 'M1': 8.62, 'M2': 3.58, 'M1+M2': 12.03}

    stages = ['VQ 上界\n(真实→VQ→解码→回译)',
              '自由翻译生成\n(文本→token→姿态→回译)']
    x_pos = [0, 1]

    fig = plt.figure(figsize=(18, 7), facecolor='white')
    gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.3], wspace=0.18,
                          left=0.07, right=0.97, top=0.95, bottom=0.13)
    ax = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    series = [
        ('基线 (单码本 4096)',     C_BASE, '--', VQ_CEIL['baseline'], TRANS['baseline']),
        ('M1 (多流)',              C_M1,   '-.', VQ_CEIL['M1'],       TRANS['M1']),
        ('M2 (残差量化)',          '#9467bd', ':', VQ_CEIL['M2'],     TRANS['M2']),
        ('本文方法: M1+M2 (MSR)',  C_MSR,  '-',  VQ_CEIL['M1+M2'],    TRANS['M1+M2']),
    ]

    for lbl, c, ls, vq, tr in series:
        ys = [vq, tr]
        ax.plot(x_pos, ys, color=c, ls=ls, linewidth=2.3, marker='o',
                markersize=9, label=lbl, alpha=0.95)
        for xi, yi in zip(x_pos, ys):
            ax.annotate(f'{yi:.2f}', (xi, yi), xytext=(8, 6),
                        textcoords='offset points', fontsize=9, color=c, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_ylabel('BLEU-4 (SLRTP 字符级, CSL DEV)', fontsize=12)
    ax.set_ylim(0, 13)
    ax.set_xlim(-0.25, 1.25)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=10, frameon=True)
    ax.set_title('(a) 误差链路: VQ 上界 → 自由生成', fontsize=12, color='#444',
                 loc='left', pad=8)

    ax_tbl.axis('off')
    ax_tbl.set_xlim(0, 1); ax_tbl.set_ylim(0, 1)
    ax_tbl.set_title('(b) 翻译损失 = VQ 上界 − 自由生成 (BLEU-4)', fontsize=11, pad=12, color='#444')

    rows = []
    for lbl, c, ls, vq, tr in series:
        short = {'基线 (单码本 4096)':    '基线',
                 'M1 (多流)':             'M1',
                 'M2 (残差量化)':         'M2',
                 '本文方法: M1+M2 (MSR)': 'M1+M2 (本文)'}.get(lbl, lbl[:12])
        rows.append((short, c, vq, tr, vq - tr))

    col_x = [0.04, 0.36, 0.58, 0.80]
    headers = ['模型', 'VQ 上界', '自由生成', '翻译损失']
    row_h = 0.12
    top_y = 0.84
    for cx, h in zip(col_x, headers):
        ax_tbl.text(cx, top_y, h, fontsize=11, fontweight='bold',
                    ha='left', va='center', color='#222')
    ax_tbl.plot([0.02, 0.98], [top_y - row_h*0.45]*2, color='#666', linewidth=1.0)
    for i, (short, c, vq, tr, gap) in enumerate(rows):
        y = top_y - (i + 1) * row_h
        bg_color = '#d9efd5' if '本文' in short else ('#f9f9f9' if i % 2 == 0 else 'white')
        ax_tbl.add_patch(plt.Rectangle((0.02, y - row_h*0.45), 0.96, row_h*0.85,
                                        facecolor=bg_color, edgecolor='none', zorder=0))
        ax_tbl.text(col_x[0], y, short, fontsize=11, ha='left', va='center',
                     color=c, fontweight='bold')
        ax_tbl.text(col_x[1], y, f'{vq:.2f}', fontsize=11, ha='left', va='center', family='monospace')
        ax_tbl.text(col_x[2], y, f'{tr:.2f}', fontsize=11, ha='left', va='center', family='monospace')
        ax_tbl.text(col_x[3], y, f'{gap:.2f}', fontsize=11, ha='left', va='center', family='monospace',
                     fontweight='bold')

    ax_tbl.text(0.5, 0.08,
                '两列均为 CSL-Daily 数据集通过 SLRTP 评估模型评价的验证集实测结果 (字符级 BLEU-4)。\n'
                'VQ 上界 = 真实姿态经量化器编解码后回译。',
                fontsize=10, ha='center', va='center', color='#555', style='italic',
                bbox=dict(boxstyle='round,pad=0.5', fc='#f5f5f5', ec='#bbbbbb'))

    out = OUT / 'fig6_csl.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK] {out.name}')


# ============================================================
# main
# ============================================================

def main():
    print('Loading data...')
    gt    = load_gt()
    gt_arr = load_gt_arr(gt)
    gens  = {tag: load_gen(tag) for tag in ['baseline', 'M1', 'M1M2']}
    toks_base = load_tokens('baseline')
    # CSL: M1M2 token dir 已经是 interleaved 格式 (162 = 27 × 6)
    toks_msr  = load_tokens('M1M2')
    print('Data loaded.')

    fig_skeleton_grid(gt_arr, gens)
    fig_trajectory_3d(gt_arr, gens)
    fig_velocity_profile(gt_arr, gens)
    fig_token_bands(toks_base, toks_msr)
    fig_codebook_heat(toks_base, toks_msr)
    fig_error_chain()
    print('All figures written to', OUT)


if __name__ == '__main__':
    main()

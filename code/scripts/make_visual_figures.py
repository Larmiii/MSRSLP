"""生成论文中真正需要可视化才能讲清的图（非 table-replaceable）。

A. fig_keypoint_layout.png   — 178 关键点按部位分流着色（解释 M1 动机）
B. fig_token_structure.png   — MSR token 序列交错结构示意图
C. fig_pose_compare.png      — GT / Baseline / M1+M2 多帧 stick figure 对比
D. fig_wrist_trajectory.png  — 右手腕 xyz 时序轨迹对比
E. fig_codebook_usage.png    — 6 子流码本利用率热力图
"""
from __future__ import annotations
import argparse, gzip, pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

for cn in ['SimHei', 'Microsoft YaHei', 'SimSun']:
    if cn in {f.name for f in fm.fontManager.ttflist}:
        plt.rcParams['font.sans-serif'] = [cn, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 11, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'legend.fontsize': 10,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})

# 三流颜色 (跟 fig5/6 训练曲线一致)
C_BODY = '#1f77b4'   # 蓝
C_FACE = '#ff7f0e'   # 橙
C_HAND = '#2ca02c'   # 绿
C_GT   = '#888888'
C_BASE = '#888888'
C_MSR  = '#2ca02c'

# 178 关键点划分
IDX_BODY = list(range(0, 8))
IDX_FACE = list(range(8, 136))
IDX_LH   = list(range(136, 157))
IDX_RH   = list(range(157, 178))

# MediaPipe Hand 拓扑 (21 keypoint, 20 bone)
HAND_BONES_LOCAL = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
# 推测 body 连接 (Pose 8-kpt 子集，相邻索引连接, 这是经验性的)
BODY_BONES = [(i, i + 1) for i in range(7)]


# ============================================================
# A. 关键点分流示意图
# ============================================================

def fig_keypoint_layout(out_dir: Path, sample_pose: np.ndarray):
    """单帧 178 keypoint 着色 (body/face/hand 三色) + 划分饼图统计。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5),
                                gridspec_kw={'width_ratios': [1.5, 1]})

    # 左：3D 散点 (按 y 上下 + x 左右 投影)
    ax = axes[0]
    x = sample_pose[:, 0]
    y = sample_pose[:, 1]  # 不翻转

    ax.scatter(x[IDX_FACE], y[IDX_FACE], c=C_FACE, s=10, alpha=0.75,
                label=f'面部 face ({len(IDX_FACE)} 点)', zorder=3)
    ax.scatter(x[IDX_BODY], y[IDX_BODY], c=C_BODY, s=80, alpha=0.95,
                edgecolors='black', linewidths=0.5,
                label=f'躯干 body ({len(IDX_BODY)} 点)', zorder=5)
    ax.scatter(x[IDX_LH + IDX_RH], y[IDX_LH + IDX_RH], c=C_HAND, s=28, alpha=0.85,
                edgecolors='black', linewidths=0.3,
                label=f'手部 hand ({len(IDX_LH) + len(IDX_RH)} 点)', zorder=4)
    # Hand 拓扑连接 (MediaPipe Hand 21-kpt)
    for off in [136, 157]:
        for a, b in HAND_BONES_LOCAL:
            ax.plot([x[off + a], x[off + b]], [y[off + a], y[off + b]],
                     color=C_HAND, linewidth=1.0, alpha=0.5, zorder=2)

    ax.set_aspect('equal')
    # 留出 y 上方空间放 legend, 避免跟散点重叠
    cur_ylim = ax.get_ylim()
    ax.set_ylim(cur_ylim[0], cur_ylim[1] + (cur_ylim[1] - cur_ylim[0]) * 0.22)
    ax.legend(loc='upper left', frameon=True, fontsize=10,
               bbox_to_anchor=(0.0, 1.0))
    ax.set_xlabel('x（左右）'); ax.set_ylabel('y（上下）')
    ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('178 关键点单帧投影（示例样本）', fontsize=11)

    # 右：按部位的码本分配饼图 (PHIX M1+M2 配置)
    ax2 = axes[1]
    sizes_kpt = [len(IDX_BODY), len(IDX_FACE), len(IDX_LH) + len(IDX_RH)]
    sizes_codes = [64, 256, 1024]   # PHIX M1 hand-priority 码本
    labels = ['身体 body\n8 点', '面部 face\n128 点', '手部 hand\n42 点']
    colors_pie = [C_BODY, C_FACE, C_HAND]

    # 双层环形：内圈 = 关键点比例，外圈 = 码本比例
    inner_size = 0.35
    outer_size = 0.25
    # 外圈：码本
    wedges1, _ = ax2.pie(sizes_codes, radius=1.0, colors=colors_pie,
                            wedgeprops=dict(width=outer_size, edgecolor='white', linewidth=1.5),
                            startangle=90, counterclock=False)
    # 内圈：关键点
    wedges2, _ = ax2.pie(sizes_kpt, radius=1.0 - outer_size - 0.02, colors=colors_pie,
                            wedgeprops=dict(width=inner_size, edgecolor='white', linewidth=1.5),
                            startangle=90, counterclock=False)
    # 内圈百分比
    for w, n, lab in zip(wedges2, sizes_kpt, labels):
        ang = (w.theta1 + w.theta2) / 2 * np.pi / 180
        r = (1.0 - outer_size - 0.02 - inner_size / 2)
        ax2.text(r * np.cos(ang), r * np.sin(ang),
                  f'{n / sum(sizes_kpt) * 100:.0f}%',
                  ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    # 外圈百分比
    for w, n in zip(wedges1, sizes_codes):
        ang = (w.theta1 + w.theta2) / 2 * np.pi / 180
        r = 1.0 - outer_size / 2
        ax2.text(r * np.cos(ang), r * np.sin(ang),
                  f'{n / sum(sizes_codes) * 100:.0f}%',
                  ha='center', va='center', fontsize=10, fontweight='bold')

    ax2.text(0, 0, '内圈：关键点占比\n外圈：码本占比\n（手优先 hand-priority）',
              ha='center', va='center', fontsize=9.5)
    ax2.set_title('PHIX 上的码本分配（M1+M2 配置）', fontsize=11)

    fig.tight_layout()
    fig.savefig(out_dir / 'fig_keypoint_layout.png')
    plt.close(fig)
    print('  [OK] fig_keypoint_layout.png')


# ============================================================
# B. MSR token 交错序列结构
# ============================================================

def fig_token_structure(out_dir: Path):
    """schematic: 显示 6 substream 如何按时间交错为一条 token 序列。"""
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.5),
                                gridspec_kw={'height_ratios': [1.4, 1.8],
                                              'hspace': 0.55})

    # 上：子流 + ID 区间。窄块用引出线标注，避免重叠
    ax_top = axes[0]
    streams = ['body_base', 'body_res', 'hand_base', 'hand_res', 'face_base', 'face_res']
    codes = [32, 32, 512, 512, 128, 128]
    colors_substream = ['#1f77b4', '#7faedc', '#2ca02c', '#7dc97d',
                          '#ff7f0e', '#ffc080']
    offsets = [0]
    for c in codes:
        offsets.append(offsets[-1] + c)
    total = sum(codes)

    # 颜色块：宽块块内放完整名，窄块用引出线指向块上方标注名 + 大小
    # 用 axes 横坐标系算位置，宽块阈值放宽到 8%
    NARROW_FRAC = 0.08

    for i, (name, c, col) in enumerate(zip(streams, codes, colors_substream)):
        ax_top.barh(0, c, left=offsets[i], height=0.6, color=col,
                     edgecolor='black', linewidth=0.5)
        center_x = offsets[i] + c / 2
        # 宽块块内大字标注 + K
        if c / total >= NARROW_FRAC:
            ax_top.text(center_x, 0, f'{name}\nK={c}', ha='center', va='center',
                         fontsize=10, fontweight='bold')

    # 窄块用引出线：从块中心斜上方引到块正上方再水平放置文字
    narrow_blocks = [(i, name, codes[i], offsets[i] + codes[i] / 2)
                     for i, (name, c) in enumerate(zip(streams, codes))
                     if c / total < NARROW_FRAC]
    # 引出线 y 高度梯度，避免标签互相重叠
    for k, (i, name, c, cx) in enumerate(narrow_blocks):
        y_top = 1.0 + (k % 2) * 0.4   # 偶数低、奇数高，交错避碰
        ax_top.annotate(f'{name}, K={c}',
                          xy=(cx, 0.3), xytext=(cx, y_top),
                          fontsize=9, ha='center', va='bottom',
                          arrowprops=dict(arrowstyle='-', color='black',
                                            lw=0.6, alpha=0.7))

    ax_top.set_xlim(-30, total + 30)
    ax_top.set_ylim(-0.5, 2.0)
    ax_top.set_xlabel('token ID 编号空间（总码本规模 = 1344）')
    ax_top.set_yticks([])
    ax_top.spines['left'].set_visible(False)
    ax_top.set_title('子流码本 ID 划分', fontsize=11)

    # 下：交错 token 序列
    ax_bot = axes[1]
    T = 6   # 显示 6 个时刻
    Sub = 6
    for t in range(T):
        for s in range(Sub):
            pos = t * Sub + s
            ax_bot.add_patch(mpatches.Rectangle((pos, 0), 1, 1,
                                                  facecolor=colors_substream[s],
                                                  edgecolor='black', linewidth=0.4))
            label = f'$z_{{{streams[s][0]}{t+1}}}^{{{streams[s].split("_")[1][0]}}}$'
            ax_bot.text(pos + 0.5, 0.5, label, ha='center', va='center',
                          fontsize=8)
        # 时刻分隔
        if t < T - 1:
            ax_bot.axvline((t + 1) * Sub, color='black', linestyle='--',
                            linewidth=0.5, alpha=0.6)
    ax_bot.set_xlim(0, T * Sub)
    ax_bot.set_ylim(0, 1)
    ax_bot.set_xticks([t * Sub + Sub / 2 for t in range(T)])
    ax_bot.set_xticklabels([f't={t+1}' for t in range(T)], fontsize=10)
    ax_bot.set_yticks([])
    ax_bot.spines['left'].set_visible(False)
    ax_bot.set_title('AR Transformer 输入 token 序列（6 子流按时间交错）', fontsize=11)
    ax_bot.set_xlabel('位置索引（k = t × 6 + 子流编号）')

    fig.tight_layout()
    fig.savefig(out_dir / 'fig_token_structure.png', dpi=140)
    plt.close(fig)
    print('  [OK] fig_token_structure.png')


# ============================================================
# C. 姿态对比 stick figure
# ============================================================

def _stick(ax, pose: np.ndarray, color_body=C_BODY, color_hand=C_HAND,
            color_face=C_FACE, alpha=1.0, marker_face=False):
    x, y = pose[:, 0], -pose[:, 1]
    # face 点
    if marker_face:
        ax.scatter(x[IDX_FACE], y[IDX_FACE], c=color_face, s=2, alpha=alpha * 0.5, zorder=2)
    # body
    ax.scatter(x[IDX_BODY], y[IDX_BODY], c=color_body, s=35, alpha=alpha,
                edgecolors='black', linewidths=0.3, zorder=4)
    for a, b in BODY_BONES:
        ax.plot([x[a], x[b]], [y[a], y[b]], color=color_body, linewidth=1.5,
                 alpha=alpha * 0.8)
    # hands
    for off in [136, 157]:
        ax.scatter(x[off:off+21], y[off:off+21], c=color_hand, s=15,
                    alpha=alpha, edgecolors='black', linewidths=0.2, zorder=4)
        for a, b in HAND_BONES_LOCAL:
            ax.plot([x[off + a], x[off + b]], [y[off + a], y[off + b]],
                     color=color_hand, linewidth=0.9, alpha=alpha * 0.7)


def _hand_subplot(ax, pose: np.ndarray, hand_off: int):
    """单只手 21-keypoint stick figure 子图。pose: (178, 3)"""
    # 局部归一化：以手腕 (kpt off+0) 为中心
    h = pose[hand_off:hand_off + 21].copy()
    h -= h[0]
    x, y = h[:, 0], h[:, 1]   # 投影 xy 平面
    ax.scatter(x, y, c=C_HAND, s=22, edgecolors='black', linewidths=0.4, zorder=3)
    for a, b in HAND_BONES_LOCAL:
        ax.plot([x[a], x[b]], [y[a], y[b]], color=C_HAND, linewidth=1.5,
                 alpha=0.85, zorder=2)
    # 手腕高亮
    ax.scatter([0], [0], c='red', s=40, edgecolors='black', linewidths=0.5, zorder=4)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False)


def fig_pose_compare(out_dir: Path, gt_pose: np.ndarray, baseline_pose: np.ndarray,
                       m1m2_pose: np.ndarray, text_sample: str = ''):
    """3 行 × 4 列：GT / 基线 / M1+M2 在 4 时间点的【右手 21 关键点】对比。

    PHIX 178-kpt 数据中只有 hand 的 MediaPipe 21-kpt 拓扑是明确标准的，
    body 8-kpt 子集的连接 topology 不确定，因此只可视化手部更清晰。
    """
    T_steps = 4
    rows = [('真实 GT', gt_pose), ('基线', baseline_pose), ('M1+M2', m1m2_pose)]
    Tmin = min(len(p) for _, p in rows)
    idx = np.linspace(0, Tmin - 1, T_steps).astype(int)
    HAND_OFFSET = 157   # right hand

    fig, axes = plt.subplots(3, T_steps, figsize=(10, 7.5))
    for r, (rname, pose) in enumerate(rows):
        for c, ti in enumerate(idx):
            ax = axes[r, c]
            _hand_subplot(ax, pose[ti], HAND_OFFSET)
            if r == 0:
                ax.set_title(f't = {ti+1}/{Tmin} 帧', fontsize=11)
            if c == 0:
                ax.set_ylabel(rname, fontsize=14, fontweight='bold')
    if text_sample:
        fig.suptitle(f'右手 21 关键点对比  文本："{text_sample[:50]}…"',
                       fontsize=11, y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig_pose_compare.png')
    plt.close(fig)
    print('  [OK] fig_pose_compare.png')


# ============================================================
# D. 手腕轨迹对比
# ============================================================

def fig_wrist_trajectory(out_dir: Path, gt_pose: np.ndarray,
                            baseline_pose: np.ndarray, m1m2_pose: np.ndarray):
    """对比右手腕 (kpt 157) 的 xyz 时序。"""
    WRIST_RH = 157   # right hand wrist (本数据 kpt 157 是 right hand 起点)
    T = min(len(p) for p in [gt_pose, baseline_pose, m1m2_pose])
    t = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 6.5), sharex=True)
    labels = ['x（左右）', 'y（上下）', 'z（前后/深度）']
    for d, (ax, lab) in enumerate(zip(axes, labels)):
        ax.plot(t, gt_pose[:T, WRIST_RH, d], color='black', linewidth=1.8,
                 label='真实 GT', linestyle='-')
        ax.plot(t, baseline_pose[:T, WRIST_RH, d], color=C_BASE, linewidth=1.5,
                 label='基线', linestyle='--', alpha=0.85)
        ax.plot(t, m1m2_pose[:T, WRIST_RH, d], color=C_MSR, linewidth=1.5,
                 label='M1+M2', alpha=0.95)
        ax.set_ylabel(lab)
        ax.grid(True, linestyle='--', alpha=0.4)
        if d == 0:
            ax.legend(loc='upper right', frameon=True, fontsize=9)
    axes[-1].set_xlabel('帧编号')
    fig.tight_layout()
    fig.savefig(out_dir / 'fig_wrist_trajectory.png')
    plt.close(fig)
    print('  [OK] fig_wrist_trajectory.png')


# ============================================================
# E. 码本利用率热力图
# ============================================================

def _token_freq(cache_path: Path, key: str, K: int) -> np.ndarray:
    """返回归一化后频率直方图 (K,)。"""
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    toks = []
    for sid, v in cache.items():
        if key in v:
            toks.append(np.asarray(v[key]).ravel())
    if not toks:
        return np.zeros(K)
    toks = np.concatenate(toks)
    h, _ = np.histogram(toks, bins=K, range=(0, K))
    return h / max(h.sum(), 1)


def fig_codebook_usage(out_dir: Path, msr_tok_dir: Path):
    """单图：MSR 6 子流的码字使用分布对比，揭示子流功能分化。

    x 轴：码字按使用频率排序后的归一化索引
    y 轴：使用频率 (对数轴)
    每条曲线代表一个子流；曲线越平坦表示码字利用越均匀。
    """
    sub_keys = ['tokens_body_base', 'tokens_body_res',
                  'tokens_hand_base', 'tokens_hand_res',
                  'tokens_face_base', 'tokens_face_res']
    sub_labels = ['body_base', 'body_res', 'hand_base', 'hand_res',
                   'face_base', 'face_res']
    sub_K = [32, 32, 512, 512, 128, 128]
    sub_colors = ['#1f77b4', '#7faedc', '#2ca02c', '#7dc97d',
                    '#ff7f0e', '#ffc080']
    sub_lstyle = ['-', '--', '-', '--', '-', '--']

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    p = msr_tok_dir / 'train_tokens.pt'
    if not p.exists():
        print(f'  [SKIP] fig_codebook_usage — 缺 {p}')
        return
    for sk, lab, K, col, ls in zip(sub_keys, sub_labels, sub_K, sub_colors, sub_lstyle):
        h = _token_freq(p, sk, K)
        sorted_h = np.sort(h)[::-1]
        active = int((h > 1e-6).sum())
        ax.plot(np.arange(K) / K, sorted_h,
                 label=f'{lab}  (K={K}, 活跃 {active}, 占{active/K*100:.0f}%)',
                 color=col, linewidth=1.7, linestyle=ls)
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 1e-1)
    ax.set_xlim(0, 1)
    ax.set_xlabel('码字排序索引（按使用频率降序，归一化到 [0, 1]）')
    ax.set_ylabel('使用频率 (对数轴)')
    ax.legend(loc='lower left', fontsize=8.5, frameon=True, ncol=1)
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig_codebook_usage.png')
    plt.close(fig)
    print('  [OK] fig_codebook_usage.png')


# ============================================================
# 主流程
# ============================================================

def load_pred_pose(pickle_path: Path, sid: str) -> np.ndarray:
    with gzip.open(pickle_path, 'rb') as f:
        preds = pickle.load(f)
    for p in preds:
        if p['name'] == sid:
            sign = p['sign']
            if torch.is_tensor(sign):
                sign = sign.numpy()
            if sign.ndim == 2:
                sign = sign.reshape(sign.shape[0], 178, 3)
            return sign.astype(np.float32)
    raise KeyError(sid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release-root', default=r'D:/Graduate thesis/sign_slp_paper_release')
    ap.add_argument('--out-dir',      default=r'D:/Graduate thesis/eggroll_v2/figures/paper')
    ap.add_argument('--gt-split', default='test')
    args = ap.parse_args()

    release = Path(args.release_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载一个 GT 样本作 keypoint layout / pose compare 用
    gt_pt = release / 'data' / 'phix' / f'phix_lift3d.{args.gt_split}.pt'
    gt = torch.load(gt_pt, map_location='cpu', weights_only=False)
    # 选一个手部动作丰富的样本：取序列长度中位数附近的
    sids_sorted = sorted(gt.keys(), key=lambda s: gt[s]['poses_3d'].shape[0])
    sid = sids_sorted[len(sids_sorted) // 2]
    gt_pose = gt[sid]['poses_3d']
    if torch.is_tensor(gt_pose):
        gt_pose = gt_pose.numpy()
    gt_pose = gt_pose.astype(np.float32)
    text_sample = gt[sid].get('text', '')
    print(f'[*] sample sid = {sid}, T={len(gt_pose)}, text="{text_sample[:60]}"')

    # 加载 baseline / M1+M2 SLP 预测
    base_pickle = release / 'results' / 'phix_baseline' / f'{args.gt_split}.pickle'
    msr_pickle  = release / 'results' / 'phix_M1M2' / f'{args.gt_split}.pickle'

    # A. keypoint layout (用第一帧)
    fig_keypoint_layout(out_dir, gt_pose[len(gt_pose)//2])

    # B. token interleaving schematic
    fig_token_structure(out_dir)

    # D. 手腕轨迹对比（C 删除：PHIX hand topology 非 MediaPipe 标准，stick figure 无法清晰）
    if base_pickle.exists() and msr_pickle.exists():
        try:
            base_pose = load_pred_pose(base_pickle, sid)
            msr_pose  = load_pred_pose(msr_pickle, sid)
            fig_wrist_trajectory(out_dir, gt_pose, base_pose, msr_pose)
        except KeyError:
            print(f'  [SKIP] D — sid {sid} 在 SLP pickle 中不存在')
    else:
        print(f'  [SKIP] D — SLP pickle 不存在')

    # E. 码本利用率 (MSR 6 子流)
    msr_tok_dir = release / 'checkpoints' / 'phix' / 'tokens' / 'vq_M1M2'
    fig_codebook_usage(out_dir, msr_tok_dir)

    print(f'\n[DONE] 输出至 {out_dir}')


if __name__ == '__main__':
    main()

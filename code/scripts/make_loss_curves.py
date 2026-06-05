"""训练损失曲线图 (中文, 2x2)。

解析训练日志，画两数据集 × 两阶段的训练/验证损失：
  (a) VQ-VAE 重建损失 · PHOENIX14T      (b) VQ-VAE 重建损失 · CSL-Daily
  (c) AR Transformer 交叉熵 · PHOENIX14T (d) AR Transformer 交叉熵 · CSL-Daily

实线 = 训练损失，虚线+圆点 = 验证集损失。四种配置用论文统一配色。
CSL 的 VQ baseline/M1/M1M2 训练日志由 _relog/ 用原 checkpoint 配置复跑得到，M2 用已存日志。
输出 figures/paper/fig_loss.png
"""
from __future__ import annotations
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 11
mpl.rcParams['savefig.dpi'] = 400
mpl.rcParams['figure.dpi'] = 150

# 仓库根目录（从脚本位置推导，可移植）
ROOT = Path(__file__).resolve().parents[2]
# 图输出目录：默认写到仓库内 figures/paper；生成论文图时用环境变量 SLP_FIG_OUT 覆盖。
OUT  = Path(os.environ.get('SLP_FIG_OUT', str(ROOT / 'figures' / 'paper')))
OUT.mkdir(parents=True, exist_ok=True)

C_BASE = '#d62728'   # 基线
C_M1   = '#1f77b4'   # MSC
C_M2   = '#9467bd'   # RVQ
C_MSR  = '#2ca02c'   # MSR (本文)

# (中文名, 颜色, 日志相对路径)
VQ_PHIX = [
    ('基线',      C_BASE, 'logs/phix_vq_baseline.train.log'),
    ('MSC',       C_M1,   'logs/phix_vq_M1.train.log'),
    ('RVQ',       C_M2,   'logs/phix_vq_M2.train.log'),
    ('MSR (本文)', C_MSR, 'logs/phix_vq_M1M2.train.log'),
]
VQ_CSL = [
    ('基线',      C_BASE, 'checkpoints/csl/vq/_relog/relog_baseline/train.log'),
    ('MSC',       C_M1,   'checkpoints/csl/vq/_relog/relog_M1/train.log'),
    ('RVQ',       C_M2,   'logs/csl_vq_M2.train.log'),
    ('MSR (本文)', C_MSR, 'checkpoints/csl/vq/_relog/relog_M1M2/train.log'),
]
TR_PHIX = [
    ('基线',      C_BASE, 'logs/phix_trans_baseline.train.log'),
    ('MSC',       C_M1,   'logs/phix_trans_M1.train.log'),
    ('RVQ',       C_M2,   'logs/phix_trans_M2.train.log'),
    ('MSR (本文)', C_MSR, 'logs/phix_trans_M1M2.train.log'),
]
TR_CSL = [
    ('基线',      C_BASE, 'logs/csl_trans_baseline_large.train.log'),
    ('MSC',       C_M1,   'logs/csl_trans_M1_large.train.log'),
    ('RVQ',       C_M2,   'logs/csl_trans_M2_large.train.log'),
    ('MSR (本文)', C_MSR, 'logs/csl_trans_M1M2_large.train.log'),
]

RE_TRAIN = {
    'recon': re.compile(r'iter\s+(\d+)\s+\|\s+lr\s+\S+\s+\|\s+recon\s+([\d.]+)'),
    'ce':    re.compile(r'iter\s+(\d+)\s+\|\s+lr\s+\S+\s+\|\s+ce\s+([\d.]+)'),
}
RE_DEV = {
    'recon': re.compile(r'>>\s*DEV iter\s+(\d+)\s+recon=([\d.]+)'),
    'ce':    re.compile(r'>>\s*DEV iter\s+(\d+)\s+ce=([\d.]+)'),
}


def parse(relpath, metric):
    p = ROOT / relpath
    if not p.exists():
        print(f'[WARN] missing log: {relpath}')
        return np.empty((0, 2)), np.empty((0, 2))
    txt = p.read_text(encoding='utf-8', errors='ignore')
    tr = [(int(m.group(1)), float(m.group(2))) for m in RE_TRAIN[metric].finditer(txt)]
    dv = [(int(m.group(1)), float(m.group(2))) for m in RE_DEV[metric].finditer(txt)]
    tr = np.array(sorted(tr)) if tr else np.empty((0, 2))
    dv = np.array(sorted(dv)) if dv else np.empty((0, 2))
    return tr, dv


def panel(ax, configs, metric, title, ylab, logy=False):
    for name, col, fname in configs:
        tr, dv = parse(fname, metric)
        if len(tr):
            ax.plot(tr[:, 0] / 1000, tr[:, 1], color=col, lw=1.6, alpha=0.9, label=name)
        if len(dv):
            ax.plot(dv[:, 0] / 1000, dv[:, 1], color=col, lw=1.3, ls='--',
                    marker='o', ms=4, alpha=0.85)
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel('训练迭代步 (×10³)', fontsize=11)
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_title(title, fontsize=12, color='#333')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='upper right', frameon=True, framealpha=0.95)


fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor='white')
panel(axes[0, 0], VQ_PHIX, 'recon', '(a) VQ-VAE 重建损失 · PHOENIX14T', '重建损失', logy=True)
panel(axes[0, 1], VQ_CSL,  'recon', '(b) VQ-VAE 重建损失 · CSL-Daily',  '重建损失', logy=True)
panel(axes[1, 0], TR_PHIX, 'ce',    '(c) AR Transformer 交叉熵 · PHOENIX14T', '交叉熵损失')
panel(axes[1, 1], TR_CSL,  'ce',    '(d) AR Transformer 交叉熵 · CSL-Daily',  '交叉熵损失')

fig.text(0.5, 0.005, '实线 = 训练损失，虚线+圆点 = 验证集损失（每个评测间隔一个点）',
         ha='center', va='bottom', fontsize=10, color='#555', style='italic')
fig.tight_layout(rect=[0, 0.025, 1, 1])

out = OUT / 'fig_loss.png'
fig.savefig(out, dpi=400, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'[OK] {out}')

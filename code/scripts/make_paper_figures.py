"""生成论文中所有图，输出到 eggroll_v2/figures/paper/。

依赖：matplotlib (中文字体 SimHei/Microsoft YaHei)
运行：python make_paper_figures.py [--release-root <path>]
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- 中文字体设置 ---
for cn in ['SimHei', 'Microsoft YaHei', 'SimSun', 'STSong']:
    if cn in {f.name for f in fm.fontManager.ttflist}:
        plt.rcParams['font.sans-serif'] = [cn, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# --- 颜色方案 (色觉友好) ---
COLORS = {
    'baseline': '#888888',
    'M1':       '#1f77b4',
    'M2':       '#ff7f0e',
    'M1M2':     '#2ca02c',
    'ceiling':  '#bbbbbb',
    'sota':     '#d62728',
}


# ============================================================
# 数据加载工具
# ============================================================

def load_slrtp_json(slrtp_results_dir: Path, tag: str) -> dict | None:
    p = slrtp_results_dir / f'{tag}.json'
    if not p.exists():
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def collect_main_results(slrtp_results_dir: Path):
    """收集 4 ablation × 2 dataset × dev/test 的 BLEU。
    PHIX tag 模式: v2_{variant}_{split}
    CSL tag 模式: csl_v1_{variant}_{split}
    """
    out = {}
    for variant in ['baseline', 'M1', 'M2', 'M1M2']:
        for split in ['dev', 'test']:
            for ds, tag_pat in [('phix', f'v2_{variant}_{split}'),
                                  ('csl',  f'csl_v1_{variant}_{split}')]:
                # CSL JSON 在 results/_archive/_slrtp_csl_*/
                if ds == 'csl':
                    release_root = slrtp_results_dir.parent.parent.parent
                    csl_dir = release_root / 'results' / '_archive' / f'_slrtp_csl_{variant}'
                    p = csl_dir / f'csl_v1_{variant}_{split}.json'
                    if not p.exists():
                        continue
                    r = json.loads(p.read_text(encoding='utf-8'))
                else:
                    r = load_slrtp_json(slrtp_results_dir, tag_pat)
                    if r is None:
                        continue
                out[(ds, variant, split)] = r
    return out


def collect_vq_ceiling(slrtp_results_dir: Path):
    """v2_vqceil_{variant}_{split}.json"""
    out = {}
    for variant in ['baseline', 'M1', 'M2', 'M1M2']:
        for split in ['dev', 'test']:
            r = load_slrtp_json(slrtp_results_dir, f'v2_vqceil_{variant}_{split}')
            if r is not None:
                out[(variant, split)] = r
    return out


def parse_train_log(log_path: Path):
    """从 train.log 解析 iter/dev_ce 序列。
    格式: ">> DEV iter NNNN dev_ce X.XXX" 或类似。
    """
    iters_ce, vals_ce = [], []
    iters_rec, vals_rec = [], []
    if not log_path.exists():
        return iters_ce, vals_ce, iters_rec, vals_rec
    for line in log_path.read_text(errors='ignore').splitlines():
        m = re.match(r'.*DEV iter\s+(\d+)\s+(?:recon=([\d.]+)|dev_ce=([\d.]+))', line)
        if m is None:
            m2 = re.match(r'.*DEV iter\s+(\d+).*?(?:recon|ce)[\s=]+([\d.]+)', line)
            if m2 is None:
                continue
            it = int(m2.group(1)); v = float(m2.group(2))
            if 'recon' in line:
                iters_rec.append(it); vals_rec.append(v)
            else:
                iters_ce.append(it); vals_ce.append(v)
            continue
        it = int(m.group(1))
        if m.group(2):
            iters_rec.append(it); vals_rec.append(float(m.group(2)))
        if m.group(3):
            iters_ce.append(it); vals_ce.append(float(m.group(3)))
    return iters_ce, vals_ce, iters_rec, vals_rec


# ============================================================
# 图 1: PHIX/CSL 主结果柱状图
# ============================================================

def fig1_main_results(out_dir: Path, results: dict):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    variants = ['baseline', 'M1', 'M2', 'M1M2']
    labels = ['基线', '+ M1', '+ M2', '+ M1+M2']

    for ax, ds, title in zip(axes, ['phix', 'csl'], ['PHIX-14T', 'CSL-Daily']):
        vals = []
        for v in variants:
            r = results.get((ds, v, 'test'))
            vals.append(r['bleu']['bleu4'] if r else 0)
        colors = [COLORS[v] for v in variants]
        bars = ax.bar(labels, vals, color=colors, edgecolor='black', linewidth=0.6)
        # 数字标注
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                     f'{v:.2f}', ha='center', va='bottom', fontsize=10)
        ax.set_ylabel('BLEU-4 (测试集)')
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_dir / 'fig1_main_bleu.png')
    plt.close(fig)
    print(f'  [OK] fig1_main_bleu.png')


# ============================================================
# 图 2: VQ Ceiling vs Trans
# ============================================================

def fig2_vq_ceiling(out_dir: Path, results: dict, vqceil: dict):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    variants = ['baseline', 'M1', 'M2', 'M1M2']
    labels = ['基线', '+ M1', '+ M2', '+ M1+M2']

    trans_vals = [results[('phix', v, 'test')]['bleu']['bleu4'] for v in variants]
    ceil_vals = [vqceil[(v, 'test')]['bleu']['bleu4'] for v in variants]

    x = np.arange(len(variants))
    w = 0.35
    ax.bar(x - w/2, ceil_vals, w, color=COLORS['ceiling'], label='VQ 上限', edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, trans_vals, w, color=COLORS['M1M2'], label='AR Trans 实测', edgecolor='black', linewidth=0.5)

    for i, (c, t) in enumerate(zip(ceil_vals, trans_vals)):
        ax.text(i - w/2, c + 0.15, f'{c:.2f}', ha='center', fontsize=9)
        ax.text(i + w/2, t + 0.15, f'{t:.2f}', ha='center', fontsize=9)
        # 百分比
        pct = t / c * 100
        ax.text(i, max(c, t) + 1.0, f'{pct:.0f}%', ha='center', fontsize=9, color='gray')

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('BLEU-4 (PHIX 测试集)')
    ax.legend(loc='upper left', frameon=True)
    ax.set_ylim(0, max(ceil_vals) * 1.20)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig2_vq_ceiling.png')
    plt.close(fig)
    print(f'  [OK] fig2_vq_ceiling.png')


# ============================================================
# 图 3: 模块协同效应
# ============================================================

def fig3_synergy(out_dir: Path, results: dict):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, ds, title in zip(axes, ['phix', 'csl'], ['PHIX-14T', 'CSL-Daily']):
        baseline = results[(ds, 'baseline', 'test')]['bleu']['bleu4']
        m1 = results[(ds, 'M1', 'test')]['bleu']['bleu4']
        m2 = results[(ds, 'M2', 'test')]['bleu']['bleu4']
        m1m2 = results[(ds, 'M1M2', 'test')]['bleu']['bleu4']

        d_m1 = m1 - baseline
        d_m2 = m2 - baseline
        d_m1m2 = m1m2 - baseline
        d_additive = d_m1 + d_m2
        d_synergy = d_m1m2 - d_additive

        items = ['仅 M1', '仅 M2', '相加预测', '实测 M1+M2']
        deltas = [d_m1, d_m2, d_additive, d_m1m2]
        colors = [COLORS['M1'], COLORS['M2'], '#cccccc', COLORS['M1M2']]
        bars = ax.bar(items, deltas, color=colors, edgecolor='black', linewidth=0.6)
        ymax = max(deltas + [d_synergy])
        for b, v in zip(bars, deltas):
            if v >= 0:
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + ymax * 0.02,
                         f'+{v:.2f}', ha='center', va='bottom', fontsize=10)
            else:
                ax.text(b.get_x() + b.get_width()/2, b.get_height() - ymax * 0.04,
                         f'{v:+.2f}', ha='center', va='top', fontsize=10)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_ylabel('ΔBLEU-4 (相对基线)')
        ax.set_title(f'{title}  (基线 {baseline:.2f} → M1+M2 {m1m2:.2f})')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        # 给 y 轴留余量以容纳协同箭头
        ax.set_ylim(min(deltas) - ymax * 0.15, ymax * 1.30)
        # 标注协同增量：箭头从相加预测指向实测，水平放置在右上方
        ax.annotate(f'协同增量 +{d_synergy:.2f}',
                     xy=(3, d_m1m2 * 0.55), xytext=(1.6, ymax * 1.10),
                     fontsize=10, color='red', ha='center',
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.0,
                                       connectionstyle='arc3,rad=-0.15'))

    fig.tight_layout()
    fig.savefig(out_dir / 'fig3_synergy.png')
    plt.close(fig)
    print(f'  [OK] fig3_synergy.png')


# ============================================================
# 图 4: SOTA 全景对比 (PHIX TEST B4)
# ============================================================

def fig4_sota(out_dir: Path):
    methods = [
        ('Progressive Transformer (2020)', 4.38, 'gloss-supervised', 'lightgray'),
        ('MDM (2023)',                     7.55, 'gloss-supervised', 'lightgray'),
        ('T2M-GPT (2023)',                 8.01, 'gloss-free',        'lightblue'),
        ('本文 M1+M2',                      8.97, 'strict gloss-free', '#2ca02c'),
        ('T2S-GPT (2024)',                11.87, 'gloss-free',        'lightblue'),
    ]
    methods_sorted = sorted(methods, key=lambda x: x[1])
    names = [m[0] for m in methods_sorted]
    vals = [m[1] for m in methods_sorted]
    colors = [m[3] for m in methods_sorted]

    fig, ax = plt.subplots(figsize=(8, 4.0))
    bars = ax.barh(names, vals, color=colors, edgecolor='black', linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(v + 0.15, b.get_y() + b.get_height()/2, f'{v:.2f}',
                 va='center', fontsize=10)
    ax.set_xlabel('PHIX 测试集 BLEU-4')
    ax.set_xlim(0, max(vals) * 1.15)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig4_sota.png')
    plt.close(fig)
    print(f'  [OK] fig4_sota.png')


# ============================================================
# 图 5/6: 训练曲线
# ============================================================

def fig5_vq_train_curves(out_dir: Path, logs_dir: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    variants = [('baseline', '基线'), ('M1', '+ M1'), ('M2', '+ M2'), ('M1M2', '+ M1+M2')]
    for v, label in variants:
        log = logs_dir / f'phix_vq_{v}.train.log'
        _, _, iters, vals = parse_train_log(log)
        if not iters:
            continue
        ax.plot(iters, vals, label=label, color=COLORS[v], linewidth=1.8, marker='o', markersize=3)
    ax.set_xlabel('训练 iter')
    ax.set_ylabel('验证集重建损失 (dev recon)')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.4)
    # 改用 linear scale + 0 起点，避免 log mathtext 与中文字体冲突
    ax.set_ylim(0, 0.30)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    fig.tight_layout()
    fig.savefig(out_dir / 'fig5_vq_train_curves.png')
    plt.close(fig)
    print(f'  [OK] fig5_vq_train_curves.png')


def fig6_trans_train_curves(out_dir: Path, logs_dir: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    variants = [('baseline', '基线'), ('M1', '+ M1'), ('M2', '+ M2'), ('M1M2', '+ M1+M2')]
    for v, label in variants:
        log = logs_dir / f'phix_trans_{v}.train.log'
        iters, vals, _, _ = parse_train_log(log)
        if not iters:
            continue
        ax.plot(iters, vals, label=label, color=COLORS[v], linewidth=1.8, marker='o', markersize=3)
    ax.set_xlabel('训练 iter')
    ax.set_ylabel('验证集交叉熵 (dev CE)')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig6_trans_train_curves.png')
    plt.close(fig)
    print(f'  [OK] fig6_trans_train_curves.png')


# ============================================================
# 图 7: 骨段稳定性 CV
# ============================================================

def fig7_skeleton_cv(out_dir: Path):
    # 来自论文 5.3 节 PHIX TEST 数据
    data = {
        '基线':    {'body': 0.046, 'LH': 0.030, 'RH': 0.039, 'all': 0.036},
        '+ M1':   {'body': 0.038, 'LH': 0.032, 'RH': 0.048, 'all': 0.040},
        '+ M2':   {'body': 0.050, 'LH': 0.034, 'RH': 0.047, 'all': 0.042},
        '+ M1+M2': {'body': 0.042, 'LH': 0.037, 'RH': 0.054, 'all': 0.045},
    }
    parts = ['body', 'LH', 'RH', 'all']
    part_labels = ['身体', '左手', '右手', '全部']
    variants = list(data.keys())
    colors = [COLORS['baseline'], COLORS['M1'], COLORS['M2'], COLORS['M1M2']]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(parts))
    w = 0.20
    for i, (v, c) in enumerate(zip(variants, colors)):
        vals = [data[v][p] for p in parts]
        offset = (i - 1.5) * w
        bars = ax.bar(x + offset, vals, w, label=v, color=c, edgecolor='black', linewidth=0.4)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,
                     f'{val:.3f}', ha='center', fontsize=7, rotation=0)
    ax.set_xticks(x); ax.set_xticklabels(part_labels)
    ax.set_ylabel('骨段长度方差系数 (CV)')
    ax.legend(loc='upper left', ncol=2, frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_ylim(0, max(d['RH'] for d in data.values()) * 1.30)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig7_skeleton_cv.png')
    plt.close(fig)
    print(f'  [OK] fig7_skeleton_cv.png')


# ============================================================
# main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release-root', default=r'D:/Graduate thesis/sign_slp_paper_release')
    ap.add_argument('--out-dir',      default=r'D:/Graduate thesis/eggroll_v2/figures/paper')
    args = ap.parse_args()

    release = Path(args.release_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slrtp_results = release / 'bt_eval_kit' / 'slrtp_official' / 'results'
    logs_dir = release / 'logs'

    print(f'[*] 收集数据 from {slrtp_results}')
    results = collect_main_results(slrtp_results)
    vqceil  = collect_vq_ceiling(slrtp_results)
    print(f'    主结果 keys: {len(results)}; VQ ceiling keys: {len(vqceil)}')

    print(f'[*] 生成图到 {out_dir}')
    if any((ds, v, 'test') in results for ds in ['phix', 'csl'] for v in ['baseline', 'M1', 'M2', 'M1M2']):
        fig1_main_results(out_dir, results)
        fig3_synergy(out_dir, results)
    else:
        print('  [SKIP] fig1/fig3 — 缺少主结果数据')

    if vqceil:
        fig2_vq_ceiling(out_dir, results, vqceil)
    else:
        print('  [SKIP] fig2 — 缺少 VQ ceiling 数据')

    fig4_sota(out_dir)
    fig5_vq_train_curves(out_dir, logs_dir)
    fig6_trans_train_curves(out_dir, logs_dir)
    fig7_skeleton_cv(out_dir)

    print(f'\n[DONE] 全部图保存至 {out_dir}')


if __name__ == '__main__':
    main()

"""3D keypoint 可视化（透明背景）：
  - fig_3d_gt_t1/t2/t3.png : 三张连续帧全身 178 kpt 3D 散点
  - fig_3d_body.png        : 单部位 body (8 kpt + 骨架)
  - fig_3d_hand.png        : 单部位 hand (42 kpt 双手, world)
  - fig_3d_face.png        : 单部位 face (86 kpt 点云)

数据：PHIX dev 样本 10August_2011_Wednesday_tagesschau-6304 (T=182)
布局：每图独立 3D，透明背景
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False

ROOT = Path(r'D:/Graduate thesis/sign_slp_paper_release')
OUT  = Path(r'D:/Graduate thesis/eggroll_v2/figures/paper/3d_kpt')
OUT.mkdir(parents=True, exist_ok=True)

# PHIX 178-kpt 布局
IDX_BODY     = list(range(0, 8))
IDX_LH_WORLD = list(range(8, 29))
IDX_RH_WORLD = list(range(29, 50))
IDX_FACE     = list(range(50, 136))     # 86 个，face-local，需要重锚

C_BODY = '#3d5a80'
C_HAND = '#ee6c4d'
C_FACE = '#3d6ea0'   # 进一步加深

BODY_BONES = [(0,1),(1,2),(3,4),(4,5),(0,3),(0,6),(3,7),(6,7)]


def reanchor_face(face_local: np.ndarray, sh_x: float, sh_y: float, sh_w: float) -> np.ndarray:
    """把 face-local 86 点重锚到肩部正上方，缩放到合理尺寸。"""
    cx = face_local[:, 0].mean()
    cy = face_local[:, 1].mean()
    cz = face_local[:, 2].mean()
    fw = face_local[:, 0].max() - face_local[:, 0].min()
    target_w = sh_w * 0.55
    scale = target_w / max(fw, 1e-6)
    out = (face_local - np.array([cx, cy, cz])) * scale
    out[:, 0] += sh_x
    out[:, 1] += sh_y - sh_w * 0.55
    # z 保持相对深度，加 body 中心 z
    out[:, 2] += 0.0
    return out


def setup_3d(ax, all_pts, elev=12, azim=-65, show_axes=False):
    """统一 3D 轴样式：透明背景、无外框、等比例。"""
    ax.patch.set_alpha(0.0)
    # 完全关闭 axis（包括三面 pane、轴线、刻度、外框立方体）
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    rng = all_pts.max(axis=0) - all_pts.min(axis=0)
    mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    half = rng.max() / 2 * 1.05
    ax.set_xlim(mid[0]-half, mid[0]+half)
    ax.set_ylim(mid[1]-half, mid[1]+half)
    ax.set_zlim(mid[2]-half, mid[2]+half)


def plot_full_3d(p178, out_path, elev=10, azim=-70):
    """画全身 178 kpt (body + hands world + face 重锚)，无标题无轴刻度。"""
    p = p178.copy()
    sh_x = (p[0, 0] + p[3, 0]) / 2
    sh_y = (p[0, 1] + p[3, 1]) / 2
    sh_w = abs(p[3, 0] - p[0, 0])
    p[IDX_FACE, :3] = reanchor_face(p[IDX_FACE, :3], sh_x, sh_y, sh_w)

    fig = plt.figure(figsize=(6, 7), facecolor='none')
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection='3d')

    body = p[IDX_BODY];   pts_body = np.stack([body[:, 0], body[:, 2], -body[:, 1]], axis=1)
    lh   = p[IDX_LH_WORLD]; pts_lh   = np.stack([lh[:, 0], lh[:, 2], -lh[:, 1]], axis=1)
    rh   = p[IDX_RH_WORLD]; pts_rh   = np.stack([rh[:, 0], rh[:, 2], -rh[:, 1]], axis=1)
    face = p[IDX_FACE];   pts_face = np.stack([face[:, 0], face[:, 2], -face[:, 1]], axis=1)

    ax.scatter(pts_face[:, 0], pts_face[:, 1], pts_face[:, 2],
               c=C_FACE, s=6, alpha=0.95, edgecolors='none')
    for a, b in BODY_BONES:
        ax.plot([pts_body[a, 0], pts_body[b, 0]],
                [pts_body[a, 1], pts_body[b, 1]],
                [pts_body[a, 2], pts_body[b, 2]],
                color=C_BODY, linewidth=2.0, alpha=0.9)
    ax.scatter(pts_body[:, 0], pts_body[:, 1], pts_body[:, 2],
               c=C_BODY, s=42, alpha=0.95, edgecolors='white', linewidths=0.6)
    for pts in (pts_lh, pts_rh):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=C_HAND, s=14, alpha=0.9, edgecolors='white', linewidths=0.3)

    all_pts = np.concatenate([pts_body, pts_lh, pts_rh, pts_face], axis=0)
    setup_3d(ax, all_pts, elev=elev, azim=azim, show_axes=False)
    plt.savefig(out_path, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'  [OK] {out_path.name}')


def plot_part_only(pts_3d, color, marker_size, out_path, bones=None,
                    elev=10, azim=-70):
    """单部位 3D，无标题无轴。pts_3d shape (N, 3) — 已 reanchor。"""
    pts = np.stack([pts_3d[:, 0], pts_3d[:, 2], -pts_3d[:, 1]], axis=1)
    fig = plt.figure(figsize=(5.5, 6), facecolor='none')
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=color, s=marker_size, alpha=0.95,
               edgecolors='white', linewidths=0.5)

    if bones is not None:
        for a, b in bones:
            ax.plot([pts[a, 0], pts[b, 0]],
                    [pts[a, 1], pts[b, 1]],
                    [pts[a, 2], pts[b, 2]],
                    color=color, linewidth=1.8, alpha=0.9)

    setup_3d(ax, pts, elev=elev, azim=azim, show_axes=False)
    plt.savefig(out_path, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'  [OK] {out_path.name}')


def main():
    gt = torch.load(ROOT / 'data/phix/phix_lift3d.dev.pt',
                     map_location='cpu', weights_only=False)
    name = '10August_2011_Wednesday_tagesschau-6304'
    p_all = gt[name]['poses_3d'].numpy()      # (T, 178, 3)
    T = p_all.shape[0]
    print(f'sample {name}, T={T}')

    # 3 张帧：明显分开（25% / 55% / 85%）以展示动作进程
    frames = [int(T * 0.25), int(T * 0.55), int(T * 0.85)]
    print(f'选定 3 帧 (t=25/55/85%): {frames}')

    # 1-3. 三张全身（不同时间点）
    for i, f in enumerate(frames, 1):
        plot_full_3d(p_all[f], OUT / f'fig_3d_gt_t{i}.png')

    # 4-6. 单部位（用中间帧）
    f_mid = frames[1]
    plot_part_only(p_all[f_mid, IDX_BODY], C_BODY, 90,
                    OUT / 'fig_3d_body.png', bones=BODY_BONES)

    hand_pts = p_all[f_mid, IDX_LH_WORLD + IDX_RH_WORLD]
    plot_part_only(hand_pts, C_HAND, 30,
                    OUT / 'fig_3d_hand.png')

    sh_x = (p_all[f_mid, 0, 0] + p_all[f_mid, 3, 0]) / 2
    sh_y = (p_all[f_mid, 0, 1] + p_all[f_mid, 3, 1]) / 2
    sh_w = abs(p_all[f_mid, 3, 0] - p_all[f_mid, 0, 0])
    face_world = reanchor_face(p_all[f_mid, IDX_FACE], sh_x, sh_y, sh_w)
    # 与全身图 fig_3d_gt_t2 用同一视角 (elev=10, azim=-70)
    plot_part_only(face_world, C_FACE, 35,
                    OUT / 'fig_3d_face.png', elev=10, azim=-70)


if __name__ == '__main__':
    main()

"""Inspect PHIX 178-keypoint layout: print shape, range, and first frame coords."""
import sys, torch, numpy as np
gt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
sid = list(gt.keys())[0]
p = gt[sid]['poses_3d']
if not torch.is_tensor(p):
    p = torch.tensor(p)
print(f'shape: {p.shape}')
print(f'first frame x range: {p[0,:,0].min():.3f} ~ {p[0,:,0].max():.3f}')
print(f'first frame y range: {p[0,:,1].min():.3f} ~ {p[0,:,1].max():.3f}')
print(f'first frame z range: {p[0,:,2].min():.3f} ~ {p[0,:,2].max():.3f}')
# Print specific keypoints for layout check
for idx, name in [(0, 'kpt 0'), (3, 'kpt 3'), (7, 'kpt 7'),
                    (8, 'kpt 8'), (135, 'kpt 135'),
                    (136, 'kpt 136'), (156, 'kpt 156'),
                    (157, 'kpt 157'), (177, 'kpt 177')]:
    print(f'  {name}: x={p[0,idx,0]:.3f} y={p[0,idx,1]:.3f} z={p[0,idx,2]:.3f}')

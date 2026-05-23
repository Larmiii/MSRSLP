import torch, sys
gt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
sids = list(gt.keys())
print(f'num samples: {len(sids)}')
print(f'first sid: {sids[0]}')
print(f'keys per sample: {list(gt[sids[0]].keys())}')
p = gt[sids[0]]['poses_3d']
print(f'poses_3d type: {type(p)}, shape: {p.shape if hasattr(p, "shape") else "n/a"}')
print(f'text: {gt[sids[0]].get("text", "")[:60]}')

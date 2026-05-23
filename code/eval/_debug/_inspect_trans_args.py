import torch, json, sys
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
a = ck.get('args', {})
d = a if isinstance(a, dict) else vars(a)
print(json.dumps(d, indent=2, default=str))

import torch, json, sys
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
a = ck['args']
out = {}
for k in ['dataname','nb_code','nb_code_residual','nb_base_body','nb_base_hand','nb_base_face',
          'nb_res_body','nb_res_hand','nb_res_face','code_dim','output_emb_width',
          'down_t','stride_t','width','depth','dilation_growth_rate','vq_act','quantizer',
          'batch_size','window_size','total_iter','lr','lr_scheduler','gamma','commit',
          'loss_vel','recons_loss','mu','beta']:
    v = a.get(k) if isinstance(a, dict) else getattr(a, k, None)
    out[k] = v
print(json.dumps(out, indent=2, default=str))

"""Verified dataset-specific grouping and matched shared-decoder VQ models.

Inputs and decoder outputs retain original dataset coordinates. Random grouping
permutes whole XYZ keypoints only. Legacy checkpoints must use explicit legacy
classes, never a silently changed keypoint interpretation.
"""
from copy import deepcopy
import numbers
import numpy as np
import torch
import torch.nn as nn
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset

LAYOUT_VERSION = "reviewer55_verified_v1"
STREAM_ORDER = ("body", "hand", "face")
STREAM_SIZES = {"body":8, "hand":42, "face":128}
VERIFIED_STREAM_INDICES = {
    "phix_lift3d": {"body":tuple(range(8)), "hand":tuple(range(8,50)), "face":tuple(range(50,178))},
    "csl_lift3d": {"body":tuple(range(8)), "hand":tuple(range(136,178)), "face":tuple(range(8,136))},
}
def _arg(args, key, default=None):
    return args.get(key,default) if isinstance(args,dict) else getattr(args,key,default)
def _set_arg(args,key,value):
    if isinstance(args,dict): args[key]=value
    else: setattr(args,key,value)

def resolve_grouping_spec(args,dataset_name):
    version=_arg(args,"layout_version")
    if version != LAYOUT_VERSION:
        raise ValueError(f"Expected layout_version={LAYOUT_VERSION!r}, got {version!r}. "
                         "Unversioned checkpoints require an explicit LegacyMultiStream class.")
    if dataset_name not in VERIFIED_STREAM_INDICES:
        raise ValueError(f"No verified 178-point mapping for {dataset_name!r}")
    grouping=_arg(args,"grouping")
    if grouping not in {"anatomical","random"}:
        raise ValueError("grouping must be explicitly 'anatomical' or 'random'")
    seed=_arg(args,"grouping_seed",20260910)
    if isinstance(seed,bool) or not isinstance(seed,numbers.Integral) or int(seed)<0:
        raise ValueError("grouping_seed must be a nonnegative integer")
    seed=int(seed)
    if grouping=="anatomical":
        indices={n:list(VERIFIED_STREAM_INDICES[dataset_name][n]) for n in STREAM_ORDER}
        permutation=[p for n in STREAM_ORDER for p in indices[n]]
    else:
        permutation=np.random.Generator(np.random.PCG64(seed)).permutation(178).tolist()
        indices={}; offset=0
        for name in STREAM_ORDER:
            count=STREAM_SIZES[name]
            indices[name]=permutation[offset:offset+count]; offset+=count
    saved=_arg(args,"grouping_permutation")
    if saved is not None:
        if not isinstance(saved,(list,tuple)) or any(isinstance(p,bool) or not isinstance(p,numbers.Integral) for p in saved):
            raise ValueError("grouping_permutation must contain integer keypoint indices")
        saved=[int(p) for p in saved]
        if len(saved)!=178 or sorted(saved)!=list(range(178)):
            raise ValueError("grouping_permutation must be a bijection of 0..177")
        if saved != permutation:
            raise ValueError("Saved grouping_permutation disagrees with layout/grouping/seed")
    spec={"layout_version":version,"dataset_name":dataset_name,"grouping":grouping,
          "grouping_seed":seed,"permutation_algorithm":"numpy.PCG64.whole_keypoints_v1",
          "stream_order":list(STREAM_ORDER),"stream_indices":indices,
          "stream_sizes":dict(STREAM_SIZES),"grouping_permutation":permutation,
          "inverse_permutation":np.argsort(permutation).tolist(),
          "coordinate_triplets_preserved":True,"decoder_output_order":"original_dataset"}
    saved_spec=_arg(args,"grouping_spec")
    if saved_spec is not None and saved_spec != spec:
        raise ValueError("Saved grouping_spec disagrees with the requested grouping")
    for key in ("layout_version","grouping","grouping_seed","grouping_permutation"):
        _set_arg(args,key,deepcopy(spec[key]))
    _set_arg(args,"grouping_spec",deepcopy(spec))
    return spec

def split_grouped_pose(pose,spec):
    if pose.ndim!=3 or pose.shape[-1]!=534:
        raise ValueError(f"Expected (B,T,534), got {tuple(pose.shape)}")
    batch,time,_=pose.shape
    points=pose.reshape(batch,time,178,3)
    return {n:points.index_select(2,torch.as_tensor(spec["stream_indices"][n],device=pose.device)).reshape(
            batch,time,STREAM_SIZES[n]*3) for n in spec["stream_order"]}

def merge_grouped_pose(streams,spec):
    """Audit inverse: place all branch XYZ outputs back in original point order."""
    first=streams[spec["stream_order"][0]]; batch,time=first.shape[:2]
    out=first.new_zeros(batch,time,178,3)
    for n in spec["stream_order"]:
        if tuple(streams[n].shape)!=(batch,time,STREAM_SIZES[n]*3):
            raise ValueError(f"Invalid stream shape for {n}")
        idx=torch.as_tensor(spec["stream_indices"][n],device=first.device)
        out=out.index_copy(2,idx,streams[n].reshape(batch,time,STREAM_SIZES[n],3))
    return out.reshape(batch,time,534)

def _quant(kind,size,dim,args):
    if kind=="ema_reset": return QuantizeEMAReset(size,dim,args)
    if kind=="orig": return Quantizer(size,dim,1.0)
    if kind=="ema": return QuantizeEMA(size,dim,args)
    if kind=="reset": return QuantizeReset(size,dim,args)
    raise ValueError(f"Unknown quantizer {kind!r}")

class _StreamEncoder(nn.Module):
    def __init__(self,input_dim,code_dim,down_t,stride_t,width,depth,dilation_growth_rate,activation,norm):
        super().__init__()
        self.encoder=Encoder(input_dim,code_dim,down_t,stride_t,width,depth,dilation_growth_rate,
                             activation=activation,norm=norm)
    def forward(self,x):
        return self.encoder(x.permute(0,2,1).float())

class SharedGroupedVQVAE(nn.Module):
    """Same branches/fusion/decoder for one-level and two-level quantization."""
    def __init__(self,args,dataset_name,*,levels,code_dim,output_emb_width,down_t,stride_t,
                 width,depth,dilation_growth_rate,activation,norm,stream_codes_base,
                 stream_codes_residual=None,enc_widths=None):
        super().__init__()
        if levels not in {1,2}: raise ValueError("levels must be 1 or 2")
        if code_dim!=output_emb_width:
            raise ValueError("Shared decoder requires code_dim == output_emb_width")
        self.grouping_spec=resolve_grouping_spec(args,dataset_name)
        self.stream_order=tuple(self.grouping_spec["stream_order"])
        self.stream_indices={n:tuple(self.grouping_spec["stream_indices"][n]) for n in self.stream_order}
        self.stream_dims={n:3*len(self.stream_indices[n]) for n in self.stream_order}
        self.dataset_name=dataset_name
        self.layout_version=LAYOUT_VERSION
        self.grouping=self.grouping_spec["grouping"]
        self.grouping_seed=self.grouping_spec["grouping_seed"]
        self.grouping_permutation=tuple(self.grouping_spec["grouping_permutation"])
        self.code_dim=code_dim; self.input_dim=534; self.num_quantization_levels=levels
        self.architecture_version="shared_decoder_v1"
        self.model_kind="pamvq_shared" if levels==1 else "pamrvq_shared"
        self.quant=_arg(args,"quantizer")
        self.enc_widths=({"body":max(128,width//4),"hand":max(256,width//2),"face":width}
                         if enc_widths is None else dict(enc_widths))
        for n in self.stream_order:
            if n not in self.enc_widths or int(self.enc_widths[n])<=0:
                raise ValueError(f"Invalid encoder width for {n}")
            self.register_buffer("_group_index_"+n,torch.tensor(self.stream_indices[n],dtype=torch.long),persistent=False)
        def codes(values):
            if values is None or set(values)!=set(self.stream_order):
                raise ValueError("Codebook sizes must name exactly body, hand, face")
            result={}
            for n in self.stream_order:
                v=values[n]
                if isinstance(v,bool) or not isinstance(v,numbers.Integral) or int(v)<=0:
                    raise ValueError(f"Invalid codebook size for {n}: {v!r}")
                result[n]=int(v)
            return result
        base=codes(stream_codes_base)
        residual=codes(stream_codes_residual) if levels==2 else None
        self.stream_codes_base=base
        if levels==1:
            self.stream_codes=base; self.quantizers=nn.ModuleDict()
        else:
            self.stream_codes_residual=residual
            self.quantizers_base=nn.ModuleDict(); self.quantizers_res=nn.ModuleDict()
        self.encoders=nn.ModuleDict()
        for n in self.stream_order:
            self.encoders[n]=_StreamEncoder(self.stream_dims[n],code_dim,down_t,stride_t,
                                           self.enc_widths[n],depth,dilation_growth_rate,activation,norm)
            if levels==1: self.quantizers[n]=_quant(self.quant,base[n],code_dim,args)
            else:
                self.quantizers_base[n]=_quant(self.quant,base[n],code_dim,args)
                self.quantizers_res[n]=_quant(self.quant,residual[n],code_dim,args)
        self.proj_in=nn.Conv1d(3*code_dim,code_dim,kernel_size=1)
        self.decoder=Decoder(534,output_emb_width,down_t,stride_t,width,depth,dilation_growth_rate,
                             activation=activation,norm=norm)
        self.token_order=(tuple(self.stream_order) if levels==1 else
                          tuple(f"{n}_{layer}" for n in self.stream_order for layer in ("base","res")))
        for key in ("architecture_version","model_kind","num_quantization_levels"):
            _set_arg(args,key,getattr(self,key))

    def export_grouping_metadata(self):
        return deepcopy(self.grouping_spec)
    def get_extra_state(self):
        return {"model_kind":self.model_kind,"architecture_version":self.architecture_version,
                "num_quantization_levels":self.num_quantization_levels,"grouping_spec":self.export_grouping_metadata()}
    def _validate_extra_state(self,state):
        if not isinstance(state,dict) or state != self.get_extra_state():
            raise RuntimeError("Checkpoint layout/grouping/architecture metadata mismatch. "
                               "Use matching saved args or an explicit LegacyMultiStream class.")
    def set_extra_state(self,state):
        self._validate_extra_state(state)
    def load_state_dict(self,state_dict,strict=True,assign=False):
        if "_extra_state" not in state_dict:
            raise RuntimeError("Unversioned state_dict rejected. Use an explicit LegacyMultiStreamVQVAE "
                               "or LegacyMultiStreamResidualVQVAE path for old checkpoints.")
        self._validate_extra_state(state_dict["_extra_state"])
        return super().load_state_dict(state_dict,strict=strict,assign=assign)
    def _streams_in_order(self):
        return list(self.stream_order)
    def _split_pose(self,x):
        if x.ndim!=3 or x.shape[-1]!=534:
            raise ValueError(f"Expected (B,T,534), got {tuple(x.shape)}")
        batch,time,_=x.shape; points=x.reshape(batch,time,178,3)
        return {n:points.index_select(2,getattr(self,"_group_index_"+n)).reshape(batch,time,self.stream_dims[n])
                for n in self.stream_order}
    def _decode_latents(self,quantized):
        # Full-pose decoder channels match original dataset order, also for random groups.
        return self.decoder(self.proj_in(torch.cat(quantized,dim=1))).permute(0,2,1)
    def forward(self,x):
        split=self._split_pose(x); quantized=[]; total_loss=0.0; ppls={}
        for n in self.stream_order:
            z=self.encoders[n](split[n])
            if self.num_quantization_levels==1:
                q,loss,ppl=self.quantizers[n](z); quantized.append(q); total_loss=total_loss+loss; ppls[n]=ppl
            else:
                base,lb,pb=self.quantizers_base[n](z)
                residual,lr,pr=self.quantizers_res[n](z-base.detach())
                quantized.append(base+residual); total_loss=total_loss+lb+lr
                ppls[n+"_base"]=pb; ppls[n+"_res"]=pr
        return self._decode_latents(quantized),total_loss,ppls
    @staticmethod
    def _indices(q,z):
        batch,dim,time=z.shape
        flat=z.permute(0,2,1).contiguous().view(-1,dim)
        return q.quantize(flat).view(batch,time)
    def encode(self,x):
        split=self._split_pose(x); tokens={}
        for n in self.stream_order:
            z=self.encoders[n](split[n])
            if self.num_quantization_levels==1:
                tokens[n]=self._indices(self.quantizers[n],z)
            else:
                base=self._indices(self.quantizers_base[n],z)
                base_q=self.quantizers_base[n].dequantize(base).permute(0,2,1)
                residual=self._indices(self.quantizers_res[n],z-base_q)
                tokens[n+"_base"]=base; tokens[n+"_res"]=residual
        return tokens
    def forward_decoder(self,tokens_dict):
        if set(tokens_dict)!=set(self.token_order):
            raise ValueError(f"Token keys must be exactly {self.token_order}")
        shapes={tuple(x.shape) for x in tokens_dict.values()}
        if len(shapes)!=1 or len(next(iter(shapes)))!=2:
            raise ValueError("All token streams must have matching (B,T) shape")
        quantized=[]
        for n in self.stream_order:
            if self.num_quantization_levels==1: q=self.quantizers[n].dequantize(tokens_dict[n])
            else:
                q=self.quantizers_base[n].dequantize(tokens_dict[n+"_base"])
                q=q+self.quantizers_res[n].dequantize(tokens_dict[n+"_res"])
            quantized.append(q.permute(0,2,1))
        return self._decode_latents(quantized)


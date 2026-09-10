"""Grouped local mask generator and action-only training objectives.

The model, sampler and losses are the numerical implementation used in the
supplementary interface experiment. This is a local DFM-style implementation.
"""
from pathlib import Path
from types import SimpleNamespace
import math,sys
import numpy as np
import torch
PUBLIC=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PUBLIC/'runtime/code/src'))
from models.t2m_dfm_cross import CrossAttnDFM

def require(condition,message):
    if not condition:raise ValueError(message)
audit=SimpleNamespace(require=require)


def validate_tokens(tokens,p):
 raw=np.asarray(tokens)
 if not np.issubdtype(raw.dtype,np.integer):raise ValueError('Action token IDs must have integer dtype')
 values=np.asarray(raw,dtype=np.int64)
 if values.ndim!=1 or not len(values) or len(values)>p['max_motion_tokens'] or len(values)%p['token_group_size']:
  raise ValueError('Action tokens must be nonempty complete groups within480 budget')
 offsets=np.cumsum([0]+p['codebook_sizes'][:-1])
 for i,(offset,size) in enumerate(zip(offsets,p['codebook_sizes'])):
  selected=values[i::6]
  if np.any(selected<offset) or np.any(selected>=offset+size):raise ValueError('Action cache contains sentinel or wrong-group code')
 return values



def collate_actions(rows,tokenizer_pad_id):
 from torch.nn.utils.rnn import pad_sequence
 lengths=torch.tensor([len(r['tokens']) for r in rows],dtype=torch.long)
 maximum=int(lengths.max());clean=torch.zeros(len(rows),maximum,dtype=torch.long)
 for i,r in enumerate(rows):clean[i,:len(r['tokens'])]=torch.as_tensor(r['tokens'],dtype=torch.long)
 pad_mask=torch.arange(maximum)[None,:]>=lengths[:,None]
 return {'sample_ids':[r['sid'] for r in rows],
         'input_ids':pad_sequence([r['input_ids'] for r in rows],batch_first=True,padding_value=tokenizer_pad_id),
         'attention_mask':pad_sequence([r['attention_mask'] for r in rows],batch_first=True,padding_value=0),
         'clean_tokens':clean,'pad_mask':pad_mask,'motion_length':lengths,
         'length_loss_valid':torch.tensor([not r['source_truncated'] for r in rows],dtype=torch.bool)}



class GroupedDFM(CrossAttnDFM):
 """Same parameters and frozen sample update, with inference-only group support."""
 def __init__(self,p,**overrides):
  args=dict(p['architecture']);args.update(overrides)
  super().__init__(num_vq=p['vocabulary_size'],block_size=p['position_capacity'],**args)
  self.group_sizes=tuple(p['codebook_sizes']);self.group_offsets=tuple(np.cumsum([0]+list(self.group_sizes[:-1])))
  self._restrict_sampling=False
 def forward(self,*args,**kwargs):
  logits=super().forward(*args,**kwargs)
  if self._restrict_sampling:
   classes=torch.arange(self.num_vq,device=logits.device)[None,:]
   group=torch.arange(logits.shape[1],device=logits.device)%len(self.group_sizes)
   offsets=torch.tensor(self.group_offsets,device=logits.device)[group,None]
   sizes=torch.tensor(self.group_sizes,device=logits.device)[group,None]
   allowed=(classes>=offsets)&(classes<offsets+sizes)
   logits=logits.masked_fill(~allowed[None,:,:],-torch.inf)
  return logits
 @torch.no_grad()
 def sample(self,*args,**kwargs):
  previous=self._restrict_sampling;self._restrict_sampling=True
  try:return super().sample(*args,**kwargs)
  finally:self._restrict_sampling=previous



def length_target_loss(log_prediction,lengths,valid):
 values=torch.nn.functional.smooth_l1_loss(log_prediction,torch.log(lengths.float()),reduction='none')
 if valid.any():return values[valid].mean()
 return log_prediction.sum()*0.0



def masked_ce(logits,clean,loss_mask):
 values=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),clean.reshape(-1),reduction='none')
 weights=loss_mask.reshape(-1).float()
 if weights.sum()>0:return (values*weights).sum()/weights.sum()
 return logits.sum()*0.0



def fixed_length(log_prediction,p):
 value=float(log_prediction)
 if not math.isfinite(value):raise ValueError('Nonfinite text length prediction')
 if value>700:raw=None;rounded=p['max_motion_tokens'];reason='text_length_overflow_clamped_to_cap'
 else:
  raw=math.exp(value)*p['sampling']['length_scale'];rounded=int(round(raw/6))*6
  reason='upper_cap' if rounded>p['max_motion_tokens'] else ('minimum_one_group' if rounded<6 else 'text_length_head')
 length=max(6,min(rounded,p['max_motion_tokens']))
 audit.require(length%6==0 and length<=p['position_capacity'],'Length cap broke complete group')
 return length,{'log_predicted_motion_tokens':value,'predicted_motion_tokens_before_rounding':raw,'rounded_motion_tokens_before_clamp':rounded,'motion_tokens':length,'pose_frames':length//6*4,'length_decision':reason}



def load_text(p,device):
 from transformers import MBart50TokenizerFast,MBartModel
 cfg=p['text_encoder'];kwargs={'revision':cfg['revision'],'local_files_only':True}
 tokenizer=MBart50TokenizerFast.from_pretrained(cfg['name'],src_lang=cfg['language'],**kwargs)
 full=MBartModel.from_pretrained(cfg['name'],**kwargs);encoder=full.encoder.to(device).eval();del full
 for parameter in encoder.parameters():parameter.requires_grad=False
 return tokenizer,encoder


"""Text-length-head generation with the local grouped DFM-style mask model."""
from pathlib import Path
import argparse,contextlib,copy,gzip,hashlib,json,os,pickle,sys
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
PUBLIC=Path(__file__).resolve().parents[2]

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    ap.add_argument('--checkpoint',type=Path,required=True,help='Completed local mask-model checkpoint')
    ap.add_argument('--vq-checkpoint',type=Path,required=True,help='Matching revised PHX PAMRVQ checkpoint, with mean.npy/std.npy beside it')
    ap.add_argument('--text-model',type=Path,help='Local mBART snapshot; otherwise use the already-cached pinned revision')
    ap.add_argument('--input-manifest',type=Path,required=True,help='ID/text-only manifest from references/prepare_inputs.py')
    ap.add_argument('--split',choices=['dev','test'],required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--dry-run',action='store_true',help='Validate configuration/inputs without loading checkpoints or generating poses')
    a=ap.parse_args(argv)
    if not 1<=a.threads<=4:ap.error('--threads must be in1..4')
    if a.out.exists() and any(a.out.iterdir()):ap.error('Output must be new or empty')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'references'))
    from reference_common import load_text_manifest,artifact,write_json,sha256
    p=json.loads(a.config.read_text(encoding='utf-8'))
    rows=load_text_manifest(a.input_manifest,'phix',a.split)
    assert (p['token_group_size'],p['max_motion_tokens'],p['max_pose_frames'],p['position_capacity'])==(6,480,320,512)
    assert p['sampling']['n_steps']==50 and all(p['sampling'][k]==1.0 for k in ('cfg_scale','length_scale','temperature'))
    if a.dry_run:
        print(json.dumps({'dry_run':True,'samples':len(rows),'dataset':'phix','split':a.split,'configuration':p,
                          'checkpoint_exists':a.checkpoint.is_file(),'vq_checkpoint_exists':a.vq_checkpoint.is_file(),'model_loaded':False}));return
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    sys.path.insert(0,str(PUBLIC/'tools'))
    import generate_motion as gm
    gm.prepare_imports(PUBLIC/'runtime')
    import grouped_mask as s
    device=torch.device(a.device)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    saved=checkpoint['args']['dfm_protocol']
    for key in ('dataset','vq_variant','layout_version','grouping','token_order','codebook_sizes','vocabulary_size',
                'max_pose_frames','token_group_size','max_motion_tokens','position_capacity','architecture','sampling','text_encoder'):
        if saved[key]!=p[key]:raise ValueError('Checkpoint/config mismatch: '+key)
    bound=checkpoint['args']['contract']['inputs']
    if bound['vq_checkpoint']['sha256']!=sha256(a.vq_checkpoint):raise ValueError('The mask model was trained with another quantizer')
    model=s.GroupedDFM(p);model.load_state_dict(checkpoint['model'],strict=True);model.to(device).eval()
    @contextlib.contextmanager
    def cpu_constructor():
        original=torch.Tensor.cuda
        def cpu_only(tensor,*args,**kwargs):
            if tensor.device.type!='cpu':raise ValueError('CPU constructor compatibility is limited to CPU buffers')
            return tensor
        torch.Tensor.cuda=cpu_only
        try:yield
        finally:torch.Tensor.cuda=original
    with cpu_constructor() if device.type=='cpu' else contextlib.nullcontext():
        vq,_,layout,factor,mean,std,stats=gm.load_vq(a.vq_checkpoint,'pamrvq','phix',device)
    if list(layout.names)!=p['token_order'] or list(layout.sizes)!=p['codebook_sizes'] or factor!=4:raise ValueError('VQ layout differs')
    for name in ('mean','std'):
        path=a.vq_checkpoint.parent/(name+'.npy')
        if sha256(path)!=bound[name]['sha256']:raise ValueError('Saved VQ normalization differs: '+name)
    text_config=copy.deepcopy(p)
    if a.text_model:text_config['text_encoder'].update(name=str(a.text_model.resolve()),revision=None)
    tokenizer,encoder=s.load_text(text_config,device)
    a.out.mkdir(parents=True,exist_ok=True)
    inputs={'checkpoint':artifact(a.checkpoint),'vq_checkpoint':artifact(a.vq_checkpoint),'normalization':stats,
            'text_input':artifact(a.input_manifest),'config':artifact(a.config),'sampler':artifact(s.__file__),
            'model':artifact(PUBLIC/'runtime/code/src/models/t2m_dfm_cross.py'),'vq_interface':artifact(gm.__file__)}
    predictions=[];metadata=[]
    with torch.no_grad():
        for row in rows:
            sid,text=row['id'],row['text']
            encoded=tokenizer(text,truncation=True,max_length=p['text_encoder']['max_text_len'],return_tensors='pt')
            text_ids=encoded['input_ids'].to(device);attention=encoded['attention_mask'].to(device)
            memory=encoder(input_ids=text_ids,attention_mask=attention).last_hidden_state
            projected,kpm=model.encode_text_memory(memory,attention)
            log_length=model.predict_motion_length(projected,kpm)
            length,length_info=s.fixed_length(log_length.item(),p)
            seed=gm.sample_seed(p['sampling']['generation_seed'],'phix',a.split,sid)
            with gm.independent_torch_rng(seed,device):
                generated=model.sample(memory,attention,lengths=torch.tensor([length],device=device),
                    n_steps=p['sampling']['n_steps'],cfg_scale=p['sampling']['cfg_scale'],temperature=p['sampling']['temperature'])
            tokens=layout.validate(generated[0,:length].detach().cpu().tolist())
            motion=gm.decode_tokens(vq,'pamrvq',tokens,layout,mean,std,device,factor)
            if len(motion)!=length//6*4 or len(motion)>320:raise ValueError('Generated frame budget differs')
            predictions.append({'name':sid,'signer':'','text':text,'gloss':'','sign':torch.from_numpy(motion.copy())})
            metadata.append({'sample_id':sid,'sample_seed':seed,'input_text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                             'length_source':'text_head_only',**length_info,'tokens':tokens,'generated_frames':len(motion)})
    path=a.out/(a.split+'.pickle')
    with path.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,filename='',mode='wb',mtime=0) as f:pickle.dump(predictions,f,protocol=pickle.HIGHEST_PROTOCOL)
    write_json(a.out/'samples.json',metadata)
    write_json(a.out/'manifest.json',{'status':'completed','dataset':'phix','split':a.split,'samples':len(rows),'inputs':inputs,
        'sampling':p['sampling'],'device':str(device),'length_uses_ground_truth_frames':False,
        'predictions':artifact(path),'sample_metadata':artifact(a.out/'samples.json'),
        'cpu_constructor_compatibility':device.type=='cpu','text_model_location':str(a.text_model) if a.text_model else p['text_encoder'],
        'role':'Local mask-generator interface experiment, not an equal-architecture AR comparison'})
    print('Generated',len(rows),'samples in',a.out)

if __name__=='__main__':main()

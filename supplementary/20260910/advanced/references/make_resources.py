"""Hash caller-supplied checkpoints and normalization; does not copy weights."""
from pathlib import Path
import argparse

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['phix','csl'],required=True)
    p.add_argument('--method',choices=['mdm','momask','motiongpt_style'],required=True)
    for name in ('mean','std','vq','base','residual','mdm','mbart-directory','mt5-directory'):
        p.add_argument('--'+name,type=Path,required=name in ('mean','std'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('Refusing to overwrite resource manifest')
    from reference_common import VERSION,artifact,write_json
    def required(value,label):
        if value is None:raise ValueError('Missing --'+label)
        return value
    ck={};result={'schema_version':VERSION,'datasets':{a.dataset:{'prepared_data':{'Mean.npy':artifact(a.mean),'Std.npy':artifact(a.std)},'checkpoints':ck}}}
    if a.method=='mdm':ck['mdm_'+a.dataset]=artifact(required(a.mdm,'mdm'))
    else:
        ck['momask_vq_'+a.dataset]=artifact(required(a.vq,'vq'))
        if a.method=='momask':
            ck['momask_mtrans_'+a.dataset]=artifact(required(a.base,'base'))
            ck['momask_rtrans_'+a.dataset]=artifact(required(a.residual,'residual'))
        else:
            folder=required(a.mt5_directory,'mt5-directory').resolve()
            weights=[folder/name for name in ('model.safetensors','pytorch_model.bin') if (folder/name).is_file()]
            if len(weights)!=1:raise ValueError('Expected one local mT5 weight file, as in the evaluated adaptation')
            files={f.name:{**artifact(f),'exists':True} for f in folder.iterdir() if f.is_file()}
            ck['mgpt_'+a.dataset]={'hf_best':{'directory':str(folder),'weights':artifact(weights[0]),'files':files}}
    if a.method!='motiongpt_style':
        folder=required(a.mbart_directory,'mbart-directory').resolve()
        files=[f for f in folder.iterdir() if f.is_file() and f.suffix in ('.json','.model','.bin','.safetensors','.txt')]
        if not any(f.suffix in ('.bin','.safetensors') for f in files):raise ValueError('Local mBART weights missing')
        result.update(mbart_directory=str(folder),mbart_files=[artifact(f) for f in sorted(files)])
    write_json(a.out,result);print('Wrote',a.out)

if __name__=='__main__':main()

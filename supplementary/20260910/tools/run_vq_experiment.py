"""Run a declared VQ experiment in the isolated workspace with an audit trail."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,sys,time,zipfile

BASE=Path(__file__).resolve().parents[1]
SRC=BASE/'runtime'/'code'/'src'
PYTHON=Path(sys.executable)
COMMON=['--batch-size','128','--window-size','64','--total-iter','100000',
        '--warm-up-iter','1000','--lr','2e-4','--lr-scheduler','50000','80000',
        '--gamma','0.05','--weight-decay','0','--commit','0.02','--loss-vel','0.1',
        '--recons-loss','l2','--code-dim','512','--output-emb-width','512',
        '--stride-t','2','--width','512','--depth','3','--dilation-growth-rate','3',
        '--quantizer','ema_reset','--print-iter','500','--eval-iter','2000',
        '--save-iter','20000','--num-workers','0','--device','cuda',
        '--early-stop-patience','4','--early-stop-min-delta','0.001',
        '--min-iter-before-early-stop','5000']

def stamp():return datetime.datetime.now().astimezone().isoformat(timespec='seconds')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def save(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',choices=['phix','csl'],required=True)
    ap.add_argument('--variant',choices=['baseline','single_large','tlrvq','pamvq','pamrvq','pamrvq_random'],required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--grouping-seed',type=int,default=20260910)
    ap.add_argument('--data-root',type=Path,default=Path(os.environ.get('SLP_SIGN_DATA_DIR',str(BASE/'runtime/data'))))
    ap.add_argument('--run-root',type=Path,default=BASE/'runs')
    ap.add_argument('--python',type=Path,default=PYTHON)
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    name=f'{a.dataset}_{a.variant}_s{a.seed}'
    out=a.run_root.resolve()/name
    ckbase=out/'checkpoints'
    if a.variant in ('baseline','single_large'):
        script='train_vq_sign.py'
        k=4096 if a.dataset=='csl' else (1024 if a.variant=='single_large' else 512)
        extra=['--nb-code',str(k)]
    elif a.variant=='tlrvq':
        script='train_vq_sign_rvq.py';k=2048 if a.dataset=='csl' else 512
        extra=['--nb-code',str(k),'--nb-code-residual',str(k)]
    elif a.variant=='pamvq':
        script='train_vq_sign_ms.py'
        body,hand,face=(1024,1024,1024) if a.dataset=='csl' else (64,1024,256)
        extra=['--nb-code','256','--nb-code-body',str(body),'--nb-code-hand',str(hand),'--nb-code-face',str(face),'--hand-weight','1']
    else:
        script='train_vq_sign_msr.py'
        body,hand,face=(512,512,512) if a.dataset=='csl' else (32,512,128)
        extra=[]
        for layer in ['base','res']:
            for part,k in [('body',body),('hand',hand),('face',face)]:extra.extend([f'--nb-{layer}-{part}',str(k)])
    if a.variant in ('pamvq','pamrvq','pamrvq_random'):
        extra+=['--layout-version','reviewer55_verified_v1','--grouping','random' if a.variant=='pamrvq_random' else 'anatomical','--grouping-seed',str(a.grouping_seed)]
    command=[str(a.python),'-u',str(SRC/script),'--dataname',a.dataset+'_lift3d','--exp-name','model','--out-dir',str(ckbase),'--down-t','2' if a.dataset=='phix' else '1','--seed',str(a.seed),*COMMON,*extra]
    if a.dry_run:
        print(json.dumps({'stage':'vq','dataset':a.dataset,'variant':a.variant,'seed':a.seed,'data_root':str(a.data_root.resolve()),'run':str(out),'command':command,'dry_run':True},indent=2))
        return
    if out.is_relative_to(a.data_root.resolve()):raise ValueError('Output must not be inside data input directory')
    if out.exists():raise RuntimeError(f'Refusing to overwrite experiment: {out}')
    stem='phix_lift3d' if a.dataset=='phix' else 'csl_daily_lift3d'
    for split in ('train','dev'):
        if not (a.data_root/a.dataset/f'{stem}.{split}.pt').is_file():raise FileNotFoundError(a.data_root/a.dataset/f'{stem}.{split}.pt')
    os.environ['SLP_SIGN_DATA_DIR']=str(a.data_root.resolve())
    out.mkdir(parents=True)
    source_files={str(p.relative_to(SRC)):sha(p) for p in SRC.rglob('*.py') if '__pycache__' not in p.parts}
    with zipfile.ZipFile(out/'source_at_start.zip','w',zipfile.ZIP_DEFLATED) as z:
        for rel in source_files:z.write(SRC/rel,rel)
    manifest={'schema':'reviewer55.vq_run.v1','status':'running','started_at':stamp(),'dataset':a.dataset,'variant':a.variant,'training_seed':a.seed,'grouping_seed':a.grouping_seed,'command':command,'cwd':str(SRC),'source_sha256':source_files,'model_selection':'lowest deterministic center-window validation coordinate MSE, with common early stopping','notes':['All variants share coordinate L2 and the inherited auxiliary coordinate loss. No extra hand-weighting in the revised controlled comparisons.','Training window length64. Short sequences are excluded from first-stage training/validation exactly as in the inherited loader. Full available dev/test sequences are retained for downstream evaluation.','This is a revised, traceable experiment. It is not a claim to reproduce an untraced historical table entry.']}
    save(out/'manifest.json',manifest)
    env=os.environ.copy();env['PYTHONUTF8']='1';env['PYTHONIOENCODING']='utf-8';env['TOKENIZERS_PARALLELISM']='false'
    began=time.monotonic()
    with (out/'console.log').open('w',encoding='utf-8') as log:
        proc=subprocess.Popen(command,cwd=SRC,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
        manifest['pid']=proc.pid;save(out/'manifest.json',manifest)
        for line in proc.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        code=proc.wait()
    manifest.update(status='completed' if code==0 else 'failed',exit_code=code,finished_at=stamp(),elapsed_seconds=round(time.monotonic()-began,3))
    manifest['outputs']=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in (ckbase/'model').glob('*') if p.is_file()]
    if code==0 and not (ckbase/'model'/'best.pt').exists():manifest.update(status='failed',error='best checkpoint missing')
    save(out/'manifest.json',manifest)
    print(json.dumps({'run':name,'status':manifest['status'],'elapsed_seconds':manifest['elapsed_seconds']},ensure_ascii=False),flush=True)
    return 0 if manifest['status']=='completed' else 1

if __name__=='__main__':raise SystemExit(main())

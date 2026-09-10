"""Frozen, dataset-specific back-translation with traceable paired text outputs."""
from pathlib import Path
import argparse,datetime,gzip,hashlib,json,os,sys,time
import numpy as np
import torch

BASE=Path(__file__).resolve().parents[1]
OFFICIAL=BASE/'runtime'/'bt_eval_kit'/'slrtp_official'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat(timespec='seconds')
def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def tensor_pose(x):
    t=torch.as_tensor(x,device='cpu',dtype=torch.float32)
    if t.ndim==2:t=t.reshape(t.shape[0],178,3)
    if t.ndim!=3 or tuple(t.shape[1:])!=(178,3) or len(t)==0 or not torch.isfinite(t).all():
        raise ValueError('Invalid, empty or nonfinite178-point pose')
    return t

def main(argv=None):
    global OFFICIAL
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',choices=['phix','csl'],required=True)
    ap.add_argument('--split',choices=['dev','test'],required=True)
    ap.add_argument('--pred',type=Path,help='omit only for real-pose evaluator calibration')
    ap.add_argument('--kind',choices=['generated','reconstruction','ground_truth','reference_method'],required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,default=BASE/'configs/protocol.json')
    ap.add_argument('--data-root',type=Path,default=Path(os.environ.get('SLP_SIGN_DATA_DIR',str(BASE/'runtime/data'))))
    ap.add_argument('--evaluator-dir',type=Path,required=True,help='External frozen evaluator directory with best.ckpt/config/vocab files')
    ap.add_argument('--official-dir',type=Path,default=OFFICIAL)
    a=ap.parse_args(argv)
    OFFICIAL=a.official_dir.resolve()
    sys.path.insert(0,str(OFFICIAL))
    from back_translation.back_translate import make_back_translation_model,back_translate
    from metrics import bleu,chrf,rouge,wer
    out=a.out.resolve()
    if out.exists() and any(out.iterdir()):raise RuntimeError('Evaluation output must be empty')
    if (a.pred is None)!=(a.kind=='ground_truth'):raise ValueError('Ground-truth calibration must omit --pred; all other kinds require --pred')
    out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(a.protocol.read_text(encoding='utf-8'))
    model_dir=a.evaluator_dir.resolve()
    for f in ['best.ckpt','config.yaml','gls.vocab','txt.vocab']:
        if not (model_dir/f).is_file():raise FileNotFoundError(model_dir/f)
    stem='phix_lift3d' if a.dataset=='phix' else 'csl_daily_lift3d'
    gt_path=a.data_root.resolve()/a.dataset/f'{stem}.{a.split}.pt'
    gt=torch.load(gt_path,map_location='cpu',weights_only=False)
    ids=list(gt)
    refs=[gt[sid]['text'] for sid in ids]
    if a.pred is None:
        poses=[tensor_pose(gt[sid]['poses_3d']) for sid in ids]
    else:
        import pickle
        with gzip.open(a.pred,'rb') as f:raw=pickle.load(f)
        pred={}
        for row in raw:
            sid=row['name']
            if sid in pred:raise ValueError(f'Duplicate prediction ID: {sid}')
            pred[sid]=tensor_pose(row['sign'])
        if set(pred)!=set(ids):raise ValueError(f'Prediction/reference IDs differ: missing{len(set(ids)-set(pred))}, extra{len(set(pred)-set(ids))}')
        poses=[pred[sid] for sid in ids]
    native_lengths=[len(p) for p in poses]
    if a.dataset=='phix':poses=[p[::2] for p in poses]
    manifest={'schema':'reviewer55.backtranslation.v1','status':'running','started_at':now(),'dataset':a.dataset,'split':a.split,'kind':a.kind,'samples':len(ids),'complete_split':True,'protocol_path':str(a.protocol.resolve()),'protocol_sha256':sha(a.protocol),'reference_path':str(gt_path),'reference_sha256':sha(gt_path),'prediction_path':str(a.pred.resolve()) if a.pred else None,'prediction_sha256':sha(a.pred) if a.pred else sha(gt_path),'evaluator_dir':str(model_dir),'evaluator_files':{f:sha(model_dir/f) for f in ['best.ckpt','config.yaml','gls.vocab','txt.vocab']},'evaluator_source':{str(p.relative_to(OFFICIAL)):sha(p) for p in OFFICIAL.rglob('*.py') if '__pycache__' not in p.parts},'wrapper_sha256':sha(__file__),'beam_size':3,'beam_alpha':-1,'prediction_subsampling':2 if a.dataset=='phix' else 1,'native_prediction_length':{'min':min(native_lengths),'mean':float(np.mean(native_lengths)),'max':max(native_lengths)},'pose_metrics':'separate verified dataset-specific skeleton protocol, not the official PHX-only joint indices on CSL'}
    save(out/'manifest.json',manifest)
    began=time.monotonic()
    try:
        bt=make_back_translation_model(model_dir)
        predictions=back_translate(bt,poses)
        if len(predictions)!=len(ids):raise RuntimeError('Back-translation output count mismatch')
        def norm(s):return ' '.join(list(s.replace(' ',''))) if a.dataset=='csl' else s
        eval_refs=[norm(s) for s in refs];eval_preds=[norm(s) for s in predictions]
        scores={'bleu':bleu(hypotheses=eval_preds,references=eval_refs),'chrf':chrf(hypotheses=eval_preds,references=eval_refs),'rouge':rouge(hypotheses=eval_preds,references=eval_refs),'wer':wer(hypotheses=eval_preds,references=eval_refs),'n_samples':len(ids)}
        with (out/'texts.jsonl').open('w',encoding='utf-8') as f:
            for sid,r,p,er,ep in zip(ids,refs,predictions,eval_refs,eval_preds):
                f.write(json.dumps({'id':sid,'reference':er,'prediction':ep,'raw_reference':r,'raw_prediction':p},ensure_ascii=False)+'\n')
        save(out/'metrics.json',scores)
        manifest.update(status='completed',finished_at=now(),elapsed_seconds=round(time.monotonic()-began,3),scores=scores,outputs={f:sha(out/f) for f in ['metrics.json','texts.jsonl']})
        save(out/'manifest.json',manifest)
        print(json.dumps({'dataset':a.dataset,'split':a.split,'kind':a.kind,**scores,'out':str(out)},ensure_ascii=False),flush=True)
    except BaseException as e:
        manifest.update(status='failed',finished_at=now(),error=f'{type(e).__name__}: {e}')
        save(out/'manifest.json',manifest)
        raise

if __name__=='__main__':main()

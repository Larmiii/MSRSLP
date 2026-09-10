"""Apply the two fixed offline treatments to complete, unpadded pose sequences."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import argparse,json,sys
import numpy as np
import skeleton_metrics_cpu as sm
import projection_core as core

HERE=Path(__file__).resolve().parent

def topology(config):
    primary=[dict(e) for e in config['edges'] if e['kind']=='rigid']
    palms=[dict(e) for e in config['edges'] if e['group'] in ('right_palm_distances','left_palm_distances')]
    arms=[e for e in primary if e['group']=='arm_segments']
    fingers=[e for e in primary if 'finger_segments' in e['group']]
    core.require(len(primary)==34 and len(palms)==10 and len(fingers)==30,'Unexpected segment sets')
    core.require([(e['a'],e['b']) for e in arms]==[(0,1),(1,2),(3,4),(4,5)],'Unexpected arm FK order')
    core.require(all(e['kind']=='reference' for e in palms),'Palm distances must remain reference distances')
    initialized={0,3,6,7,8,29}
    for edge in arms+palms+fingers:
        core.require(edge['a'] in initialized,'Edges must follow parent-before-child order')
        initialized.add(edge['b'])
    core.require(initialized==set(range(50)),'Uncovered body/hand point')
    return {'primary':primary,'palms':palms,'arms':arms,'fingers':fingers}

def project_source(raw,reference,edges,permutation,treatment):
    original=core.strict_pose(raw)
    canonical=original[:,permutation,:]
    result,events,_=core.project_pose(canonical,reference,edges,treatment)
    restored=np.empty_like(result);restored[:,permutation,:]=result
    core.require(np.array_equal(restored[:,permutation[50:]],original[:,permutation[50:]].astype(np.float32)),'Face changed')
    core.require(restored.shape==original.shape,'Frame count or pose shape changed')
    return restored,events

def save(path,value):
    path.write_text(json.dumps(sm.clean_json(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',choices=['phix','csl'],required=True)
    ap.add_argument('--train',type=Path,required=True,help='Original TRAIN poses in the same source layout and coordinate units')
    ap.add_argument('--pred',type=Path,required=True,help='Complete unpadded generated sequences; no automatic inverse normalization')
    ap.add_argument('--config',type=Path,help='Default: supplied dataset-specific configuration')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--diagnostics',action='store_true',help='Also compute before/after E4 metrics using this same TRAIN reference')
    args=ap.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):ap.error('Output must be new or empty')
    cfg_path=args.config or HERE/'configurations'/f'{args.dataset}_segments.json'
    config=sm.validate_config(json.loads(cfg_path.read_text(encoding='utf-8')))
    permutation=np.asarray(config['canonical_to_source']);edges=topology(config)
    expected=list(range(178)) if args.dataset=='phix' else list(range(8))+list(range(157,178))+list(range(136,157))+list(range(8,136))
    core.require(permutation.tolist()==expected,'Dataset source layout differs')
    inputs={'train':sm.file_info(args.train),'predictions':sm.file_info(args.pred),'config':sm.file_info(cfg_path),
            'algorithm':sm.file_info(core.__file__),'entry_point':sm.file_info(__file__),'metrics':sm.file_info(sm.__file__)}
    # The unchanged fitting function canonicalizes TRAIN with the same mapping.
    reference=core.fit_reference(sm.load_samples(args.train),config,edges)
    args.out.mkdir(parents=True,exist_ok=True)
    save(args.out/'train_reference.json',{'inputs':inputs,'reference':reference})
    import torch
    torch.set_num_threads(1)
    samples=list(sm.load_samples(args.pred));ids=[x[0] for x in samples]
    core.require(samples and len(ids)==len(set(ids)),'Empty predictions or duplicate IDs')
    outputs={}
    for treatment in core.TREATMENTS:
        records=[];metadata=[]
        for sid,pose,meta in samples:
            if 'length' in meta and int(meta['length'])!=len(pose):raise ValueError('Padded input: provide unpadded sequences explicitly')
            result,events=project_source(pose,reference,edges,permutation,treatment)
            records.append({'name':sid,'signer':meta.get('signer',''),'text':meta.get('text',''),'gloss':meta.get('gloss',''),
                            'sign':torch.from_numpy(result.reshape(len(result),534).copy())})
            metadata.append({'sample_id':sid,'frames':len(result),'fallback_events':events})
        path=args.out/(treatment+'.pickle');core.write_slt(path,records)
        save(args.out/(treatment+'.samples.json'),metadata)
        outputs[treatment]=sm.file_info(path)
    if args.diagnostics:
        for name,path in [('raw',args.pred)]+[(k,Path(v['path'])) for k,v in outputs.items()]:
            results,per_sample,per_edge=sm.evaluate(sm.load_samples(path),config,reference['reference_statistics'],reference_split='train')
            folder=args.out/('diagnostics_'+name);folder.mkdir()
            save(folder/'summary.json',{'provenance':inputs,'condition':name,'results':results})
            sm.write_csv(folder/'per_sample_metrics.csv',per_sample);sm.write_csv(folder/'per_edge_metrics.csv',per_edge)
    for key,path in [('train',args.train),('predictions',args.pred)]:
        core.require(sm.file_info(path)==inputs[key],'Input changed during processing')
    save(args.out/'manifest.json',{'status':'completed','dataset':args.dataset,'samples':len(ids),'sample_ids':ids,'inputs':inputs,
        'outputs':outputs,'treatments':list(core.TREATMENTS),'kernel':[.25,.5,.25],
        'coordinate_units':'Released keypoint coordinates, not millimetres','target_lengths':'TRAIN temporal mean per sample, then equal-sample mean',
        'smooth_indices_source':permutation[:50].tolist(),'unchanged_face_source':permutation[50:].tolist(),
        'offline':True,'future_direction_fallback_recorded':True,
        'interpretation':'CV is reduced by construction; compare back-translation, motion amplitude and acceleration, and fixed visual examples.',
        'extension_disclosure':'The same fixed algorithm was applied to CSL after the E4 diagnostic, without tuning its kernel.' if args.dataset=='csl' else None})
    print(json.dumps({'samples':len(ids),'dataset':args.dataset,'out':str(args.out.resolve())}))

if __name__=='__main__':main()

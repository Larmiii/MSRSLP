"""Prepare ID/text inference inputs, or a TRAIN-only text-length predictor."""
from pathlib import Path
import argparse

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True,help='Trusted released pose .pt file')
    p.add_argument('--dataset',choices=['phix','csl'],required=True)
    p.add_argument('--split',choices=['train','dev','test'],required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('Refusing to overwrite output')
    import torch
    from reference_common import VERSION,artifact,fit_length,write_json
    raw=torch.load(a.data,map_location='cpu',weights_only=False)
    if not isinstance(raw,dict) or not raw:raise ValueError('Expected nonempty released sample dictionary')
    rows=[]
    for sid,row in raw.items():
        text=row.get('text','') or ''
        if not isinstance(sid,str) or not isinstance(text,str):raise ValueError('Invalid ID/text')
        item={'id':sid,'text':text}
        if a.split=='train':
            pose=row.get('poses_3d',row.get('sign'))
            if pose is None:raise ValueError('Training frame count unavailable')
            item['frames']=len(pose)
        rows.append(item)
    rows.sort(key=lambda row:row['id'])
    if a.split=='train':
        result=fit_length(rows,a.dataset);result['source']=artifact(a.data)
    else:
        result={'schema_version':VERSION,'dataset':a.dataset,'split':a.split,'source':artifact(a.data),'samples':rows}
    write_json(a.out,result)
    print('Wrote',a.out,'for',len(rows),'samples')

if __name__=='__main__':main()

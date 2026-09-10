"""Fixed offline segment projection. Numerical functions retained from the experiment.

Canonical order: body0:8, right hand8:29, left hand29:50, face50:178.
The public CLI converts dataset source order before/after these functions.
"""
from pathlib import Path
import gzip,pickle
from types import SimpleNamespace
import numpy as np
import skeleton_metrics_cpu as sm

def require(condition,message):
    if not condition:raise ValueError(message)
provenance=SimpleNamespace(require=require)
EPSILON=1e-8
FLOAT32_TOL=2e-6
TREATMENTS=('bone_projection','smooth3_then_projection')


def strict_pose(raw):
    array=sm.as_array(raw)
    if array.ndim==2 and array.shape[1]==534:array=array.reshape(-1,178,3)
    if array.ndim!=3 or array.shape[1:]!=(178,3) or len(array)<1 or not np.isfinite(array).all():
        raise ValueError(f'Invalid complete PHX pose: {array.shape}')
    return np.asarray(array,dtype=np.float64)



def fit_reference(records,config,topology):
    """Equal-sample mean lengths via the unchanged common metric implementation."""
    edges=topology['primary'];a=np.array([e['a'] for e in edges]);b=np.array([e['b'] for e in edges])
    direction_sum=np.zeros((len(edges),3));direction_samples=np.zeros(len(edges),dtype=np.int64)
    direction_frames=np.zeros(len(edges),dtype=np.int64);invalid_frames=np.zeros(len(edges),dtype=np.int64)
    first_valid=[None]*len(edges);n_samples=0;n_frames=0
    def tapped():
        nonlocal n_samples,n_frames
        for sid,raw,metadata in records:
            pose=sm.canonical_pose(raw,config,metadata)
            n_samples+=1;n_frames+=len(pose)
            if len(pose)>=2:
                vectors=pose[:,b]-pose[:,a];norms=np.linalg.norm(vectors,axis=-1)
                valid=np.isfinite(vectors).all(axis=-1)&np.isfinite(norms)&(norms>EPSILON)
                for index in range(len(edges)):
                    good=np.flatnonzero(valid[:,index]);direction_frames[index]+=len(good);invalid_frames[index]+=len(pose)-len(good)
                    if len(good):
                        units=vectors[good,index]/norms[good,index,None]
                        direction_sum[index]+=units.mean(axis=0);direction_samples[index]+=1
                        if first_valid[index] is None:first_valid[index]={'sample_id':sid,'frame':int(good[0]),'unit_vector':units[0].tolist()}
            yield sid,raw,metadata
    common_reference=sm.collect_reference(tapped(),config,EPSILON)
    positions={e['name']:i for i,e in enumerate(config['edges'])}
    values=[]
    for index,e in enumerate(edges):
        ref_index=positions[e['name']]
        length=common_reference['sample_mean_reference']['mean'][ref_index]
        provenance.require(np.isfinite(length) and length>EPSILON,f'No usable training length for {e["name"]}')
        provenance.require(direction_samples[index]>0,f'No valid training direction for {e["name"]}')
        direction=direction_sum[index]/direction_samples[index];norm=np.linalg.norm(direction)
        if norm>EPSILON:
            unit=direction/norm;policy='normalized equal-sample mean of within-sample mean unit vectors'
        else:
            unit=np.array(first_valid[index]['unit_vector']);policy='training mean unit vector cancelled; deterministic first valid TRAIN frame'
        values.append({**e,'target_length':float(length),
            'length_valid_samples':common_reference['sample_mean_reference']['count'][ref_index],
            'length_between_sample_std_ddof0':common_reference['sample_mean_reference']['std_ddof0'][ref_index],
            'fallback_unit_direction':unit.tolist(),'fallback_direction_policy':policy,
            'direction_valid_samples':int(direction_samples[index]),'direction_valid_frames':int(direction_frames[index]),
            'direction_invalid_or_zero_frames':int(invalid_frames[index]),'first_valid_train_direction':first_valid[index]})
    return {'reference_split':'train','total_samples_read':n_samples,'total_frames_read':n_frames,
            'length_aggregation':'mean TRAIN sample time-mean length, equal weight per valid sample, unchanged skeleton collect_reference',
            'reference_statistics':common_reference,'segments':values}



def smooth_three(pose):
    result=pose.copy();active=pose[:,:50]
    padded=np.pad(active,((1,1),(0,0),(0,0)),mode='edge')
    result[:,:50]=.25*padded[:-2]+.5*padded[1:-1]+.25*padded[2:]
    return result



def resolve_directions(vectors,fallback,edge_name):
    lengths=np.linalg.norm(vectors,axis=-1)
    valid=np.isfinite(vectors).all(axis=-1)&np.isfinite(lengths)&(lengths>EPSILON)
    good=np.flatnonzero(valid);bad=np.flatnonzero(~valid)
    units=np.zeros_like(vectors,dtype=np.float64)
    units[good]=vectors[good]/lengths[good,None]
    records=[]
    for frame in bad:
        if len(good):
            # np.argmin breaks equal-distance ties at the smaller, past index.
            source=int(good[np.argmin(np.abs(good-frame))]);units[frame]=units[source]
            records.append({'edge':edge_name,'frame':int(frame),'reason':'zero_or_nonfinite_direction',
                            'fallback':'nearest_valid_same_sample','source_frame':source,
                            'offset_frames':source-int(frame),'uses_future':source>frame})
        else:
            units[frame]=fallback
            records.append({'edge':edge_name,'frame':int(frame),'reason':'zero_or_nonfinite_direction',
                            'fallback':'training_reference_direction','source_frame':None,
                            'offset_frames':None,'uses_future':False})
    return units,records



def project_pose(raw,reference,topology,treatment):
    if treatment not in TREATMENTS:raise ValueError('Undeclared treatment')
    original=strict_pose(raw)
    source=smooth_three(original) if treatment=='smooth3_then_projection' else original.copy()
    result=source.copy();by_name={e['name']:e for e in reference['segments']}
    directions={};fallbacks=[]
    # All directions/vectors are read before any FK coordinate update.
    for e in topology['primary']:
        ref=by_name[e['name']]
        direction,events=resolve_directions(source[:,e['b']]-source[:,e['a']],np.array(ref['fallback_unit_direction']),e['name'])
        directions[e['name']]=direction;fallbacks.extend(events)
    palm_vectors={e['name']:(source[:,e['b']]-source[:,e['a']]).copy() for e in topology['palms']}
    for e in topology['arms']:
        result[:,e['b']]=result[:,e['a']]+directions[e['name']]*by_name[e['name']]['target_length']
    result[:,8]=result[:,2];result[:,29]=result[:,5]
    for e in topology['palms']:result[:,e['b']]=result[:,e['a']]+palm_vectors[e['name']]
    for e in topology['fingers']:
        result[:,e['b']]=result[:,e['a']]+directions[e['name']]*by_name[e['name']]['target_length']
    # Save the established SLRTP float32 source layout, then validate that format.
    result=result.astype(np.float32)
    verify_constraints(original,source,result,reference,topology)
    return result,fallbacks,source



def verify_constraints(original,source,result,reference,topology):
    provenance.require(result.shape==original.shape and np.isfinite(result).all(),'Postprocess shape/nonfinite failure')
    provenance.require(np.array_equal(result[:,50:],original[:,50:].astype(np.float32)),'Face coordinates changed')
    provenance.require(np.array_equal(result[:,[0,3]],source[:,[0,3]].astype(np.float32)),'Projection moved shoulder roots')
    provenance.require(np.array_equal(result[:,8],result[:,2]) and np.array_equal(result[:,29],result[:,5]),'Duplicate wrists do not coincide')
    by_name={e['name']:e for e in reference['segments']}
    pose=result.astype(np.float64)
    for e in topology['primary']:
        lengths=np.linalg.norm(pose[:,e['b']]-pose[:,e['a']],axis=-1)
        provenance.require(np.max(np.abs(lengths-by_name[e['name']]['target_length']))<FLOAT32_TOL,'Fixed length outside serialization tolerance')
    for e in topology['palms']:
        actual=pose[:,e['b']]-pose[:,e['a']];expected=source[:,e['b']]-source[:,e['a']]
        provenance.require(np.max(np.abs(actual-expected))<FLOAT32_TOL,'Palm reference vector was constrained/changed during projection')



def write_slt(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as stream:pickle.dump(rows,stream,protocol=pickle.HIGHEST_PROTOCOL)


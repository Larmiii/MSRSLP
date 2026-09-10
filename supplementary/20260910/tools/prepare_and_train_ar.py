"""Traceable token preparation and AR execution for the frozen reviewer55 protocol.

Dry-run is read-only and launches no subprocesses. Actual steps write into unique
attempt directories and publish results only after verification. Existing results
are never replaced; completed steps may only be reused after signature and hash
verification. The VQ experiment's manifest and artifacts are not modified.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

BASE=Path(__file__).resolve().parents[1]
SRC=BASE/"runtime"/"code"/"src"
PYTHON=Path(sys.executable)
DATA=Path(os.environ.get("SLP_SIGN_DATA_DIR",str(BASE/"runtime/data")))
PROTOCOL=BASE/"configs/protocol.json"
RUNS=BASE/"runs"
LENGTHS=RUNS/"lengths"
CHAR_VOCAB=None
MBART_NAME="facebook/mbart-large-50"
SPLITS=("train","dev","test")
VARIANTS=("baseline","single_large","pamvq","tlrvq","pamrvq","pamrvq_random")

def stamp():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def json_save(path,obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
    os.replace(tmp,path)

def metadata(path,hash_file=True):
    path=Path(path)
    value={"path":str(path),"exists":path.is_file()}
    if path.is_file():
        value["bytes"]=path.stat().st_size
        if hash_file:value["sha256"]=sha(path)
    return value

def source_files(kind):
    paths={Path(__file__)}
    if kind=="tokenize":
        paths.update(SRC.glob("tokenize_sign*.py"))
        paths.update((SRC/"models").glob("*.py"))
        paths.add(SRC/"dataset"/"dataset_VQ_sign.py")
    elif kind=="interleave":
        paths.update(SRC.glob("interleave*tokens.py"))
        paths.add(SRC/"dataset"/"dataset_TM_sign_msr.py")
        paths.add(SRC/"models"/"vqvae_multistream.py")
    elif kind=="train":
        paths.update(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)
    return {str(p):sha(p) for p in sorted(paths)}

def declared_layout(dataset,variant):
    if variant in ("baseline","single_large"):
        size=4096 if dataset=="csl" else (1024 if variant=="single_large" else 512)
        order=["global"]; sizes=[size]
    elif variant=="tlrvq":
        order=["base","res"]; sizes=[512,512] if dataset=="phix" else [2048,2048]
    elif variant=="pamvq":
        order=["body","hand","face"]
        sizes=[64,1024,256] if dataset=="phix" else [1024]*3
    else:
        order=["body_base","body_res","hand_base","hand_res","face_base","face_res"]
        sizes=[32,32,512,512,128,128] if dataset=="phix" else [512]*6
    return {"token_order":order,"codebook_sizes":sizes,"group_size":len(order),"num_vq":sum(sizes),
            "stride_power":4 if dataset=="phix" else 2}

def read_vq_layout(path,dataset,variant,seed):
    import torch
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    a=checkpoint["args"]; a=a if isinstance(a,dict) else vars(a)
    if a.get("dataname")!=dataset+"_lift3d":
        raise ValueError("VQ dataset does not match the requested run")
    if int(a.get("seed",seed))!=seed:
        raise ValueError("VQ seed does not match the requested run")
    stride=int(a["stride_t"])**int(a["down_t"])
    if stride != (4 if dataset=="phix" else 2):
        raise ValueError("VQ temporal stride violates the frozen protocol")
    result=declared_layout(dataset,variant)
    if variant in ("baseline","single_large"):
        sizes=[int(a["nb_code"])]
    elif variant=="tlrvq":
        sizes=[int(a["nb_code"]),int(a["nb_code_residual"])]
    else:
        extra=checkpoint["model"].get("_extra_state")
        if not isinstance(extra,dict) or extra.get("grouping_spec")!=a.get("grouping_spec"):
            raise ValueError("VQ grouping metadata missing or inconsistent")
        spec=a["grouping_spec"]
        expected_group="random" if variant=="pamrvq_random" else "anatomical"
        if spec.get("layout_version")!="reviewer55_verified_v1" or spec.get("grouping")!=expected_group:
            raise ValueError("VQ layout/grouping does not match the declared variant")
        if spec.get("stream_order")!=["body","hand","face"]:
            raise ValueError("VQ stream order differs from the frozen protocol")
        if int(spec.get("grouping_seed",-1))!=20260910:
            raise ValueError("VQ grouping seed differs from the frozen protocol")
        if variant=="pamvq":
            sizes=[int(a.get("nb_code_"+n) or a["nb_code"]) for n in spec["stream_order"]]
            expected_kind="pamvq_shared"
        else:
            sizes=[int(a["nb_"+layer+"_"+n]) for n in spec["stream_order"] for layer in ("base","res")]
            expected_kind="pamrvq_shared"
        if extra.get("model_kind")!=expected_kind:
            raise ValueError("VQ shared-decoder model kind differs from the requested variant")
        result["grouping_spec"]=deepcopy(spec)
    if sizes!=result["codebook_sizes"]:
        raise ValueError(f"VQ codebook sizes {sizes} differ from declared {result['codebook_sizes']}")
    result["vq_args"]=a
    return result

def load_lengths(path):
    value=json_read(path)
    if not isinstance(value,dict) or not all(isinstance(k,str) and isinstance(v,int) and not isinstance(v,bool) and v>0 for k,v in value.items()):
        raise ValueError(f"Expected sample_id -> positive original frame count in {path}")
    return value

def validate_and_annotate_cache(cache,lengths,layout,max_frames=320):
    """Pure CPU transform; no token changes. It also rejects lost/extra samples."""
    import numpy as np
    import torch
    if not isinstance(cache,dict) or set(cache)!=set(lengths):
        absent=sorted(set(lengths)-set(cache))[:5] if isinstance(cache,dict) else []
        extra=sorted(set(cache)-set(lengths))[:5] if isinstance(cache,dict) else []
        raise ValueError(f"Cache coverage mismatch. Missing={absent}, extra={extra}")
    groups=layout["group_size"]; sizes=layout["codebook_sizes"]
    offsets=np.cumsum([0]+sizes[:-1]).tolist()
    budget=(max_frames//layout["stride_power"])*groups
    prepared={}; observed_lengths=[]; over=0
    for sid,record in cache.items():
        tokens=record["tokens"]
        if torch.is_tensor(tokens):tokens=tokens.detach().cpu().numpy()
        tokens=np.asarray(tokens)
        if tokens.ndim!=1 or not np.issubdtype(tokens.dtype,np.integer):
            raise ValueError(f"{sid}: tokens must be a one-dimensional integer array")
        tokens=tokens.astype(np.int64,copy=False)
        original=lengths[sid]; used=min(original,max_frames)
        if int(record.get("T_orig",-1))!=used:
            raise ValueError(f"{sid}: tokenizer T_orig={record.get('T_orig')} differs from expected {used}")
        if len(tokens)%groups or len(tokens)>budget:
            raise ValueError(f"{sid}: incomplete group or action-token budget exceeded")
        expected=(used//layout["stride_power"])*groups
        if len(tokens)!=expected:
            raise ValueError(f"{sid}: {len(tokens)} tokens, expected {expected} for the recorded stride")
        if not len(tokens):
            raise ValueError(f"{sid}: empty token sequence requires an explicit short-sequence policy")
        for j,(offset,size) in enumerate(zip(offsets,sizes)):
            part=tokens[j::groups]
            if np.any(part<offset) or np.any(part>=offset+size):
                raise ValueError(f"{sid}: illegal token in group {layout['token_order'][j]}")
        item=dict(record)
        item.update(tokens=tokens,source_frames=original,source_truncated=original>max_frames,
                    tokenized_frames=used,T_orig=used,T_tok=len(tokens),
                    token_group_size=groups,token_order=list(layout["token_order"]))
        prepared[sid]=item
        observed_lengths.append(len(tokens));over+=int(original>max_frames)
    summary={"samples":len(prepared),"source_truncated_samples":over,
             "natural_terminal_samples":len(prepared)-over,"max_pose_frames":max_frames,
             "max_motion_tokens":budget,"token_min":min(observed_lengths),
             "token_max":max(observed_lengths),"token_mean":sum(observed_lengths)/len(observed_lengths)}
    return prepared,summary

def tokenize_command(plan,outdir):
    scripts={"baseline":"tokenize_sign.py","single_large":"tokenize_sign.py",
             "pamvq":"tokenize_sign_ms.py","tlrvq":"tokenize_sign_rvq.py",
             "pamrvq":"tokenize_sign_msr.py","pamrvq_random":"tokenize_sign_msr.py"}
    return [str(PYTHON),"-u",str(SRC/scripts[plan["variant"]]),
            "--dataname",plan["dataset"]+"_lift3d","--vq-ckpt",plan["vq_path"],
            "--stats-dir",str(Path(plan["vq_path"]).parent),"--mean-name","mean.npy","--std-name","std.npy",
            "--splits",",".join(SPLITS),"--max-len","320","--device","cuda","--out-dir",str(outdir)]

def interleave_command(plan,indir,outdir):
    variant=plan["variant"]; sizes=plan["layout"]["codebook_sizes"]
    common=["--in-dir",str(indir),"--out-dir",str(outdir),"--splits",",".join(SPLITS)]
    if variant=="pamvq":
        script="interleave_ms_tokens.py"
        extra=["--dataname",plan["dataset"]+"_lift3d","--stream-order","body,hand,face",
               "--nb-body",str(sizes[0]),"--nb-hand",str(sizes[1]),"--nb-face",str(sizes[2])]
    elif variant=="tlrvq":
        script="interleave_rvq_tokens.py";extra=["--nb-base",str(sizes[0]),"--nb-res",str(sizes[1])]
    elif variant in ("pamrvq","pamrvq_random"):
        script="interleave_msr_tokens.py";extra=[]
        for name,size in zip(plan["layout"]["token_order"],sizes):
            part,layer=name.split("_")
            extra += [f"--nb-{layer}-{part}",str(size)]
    else:return None
    return [str(PYTHON),"-u",str(SRC/script),*common,*extra]

def train_command(plan,outdir):
    ds=plan["dataset"];layout=plan["layout"];stage2=plan["protocol"]["second_stage"]
    command=[str(PYTHON),"-u",str(SRC/"train_trans_sign_cross.py"),
             "--dataname",ds+"_lift3d","--tokens-dir",str(Path(plan["run"])/"tokens/interleaved"),
             "--vq-ckpt",plan["vq_path"],"--exp-name","model","--out-dir",str(outdir),
             "--num-vq",str(layout["num_vq"]),"--block-size",str(stage2["position_capacity"][ds]),
             "--token-group-size",str(layout["group_size"]),"--max-pose-frames","320",
             "--token-order",",".join(layout["token_order"]),"--protocol-path",str(PROTOCOL),
             "--num-layers",str(stage2["ar_layers"][ds]),"--embed-dim","512","--n-head","8",
             "--fc-rate","4","--drop-out-rate","0.1","--batch-size","16",
             "--warm-up-iter","1000","--lr","1e-4","--weight-decay","1e-5",
             "--print-iter","200","--eval-iter","1000","--save-iter","10000",
             "--early-stop-patience","8" if ds=="phix" else "6",
             "--early-stop-min-delta","0.001","--max-no-improve-iter","8000",
             "--motion-token-mask-prob","0","--seed",str(plan["seed"]),"--device","cuda",
             "--num-workers","0","--gloss-supervised","0","--predict-length","0","--align-dim","0"]
    if ds=="phix":
        command += ["--text-encoder","mbart","--mbart-name",MBART_NAME,
                    "--lang","de_DE","--text-dim","1024","--max-text-len","128","--freeze-text","1",
                    "--total-iter","20000","--lr-scheduler","10000","15000","--gamma","0.1",
                    "--min-iter-before-early-stop","3000"]
    else:
        command += ["--text-encoder","char","--char-vocab-path",str(CHAR_VOCAB or DATA/"csl/char_vocab/txt.vocab"),
                    "--char-enc-layers","2","--char-enc-dim","512","--char-enc-heads","8",
                    "--lang","zh_CN","--text-dim","512","--max-text-len","128","--freeze-text","1",
                    "--total-iter","50000","--lr-scheduler","20000","35000","--gamma","0.3",
                    "--min-iter-before-early-stop","36000"]
    return command

def build_plan(dataset,variant,seed,dry_run=False):
    protocol=json_read(PROTOCOL)
    if protocol["version"]!="reviewer55_verified_v1" or protocol["second_stage"]["max_pose_frames"]!=320:
        raise ValueError("Unexpected protocol version or pose-frame budget")
    if protocol["second_stage"]["group_order"]!=["body","hand","face"]:
        raise ValueError("Unexpected protocol stream order")
    if seed not in protocol["training_seeds"]:
        raise ValueError("Seed is not predeclared in the protocol")
    run=RUNS/f"{dataset}_{variant}_s{seed}"
    vq=run/"checkpoints/model/best.pt"
    required=[vq,vq.parent/"mean.npy",vq.parent/"std.npy",run/"manifest.json",
              *[LENGTHS/f"{dataset}_{s}_lengths.json" for s in SPLITS]]
    missing=[str(p) for p in required if not p.is_file()]
    warnings=[]
    vq_manifest=json_read(run/"manifest.json") if (run/"manifest.json").is_file() else None
    if vq_manifest and vq_manifest.get("status")!="completed":
        warnings.append("VQ run is not completed")
    if missing or warnings:
        if not dry_run:raise RuntimeError(f"VQ/length inputs not ready: missing={missing}, warnings={warnings}")
        layout=declared_layout(dataset,variant)
    else:
        expected=next((x["sha256"] for x in vq_manifest.get("outputs",[]) if Path(x["path"])==vq),None)
        if expected is None or sha(vq)!=expected:
            raise RuntimeError("VQ checkpoint hash does not match its completed run manifest")
        layout=read_vq_layout(vq,dataset,variant,seed)
    training_warnings=[]
    if dataset=="csl" and not (CHAR_VOCAB or DATA/"csl/char_vocab/txt.vocab").is_file():
        training_warnings.append("CSL character vocabulary is missing")
    latent=320//layout["stride_power"]
    if protocol["second_stage"]["position_capacity"][dataset] <= latent*layout["group_size"]:
        raise ValueError("Position capacity must include the full motion-token budget plus EOS")
    plan={"schema":"reviewer55.ar_pipeline.v1","dataset":dataset,"variant":variant,"seed":seed,
          "run":str(run),"vq_path":str(vq),"vq_sha256":sha(vq) if vq.is_file() and not missing and not warnings else None,
          "protocol":protocol,"protocol_path":str(PROTOCOL),"protocol_sha256":sha(PROTOCOL),
          "layout":layout,"max_motion_tokens":latent*layout["group_size"],
          "data_sources":{s:str(DATA/dataset/(f'phix_lift3d.{s}.pt' if dataset=="phix" else f'csl_daily_lift3d.{s}.pt')) for s in SPLITS},
          "length_sources":{s:str(LENGTHS/f"{dataset}_{s}_lengths.json") for s in SPLITS},
          "missing_inputs":missing,"preflight_warnings":warnings,"training_warnings":training_warnings}
    plan["commands"]={
        "tokenize":tokenize_command(plan,"<attempt>/raw"),
        "interleave":interleave_command(plan,run/"tokens/raw","<attempt>/interleaved_raw"),
        "train":train_command(plan,"<attempt>/ar")}
    return plan

def verify_outputs(record):
    outputs=record.get("outputs",[])
    return bool(outputs) and all(Path(x["path"]).is_file() and sha(x["path"])==x["sha256"] for x in outputs)

class Pipeline:
    def __init__(self,plan):
        self.plan=plan;self.run=Path(plan["run"]).resolve()
        self.path=self.run/"ar_pipeline_manifest.json"
        identity={k:plan[k] for k in ("dataset","variant","seed","vq_sha256","protocol_sha256")}
        if self.path.exists():
            self.state=json_read(self.path)
            if self.state["identity"]!=identity:
                raise RuntimeError("Existing AR pipeline belongs to different inputs; refusing to overwrite")
        else:
            self.state={"schema":"reviewer55.ar_pipeline.v1","created_at":stamp(),"identity":identity,
                        "steps":{},"history":[],"plan":plan}
        packages={}
        for name in ("torch","numpy","transformers","tokenizers"):
            try: packages[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: packages[name]=None
        self.state["runner_runtime"]={"python":sys.version,"executable":sys.executable,
                                      "subprocess_python":str(PYTHON),"packages":packages}
        self.state["status"]="ready"
        json_save(self.path,self.state)

    def safe(self,path):
        path=Path(path).resolve()
        if not path.is_relative_to(self.run) or path==self.run:
            raise RuntimeError(f"Output escapes run directory: {path}")
        return path

    def inventory(self,directory):
        return [metadata(p) for p in sorted(Path(directory).rglob("*")) if p.is_file()]

    def snapshot(self):
        files={str(p.relative_to(SRC)).replace("\\","/"):p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts}
        files["__runner__/prepare_and_train_ar.py"]=Path(__file__)
        files["__protocol__/experiment_protocol_v1.json"]=PROTOCOL
        hashes={k:sha(p) for k,p in sorted(files.items())}
        path=self.safe(self.run/"ar_pipeline_sources"/("source_"+digest(hashes)[:16]+".zip"))
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            with zipfile.ZipFile(path,"x",zipfile.ZIP_DEFLATED) as archive:
                for name,p in sorted(files.items()):
                    info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                    archive.writestr(info,p.read_bytes())
        with zipfile.ZipFile(path) as archive:
            if {n:hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()}!=hashes:
                raise RuntimeError("Source archive integrity check failed")
        return {"zip":metadata(path),"file_sha256":hashes}

    def execute(self,key,final_dir,descriptor,inputs,action):
        final_dir=self.safe(final_dir)
        contract={"operation":key,"descriptor":descriptor,"inputs":inputs}
        signature=digest(contract)
        old=self.state["steps"].get(key)
        if old and old.get("status")=="completed":
            if old.get("signature")!=signature or not verify_outputs(old):
                raise RuntimeError(f"{key}: completed step changed or outputs altered; refusing reuse/overwrite")
            print(json.dumps({"step":key,"status":"skipped_verified"}),flush=True)
            return
        if final_dir.exists():
            raise RuntimeError(f"{key}: unverified existing output directory {final_dir}; refusing overwrite")
        attempt=self.safe(self.run/"ar_pipeline_attempts"/(key+"_"+uuid.uuid4().hex[:12]))
        attempt.mkdir(parents=True)
        staged=attempt/"result"
        record={"status":"running","signature":signature,"contract":contract,"started_at":stamp(),
                "attempt_directory":str(attempt),"final_directory":str(final_dir),
                "source_snapshot":self.snapshot(),"console":str(attempt/"console.log")}
        if old:self.state["history"].append({"step":key,"record":old})
        self.state["steps"][key]=record;self.state["status"]="running";json_save(self.path,self.state)
        start=time.monotonic()
        try:
            action(staged,attempt,record)
            for filename,expected in inputs.items():
                if not Path(filename).is_file() or sha(filename)!=expected:
                    raise RuntimeError(f"{key}: input/source changed during execution: {filename}")
            if not staged.is_dir() or not any(staged.rglob("*")):
                raise RuntimeError(f"{key}: no output artifacts produced")
            final_dir.parent.mkdir(parents=True,exist_ok=True)
            self.safe(staged);self.safe(final_dir)
            if final_dir.exists():raise RuntimeError("Output appeared while executing; refusing overwrite")
            os.rename(staged,final_dir)
            record.update(status="completed",finished_at=stamp(),elapsed_seconds=round(time.monotonic()-start,3),
                          outputs=self.inventory(final_dir))
            self.state["status"]="ready"
        except BaseException as error:
            record.update(status="failed",finished_at=stamp(),elapsed_seconds=round(time.monotonic()-start,3),
                          error=repr(error),partial_outputs=self.inventory(attempt))
            self.state["status"]="failed"
            json_save(self.path,self.state)
            raise
        json_save(self.path,self.state)

    def subprocess(self,command,attempt,record):
        record.update(command=command,cwd=str(SRC),python=str(PYTHON))
        json_save(self.path,self.state)
        env=os.environ.copy()
        env.update(PYTHONUTF8="1",PYTHONIOENCODING="utf-8",PYTHONDONTWRITEBYTECODE="1",TOKENIZERS_PARALLELISM="false")
        record["environment"]={k:env.get(k) for k in ("PYTHONUTF8","PYTHONIOENCODING","PYTHONDONTWRITEBYTECODE","TOKENIZERS_PARALLELISM","CUDA_VISIBLE_DEVICES")}
        with (attempt/"console.log").open("x",encoding="utf-8") as log:
            process=subprocess.Popen(command,cwd=SRC,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                     text=True,encoding="utf-8",errors="replace",bufsize=1)
            record["pid"]=process.pid;json_save(self.path,self.state)
            try:
                for line in process.stdout:
                    print(line,end="",flush=True);log.write(line);log.flush()
                returncode=process.wait()
            except BaseException:
                process.terminate();process.wait()
                raise
        record["return_code"]=returncode
        if returncode:raise RuntimeError(f"Subprocess failed with exit code {returncode}")

    def inputs(self,kind,extra=()):
        paths=[Path(self.plan["vq_path"]),Path(self.plan["vq_path"]).parent/"mean.npy",
               Path(self.plan["vq_path"]).parent/"std.npy",PROTOCOL,*map(Path,extra)]
        values={str(p):sha(p) for p in paths}
        values.update(source_files(kind))
        return values

    def prepare_tokens(self):
        import torch
        plan=self.plan;raw=self.run/"tokens/raw";inter=self.run/"tokens/interleaved_raw";final=self.run/"tokens/interleaved"
        def tokenize(staged,attempt,record):
            self.subprocess(tokenize_command(plan,staged),attempt,record)
            for split in SPLITS:
                if not (staged/f"{split}_tokens.pt").is_file():raise RuntimeError("Tokenizer output missing")
        self.execute("tokenize",raw,plan["commands"]["tokenize"],
                     self.inputs("tokenize",[*plan["length_sources"].values(),*plan["data_sources"].values()]),tokenize)
        source=raw
        if plan["variant"] not in ("baseline","single_large"):
            def interleave(staged,attempt,record):
                self.subprocess(interleave_command(plan,raw,staged),attempt,record)
                for split in SPLITS:
                    if not (staged/f"{split}_tokens.pt").is_file():raise RuntimeError("Interleaver output missing")
            self.execute("interleave",inter,plan["commands"]["interleave"],
                         self.inputs("interleave",[raw/f"{s}_tokens.pt" for s in SPLITS]),interleave)
            source=inter
        def annotate(staged,attempt,record):
            staged.mkdir()
            reports=[]
            for split in SPLITS:
                inp=source/f"{split}_tokens.pt";length_path=Path(plan["length_sources"][split])
                cache=torch.load(inp,map_location="cpu",weights_only=False)
                prepared,summary=validate_and_annotate_cache(cache,load_lengths(length_path),plan["layout"])
                target=staged/f"{split}_tokens.pt";torch.save(prepared,target)
                sidecar={"schema":"reviewer55.token_cache.v1","dataset":plan["dataset"],"variant":plan["variant"],
                         "split":split,"seed":plan["seed"],"vq_sha256":plan["vq_sha256"],
                         "protocol_sha256":plan["protocol_sha256"],"input_cache":metadata(inp),
                         "source_lengths":metadata(length_path),"source_data":plan["data_sources"][split],
                         "source_data_sha256":self.state["steps"]["tokenize"]["contract"]["inputs"][plan["data_sources"][split]],
                         "token_order":plan["layout"]["token_order"],"codebook_sizes":plan["layout"]["codebook_sizes"],
                         "token_group_size":plan["layout"]["group_size"],"grouping_spec":plan["layout"].get("grouping_spec"),
                         "cache_sha256":sha(target),"summary":summary}
                json_save(staged/f"{split}_tokens.meta.json",sidecar);reports.append(sidecar)
            record["cache_summaries"]=[x["summary"] for x in reports]
            (attempt/"console.log").write_text(json.dumps(record["cache_summaries"],indent=2),encoding="utf-8")
        self.execute("annotate",final,{"max_pose_frames":320,"layout":plan["layout"],
                     "source_frames":"original length audit","source_truncated":"source_frames > 320"},
                     self.inputs("annotate",[*[source/f"{s}_tokens.pt" for s in SPLITS],*plan["length_sources"].values()]),annotate)

    def train(self):
        import torch
        plan=self.plan;tokens=self.run/"tokens/interleaved"
        if plan.get("training_warnings"):
            raise RuntimeError("; ".join(plan["training_warnings"]))
        done=self.state["steps"].get("annotate")
        if not done or done.get("status")!="completed" or not verify_outputs(done):
            raise RuntimeError("AR training requires a completed, hash-verified annotated cache step")
        extras=[]
        for split in SPLITS:
            cache=tokens/f"{split}_tokens.pt";meta=tokens/f"{split}_tokens.meta.json"
            m=json_read(meta)
            if m["cache_sha256"]!=sha(cache) or m["vq_sha256"]!=plan["vq_sha256"] or m["protocol_sha256"]!=plan["protocol_sha256"]:
                raise RuntimeError("Cache sidecar provenance mismatch")
            extras += [cache,meta]
        if plan["dataset"]=="csl":
            vocab=CHAR_VOCAB or DATA/"csl/char_vocab/txt.vocab"
            if not vocab.is_file():raise FileNotFoundError(vocab)
            extras.append(vocab)
        def train(staged,attempt,record):
            self.subprocess(train_command(plan,staged),attempt,record)
            p=staged/"model/best.pt"
            if not p.is_file():raise RuntimeError("AR best checkpoint missing")
            checkpoint=torch.load(p,map_location="cpu",weights_only=False)
            a=checkpoint["args"];a=a if isinstance(a,dict) else vars(a)
            expected={"vq_sha256":plan["vq_sha256"],"protocol_sha256":plan["protocol_sha256"],
                      "token_order":plan["layout"]["token_order"],"token_group_size":plan["layout"]["group_size"],
                      "max_pose_frames":320,"block_size":plan["protocol"]["second_stage"]["position_capacity"][plan["dataset"]],
                      "seed":plan["seed"]}
            for key,value in expected.items():
                if a.get(key)!=value:raise RuntimeError(f"AR checkpoint field mismatch: {key}")
            for split in ("train","dev"):
                if a.get("token_cache_sha256",{}).get(split)!=sha(tokens/f"{split}_tokens.pt"):
                    raise RuntimeError("AR checkpoint token cache hash mismatch")
            record["best_checkpoint_dev_ce"]=checkpoint.get("dev_ce")
            record["best_checkpoint_iteration"]=checkpoint.get("iter")
        self.execute("train",self.run/"checkpoints/ar",plan["commands"]["train"],self.inputs("train",extras),train)

@contextmanager
def run_lock(run):
    path=Path(run)/"ar_pipeline.lock"
    with path.open("x",encoding="utf-8") as f:
        json.dump({"pid":os.getpid(),"started_at":stamp()},f)
    try:yield
    finally:path.unlink()

def configure_paths(args):
    global DATA,RUNS,LENGTHS,PROTOCOL,PYTHON,CHAR_VOCAB,MBART_NAME
    DATA=args.data_root.expanduser().resolve();RUNS=args.run_root.expanduser().resolve()
    if RUNS.is_relative_to(DATA):raise ValueError("Run outputs must not be inside data input directory")
    LENGTHS=RUNS/"lengths";PROTOCOL=args.protocol.resolve();PYTHON=args.python.resolve()
    CHAR_VOCAB=args.char_vocab_path.resolve() if args.char_vocab_path else None
    MBART_NAME=args.mbart_name
    os.environ["SLP_SIGN_DATA_DIR"]=str(DATA)

def prepare_lengths(dataset):
    """Create frame-count metadata from user-supplied source files, without changing poses."""
    import torch
    LENGTHS.mkdir(parents=True,exist_ok=True)
    for split in SPLITS:
        stem="phix_lift3d" if dataset=="phix" else "csl_daily_lift3d"
        source=DATA/dataset/f"{stem}.{split}.pt"
        if not source.is_file():raise FileNotFoundError(source)
        identity=metadata(source);target=LENGTHS/f"{dataset}_{split}_lengths.json"
        sidecar=target.with_suffix(".source.json")
        if target.exists() or sidecar.exists():
            if not target.is_file() or not sidecar.is_file():raise RuntimeError("Incomplete length metadata: "+str(target))
            old=json_read(sidecar)
            if old["source"]["sha256"]!=identity["sha256"] or old["lengths_sha256"]!=sha(target):raise RuntimeError("Length metadata/input changed: "+str(target))
            continue
        raw=torch.load(source,map_location="cpu",weights_only=False)
        if not isinstance(raw,dict) or not raw:raise ValueError("Expected source ID dictionary")
        lengths={}
        for sid,row in raw.items():
            if not isinstance(sid,str) or "poses_3d" not in row:raise ValueError("Missing sample ID or poses_3d")
            length=len(row["poses_3d"])
            if length<1:raise ValueError("Empty pose sequence: "+sid)
            lengths[sid]=int(length)
        json_save(target,lengths)
        json_save(sidecar,{"source":identity,"lengths_sha256":sha(target),"samples":len(lengths)})
        del raw

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=["phix","csl"],required=True)
    parser.add_argument("--variant",choices=VARIANTS,required=True)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--stage",choices=["tokenize","train","both"],default="both")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--data-root",type=Path,default=DATA)
    parser.add_argument("--run-root",type=Path,default=RUNS)
    parser.add_argument("--protocol",type=Path,default=PROTOCOL)
    parser.add_argument("--python",type=Path,default=PYTHON)
    parser.add_argument("--char-vocab-path",type=Path)
    parser.add_argument("--mbart-name",default=MBART_NAME)
    args=parser.parse_args(argv)
    configure_paths(args)
    if not args.dry_run:
        run=RUNS/f"{args.dataset}_{args.variant}_s{args.seed}"
        if not (run/"checkpoints/model/best.pt").is_file():raise FileNotFoundError("Run VQ training first: "+str(run))
        prepare_lengths(args.dataset)
    plan=build_plan(args.dataset,args.variant,args.seed,args.dry_run)
    if args.dry_run:
        plan["requested_stage"]=args.stage;plan["dry_run"]=True
        plan["layout"].pop("vq_args",None)
        print(json.dumps(plan,ensure_ascii=False,indent=2))
        return 0
    if plan["preflight_warnings"]:
        raise RuntimeError("; ".join(plan["preflight_warnings"]))
    with run_lock(plan["run"]):
        pipeline=Pipeline(plan)
        try:
            if args.stage in ("tokenize","both"):pipeline.prepare_tokens()
            if args.stage in ("train","both"):pipeline.train()
            pipeline.state.update(status="completed",last_completed_stage=args.stage,updated_at=stamp())
            json_save(pipeline.path,pipeline.state)
        except BaseException as error:
            pipeline.state.update(status="failed",last_error=repr(error),updated_at=stamp())
            json_save(pipeline.path,pipeline.state)
            raise
    print(json.dumps({"run":plan["run"],"stage":args.stage,"status":"completed"},ensure_ascii=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())

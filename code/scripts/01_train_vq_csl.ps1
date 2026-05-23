# Train all CSL VQs (baseline / M1 / M2-single / M1+M2) — for ablation
# Run from project root.

param(
    [string]$Py = "C:/Python314/python.exe",
    [ValidateSet('baseline', 'M1', 'M2', 'M1M2', 'all')][string]$Which = 'all'
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = "1"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
Set-Location (Join-Path $RepoRoot "code/src")

$CKPT_DIR = Join-Path $RepoRoot "checkpoints/csl/vq"
New-Item -ItemType Directory -Force -Path $CKPT_DIR | Out-Null

function Train-VQ-Baseline {
    Write-Host "=== Training baseline VQ ==="
    & $Py train_vq_sign.py --dataname csl_lift3d `
        --exp-name vq_baseline `
        --nb-code 4096 --code-dim 256 --output-emb-width 256 `
        --down-t 1 --stride-t 1 --width 256 --depth 3 --dilation-growth-rate 3 `
        --batch-size 64 --total-iter 100000 --lr 2e-4 --gamma 0.1 --lr-scheduler 60000 `
        --out-dir $CKPT_DIR
}

function Train-VQ-M1 {
    Write-Host "=== Training M1 (multi-stream) VQ ==="
    & $Py train_vq_sign_ms.py --dataname csl_lift3d `
        --exp-name vq_M1 `
        --nb-code 1024 --n-streams 3 `
        --code-dim 256 --output-emb-width 256 `
        --down-t 1 --stride-t 1 --width 256 --depth 3 --dilation-growth-rate 3 `
        --batch-size 64 --total-iter 100000 --lr 2e-4 --gamma 0.1 --lr-scheduler 60000 `
        --out-dir $CKPT_DIR
}

function Train-VQ-M2 {
    Write-Host "=== Training M2 (single-stream residual) VQ ==="
    & $Py train_vq_sign_rvq.py --dataname csl_lift3d `
        --exp-name vq_M2 `
        --nb-base 1024 --nb-res 1024 `
        --code-dim 256 --output-emb-width 256 `
        --down-t 1 --stride-t 1 --width 256 --depth 3 --dilation-growth-rate 3 `
        --batch-size 64 --total-iter 100000 --lr 2e-4 --gamma 0.1 --lr-scheduler 60000 `
        --out-dir $CKPT_DIR
}

function Train-VQ-M1M2 {
    Write-Host "=== Training M1+M2 (multi-stream residual) VQ ==="
    & $Py train_vq_sign_msr.py --dataname csl_lift3d `
        --exp-name vq_M1M2 `
        --nb-base-body 512 --nb-base-hand 512 --nb-base-face 512 `
        --nb-res-body 512  --nb-res-hand 512  --nb-res-face 512 `
        --code-dim 256 --output-emb-width 256 `
        --down-t 1 --stride-t 1 --width 256 --depth 3 --dilation-growth-rate 3 `
        --batch-size 64 --total-iter 100000 --lr 2e-4 --gamma 0.1 --lr-scheduler 60000 `
        --out-dir $CKPT_DIR
}

switch ($Which) {
    'baseline' { Train-VQ-Baseline }
    'M1' { Train-VQ-M1 }
    'M2' { Train-VQ-M2 }
    'M1M2' { Train-VQ-M1M2 }
    'all' {
        Train-VQ-Baseline
        Train-VQ-M1
        Train-VQ-M2
        Train-VQ-M1M2
    }
}

# Train large gloss-free Trans for one of {baseline, M1, M2, M1M2} on CSL.

param(
    [Parameter(Mandatory=$true)][ValidateSet('baseline', 'M1', 'M2', 'M1M2')][string]$Variant,
    [string]$Py = "C:/Python314/python.exe",
    [string]$Suffix = "large_v1"
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = "1"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
Set-Location (Join-Path $RepoRoot "code/src")

# Map variant → (tokens_dir, vq_ckpt, num_vq, block_size)
$config = @{
    'baseline' = @{ tok = "vq_baseline"; vq = "vq_baseline.pt"; num_vq = 4096; block = 160 }
    'M1'       = @{ tok = "vq_M1";       vq = "vq_M1.pt";       num_vq = 3072; block = 320 }
    'M2'       = @{ tok = "vq_M2";       vq = "vq_M2.pt";       num_vq = 2048; block = 320 }
    'M1M2'     = @{ tok = "vq_M1M2";     vq = "vq_M1M2.pt";     num_vq = 3072; block = 480 }
}
$c = $config[$Variant]

$EXP = "trans_${Variant}_${Suffix}_glossfree"
$OUT_DIR = Join-Path $RepoRoot "checkpoints/csl/trans/_train_${EXP}"
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null

Write-Host "=== Training Trans ($Variant) — output: $EXP ==="
Write-Host "    tokens: $($c.tok), num_vq: $($c.num_vq), block_size: $($c.block)"

& $Py train_trans_sign_cross.py `
    --dataname csl_lift3d `
    --tokens-dir "$(Join-Path $RepoRoot "checkpoints/csl/tokens/$($c.tok)")" `
    --vq-ckpt "$(Join-Path $RepoRoot "checkpoints/csl/vq/$($c.vq)")" `
    --exp-name $EXP `
    --text-encoder char --gloss-supervised 0 `
    --num-vq $c.num_vq --block-size $c.block `
    --num-layers 8 --embed-dim 512 --n-head 8 --fc-rate 4 --drop-out-rate 0.1 `
    --batch-size 16 --total-iter 50000 `
    --warm-up-iter 1000 `
    --lr 1e-4 --lr-scheduler 20000 35000 --gamma 0.3 `
    --eval-iter 1000 --print-iter 200 --save-iter 10000 `
    --early-stop-patience 6 --early-stop-min-delta 0.001 `
    --min-iter-before-early-stop 36000 --max-no-improve-iter 8000 `
    --motion-token-mask-prob 0.0 `
    --out-dir $OUT_DIR

Write-Host ""
Write-Host "[OK] Done. Rename best.pt:"
Write-Host "  cp `"$OUT_DIR/$EXP/best.pt`" `"$(Join-Path $RepoRoot 'checkpoints/csl/trans/trans_'+$Variant+'_large.pt')`""

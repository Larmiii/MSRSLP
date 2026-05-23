# PHIX v2 full pipeline: tokenize → interleave → train 4 trans → eval → BT
# Assumes 4 v2 VQ ckpts exist at $VQ_ROOT.
# Run from any cwd; pulls release root from script location.

$ErrorActionPreference = 'Continue'  # python stderr would otherwise abort the script
$env:PYTHONUTF8 = "1"
$Py = "C:\Python314\python.exe"
$Py37 = "C:/Users/22949/miniconda3/envs/slt37/python.exe"  # signjoey BT
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$VQ_ROOT = "$RepoRoot\checkpoints\phix\vq_v2"
$TOK_ROOT = "$RepoRoot\checkpoints\phix\tokens_v2"
$TR_ROOT = "$RepoRoot\checkpoints\phix\trans_v2"
$RES_ROOT = "$RepoRoot\results"
New-Item -ItemType Directory -Force -Path $TOK_ROOT, $TR_ROOT | Out-Null

# Sanity: ensure all 4 VQs present
foreach ($v in @('baseline_v2', 'M1_v2', 'M2_v2', 'M1M2_v2')) {
    if (-not (Test-Path "$VQ_ROOT\vq_$v.pt")) {
        Write-Error "Missing $VQ_ROOT\vq_$v.pt"; exit 1
    }
}
Write-Output "[OK] All 4 v2 VQ ckpts present"

# === STAGE 1: tokenize ===
Set-Location "$RepoRoot\code\src"
Write-Output ""
Write-Output "=========================================="
Write-Output "STAGE 1: TOKENIZE"
Write-Output "=========================================="
foreach ($pair in @(
    @('tokenize_sign.py',     'baseline_v2'),
    @('tokenize_sign_ms.py',  'M1_v2'),
    @('tokenize_sign_rvq.py', 'M2_v2'),
    @('tokenize_sign_msr.py', 'M1M2_v2')
)) {
    $script = $pair[0]; $v = $pair[1]
    Write-Output ">>> tokenize $v"
    & $Py $script `
        --dataname phix_lift3d `
        --vq-ckpt "$VQ_ROOT\vq_$v.pt" `
        --stats-dir "$VQ_ROOT\stats" `
        --out-dir "$TOK_ROOT\vq_$v" 2>&1 | Select-String "saved|Error|Traceback" | Select-Object -Last 4
}

# === STAGE 2: interleave (M1, M2, M1M2) ===
Write-Output ""
Write-Output "=========================================="
Write-Output "STAGE 2: INTERLEAVE"
Write-Output "=========================================="

# M1: ms (3 streams), find per-stream codes from VQ args
$M1_args = & $Py -c "import torch; ck = torch.load(r'$VQ_ROOT\vq_M1_v2.pt', map_location='cpu', weights_only=False); a = ck['args']; print(a.get('nb_code_body') or a.get('nb_code'), a.get('nb_code_hand') or a.get('nb_code'), a.get('nb_code_face') or a.get('nb_code'))"
$m1Tokens = $M1_args.Trim().Split(' ')
Write-Output ">>> M1 stream_codes: body=$($m1Tokens[0]), hand=$($m1Tokens[1]), face=$($m1Tokens[2])"
# For asymmetric ms, interleave script uses --nb-per-stream uniform. Need a per-stream interleave or pack with offsets manually.
# The existing interleave_ms_tokens.py only takes single --nb-per-stream. We need to extend.

# M1 asymmetric: pass per-stream codes explicitly
& $Py interleave_ms_tokens.py `
    --in-dir "$TOK_ROOT\vq_M1_v2" `
    --out-dir "$TOK_ROOT\vq_M1_v2_interleaved" `
    --dataname phix_lift3d `
    --nb-body $m1Tokens[0] --nb-hand $m1Tokens[1] --nb-face $m1Tokens[2] 2>&1 | Select-String "OK|DONE|Error|Traceback|num_vq" | Select-Object -Last 4

# M2: rvq
& $Py interleave_rvq_tokens.py `
    --in-dir "$TOK_ROOT\vq_M2_v2" `
    --out-dir "$TOK_ROOT\vq_M2_v2_interleaved" `
    --nb-base 512 --nb-res 512 2>&1 | Select-String "OK|DONE|Error|Traceback" | Select-Object -Last 4

# M1M2: msr
& $Py interleave_msr_tokens.py `
    --in-dir "$TOK_ROOT\vq_M1M2_v2" `
    --out-dir "$TOK_ROOT\vq_M1M2_v2_interleaved" `
    --nb-base-body 32 --nb-res-body 32 `
    --nb-base-hand 512 --nb-res-hand 512 `
    --nb-base-face 128 --nb-res-face 128 2>&1 | Select-String "OK|DONE|Error|Traceback" | Select-Object -Last 4

# === STAGE 3: trans ===
Write-Output ""
Write-Output "=========================================="
Write-Output "STAGE 3: TRAIN 4 TRANS (mBART, 9L, 20K iter)"
Write-Output "=========================================="

# num_vq per variant
$NUM_BASELINE = 512
$NUM_M1 = [int]$m1Tokens[0] + [int]$m1Tokens[1] + [int]$m1Tokens[2]
$NUM_M2 = 1024
$NUM_M1M2 = 32 + 32 + 512 + 512 + 128 + 128
Write-Output "num_vq: baseline=$NUM_BASELINE M1=$NUM_M1 M2=$NUM_M2 M1M2=$NUM_M1M2"

foreach ($cfg in @(
    @('baseline', 'vq_baseline_v2', 'tokens_v2/vq_baseline_v2',            $NUM_BASELINE, 160, 1),
    @('M1',       'vq_M1_v2',       'tokens_v2/vq_M1_v2_interleaved',      $NUM_M1,       320, 3),
    @('M2',       'vq_M2_v2',       'tokens_v2/vq_M2_v2_interleaved',      $NUM_M2,       160, 2),
    @('M1M2',     'vq_M1M2_v2',     'tokens_v2/vq_M1M2_v2_interleaved',    $NUM_M1M2,     320, 6)
)) {
    $name = $cfg[0]; $vq = $cfg[1]; $tokDir = $cfg[2]; $numVq = $cfg[3]; $blockSize = $cfg[4]
    $exp = "trans_${name}_v2_mbart"
    $OUT = "$TR_ROOT\_train_$exp"
    New-Item -ItemType Directory -Force -Path $OUT | Out-Null
    Write-Output ""
    Write-Output ">>> TRAIN $exp (num_vq=$numVq, block=$blockSize)"

    & $Py train_trans_sign_cross.py `
        --dataname phix_lift3d `
        --tokens-dir "$RepoRoot\checkpoints\phix\$tokDir" `
        --vq-ckpt "$VQ_ROOT\$vq.pt" `
        --exp-name $exp --out-dir $OUT `
        --text-encoder mbart --mbart-name "facebook/mbart-large-50" --lang de_DE `
        --num-vq $numVq `
        --embed-dim 512 --text-dim 1024 `
        --num-layers 9 --n-head 8 `
        --block-size $blockSize --drop-out-rate 0.1 --fc-rate 4 `
        --batch-size 16 --total-iter 20000 --warm-up-iter 1000 `
        --lr 1e-4 --lr-scheduler 10000 15000 --gamma 0.1 `
        --weight-decay 1e-5 --max-text-len 128 --freeze-text 1 `
        --print-iter 200 --eval-iter 1000 --save-iter 5000 `
        --early-stop-patience 8 --early-stop-min-delta 0.001 `
        --min-iter-before-early-stop 3000 --max-no-improve-iter 8000 `
        --seed 42 2>&1 | Select-String "best dev ce|EARLY|done|Traceback|OOM|Error" | Select-Object -Last 5

    # Copy ckpt
    if (Test-Path "$OUT\$exp\best.pt") {
        Copy-Item "$OUT\$exp\best.pt" "$TR_ROOT\trans_${name}_v2_mbart.pt" -Force
        Copy-Item "$OUT\$exp\train.log" "$RepoRoot\logs\phix_trans_${name}_v2.train.log" -Force
        Write-Output "[OK] $name trans copied"
    }
}

Write-Output ""
Write-Output "=========================================="
Write-Output "ALL DONE. Trans ckpts in $TR_ROOT"
Write-Output "=========================================="
Get-ChildItem $TR_ROOT -Filter "trans_*_v2_mbart.pt" | Select-Object Name, @{n='MB';e={[math]::Round($_.Length/1MB,1)}}

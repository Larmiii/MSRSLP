$ErrorActionPreference = "Stop"
$PY = "C:\Users\22949\miniconda3\envs\uplift310\python.exe"
$ROOT = "D:\Graduate thesis\sign_slp_paper_release"
$env:LIFT3D_DATA_DIR = "$ROOT\data\phix"

$VQ = "$ROOT\checkpoints\phix\vq\vq_M1M2.pt"
$TRANS = "$ROOT\checkpoints\phix\trans\trans_M1M2.pt"
$BT_MODEL = "$ROOT\bt_eval_kit\slrtp_official\backTranslation_PHIX_model"
$SLRTP_REPO = "$ROOT\bt_eval_kit\slrtp_official"
$GT_DEV = "$SLRTP_REPO\data_official\dev.pt"
$GT_TEST = "$SLRTP_REPO\data_official\test.pt"
$OUT_BASE = "$ROOT\results\_sweep_slrtp"
New-Item -ItemType Directory -Path $OUT_BASE -Force | Out-Null

# Decoding configs: tag, T, k, rep_pen, max_run
$configs = @(
  @{ tag = "default";    T = 0.9; k = 20; rep = 1.5; maxrun = 4 },
  @{ tag = "low_temp";   T = 0.7; k = 20; rep = 1.5; maxrun = 4 },
  @{ tag = "wide_topk";  T = 0.9; k = 50; rep = 1.5; maxrun = 4 },
  @{ tag = "weak_rep";   T = 0.9; k = 20; rep = 1.0; maxrun = 4 },
  @{ tag = "no_maxrun";  T = 0.9; k = 20; rep = 1.5; maxrun = 0 },
  @{ tag = "no_penalty"; T = 0.9; k = 20; rep = 0.0; maxrun = 0 }
)

foreach ($cfg in $configs) {
  $tag = $cfg.tag
  $outDir = "$OUT_BASE\$tag"
  New-Item -ItemType Directory -Path $outDir -Force | Out-Null
  Write-Host "`n========== $tag : T=$($cfg.T) k=$($cfg.k) rep=$($cfg.rep) maxrun=$($cfg.maxrun) ==========" -ForegroundColor Cyan

  # Step 1: SLP generation (dev + test)
  Set-Location "$ROOT\code\eval"
  Write-Host "[1/3] Generating poses..."
  & $PY eval_cross_slt_lift3d.py --variant msr `
    --vq-ckpt $VQ --trans-ckpt $TRANS `
    --splits dev,test --out $outDir `
    --dataset phix --lang de_DE `
    --temperature $cfg.T --top-k $cfg.k `
    --rep-penalty $cfg.rep --max-run $cfg.maxrun `
    --rep-streams 6 2>&1 | Select-Object -Last 5

  # Step 2: SLRTP eval on dev
  Write-Host "[2/3] SLRTP eval on dev..."
  & $PY slrtp_eval_phix.py `
    --pred-pickle "$outDir\dev.pickle" `
    --gt-pt $GT_DEV --bt-model-dir $BT_MODEL `
    --slrtp-repo $SLRTP_REPO --tag "${tag}_dev" --fps 25 `
    --out-dir $outDir 2>&1 | Select-Object -Last 8

  # Step 3: SLRTP eval on test
  Write-Host "[3/3] SLRTP eval on test..."
  & $PY slrtp_eval_phix.py `
    --pred-pickle "$outDir\test.pickle" `
    --gt-pt $GT_TEST --bt-model-dir $BT_MODEL `
    --slrtp-repo $SLRTP_REPO --tag "${tag}_test" --fps 25 `
    --out-dir $outDir 2>&1 | Select-Object -Last 8

  # Read JSON results
  $devJson = "$SLRTP_REPO\results\${tag}_dev.json"
  $testJson = "$SLRTP_REPO\results\${tag}_test.json"
  if (Test-Path $devJson) {
    $dev = Get-Content $devJson | ConvertFrom-Json
    Write-Host "DEV BLEU-4 : $($dev.bleu.bleu4)" -ForegroundColor Green
  }
  if (Test-Path $testJson) {
    $test = Get-Content $testJson | ConvertFrom-Json
    Write-Host "TEST BLEU-4: $($test.bleu.bleu4)" -ForegroundColor Green
  }
}

Write-Host "`n========== SUMMARY ==========" -ForegroundColor Yellow
foreach ($cfg in $configs) {
  $tag = $cfg.tag
  $devJson = "$SLRTP_REPO\results\${tag}_dev.json"
  $testJson = "$SLRTP_REPO\results\${tag}_test.json"
  $devB4 = "N/A"; $testB4 = "N/A"
  if (Test-Path $devJson)  { $devB4  = [math]::Round((Get-Content $devJson | ConvertFrom-Json).bleu.bleu4, 2) }
  if (Test-Path $testJson) { $testB4 = [math]::Round((Get-Content $testJson | ConvertFrom-Json).bleu.bleu4, 2) }
  Write-Host ("{0,-12} T={1} k={2,2} rep={3} R={4}  DEV={5,5}  TEST={6,5}" -f $tag, $cfg.T, $cfg.k, $cfg.rep, $cfg.maxrun, $devB4, $testB4)
}

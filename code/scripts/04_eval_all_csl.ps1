# Run SLP + BT eval for all 4 CSL ablation configs sequentially.

param(
    [string]$Py = "C:/Python314/python.exe",
    [string]$BTPy = "C:/Users/22949/miniconda3/envs/slt37/python.exe"
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = "1"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path

# Variant config: (label, variant, vq_ckpt, trans_ckpt, rep_streams)
$configs = @(
    @{ label='baseline'; variant='base'; vq='vq_baseline.pt'; trans='trans_baseline_large.pt'; rep_streams=1 },
    @{ label='M1'; variant='ms'; vq='vq_M1.pt'; trans='trans_M1_large.pt'; rep_streams=3 },
    @{ label='M2'; variant='base'; vq='vq_M2.pt'; trans='trans_M2_large.pt'; rep_streams=2 },
    @{ label='M1M2'; variant='msr'; vq='vq_M1M2.pt'; trans='trans_M1M2_large.pt'; rep_streams=6 }
)

# Note: M2 uses 'base' eval variant since the trans output is interleaved [base|res] flat tokens,
# but VQ decoding needs RVQ-aware decoder. We need to add 'rvq' variant support to eval_cross_slt_lift3d.py.
# For now, we'll handle M2 separately or extend the eval script.

foreach ($c in $configs) {
    $label = $c.label
    $OUT_DIR = Join-Path $RepoRoot "results/$label/slp_pickles"
    $LOG_DIR = Split-Path $OUT_DIR -Parent
    New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "=== $label : SLP gen ==="
    Write-Host "============================================================"
    Set-Location (Join-Path $RepoRoot "code")
    & $Py "eval/eval_cross_slt_lift3d.py" `
        --variant $c.variant `
        --vq-ckpt "$(Join-Path $RepoRoot ('checkpoints/csl/vq/'+$c.vq))" `
        --trans-ckpt "$(Join-Path $RepoRoot ('checkpoints/csl/trans/'+$c.trans))" `
        --splits dev,test `
        --out $OUT_DIR `
        --temperature 0.9 --top-k 20 `
        --rep-penalty 2.0 --max-run 3 --rep-streams $c.rep_streams 2>&1 | Tee-Object -FilePath "$LOG_DIR/slp.log"

    Write-Host ""
    Write-Host "=== $label : BT eval ==="
    # signjoey test
    $DataDir = $OUT_DIR.Replace('\','/').TrimEnd('/')
    $BT_KIT = Join-Path $RepoRoot "bt_eval_kit"
    $CFG = Join-Path $LOG_DIR "bt_eval.yaml"
    $LOG = Join-Path $LOG_DIR "bt_eval.log"
    $TrainPickle = Join-Path $DataDir "csl_daily.train"
    if (-not (Test-Path $TrainPickle)) {
        $TrainSrc = Join-Path $BT_KIT "csl/csl_daily.train"
        cmd /c "mklink /H `"$TrainPickle`" `"$TrainSrc`"" | Out-Null
    }
    $content = Get-Content (Join-Path $BT_KIT "csl/config_template.yaml") -Raw
    $content = $content -replace "data_path: .*", "data_path: $DataDir/"
    [System.IO.File]::WriteAllText($CFG, $content, [System.Text.UTF8Encoding]::new($false))

    Push-Location (Join-Path $BT_KIT "slt")
    try {
        & $BTPy -m signjoey test $CFG --ckpt (Join-Path $BT_KIT "csl/bt_model.ckpt") > $LOG 2>&1
    } finally { Pop-Location }

    # Parse BLEU
    $logContent = Get-Content $LOG -Raw -Encoding Unicode
    Write-Host "[$label] Results:"
    $matches_ = [regex]::Matches($logContent, "BLEU-4\s+(\d+\.\d+)\s+\(BLEU-1:\s+(\d+\.\d+),\s+BLEU-2:\s+(\d+\.\d+),\s+BLEU-3:\s+(\d+\.\d+),\s+BLEU-4:\s+(\d+\.\d+)\)")
    $i = 0
    foreach ($m in $matches_) {
        $part = if ($i -le 1) { "DEV " } else { "TEST" }
        Write-Host ("  {0}: B1={1} B2={2} B3={3} B4={4}" -f $part, $m.Groups[2].Value, $m.Groups[3].Value, $m.Groups[4].Value, $m.Groups[5].Value)
        $i++
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "=== ALL DONE ==="
Write-Host "============================================================"

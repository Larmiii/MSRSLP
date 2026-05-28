# PHIX M1+M2 decoding hyperparam sweep under SLRTP-canonical eval.
# 4 configs * (dev + test gen + SLRTP eval) ≈ 2 hours total.

$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = "1"

$Py     = "C:/Python314/python.exe"
$REL    = "D:/Graduate thesis/sign_slp_paper_release"
$VQ     = "$REL/checkpoints/phix/vq/vq_M1M2.pt"
$TR     = "$REL/checkpoints/phix/trans/trans_M1M2.pt"
$SWEEP  = "$REL/results/phix_decoding_sweep"
$SLRTP_REPO = "$REL/bt_eval_kit/slrtp_official"
$BT_DIR   = "$SLRTP_REPO/backTranslation_PHIX_model"
$GT_DEV   = "$SLRTP_REPO/data_official/dev.pt"
$GT_TEST  = "$SLRTP_REPO/data_official/test.pt"

New-Item -ItemType Directory -Force -Path $SWEEP | Out-Null

# Configs: name, T, top-k, rep-penalty, max-run
$configs = @(
    @{name='default'; T=0.9; k=20; rep=1.5; mr=4},
    @{name='lowtemp'; T=0.7; k=20; rep=1.5; mr=4},
    @{name='widetop'; T=0.9; k=50; rep=1.5; mr=4},
    @{name='norep';   T=0.9; k=20; rep=1.0; mr=999}
)

$results = @{}

foreach ($c in $configs) {
    $name = $c.name
    $OUT  = "$SWEEP/$name"
    New-Item -ItemType Directory -Force -Path $OUT | Out-Null

    Write-Output ""
    Write-Output "==================================================================="
    Write-Output "=== CONFIG: $name  T=$($c.T) k=$($c.k) rep=$($c.rep) maxrun=$($c.mr)"
    Write-Output "==================================================================="

    $devPickle  = "$OUT/dev.pickle"
    $testPickle = "$OUT/test.pickle"

    # === STAGE 1: SLP generation ===
    # 默认配置可以复用已有 pickle，跳过 gen
    if ($name -eq 'default' -and (Test-Path "$REL/results/phix_M1M2/dev.pickle")) {
        Write-Output "[$name] reusing existing default pickle"
        if (-not (Test-Path $devPickle)) {
            Copy-Item "$REL/results/phix_M1M2/dev.pickle" $devPickle
        }
        if (-not (Test-Path $testPickle)) {
            Copy-Item "$REL/results/phix_M1M2/test.pickle" $testPickle
        }
    } else {
        Write-Output "[$name] SLP gen (dev + test) ..."
        & $Py "$REL/code/eval/eval_cross_slt_lift3d.py" `
            --dataset phix --variant msr `
            --vq-ckpt $VQ --trans-ckpt $TR `
            --splits dev,test `
            --out $OUT `
            --temperature $c.T --top-k $c.k `
            --rep-penalty $c.rep --max-run $c.mr `
            --rep-streams 6 2>&1 | Tee-Object -FilePath "$OUT/gen.log" | Select-String -Pattern "Error|Trace|done|saved|samples" | Select-Object -Last 8
    }

    # === STAGE 2: SLRTP eval (dev + test) ===
    foreach ($split in @('dev', 'test')) {
        $gt = if ($split -eq 'dev') { $GT_DEV } else { $GT_TEST }
        $pickle = "$OUT/$split.pickle"
        $tag = "phix_M1M2_${name}_${split}"
        Write-Output "[$name/$split] SLRTP eval ..."
        & $Py "$REL/code/eval/slrtp_eval_phix.py" `
            --pred-pickle $pickle `
            --gt-pt $gt `
            --bt-model-dir $BT_DIR `
            --slrtp-repo $SLRTP_REPO `
            --tag $tag `
            --out-dir "$OUT/slrtp" 2>&1 | Tee-Object -FilePath "$OUT/eval_${split}.log" | Select-String -Pattern "BLEU|CHRF|ROUGE|Error" | Select-Object -Last 6
    }

    # Extract BLEU-4 from json
    foreach ($split in @('dev', 'test')) {
        $jsonPath = "$SLRTP_REPO/results/phix_M1M2_${name}_${split}.json"
        if (Test-Path $jsonPath) {
            $j = Get-Content $jsonPath | ConvertFrom-Json
            $b4 = [math]::Round($j.bleu.bleu4, 2)
            Write-Output "  [$name/$split] BLEU-4 = $b4"
            $results["${name}_${split}"] = $b4
        }
    }
}

# Final summary
Write-Output ""
Write-Output "==================================================================="
Write-Output "=== FINAL SUMMARY (SLRTP-canonical BLEU-4) ==="
Write-Output "==================================================================="
Write-Output "config     T   k  rep  max-run    DEV    TEST"
foreach ($c in $configs) {
    $n = $c.name
    $dev  = $results["${n}_dev"]
    $test = $results["${n}_test"]
    Write-Output ("{0,-10} {1,4} {2,3} {3,4} {4,8}   {5,5}  {6,5}" -f $n, $c.T, $c.k, $c.rep, $c.mr, $dev, $test)
}

# Save summary CSV
$csvPath = "$SWEEP/SLRTP_grid_results.csv"
"config,T,top_k,rep_penalty,max_run,DEV_BLEU4,TEST_BLEU4" | Out-File -FilePath $csvPath -Encoding utf8
foreach ($c in $configs) {
    $n = $c.name
    "$n,$($c.T),$($c.k),$($c.rep),$($c.mr),$($results[$n+'_dev']),$($results[$n+'_test'])" | Out-File -FilePath $csvPath -Append -Encoding utf8
}
Write-Output ""
Write-Output "[OK] Summary CSV: $csvPath"

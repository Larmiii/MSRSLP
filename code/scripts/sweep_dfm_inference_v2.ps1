# 2nd-pass DFM sweep: explore higher CFG + len_mult to close gap to AR baseline.

$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = "1"

$Py     = "C:/Python314/python.exe"
$REL    = "D:/Graduate thesis/sign_slp_paper_release"
$VQ     = "$REL/checkpoints/phix/vq/vq_M1M2.pt"
$DFM    = "$REL/output_sign/dfm_phix_M1M2_v1/best.pt"
$SWEEP  = "$REL/results/phix_dfm_sweep"
$SLRTP_REPO = "$REL/bt_eval_kit/slrtp_official"
$BT_DIR   = "$SLRTP_REPO/backTranslation_PHIX_model"
$GT_DEV   = "$SLRTP_REPO/data_official/dev.pt"
$GT_TEST  = "$SLRTP_REPO/data_official/test.pt"

$configs = @(
    @{name='s24_cfg4_len10'; n_steps=24; cfg=4.0; temp=1.0; len_mult=1.0},
    @{name='s24_cfg3_len12'; n_steps=24; cfg=3.0; temp=1.0; len_mult=1.2},
    @{name='s50_cfg3_len10'; n_steps=50; cfg=3.0; temp=1.0; len_mult=1.0},
    @{name='s24_cfg3_t08';   n_steps=24; cfg=3.0; temp=0.8; len_mult=1.0}
)

$results = @{}
foreach ($c in $configs) {
    $name = $c.name
    $OUT  = "$SWEEP/$name"
    New-Item -ItemType Directory -Force -Path $OUT | Out-Null

    Write-Output ""
    Write-Output "=== DFM-v2: $name  steps=$($c.n_steps) cfg=$($c.cfg) temp=$($c.temp) len_mult=$($c.len_mult)"

    & $Py "$REL/code/eval/eval_dfm_phix.py" `
        --vq-ckpt $VQ --dfm-ckpt $DFM `
        --splits dev,test --out $OUT `
        --n-steps $c.n_steps --cfg-scale $c.cfg `
        --temperature $c.temp --len-mult $c.len_mult 2>&1 `
        | Tee-Object -FilePath "$OUT/gen.log" `
        | Select-String -Pattern "OK\]|Error|Trace" | Select-Object -Last 4

    foreach ($split in @('dev', 'test')) {
        $gt = if ($split -eq 'dev') { $GT_DEV } else { $GT_TEST }
        $tag = "phix_dfm_${name}_${split}"
        & $Py "$REL/code/eval/slrtp_eval_phix.py" `
            --pred-pickle "$OUT/$split.pickle" --gt-pt $gt `
            --bt-model-dir $BT_DIR --slrtp-repo $SLRTP_REPO `
            --tag $tag --out-dir "$OUT/slrtp" 2>&1 `
            | Tee-Object -FilePath "$OUT/eval_${split}.log" `
            | Select-String -Pattern "BLEU|Error" | Select-Object -Last 4
    }

    foreach ($split in @('dev', 'test')) {
        $jsonPath = "$SLRTP_REPO/results/phix_dfm_${name}_${split}.json"
        if (Test-Path $jsonPath) {
            $j = Get-Content $jsonPath | ConvertFrom-Json
            $b4 = [math]::Round($j.bleu.bleu4, 2)
            Write-Output "  [$name/$split] BLEU-4 = $b4"
            $results["${name}_${split}"] = $b4
        }
    }
}

Write-Output ""
Write-Output "=== SWEEP v2 SUMMARY ==="
Write-Output "AR baseline:   DEV 9.28  TEST 8.97"
Write-Output "Best v1 (cfg3): DEV 6.60  TEST 7.43"
Write-Output ""
Write-Output "config             steps  cfg  temp  len_mult   DEV    TEST"
foreach ($c in $configs) {
    $n = $c.name
    $dev  = $results["${n}_dev"]
    $test = $results["${n}_test"]
    Write-Output ("{0,-18} {1,5}  {2,3}  {3,4}  {4,8}    {5,5}  {6,5}" -f $n, $c.n_steps, $c.cfg, $c.temp, $c.len_mult, $dev, $test)
}

$csvPath = "$SWEEP/SLRTP_dfm_v2_results.csv"
"config,n_steps,cfg,temperature,len_mult,DEV_BLEU4,TEST_BLEU4" | Out-File -FilePath $csvPath -Encoding utf8
foreach ($c in $configs) {
    $n = $c.name
    "$n,$($c.n_steps),$($c.cfg),$($c.temp),$($c.len_mult),$($results[$n+'_dev']),$($results[$n+'_test'])" | Out-File -FilePath $csvPath -Append -Encoding utf8
}
Write-Output "[OK] $csvPath"

# Back-translation BLEU eval for generated SLP poses.
#
# Args:
#   -SlpDir   : Directory containing csl_daily.{dev,test} pickle gen output
#   -Dataset  : 'csl' or 'phix'
#   -ExpName  : Used to name the temp yaml + log file
#
# Output:
#   <SlpDir>/../bt_eval_log/<ExpName>.log

param(
    [Parameter(Mandatory=$true)][string]$SlpDir,
    [Parameter(Mandatory=$true)][ValidateSet('csl', 'phix')][string]$Dataset,
    [Parameter(Mandatory=$true)][string]$ExpName,
    [string]$Py = "C:/Users/22949/miniconda3/envs/slt37/python.exe"
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = "1"

# Find repo root (this script is at code/scripts/run_bt_eval.ps1)
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$BT_KIT = Join-Path $RepoRoot "bt_eval_kit"
$SLT    = Join-Path $BT_KIT  "slt"
$BT_DIR = Join-Path $BT_KIT  $Dataset
$BT_CKPT = Join-Path $BT_DIR "bt_model.ckpt"
$CFG_TMPL = Join-Path $BT_DIR "config_template.yaml"

# Sanity
if (-not (Test-Path $BT_CKPT)) { Write-Error "BT model not found: $BT_CKPT"; exit 1 }
if (-not (Test-Path $CFG_TMPL)) { Write-Error "BT config template not found: $CFG_TMPL"; exit 1 }
if (-not (Test-Path $SlpDir)) { Write-Error "SlpDir not found: $SlpDir"; exit 1 }

# Resolve SlpDir to absolute
$SlpDir = (Resolve-Path $SlpDir).Path.Replace('\', '/').TrimEnd('/')

# Ensure train pickle (for signjoey vocab loading) is in SlpDir as a hard link
if ($Dataset -eq 'csl') {
    $TrainSrc = Join-Path $BT_DIR "csl_daily.train"
    $TrainDst = Join-Path $SlpDir "csl_daily.train"
} else {
    $TrainSrc = Join-Path $BT_DIR "train.pickle"
    $TrainDst = Join-Path $SlpDir "train.pickle"
}
if (-not (Test-Path $TrainDst)) {
    Write-Host "[*] Linking train pickle: $TrainSrc -> $TrainDst"
    cmd /c "mklink /H `"$TrainDst`" `"$TrainSrc`"" | Out-Null
}

# Write run-specific yaml
$LogDir = Join-Path (Split-Path $SlpDir -Parent) "bt_eval_log"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$RunCfg = Join-Path $LogDir "${ExpName}.yaml"
$content = Get-Content $CFG_TMPL -Raw
$content = $content -replace "data_path: .*", "data_path: $SlpDir/"
$GlsVocab = (Join-Path $BT_DIR "gls.vocab").Replace('\', '/')
$TxtVocab = (Join-Path $BT_DIR "txt.vocab").Replace('\', '/')
$content = $content -replace "__GLS_VOCAB__", $GlsVocab
$content = $content -replace "__TXT_VOCAB__", $TxtVocab
[System.IO.File]::WriteAllText($RunCfg, $content, [System.Text.UTF8Encoding]::new($false))

$LogFile = Join-Path $LogDir "${ExpName}.log"
Write-Host "[*] BT eval $ExpName (log -> $LogFile)"
Write-Host "    SlpDir: $SlpDir"
Write-Host "    Dataset: $Dataset"
Write-Host "    BT model: $BT_CKPT"

Push-Location $SLT
try {
    & $Py -m signjoey test $RunCfg --ckpt $BT_CKPT > $LogFile 2>&1
} finally {
    Pop-Location
}

# Parse and print BLEU
Write-Host ""
Write-Host "=== Results ==="
$logContent = Get-Content $LogFile -Raw -Encoding Unicode
$m = [regex]::Matches($logContent, "BLEU-4\s+(\d+\.\d+)\s+\(BLEU-1:\s+(\d+\.\d+),\s+BLEU-2:\s+(\d+\.\d+),\s+BLEU-3:\s+(\d+\.\d+),\s+BLEU-4:\s+(\d+\.\d+)\)")
$i = 0
foreach ($match in $m) {
    $part = if ($i -le 1) { "DEV " } else { "TEST" }
    Write-Host ("{0}: B1={1} B2={2} B3={3} B4={4}" -f $part, $match.Groups[2].Value, $match.Groups[3].Value, $match.Groups[4].Value, $match.Groups[5].Value)
    $i++
}
$chrf = [regex]::Matches($logContent, "CHRF\s+(\d+\.\d+)\s+ROUGE\s+(\d+\.\d+)")
$i = 0
foreach ($match in $chrf) {
    $part = if ($i -le 1) { "DEV " } else { "TEST" }
    Write-Host ("{0}: CHRF={1} ROUGE={2}" -f $part, $match.Groups[1].Value, $match.Groups[2].Value)
    $i++
}

Write-Host ""
Write-Host "[OK] Full log: $LogFile"

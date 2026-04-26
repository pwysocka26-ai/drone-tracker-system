# Pakuje transfer bundle dla migracji drone-tracker-system na nowy komputer.
#
# Tworzy folder migration_bundle/ z 4 ZIP-ami (minimum stack):
#   1. memory_backup.zip       (Claude Code memory ~150 KB)
#   2. third_party_minimal.zip (opencv_contrib_install + onnxruntime-directml ~145 MB)
#   3. weights.zip             (data/weights/* ~310 MB)
#   4. test_videos.zip         (artifacts/test_videos/* ~340 MB)
#
# Parametr -Full doklada videos zewnetrzne, datasety treningowe, etc. (~9 GB).
#
# Uzycie:
#     powershell -File tools\_migrate_pack.ps1
#     powershell -File tools\_migrate_pack.ps1 -Full

param(
    [switch]$Full,
    [string]$OutDir = "migration_bundle"
)

$ErrorActionPreference = "Stop"
$REPO = "C:\Users\pwyso\drone-tracker-system"
$MEMORY = "C:\Users\pwyso\.claude\projects\C--Users-pwyso-drone-tracker-system\memory"
$BUNDLE = Join-Path $REPO $OutDir

if (Test-Path $BUNDLE) {
    Write-Host "Removing existing $BUNDLE" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BUNDLE
}
New-Item -ItemType Directory -Path $BUNDLE | Out-Null
Write-Host "Bundle dir: $BUNDLE" -ForegroundColor Cyan

function Pack-Folder {
    param([string]$Source, [string]$ZipName, [string]$Description)
    if (-not (Test-Path $Source)) {
        Write-Host "  [skip] $Description -- source not found: $Source" -ForegroundColor DarkYellow
        return
    }
    $Out = Join-Path $BUNDLE $ZipName
    Write-Host "  Packing $Description ..." -ForegroundColor White -NoNewline
    Compress-Archive -Path $Source -DestinationPath $Out -CompressionLevel NoCompression
    $sz = (Get-Item $Out).Length / 1MB
    Write-Host (" {0:N1} MB" -f $sz) -ForegroundColor Green
}

# === MINIMUM STACK ===

Write-Host "`n=== Minimum stack (~800 MB) ===" -ForegroundColor Cyan
Pack-Folder $MEMORY "memory_backup.zip" "Claude memory (~25 plikow)"

# third_party minimum: contrib (101 MB) + ORT (43 MB), pomijamy opencv prebuilt 887 MB
$tp_min = @(
    "$REPO\third_party\opencv_contrib_install",
    "$REPO\third_party\onnxruntime-directml",
    "$REPO\third_party\cmake-3.31.2-windows-x86_64"
)
$tp_existing = $tp_min | Where-Object { Test-Path $_ }
if ($tp_existing.Count -gt 0) {
    Write-Host "  Packing third_party_minimal..." -ForegroundColor White -NoNewline
    Compress-Archive -Path $tp_existing -DestinationPath (Join-Path $BUNDLE "third_party_minimal.zip") -CompressionLevel NoCompression
    $sz = (Get-Item (Join-Path $BUNDLE "third_party_minimal.zip")).Length / 1MB
    Write-Host (" {0:N1} MB" -f $sz) -ForegroundColor Green
}

Pack-Folder "$REPO\data\weights" "weights.zip" "data/weights (v4+v3 ONNX+PT)"
Pack-Folder "$REPO\artifacts\test_videos" "test_videos.zip" "artifacts/test_videos"

# === FULL STACK (opcja) ===

if ($Full) {
    Write-Host "`n=== Full stack additions (~6 GB more) ===" -ForegroundColor Cyan
    Pack-Folder "$REPO\third_party\opencv" "third_party_opencv_prebuilt.zip" "OpenCV prebuilt (fallback bez contrib)"
    Pack-Folder "$REPO\data\cvat_exports" "cvat_exports.zip" "CVAT export source (re-train v4)"
    Pack-Folder "$REPO\data\ext_rgb_drone" "ext_rgb_drone.zip" "DJI clips dji0002/0003/0005"
    Pack-Folder "$REPO\data\ext_sea" "ext_sea.zip" "Sea videos (training negs)"
    Pack-Folder "$REPO\data\roboflow_drone_v1" "roboflow_drone_v1.zip" "Roboflow public drone dataset"
    Pack-Folder "$REPO\training\v4" "training_v4_dataset.zip" "v4 dataset (4032 obrazki)"
    if (Test-Path "$REPO\data\test.mp4") {
        Copy-Item "$REPO\data\test.mp4" $BUNDLE
        Write-Host "  Copied data/test.mp4" -ForegroundColor Green
    }
    if (Test-Path "$REPO\video.mp4") {
        Copy-Item "$REPO\video.mp4" $BUNDLE
        Write-Host "  Copied video.mp4" -ForegroundColor Green
    }
}

# === Summary ===

Write-Host "`n=== DONE ===" -ForegroundColor Cyan
$total = 0.0
Get-ChildItem $BUNDLE | ForEach-Object {
    $sz = $_.Length / 1MB
    $total += $sz
    Write-Host (" {0,-40} {1,10:N1} MB" -f $_.Name, $sz)
}
Write-Host (" {0,-40} {1,10:N1} MB" -f "TOTAL", $total) -ForegroundColor Yellow
Write-Host "`nNext: skopiuj $BUNDLE na nowy komputer (USB/OneDrive/network)." -ForegroundColor Cyan
Write-Host "Na nowym komputerze: rozpakuj kazdy ZIP do oryginalnej sciezki -- patrz docs/env_setup.md sekcja 10."

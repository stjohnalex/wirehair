param(
    [Parameter(Mandatory = $true)]
    [string] $TargetPath
)

$fn = [System.IO.Path]::GetFileName($TargetPath)
if ($fn) {
    & taskkill.exe /IM $fn /F /T 2>$null | Out-Null
}

Start-Sleep -Milliseconds 500

for ($i = 0; $i -lt 40; $i++) {
    if (-not (Test-Path -LiteralPath $TargetPath)) {
        exit 0
    }
    try {
        & attrib.exe -R $TargetPath 2>$null | Out-Null
        Remove-Item -LiteralPath $TargetPath -Force -ErrorAction Stop
        exit 0
    }
    catch {
        Start-Sleep -Milliseconds 150
    }
}

exit 0

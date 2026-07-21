# Prompt 9 API smoke tests — run ONLY after Terminal 1 shows "Application startup complete."
# Usage:  cd D:\Documents\Kamran\xai-project
#         .\scripts\test_api.ps1

$ErrorActionPreference = "Stop"
$Base = "http://127.0.0.1:8000"
$Pass = 0
$Fail = 0

function Assert-Ok {
    param(
        [string]$Name,
        [int]$ExpectedStatus,
        [scriptblock]$Call
    )
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    try {
        $result = & $Call
        $status = $result.StatusCode
        if ($status -eq $ExpectedStatus) {
            Write-Host "PASS  HTTP $status" -ForegroundColor Green
            if ($result.Preview) { Write-Host $result.Preview }
            $script:Pass++
        } else {
            Write-Host "FAIL  expected HTTP $ExpectedStatus, got HTTP $status" -ForegroundColor Red
            if ($result.Preview) { Write-Host $result.Preview }
            $script:Fail++
        }
    } catch {
        $status = $null
        if ($_.Exception.Response) {
            $status = [int]$_.Exception.Response.StatusCode
        }
        if ($status -eq $ExpectedStatus) {
            Write-Host "PASS  HTTP $status (expected error)" -ForegroundColor Green
            $script:Pass++
        } else {
            Write-Host "FAIL  $_" -ForegroundColor Red
            $script:Fail++
        }
    }
}

function Get-Json {
    param([string]$Url)
    $r = Invoke-WebRequest -Uri $Url -UseBasicParsing
    $body = $r.Content | ConvertFrom-Json
    $preview = if ($body -is [array]) { "items: $($body.Count)" } else { ($r.Content.Substring(0, [Math]::Min(120, $r.Content.Length)) + "...") }
    return @{ StatusCode = $r.StatusCode; Preview = $preview; Body = $body }
}

function Post-Json {
    param([string]$Url, [hashtable]$Payload)
    $json = $Payload | ConvertTo-Json -Compress
    $r = Invoke-WebRequest -Uri $Url -Method POST -ContentType "application/json" -Body $json -UseBasicParsing
    return @{ StatusCode = $r.StatusCode; Preview = $r.Content; Body = ($r.Content | ConvertFrom-Json) }
}

# Wait for server (up to 30 s)
Write-Host "Waiting for API at $Base ..." -ForegroundColor Yellow
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $h = Invoke-WebRequest -Uri "$Base/api/health" -UseBasicParsing -TimeoutSec 2
        if ($h.StatusCode -eq 200) { $ready = $true; break }
    } catch { Start-Sleep -Seconds 1 }
}
if (-not $ready) {
    Write-Host "ERROR: API not reachable. Start Terminal 1 first:" -ForegroundColor Red
    Write-Host "  .venv\Scripts\activate" -ForegroundColor Red
    Write-Host "  uvicorn api.main:app --reload --host 127.0.0.1 --port 8000" -ForegroundColor Red
    exit 1
}
Write-Host "API is up.`n" -ForegroundColor Green

Assert-Ok "GET /api/health" 200 { Get-Json "$Base/api/health" }

Assert-Ok "GET /api/status" 200 {
    $r = Get-Json "$Base/api/status"
    $r.Preview = "steps: $($r.Body.Count), done: $(($r.Body | Where-Object done).Count)"
    $r
}

Assert-Ok "POST /api/pipeline/run (predict)" 200 {
    Post-Json "$Base/api/pipeline/run" @{ step = "predict" }
}

Assert-Ok "POST /api/pipeline/run (invalid step=status)" 400 {
    try {
        Post-Json "$Base/api/pipeline/run" @{ step = "status" }
    } catch {
        throw
    }
}

Assert-Ok "GET /api/pipeline/log" 200 {
    $r = Get-Json "$Base/api/pipeline/log"
    $r.Preview = "log lines: $($r.Body.lines.Count)"
    $r
}

Assert-Ok "GET /api/evaluate/leaderboard" 200 {
    $r = Get-Json "$Base/api/evaluate/leaderboard"
    $r.Preview = "models: $($r.Body.Count)"
    $r
}

Assert-Ok "GET /api/evaluate/detail?model=xgboost" 200 {
    $r = Get-Json "$Base/api/evaluate/detail?model=xgboost"
    $imgCount = @($r.Body.images.PSObject.Properties).Count
    $r.Preview = "metrics keys: $(@($r.Body.metrics.PSObject.Properties.Name).Count), images: $imgCount"
    $r
}

Assert-Ok "GET /api/evaluate/training-curve?model=xgboost" 200 {
    $r = Get-Json "$Base/api/evaluate/training-curve?model=xgboost"
    $hasImg = [bool]$r.Body.image
    $r.Preview = "image present: $hasImg"
    $r
}

Assert-Ok "GET /api/xai/global?model=xgboost&plot_type=beeswarm" 200 {
    $r = Get-Json "$Base/api/xai/global?model=xgboost&plot_type=beeswarm"
    $r.Preview = "image present: $([bool]$r.Body.image)"
    $r
}

Assert-Ok "GET /api/xai/local/instances" 200 {
    $r = Get-Json "$Base/api/xai/local/instances"
    $r.Preview = "instances: $($r.Body.instances.Count)"
    $r
}

Assert-Ok "GET /api/xai/local/image" 200 {
    $url = "$Base/api/xai/local/image?explainer=xgb_shap&true_class=BENIGN&outcome=correct&instance_idx=800461"
    $r = Get-Json $url
    $r.Preview = "image present: $([bool]$r.Body.image)"
    $r
}

foreach ($metric in @("fidelity", "stability", "comprehensibility")) {
    Assert-Ok "GET /api/xai/quality?metric=$metric" 200 {
        $r = Get-Json "$Base/api/xai/quality?metric=$metric"
        $r.Preview = "rows: $($r.Body.rows.Count), image: $([bool]$r.Body.image)"
        $r
    }
}

Assert-Ok "POST /api/predict" 200 {
    $filePath = (Join-Path $PSScriptRoot "..\data\manual\example_flows.csv" | Resolve-Path).Path
    $tmpBody = [System.IO.Path]::GetTempFileName()
    try {
        $statusLine = curl.exe -s -o $tmpBody -w "%{http_code}" -X POST "$Base/api/predict" -F "file=@$filePath" -F "model=xgboost"
        $status = [int]$statusLine.Trim()
        $body = Get-Content -Path $tmpBody -Raw -Encoding UTF8 | ConvertFrom-Json
        return @{
            StatusCode = $status
            Preview    = "total_rows: $($body.total_rows), predictions: $($body.predictions.Count)"
            Body       = $body
        }
    } finally {
        Remove-Item -Path $tmpBody -ErrorAction SilentlyContinue
    }
}

Assert-Ok "GET /api/reports" 200 {
    $r = Get-Json "$Base/api/reports"
    $r.Preview = "files: $($r.Body.Count)"
    $r
}

Assert-Ok "GET /api/reports/file?path=reports/model_card.md" 200 {
    $r = Invoke-WebRequest -Uri "$Base/api/reports/file?path=reports/model_card.md" -UseBasicParsing
    $body = $r.Content | ConvertFrom-Json
    return @{
        StatusCode = $r.StatusCode
        Preview    = "type: $($body.type), chars: $($body.content.Length)"
        Body       = $body
    }
}

Assert-Ok "GET /api/reports/file (404 test)" 404 {
    try {
        Invoke-WebRequest -Uri "$Base/api/reports/file?path=reports/does_not_exist.md" -UseBasicParsing
    } catch {
        if ($_.Exception.Response) {
            return @{ StatusCode = [int]$_.Exception.Response.StatusCode }
        }
        throw
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Results: $Pass passed, $Fail failed" -ForegroundColor $(if ($Fail -eq 0) { "Green" } else { "Red" })
if ($Fail -gt 0) { exit 1 }

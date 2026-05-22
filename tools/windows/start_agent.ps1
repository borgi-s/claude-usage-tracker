# Restart-on-crash wrapper for the Claude usage tracker.
# Started by Task Scheduler at user logon. Runs hidden — see agent.log for output.

$ErrorActionPreference = "Continue"
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$LogFile     = Join-Path $PSScriptRoot "agent.log"
$Venv        = Join-Path $ProjectRoot ".venv\Scripts\streamlit.exe"
$App         = Join-Path $ProjectRoot "app.py"
$Port        = 8765

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $LogFile -Value $line -Encoding utf8
}

Log "=== start_agent.ps1 starting ==="
Log "Project: $ProjectRoot"
Log "Streamlit: $Venv"

if (-not (Test-Path $Venv)) {
    Log "FATAL: streamlit not found at $Venv"
    exit 1
}
if (-not (Test-Path $App)) {
    Log "FATAL: app.py not found at $App"
    exit 1
}

# Restart loop with exponential-ish backoff capped at 5 min.
$backoff = 5
while ($true) {
    Log "Launching streamlit on port $Port (backoff=$backoff s on failure)"
    try {
        $proc = Start-Process -FilePath $Venv `
            -ArgumentList "run", $App, "--server.headless", "true", "--server.port", $Port, "--browser.gatherUsageStats", "false" `
            -WorkingDirectory $ProjectRoot `
            -RedirectStandardOutput $LogFile `
            -RedirectStandardError  $LogFile `
            -NoNewWindow `
            -PassThru `
            -Wait
        Log "Streamlit exited with code $($proc.ExitCode)"
    } catch {
        Log "Launch failed: $_"
    }
    Start-Sleep -Seconds $backoff
    $backoff = [Math]::Min($backoff * 2, 300)
}

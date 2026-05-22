# Register the user-scope logon task that starts start_agent.ps1.
# No admin required (current-user scope).
#
# Usage:
#   .\install_scheduled_task.ps1            # install or update
#   .\install_scheduled_task.ps1 -Uninstall # remove

param(
    [switch]$Uninstall,
    [string]$TaskName = "ClaudeUsageTracker"
)

$StartScript = Join-Path $PSScriptRoot "start_agent.ps1"

if ($Uninstall) {
    schtasks.exe /Delete /TN $TaskName /F
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Removed scheduled task '$TaskName'." -ForegroundColor Green
    } else {
        Write-Host "schtasks delete failed (exit $LASTEXITCODE). Task may not exist." -ForegroundColor Yellow
    }
    exit $LASTEXITCODE
}

if (-not (Test-Path $StartScript)) {
    Write-Host "FATAL: start_agent.ps1 not found at $StartScript" -ForegroundColor Red
    exit 1
}

# Quote-safe args for schtasks. Use powershell.exe (Windows PowerShell 5.1) for
# maximum compatibility — pwsh.exe isn't guaranteed to exist.
$action = 'powershell.exe -NoLogo -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $StartScript + '"'

# /SC ONLOGON      = run at logon
# /RL LIMITED      = current-user privilege (no UAC elevation, no admin)
# /F               = overwrite existing task with same name
schtasks.exe /Create /TN $TaskName /TR $action /SC ONLOGON /RL LIMITED /F

if ($LASTEXITCODE -ne 0) {
    Write-Host "schtasks /Create failed (exit $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Installed scheduled task '$TaskName'." -ForegroundColor Green
Write-Host "It will run at every logon, starting Streamlit hidden on port 8765." -ForegroundColor Green
Write-Host ""
Write-Host "To start it RIGHT NOW (without logging out):" -ForegroundColor Cyan
Write-Host "  schtasks /Run /TN $TaskName"
Write-Host ""
Write-Host "To see what it's doing:" -ForegroundColor Cyan
Write-Host "  .\show_agent_log.ps1"
Write-Host ""
Write-Host "To uninstall:" -ForegroundColor Cyan
Write-Host "  .\install_scheduled_task.ps1 -Uninstall"

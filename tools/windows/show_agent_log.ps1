# Live tail of the agent log. Run this whenever you want to "see the window".
# Ctrl+C exits.

$LogFile = Join-Path $PSScriptRoot "agent.log"

if (-not (Test-Path $LogFile)) {
    Write-Host "No agent.log yet at $LogFile" -ForegroundColor Yellow
    Write-Host "Either the agent hasn't started, or it ran from a different location."
    exit 1
}

Write-Host "Tailing $LogFile" -ForegroundColor Cyan
Write-Host "(Ctrl+C to exit)" -ForegroundColor DarkGray
Write-Host ""
Get-Content -Path $LogFile -Tail 200 -Wait

# Installs Cognitive Monitor as a Windows Task Scheduler task that starts
# automatically at logon.
#
# Usage (run in an elevated PowerShell):
#   .\scripts\install_windows_taskscheduler.ps1 -Mode full
#
param(
    [ValidateSet("full", "input-only")]
    [string]$Mode = "full"
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = (Get-Command python).Source
$DaemonScript = Join-Path $ProjectRoot "src\realtime\daemon.py"

$Action = New-ScheduledTaskAction -Execute $PythonExe `
    -Argument "`"$DaemonScript`" --mode $Mode" `
    -WorkingDirectory $ProjectRoot

$Trigger = New-ScheduledTaskTrigger -AtLogOn

$Settings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 0) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName "CognitiveMonitor" -Action $Action -Trigger $Trigger `
    -Settings $Settings -Description "Real-time cognitive state monitor" -Force

Write-Host "Installed. Manage it with:"
Write-Host "  Start-ScheduledTask -TaskName CognitiveMonitor"
Write-Host "  Stop-ScheduledTask  -TaskName CognitiveMonitor"
Write-Host "  Unregister-ScheduledTask -TaskName CognitiveMonitor -Confirm:`$false   # remove autostart"
Write-Host ""
Write-Host "Note: Windows Focus Assist has no supported scripting API, so"
Write-Host "Do Not Disturb toggling relies on the app's own notification"
Write-Host "suppression rather than the OS Focus Assist setting."

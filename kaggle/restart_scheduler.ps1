# Restart the scheduler only while it is sleeping (never mid-cycle); closes the old window, opens exactly one new one.
$hb = 'C:\Users\rohit\sih26074-v3\kaggle\scheduler_heartbeat.json'
$deadline = (Get-Date).AddMinutes(10)
while ((Get-Date) -lt $deadline) {
    $h = Get-Content $hb -Raw | ConvertFrom-Json
    $now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    if ($h.phase -eq 'sleeping' -and ($h.until - $now) -gt 20) { break }
    Start-Sleep 3
}
$old = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*kaggle\scheduler.py*' }
foreach ($p in $old) {
    Stop-Process -Id $p.ProcessId -Force
    Stop-Process -Id $p.ParentProcessId -Force -ErrorAction SilentlyContinue   # its console window
}
Start-Process powershell -ArgumentList '-NoExit', '-Command', "`$host.UI.RawUI.WindowTitle='SIH-26074 scheduler + dashboard'; cd C:\Users\rohit\sih26074-v3; python kaggle\scheduler.py"
"restarted while sleeping (stopped $($old.Count) old instance(s))"

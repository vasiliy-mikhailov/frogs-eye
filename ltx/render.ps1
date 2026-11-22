param([int]$W = 768, [int]$H = 512, [int]$F = 121, [int]$Fps = 24, [long]$Seed = 42, [string]$Prompt = "A slow dolly-in on a lighthouse at dusk, waves crashing on the rocks below, gulls calling, wind howling.")
$log = "C:\ltx\build\gpu_samples.csv"
Remove-Item $log -ErrorAction SilentlyContinue
$job = Start-Job -ScriptBlock { param($l) while ($true) { nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu,fan.speed --format=csv,noheader,nounits | Out-File -Append -Encoding ascii $l; Start-Sleep 2 } } -ArgumentList $log
$env:LTX_PROMPT = $Prompt  # via the environment: PowerShell 5.1 would mangle inner double quotes on the command line
docker run --rm --network container:ltx -e LTX_PROMPT -v C:\ltx\build:/b:ro python:3.12 python /b/render_test.py $W $H $F $Fps $Seed
Stop-Job $job; Remove-Job $job -Force
$rows = Get-Content $log | ForEach-Object { $p = $_.Split(","); [pscustomobject]@{ mem = [int]$p[0]; util = [int]$p[1]; pw = [double]$p[2]; t = [int]$p[3]; fan = [int]$p[4] } }
"GPU samples: $($rows.Count); peak VRAM used: $(($rows | Measure-Object mem -Maximum).Maximum) MiB; max util: $(($rows | Measure-Object util -Maximum).Maximum) %; max power: $(($rows | Measure-Object pw -Maximum).Maximum) W; temperature $(($rows | Measure-Object t -Minimum).Minimum)-$(($rows | Measure-Object t -Maximum).Maximum) C; max fan $(($rows | Measure-Object fan -Maximum).Maximum) %"
New-Item -ItemType Directory -Force -Path C:\ltx\out | Out-Null
docker cp ltx:/data/out/ltx C:\ltx\out\ 2>$null
Get-ChildItem C:\ltx\out\ltx -Filter *.mp4 | Sort-Object LastWriteTime | Select-Object -Last 2 | ForEach-Object { "$($_.Name) $([math]::Round($_.Length/1MB,2)) MB" }

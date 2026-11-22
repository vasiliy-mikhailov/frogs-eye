# Recreate the ltx container: ComfyUI + adapter (mounted read-only from C:\ltx\build\adapter), engine key from the profile.
$envfile = "$env:USERPROFILE\.ltx\engine.env"
if (-not (Test-Path $envfile)) { "missing $envfile"; exit 1 }
docker rm -f ltx 2>$null | Out-Null
docker run -d --name ltx --gpus all --restart unless-stopped -p 127.0.0.1:8188:8188 -p 127.0.0.1:8101:8101 `
  -v ltx-models:/models -v ltx-data:/data -v C:\ltx\build\adapter:/opt/adapter:ro --env-file $envfile `
  --entrypoint /bin/bash ltx-comfy:v0.45.0 /opt/adapter/run.sh | Out-Null
$ok = $false
for ($i = 0; $i -lt 90; $i++) { Start-Sleep 2; try { $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 http://127.0.0.1:8101/health; if ($r.StatusCode -eq 200) { $ok = $true; break } } catch {} }
"adapter healthy: $ok after $((($i + 1) * 2)) s"
docker logs --tail 5 ltx 2>&1 | ForEach-Object { "$_" }

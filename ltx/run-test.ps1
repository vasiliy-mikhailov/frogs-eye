param([string]$phase = "main")
docker run --rm --network container:ltx --env-file $env:USERPROFILE\.ltx\engine.env -v C:\ltx\build\testout:/t -v C:\ltx\build\adapter_test.py:/opt/adapter_test.py:ro --entrypoint python ltx-comfy:v0.45.0 -u /opt/adapter_test.py $phase 2>&1 | ForEach-Object { "$_" }

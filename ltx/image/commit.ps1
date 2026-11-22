# Turn the finished ltx-setup container (image/setup.sh) into ltx-comfy:v0.45.0. The production tag only moves after
# setup exited 0 and the committed candidate sees the GPU from torch and has ComfyUI-GGUF; otherwise nothing is retagged
# and ltx-setup is kept for its logs. PowerShell 5.1 strips bare inner double quotes from native arguments, hence \"
# inside the ENTRYPOINT JSON.
$st = docker inspect -f "{{.State.Status}} {{.State.ExitCode}}" ltx-setup 2>$null
if ($LASTEXITCODE -ne 0 -or "$st" -ne "exited 0") { "ltx-setup did not finish cleanly (state: '$st'); see docker logs ltx-setup"; exit 1 }
docker commit --change "WORKDIR /opt/ComfyUI" --change "EXPOSE 8188 8101" --change 'ENTRYPOINT [\"/opt/service/entrypoint.sh\"]' --change 'CMD []' ltx-setup ltx-comfy:candidate | Out-Null
if ($LASTEXITCODE -ne 0) { "docker commit failed"; exit 1 }
docker run --rm --gpus all --entrypoint python -w /opt/ComfyUI ltx-comfy:candidate -c "import os, torch, gguf; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); assert os.path.isfile('custom_nodes/ComfyUI-GGUF/nodes.py'); print('ComfyUI-GGUF present')"
if ($LASTEXITCODE -ne 0) { "the candidate image failed its check; ltx-comfy:v0.45.0 is unchanged"; docker rmi ltx-comfy:candidate | Out-Null; exit 1 }
docker tag ltx-comfy:candidate ltx-comfy:v0.45.0
docker rmi ltx-comfy:candidate | Out-Null
docker rm ltx-setup | Out-Null
"entrypoint: " + (docker image inspect -f "{{json .Config.Entrypoint}} cmd={{json .Config.Cmd}}" ltx-comfy:v0.45.0)
docker images ltx-comfy --format "{{.Repository}}:{{.Tag}} {{.Size}}"

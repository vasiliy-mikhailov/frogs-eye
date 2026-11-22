"""Download LTX-2.5 files into /models (a Docker volume) and verify each against its Hugging Face sha256.
usage: python ltx_dl.py repo:path:subdir [...]   e.g. unsloth/LTX-2.5-GGUF:distilled/ltx-2.5-22b-distilled-Q6_K.gguf:diffusion_models
Writes progress and results to /models/_dl_status.txt. Uses the stored Hugging Face login if HF_TOKEN_PATH points at one (never printed)."""
import hashlib, json, os, sys, time, urllib.request

from huggingface_hub import hf_hub_download

STATUS = "/models/_dl_status.txt"


def log(msg):
    line = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + msg
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def expected_sha(repo, path):
    from huggingface_hub import HfApi  # uses the stored login (HF_TOKEN_PATH) for gated repos
    info = HfApi().get_paths_info(repo, [path])[0]
    return info.lfs.sha256, info.size


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


for spec in sys.argv[1:]:
    repo, path, sub = spec.split(":")
    name = path.split("/")[-1]
    dest_dir = os.path.join("/models", sub)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, name)
    want, size = expected_sha(repo, path)
    if os.path.exists(dest) and os.path.getsize(dest) == size and sha256(dest) == want:
        log("OK already %s/%s" % (sub, name))
        continue
    t = time.time()
    log("START %s %s (%.2f GB)" % (repo, path, size / 1e9))
    got = hf_hub_download(repo, path, local_dir="/models/_hf/" + repo.replace("/", "__"))
    os.replace(got, dest)
    dt = time.time() - t
    ok = sha256(dest) == want
    log("%s %s/%s %.2f GB in %.0f s (%.1f MB/s) sha256 %s" % ("OK" if ok else "BAD", sub, name, size / 1e9, dt, size / 1e6 / max(dt, 1), "matches" if ok else "MISMATCH"))
    if not ok:
        os.remove(dest)
log("DONE")

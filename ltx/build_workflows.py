"""Builds the two ComfyUI API-format workflows the adapter loads (adapter/ltx25_distilled_gguf_{t2v,i2v}_api.json).

The graphs are adapted from Comfy-Org/workflow_templates video_ltx2_5_t2v / video_ltx2_5_i2v (MIT) (two-stage: half-size 8-step pass, x2 latent upsample, 3-step
refine), with the DiT swapped for the Unsloth Q6_K GGUF (ComfyUI-GGUF's UnetLoaderGGUF), the official int8 Gemma 4
text encoder, and tiled VAE decode. Nodes titled "ADAPTER ..." are the ones ltx_video_adapter.py fills per job.
Usage: python build_workflows.py   (checks every link and writes the t2v, i2v and ext JSONs into adapter/)
"""
from pathlib import Path
import json, sys
NEG = "pc game, console game, video game, cartoon, childish, ugly"
def N(cls, inputs, title):
    return {"class_type": cls, "inputs": inputs, "_meta": {"title": title}}

common = {
 "384": N("UnetLoaderGGUF", {"unet_name": "ltx-2.5-22b-distilled-Q6_K.gguf"}, "Unet Loader (GGUF) [replaces UNETLoader]"),
 "387": N("CLIPLoader", {"clip_name": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "type": "ltxv", "device": "default"}, "Load CLIP (Gemma 4 12B + proj, int8)"),
 "385": N("VAELoader", {"vae_name": "ltx-2.5-video-vae-bf16.safetensors"}, "Load VAE (video, DiffVAE)"),
 "386": N("VAELoader", {"vae_name": "ltx-2.5-audio-vae-bf16.safetensors"}, "Load VAE (audio)"),
 "371": N("LatentUpscaleModelLoader", {"model_name": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"}, "Load Latent Upscale Model"),
 "364": N("CLIPTextEncode", {"text": "A slow dolly-in on a lighthouse at dusk, waves crashing on the rocks below, gulls calling, wind howling.", "clip": ["387", 0]}, "ADAPTER positive prompt"),
 "373": N("CLIPTextEncode", {"text": NEG, "clip": ["387", 0]}, "ADAPTER negative prompt"),
 "372": N("PrimitiveInt", {"value": 1280}, "ADAPTER width"),
 "360": N("PrimitiveInt", {"value": 704}, "ADAPTER height"),
 "361": N("PrimitiveInt", {"value": 24}, "ADAPTER fps"),
 "362": N("PrimitiveInt", {"value": 121}, "ADAPTER frames (8k+1)"),
 "353": N("ComfyMathExpression", {"expression": "a/2", "values.a": ["372", 0]}, "stage-1 width = W/2"),
 "355": N("ComfyMathExpression", {"expression": "a/2", "values.a": ["360", 0]}, "stage-1 height = H/2"),
 "359": N("ComfyMathExpression", {"expression": "a", "values.a": ["361", 0]}, "fps as FLOAT(0) / INT(1)"),
 "365": N("LTXVConditioning", {"positive": ["364", 0], "negative": ["373", 0], "frame_rate": ["359", 0]}, "LTXVConditioning"),
 "356": N("EmptyLTXVLatentVideo", {"width": ["353", 1], "height": ["355", 1], "length": ["362", 0], "batch_size": 1}, "Empty LTXV Latent Video (stage 1, half size)"),
 "366": N("LTXVEmptyLatentAudio", {"frames_number": ["362", 0], "frame_rate": ["359", 1], "batch_size": 1, "audio_vae": ["386", 0]}, "Empty LTXV Latent Audio"),
 "339": N("RandomNoise", {"noise_seed": 558811532553686}, "ADAPTER seed (stage 1 noise)"),
 "352": N("KSamplerSelect", {"sampler_name": "euler_ancestral"}, "KSamplerSelect (stage 1)"),
 "388": N("LTXVDualCFGGuider", {"model": ["384", 0], "positive": ["365", 0], "negative": ["365", 1], "video_cfg": 1.0, "audio_cfg": 1.0}, "LTXV Dual CFG Guider (stage 1)"),
 "344": N("SamplerCustomAdvanced", {"noise": ["339", 0], "guider": ["388", 0], "sampler": ["352", 0], "sigmas": ["S1", 0], "latent_image": ["377", 0]}, "SamplerCustomAdvanced (stage 1, 8 steps)"),
 "367": N("LTXVSeparateAVLatent", {"av_latent": ["344", 0]}, "Separate AV Latent (after stage 1)"),
 "348": N("LTXVLatentUpsampler", {"samples": ["367", 0], "upscale_model": ["371", 0], "vae": ["385", 0]}, "LTXV Latent Upsampler x2"),
 "338": N("RandomNoise", {"noise_seed": 42}, "RandomNoise (stage 2, fixed 42 as in template)"),
 "341": N("KSamplerSelect", {"sampler_name": "euler_ancestral"}, "KSamplerSelect (stage 2)"),
 "391": N("LTXVDualCFGGuider", {"model": ["384", 0], "positive": ["365", 0], "negative": ["365", 1], "video_cfg": 1.0, "audio_cfg": 1.0}, "LTXV Dual CFG Guider (stage 2)"),
 "368": N("SamplerCustomAdvanced", {"noise": ["338", 0], "guider": ["391", 0], "sampler": ["341", 0], "sigmas": ["S2", 0], "latent_image": ["340", 0]}, "SamplerCustomAdvanced (stage 2, 3 steps)"),
 "369": N("LTXVSeparateAVLatent", {"av_latent": ["368", 0]}, "Separate AV Latent (final)"),
 "374": N("VAEDecodeTiled", {"samples": ["369", 0], "vae": ["385", 0], "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 16}, "VAE Decode (Tiled)"),
 "358": N("LTXVAudioVAEDecode", {"samples": ["369", 1], "audio_vae": ["386", 0]}, "LTXV Audio VAE Decode"),
 "370": N("CreateVideo", {"images": ["374", 0], "audio": ["358", 0], "fps": ["359", 0]}, "Create Video"),
 "75": N("SaveVideo", {"video": ["370", 0], "filename_prefix": "ltx/run", "format": "mp4", "format.codec": "h264"}, "Save Video (mp4, H.264 + AAC)"),
}

def t2v():
    g = json.loads(json.dumps(common))
    g["404"] = N("ManualSigmas", {"sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"}, "ManualSigmas (stage 1)")
    g["395"] = N("ManualSigmas", {"sigmas": "0.85, 0.7250, 0.4219, 0.0"}, "ManualSigmas (stage 2)")
    g["344"]["inputs"]["sigmas"] = ["404", 0]
    g["368"]["inputs"]["sigmas"] = ["395", 0]
    g["377"] = N("LTXVConcatAVLatent", {"video_latent": ["356", 0], "audio_latent": ["366", 0]}, "Concat AV Latent (stage 1)")
    g["340"] = N("LTXVConcatAVLatent", {"video_latent": ["348", 0], "audio_latent": ["367", 1]}, "Concat AV Latent (stage 2)")
    return g

def i2v():
    g = json.loads(json.dumps(common))
    g["339"]["inputs"]["noise_seed"] = 875362541677469
    g["397"] = N("ManualSigmas", {"sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"}, "ManualSigmas (stage 1)")
    g["396"] = N("ManualSigmas", {"sigmas": "0.85, 0.7250, 0.4219, 0.0"}, "ManualSigmas (stage 2)")
    g["344"]["inputs"]["sigmas"] = ["397", 0]
    g["368"]["inputs"]["sigmas"] = ["396", 0]
    g["395"] = N("LoadImage", {"image": "first_frame.png"}, "ADAPTER input image (Load First Frame)")
    g["351"] = N("ResizeImageMaskNode", {"input": ["395", 0], "resize_type": "scale longer dimension", "resize_type.longer_size": 1536, "scale_method": "lanczos"}, "Resize Image/Mask (longer side 1536)")
    g["350"] = N("LTXVPreprocess", {"image": ["351", 0], "img_compression": 18}, "LTXV Preprocess")
    g["357"] = N("LTXVImgToVideoInplace", {"vae": ["385", 0], "image": ["350", 0], "latent": ["356", 0], "strength": 0.7, "bypass": False}, "LTXVImgToVideoInplace (stage 1, frame 0)")
    g["349"] = N("LTXVImgToVideoInplace", {"vae": ["385", 0], "image": ["350", 0], "latent": ["348", 0], "strength": 1.0, "bypass": False}, "LTXVImgToVideoInplace (stage 2, frame 0)")
    g["377"] = N("LTXVConcatAVLatent", {"video_latent": ["357", 0], "audio_latent": ["366", 0]}, "Concat AV Latent (stage 1)")
    g["340"] = N("LTXVConcatAVLatent", {"video_latent": ["349", 0], "audio_latent": ["367", 1]}, "Concat AV Latent (stage 2)")
    g["364"]["inputs"]["text"] = "The subject in the first frame slowly turns toward the camera and smiles; soft ambient room tone."
    return g

def ext():
    """Extend: continue an existing clip. The last frames of the source (cut by the adapter into a short mp4 in the
    input folder) are VAE-encoded into the first latent frames at both stages and pinned (strength 1.0), and the
    source's last seconds of sound are encoded and pinned at the start of the audio stream; LTXVConcatAVLatent zero-
    pads the rest, unmasked, so the model continues picture and sound from there. The adapter cuts the overlap off
    the result and appends the rest to the source."""
    g = i2v()
    for nid in ("395", "351", "350"):
        del g[nid]
    g["500"] = N("LoadVideo", {"file": "overlap.mp4"}, "ADAPTER overlap clip (tail of the source video)")
    g["501"] = N("GetVideoComponents", {"video": ["500", 0]}, "Get Video Components (overlap frames + audio)")
    g["357"]["inputs"].update({"image": ["501", 0], "strength": 1.0})
    g["357"]["_meta"]["title"] = "LTXVImgToVideoInplace (stage 1, overlap frames pinned)"
    g["349"]["inputs"].update({"image": ["501", 0], "strength": 1.0})
    g["349"]["_meta"]["title"] = "LTXVImgToVideoInplace (stage 2, overlap frames pinned)"
    g["502"] = N("LTXVAudioVAEEncode", {"audio": ["501", 1], "audio_vae": ["386", 0]}, "Encode overlap audio")
    g["503"] = N("SolidMask", {"value": 0.0, "width": 1, "height": 1}, "Mask 0 = keep")
    g["504"] = N("SetLatentNoiseMask", {"samples": ["502", 0], "mask": ["503", 0]}, "Pin the overlap audio")
    g["505"] = N("LTXVConcatAVLatent", {"video_latent": ["377", 0], "audio_latent": ["504", 0]},
                 "Swap in the pinned overlap audio (zero-padded, unmasked tail is generated)")
    g["344"]["inputs"]["latent_image"] = ["505", 0]
    g["364"]["inputs"]["text"] = "The scene continues naturally from the previous shot."
    return g

# outputs per class (count) from node sources
OUTS = {"UnetLoaderGGUF":1,"CLIPLoader":1,"VAELoader":1,"LatentUpscaleModelLoader":1,"CLIPTextEncode":1,"PrimitiveInt":1,
 "ComfyMathExpression":3,"LTXVConditioning":2,"EmptyLTXVLatentVideo":1,"LTXVEmptyLatentAudio":1,"LTXVConcatAVLatent":1,
 "RandomNoise":1,"KSamplerSelect":1,"ManualSigmas":1,"LTXVDualCFGGuider":1,"SamplerCustomAdvanced":2,"LTXVSeparateAVLatent":2,
 "LTXVLatentUpsampler":1,"VAEDecodeTiled":1,"LTXVAudioVAEDecode":1,"CreateVideo":1,"SaveVideo":1,"LoadImage":2,
 "ResizeImageMaskNode":1,"LTXVPreprocess":1,"LTXVImgToVideoInplace":1,
 "LoadVideo":1,"GetVideoComponents":5,"LTXVAudioVAEEncode":1,"SolidMask":1,"SetLatentNoiseMask":1}
def check(g, name):
    used=set()
    for nid,n in g.items():
        for k,v in n["inputs"].items():
            if isinstance(v,list):
                src,idx=v
                assert src in g, (name,nid,k,src)
                assert idx < OUTS[g[src]["class_type"]], (name,nid,k,v)
                used.add(src)
    # reachability from output node
    seen=set(); stack=["75"]
    while stack:
        x=stack.pop()
        if x in seen: continue
        seen.add(x)
        stack += [v[0] for v in g[x]["inputs"].values() if isinstance(v,list)]
    unreach=set(g)-seen
    print(name, "nodes", len(g), "unreachable", unreach)
for name,g in (("t2v",t2v()),("i2v",i2v()),("ext",ext())):
    check(g,name)
    json.dump(dict(sorted(g.items(), key=lambda kv:int(kv[0]))), open(Path(__file__).resolve().parent / "adapter" / f"ltx25_distilled_gguf_{name}_api.json","w"), indent=1, ensure_ascii=False)

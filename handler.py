import os
import io
import base64
import tempfile
import torch
import numpy as np
from PIL import Image
import runpod
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video
from transformers import CLIPVisionModel

MODEL_ID = os.environ.get("WAN_MODEL_ID", "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers")
pipe = None

def load_pipeline():
    global pipe
    if pipe is not None:
        return pipe

    print(f"[Wan2.1 FLF2V] Carregando modelo {MODEL_ID} na GPU...")
    image_encoder = CLIPVisionModel.from_pretrained(
        MODEL_ID,
        subfolder="image_encoder",
        torch_dtype=torch.float32
    )
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float32
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=torch.bfloat16
    )
    # Otimizacao de VRAM para GPUs de 24GB-48GB (A5000, RTX 4090, L40S, A6000)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    print("[Wan2.1 FLF2V] Pipeline pronto!")
    return pipe

def decode_base64_image(b64_str):
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def calc_wan_dimensions(width, height):
    # Mantem proporcao 9:16 em multiplos de 16 seguros para Wan2.1
    w = (int(width) // 16) * 16
    h = (int(height) // 16) * 16
    return max(256, w), max(256, h)

def handler(job):
    job_input = job.get("input", {})

    b64_start = job_input.get("image_start_base64") or job_input.get("image_base64")
    b64_end = job_input.get("image_end_base64")

    if not b64_start:
        return {"error": "Parametro 'image_start_base64' (foto inicial) e obrigatorio."}

    prompt = job_input.get(
        "prompt",
        "Candid raw vertical 9:16 smartphone video, natural realistic movement, authentic indoor window lighting, smooth camera motion"
    )
    negative_prompt = job_input.get(
        "negative_prompt",
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    )

    width, height = calc_wan_dimensions(
        job_input.get("width", 480),
        job_input.get("height", 832)
    )
    # 81 frames a 16fps = ~5.06 segundos
    num_frames = int(job_input.get("num_frames", 81))
    num_inference_steps = int(job_input.get("num_inference_steps", 30))
    guidance_scale = float(job_input.get("guidance_scale", 5.0))

    try:
        model = load_pipeline()

        first_frame = decode_base64_image(b64_start).resize((width, height), Image.Resampling.LANCZOS)
        last_frame = None
        if b64_end:
            last_frame = decode_base64_image(b64_end).resize((width, height), Image.Resampling.LANCZOS)

        print(f"[Wan2.1 FLF2V] Gerando video {width}x{height} | {num_frames} frames | last_image={'SIM' if last_frame else 'NAO'}")

        pipe_kwargs = {
            "image": first_frame,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        if last_frame is not None:
            pipe_kwargs["last_image"] = last_frame

        with torch.inference_mode():
            output = model(**pipe_kwargs).frames[0]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_mp4:
            tmp_path = tmp_mp4.name

        export_to_video(output, tmp_path, fps=16)

        with open(tmp_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        os.remove(tmp_path)
        return {"video_base64": video_b64}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})

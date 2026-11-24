import runpod
import torch
import base64
import tempfile
import os
from io import BytesIO
from PIL import Image
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video

# Carrega o modelo na memória da GPU uma única vez quando o worker inicia
print("Carregando pipeline LTX-Video na GPU...")
pipe = LTXImageToVideoPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")
print("Modelo pronto!")

def base64_to_pil(b64_str):
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_data = base64.b64decode(b64_str)
    return Image.open(BytesIO(img_data)).convert("RGB")

def handler(job):
    job_input = job.get("input", {})
    
    image_b64 = job_input.get("image_base64")
    prompt = job_input.get("prompt", "Natural subtle handheld camera movement, realistic motion, high quality")
    negative_prompt = job_input.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted, talking, open mouth")
    
    # Resolução vertical 9:16 (múltiplos de 32 exigidos pelo modelo)
    width = int(job_input.get("width", 512))
    height = int(job_input.get("height", 896))
    # 121 frames a 24fps = ~5 segundos de vídeo
    num_frames = int(job_input.get("num_frames", 121))
    num_inference_steps = int(job_input.get("steps", 30))

    if not image_b64:
        return {"error": "Campo 'image_base64' é obrigatório."}

    init_image = base64_to_pil(image_b64).resize((width, height))

    with torch.inference_mode():
        video_frames = pipe(
            image=init_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=3.0,
        ).frames[0]

    # Salva MP4 temporário e converte para Base64 para devolver ao Node.js
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_path = tmp.name

    export_to_video(video_frames, temp_path, fps=24)

    with open(temp_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")

    os.remove(temp_path)

    return {
        "status": "ok",
        "video_base64": video_b64
    }

runpod.serverless.start({"handler": handler})
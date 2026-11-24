import runpod
import torch
import base64
import io
import os
import tempfile
from PIL import Image
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video

# Carrega o modelo na GPU ao iniciar o worker (fica em cache na VRAM)
print("Carregando LTX-Video na GPU...")
pipe = LTXImageToVideoPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")
print("Modelo LTX-Video pronto!")

def handler(job):
    job_input = job["input"]
    
    image_b64 = job_input.get("image_base64")
    if not image_b64:
        return {"error": "image_base64 é obrigatório"}
    
    # Remove cabeçalho data:image/...;base64, se vier do navegador
    if "," in image_b64:
        image_b64 = image_b64.split(",")[1]
        
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    prompt = job_input.get("prompt", "Natural UGC style product showcase, subtle handheld camera movement, realistic lighting, high quality 9:16 video")
    negative_prompt = job_input.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted")
    width = int(job_input.get("width", 512))
    height = int(job_input.get("height", 896))
    num_frames = int(job_input.get("num_frames", 121))
    num_inference_steps = int(job_input.get("num_inference_steps", 30))
    guidance_scale = float(job_input.get("guidance_scale", 3.0))
    
    image = image.resize((width, height))
    
    video_frames = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).frames[0]
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        
    export_to_video(video_frames, tmp_path, fps=24)
    
    with open(tmp_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    os.remove(tmp_path)
    
    return {"video_base64": video_b64}

runpod.serverless.start({"handler": handler})

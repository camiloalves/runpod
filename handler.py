import os
import base64
import tempfile
import torch
from io import BytesIO
from PIL import Image
import runpod
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
pipe = None

def load_pipeline():
    global pipe
    if pipe is None:
        print("Carregando Wan2.1-I2V na GPU...")
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        print("Wan2.1 pronto!")
    return pipe

def decode_b64_image(b64_str, width, height):
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return img.resize((width, height), Image.LANCZOS)

def handler(job):
    try:
        job_input = job.get("input", {})
        img_start_b64 = job_input.get("image_start_base64") or job_input.get("image_base64")
        img_end_b64 = job_input.get("image_end_base64")
        prompt = job_input.get("prompt", "Authentic smartphone UGC video, natural movement, realistic indoor lighting, smooth motion")
        neg_prompt = job_input.get("negative_prompt", "worst quality, distorted, deformed body, morphing face, blurry, jittery, text, watermark")
        
        width = int(job_input.get("width", 480))
        height = int(job_input.get("height", 832))
        num_frames = int(job_input.get("num_frames", 81))
        steps = int(job_input.get("num_inference_steps", 28))
        guidance = float(job_input.get("guidance_scale", 5.0))

        if not img_start_b64:
            return {"error": "image_start_base64 é obrigatório."}

        pipeline = load_pipeline()
        start_img = decode_b64_image(img_start_b64, width, height)
        
        kwargs = {
            "image": start_img,
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance
        }

        if img_end_b64:
            end_img = decode_b64_image(img_end_b64, width, height)
            kwargs["last_image"] = end_img

        with torch.inference_mode():
            output = pipeline(**kwargs).frames[0]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_path = tmp.name

        export_to_video(output, temp_path, fps=16)

        with open(temp_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        os.remove(temp_path)
        return {"video_base64": video_b64}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})

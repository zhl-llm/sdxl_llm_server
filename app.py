from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
from python_coreml_stable_diffusion.pipeline import StableDiffusionXLPipeline
from PIL import Image
import uuid
import io

# ======================
# CONFIG
# ======================
BASE_MODEL_DIR = "/Users/zhlsunshine/Projects/inference/models/stable-diffusion-xl-base-1.0"
MODEL_DIR = "/Users/zhlsunshine/Projects/inference/models/sdxl-core-ml"
COMPUTE_UNIT = "CPU_AND_NE"  # Best for M4

# ======================
# LOAD PIPELINE (ONCE)
# ======================
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_DIR,  # Directly pass the model directory as an argument
    compute_unit=COMPUTE_UNIT
)

app = FastAPI(title="SDXL CoreML M4 LLM 推理服务器")

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 30
    seed: Optional[int] = None

@app.post("/generate")
def generate(req: GenerateRequest):
    result = pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        num_inference_steps=req.steps,
        guidance_scale=7.5,
        seed=req.seed
    )

    image = result.images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")

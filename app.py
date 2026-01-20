from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from python_coreml_stable_diffusion.pipeline import StableDiffusionXLPipeline
from PIL import Image
import uuid
import os

# ======================
# CONFIG
# ======================
BASE_MODEL_DIR = "/Users/zhlsunshine/Projects/inference/models/stable-diffusion-xl-base-1.0"
MODEL_DIR = "/Users/zhlsunshine/Projects/inference/models/sdxl-core-ml"
OUTPUT_DIR = "outputs"
COMPUTE_UNIT = "CPU_AND_NE"  # Best for M4

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        seed=req.seed
    )

    image = result.images[0]

    filename = f"{uuid.uuid4().hex}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    image.save(path)

    return {"image_path": path}

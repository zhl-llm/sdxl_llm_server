from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from python_coreml_stable_diffusion.pipeline import StableDiffusionPipeline
from PIL import Image
import uuid
import os
import torch
from transformers import CLIPTextModel, CLIPTokenizer

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
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_DIR,  # Directly pass the model directory as an argument
    compute_unit=COMPUTE_UNIT
)

# Initialize the tokenizer and model for generating text embeddings
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

app = FastAPI(title="SDXL CoreML M4 LLM 推理服务器")

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 30
    seed: Optional[int] = None

@app.post("/generate")
def generate(req: GenerateRequest):
    # Generate text embeddings using CLIP tokenizer and model
    inputs = tokenizer(req.prompt, return_tensors="pt")
    text_embeds = text_model(**inputs).last_hidden_state

    # Initialize added_cond_kwargs with the required embeddings
    added_cond_kwargs = {
        "text_embeds": text_embeds
    }

    # Call the pipeline with additional parameters, including the added_cond_kwargs
    image = pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        num_inference_steps=req.steps,
        seed=req.seed,
        added_cond_kwargs=added_cond_kwargs  # Ensure it's properly populated
    )

    filename = f"{uuid.uuid4().hex}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    image.save(path)

    return {
        "image_path": path
    }
# Core ML SDXL Inference Service

brew install xz
pyenv install 3.8.10
pyenv virtualenv 3.8.10 sdxl-env
source ~/.pyenv/versions/sdxl-env/bin/activate

## Prepare the Model

### 1. Convert SDXL to Core ML (One-time)

```sh
git clone https://github.com/zhl-llm/ml-stable-diffusion.git
cd ml-stable-diffusion
pip install -r requirements.txt
```

#### Convert SDXL Base:

[Using Stable Diffusion XL](https://github.com/zhl-llm/ml-stable-diffusion?tab=readme-ov-file#-using-stable-diffusion-xl)

```sh
pip install -e .

python -m python_coreml_stable_diffusion.torch2coreml \
    --convert-unet \
    --convert-vae-decoder \
    --convert-text-encoder \
    --xl-version \
    --model-version /Users/zhlsunshine/Projects/inference/models/stable-diffusion-xl-base-1.0 \
    --refiner-version /Users/zhlsunshine/Projects/inference/models/stable-diffusion-xl-refiner-1.0 \
    --bundle-resources-for-swift-cli \
    --attention-implementation ORIGINAL \
    -o /Users/zhlsunshine/Projects/inference/models/stable-diffusion-core-ml-base-1-0
```

### 2. Run SDXL Inference (CLI or Service)

```sh
python -m python_coreml_stable_diffusion.pipeline \
  --prompt "a photo of an astronaut riding a horse on mars" \
  --compute-unit CPU_AND_GPU \
  -o /Users/zhlsunshine/Projects/sources/sdxl_llm_server/outputs \
  -i /Users/zhlsunshine/Projects/inference/models/stable-diffusion-core-ml-base-1-0/Resources \
  --model-version /Users/zhlsunshine/Projects/inference/models/stable-diffusion-xl-base-1.0
```

## Solution for Model Inference

### Architecuture

Browser (WebUI)
    ↓
FastAPI (REST)
    ↓
python_coreml_stable_diffusion
    ↓
Core ML (ANE / GPU)

sdxl_llm_server/backend.py


### Install the dependencies libs

```sh
pip install diffusers transformers python_coreml_stable_diffusion fastapi uvicorn pillow
```

### backend.py code writing

sdxl_llm_server/app.py

```sh
uvicorn app:app --host 0.0.0.0 --port 8000
```

Valid the backend service:

```sh
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a cinematic cyberpunk city at night"}'
```

### webui.py code writing

sdxl_llm_server/webui.py

```sh
python webui.py
```

```sh
pip install gradio requests
python webui.py
```

Open:
👉 http://localhost:7860

## Performance Expectations (SDXL Base 1.0)

On M4 / 32GB unified memory:

1024×1024
30 steps
~8–12 seconds
Memory usage: ~10–12 GB

## NOTE

https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
https://modelscope.cn/models/AI-ModelScope/stable-diffusion-xl-base-1.0/summary

pyenv virtualenv 3.12.12 mlx-env

source ~/.pyenv/versions/mlx-env/bin/activate

pip install --upgrade pip

pip install modelscope invisible_watermark transformers safetensors
pip install diffusers
pip install diffusers --upgrade

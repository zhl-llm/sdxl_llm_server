import gradio as gr
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000/generate"

def generate(prompt, negative, steps):
    import io
    print("➡️ sending request")
    response = requests.post(
        API_URL,
        json={
            "prompt": prompt,
            "negative_prompt": negative,
            "steps": steps
        },
        timeout=3600
    )
    print("⬅️ status:", response.status_code)
    print("⬅️ content length:", len(response.content))

    try:
        # Open image from bytes
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        print("❌ Failed to decode JSON:", e)
        return None

### 调试 generate 函数
# def generate(prompt, negative, steps):
#     print("🔥 CALLBACK CALLED 🔥")
#     return None

with gr.Blocks(title="SDXL CoreML M4 LLM 文生图推理服务器") as demo:
    gr.Markdown("## SDXL CoreML M4 LLM 文生图推理服务器")

    prompt = gr.Textbox(label="提示词")
    negative = gr.Textbox(label="负面提示词")
    steps = gr.Slider(10, 50, value=30, step=1)

    btn = gr.Button("创建")
    output = gr.Image()

    btn.click(generate, inputs=[prompt, negative, steps], outputs=output)

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    debug=True,
    share=False
)

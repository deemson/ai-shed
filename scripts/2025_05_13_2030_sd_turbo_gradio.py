import gradio as gr
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
import torch
from PIL.Image import Image


def generate_image(prompt: str) -> Image:
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to("cuda")
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    return image


with gr.Blocks() as bl:
    with gr.Row(equal_height=True):
        textbox = gr.Textbox("", show_label=False)
        button = gr.Button("Generate")

    image = gr.Image(height=700)
    button.click(generate_image, inputs=textbox, outputs=image)

bl.launch()

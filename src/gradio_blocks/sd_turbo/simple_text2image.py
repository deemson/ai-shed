from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
import torch
from PIL.Image import Image
import gradio as gr


def generate_image(
    prompt: str, negative_prompt: str, steps: float, guidance: float, width: float, height: float
) -> Image:
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=int(width),
        height=int(height),
        num_inference_steps=int(steps),
        guidance_scale=guidance,
    ).images[0]
    return image


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox("", label="Prompt", lines=2)
            negative_prompt_input = gr.Textbox("", label="Negative Prompt", lines=2)
            steps_input = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Steps")
            guidance_input = gr.Slider(minimum=0.0, maximum=4.0, value=1.0, step=0.1, label="Guidance")
            with gr.Row(equal_height=True):
                width_input = gr.Number(minimum=64, maximum=2048, value=512, step=1, label="Width")
                height_input = gr.Number(minimum=64, maximum=2048, value=512, step=1, label="Height")
            generate_input = gr.Button("Generate", variant="primary")
        with gr.Column():
            image_output = gr.Image()

    generate_input.click(
        generate_image,
        inputs=[prompt_input, negative_prompt_input, steps_input, guidance_input, width_input, height_input],
        outputs=image_output,
    )

if __name__ == "__main__":
    demo.launch()

from typing import cast
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
import torch
from PIL.Image import Image
import gradio as gr


def generate_image(
    prompt: str, negative_prompt: str, steps: float, guidance: float, width: float, height: float
) -> Image:
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    pipe.to("cuda")
    ouput = cast(
        StableDiffusionXLPipelineOutput,
        pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            num_inference_steps=int(steps),
            guidance_scale=guidance,
        ),
    )
    image = ouput.images[0]
    return image


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox("", label="Prompt -- what to do", lines=2)
            negative_prompt_input = gr.Textbox("", label="Negative Prompt -- what not to do", lines=2)
            steps_input = gr.Number(
                minimum=1,
                maximum=100,
                value=50,
                step=1,
                label="Steps [1, 100] -- more = better effort, but slower; 70 is practical cutoff",
            )
            guidance_input = gr.Number(
                minimum=1.0,
                maximum=20.0,
                value=7.0,
                step=0.1,
                label="Guidance [1.0, 20.0] -- more = closer follows prompts, less = more creative freedom; 12.0 is practical cutoff",
            )
            with gr.Row(equal_height=True):
                width_input = gr.Number(minimum=64, maximum=2048, value=512, step=1, label="Width [64, 2048]")
                height_input = gr.Number(minimum=64, maximum=2048, value=512, step=1, label="Height [64, 2048]")
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

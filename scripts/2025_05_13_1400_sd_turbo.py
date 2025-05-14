from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
import torch
import os

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

prompt = os.environ.get("PROMPT")
image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

image.save("2025_05_13_1400_sd_turbo.png")

from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
output = pipe(prompt=prompt, width=32, height=32, num_inference_steps=1)
pipe.enable_model_cpu_offload()
print(help(pipe.__call__))
print(type(pipe))
print(type(output))

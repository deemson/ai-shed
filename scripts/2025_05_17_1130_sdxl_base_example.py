from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "An astronaut riding a green horse"
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
output = pipe(prompt=prompt, width=256, height=256)
print(type(pipe))
print(type(output))

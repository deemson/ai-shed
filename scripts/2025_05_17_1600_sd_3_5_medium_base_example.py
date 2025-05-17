from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch

model_id = "stabilityai/stable-diffusion-3.5-medium"

nf4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id, subfolder="transformer", quantization_config=nf4_config, torch_dtype=torch.bfloat16
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    transformer=model_nf4,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
)
pipeline.enable_attention_slicing()

prompt = "woman"
output = pipeline(
    width=512,
    height=512,
    prompt=prompt,
    num_inference_steps=1,
    guidance_scale=4.5,
)

print(type(nf4_config))
print(type(model_nf4))
print(type(pipeline))
print(type(output))

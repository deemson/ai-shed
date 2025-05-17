from dataclasses import dataclass
from nicegui import ui, app as nicegui_app
import gradio
from gradio import mount_gradio_app

from gradio_blocks import sd_turbo
from gradio_blocks import sd_1_5
from gradio_blocks import sdxl1


@dataclass
class Variant:
    name: str
    url: str
    gradio_app: gradio.Blocks


@dataclass
class Model:
    name: str
    url: str
    hugging_face_link: str
    variants: list[Variant]


models: list[Model] = [
    Model(
        name="sd-turbo",
        url="sd-turbo",
        hugging_face_link="https://huggingface.co/stabilityai/sd-turbo",
        variants=[
            Variant(
                name="Simple Text to Image",
                url="simple-text-to-image",
                gradio_app=sd_turbo.simple_text2image,
            )
        ],
    ),
    Model(
        name="Stable Diffusion 1.5",
        url="sd-1-5",
        hugging_face_link="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5",
        variants=[
            Variant(
                name="Simple Text to Image",
                url="simple-text-to-image",
                gradio_app=sd_1_5.simple_text2image,
            )
        ],
    ),
    Model(
        name="Stable Diffusion XL",
        url="sdxl",
        hugging_face_link="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
        variants=[
            Variant(
                name="Simple Text to Image",
                url="simple-text-to-image",
                gradio_app=sdxl1.simple_text2image,
            )
        ],
    ),
]

index_md_lines = ["# AI Shed"]
for model in models:
    index_md_lines.extend(
        [
            f"## [{model.name}](/{model.url})",
            f"[Hugging Face Link]({model.hugging_face_link})",
            "",
            "**Generators:**",
            "",
        ]
    )
    for variant in model.variants:
        model_varian_url = f"/{model.url}/{variant.url}"
        index_md_lines.extend([f"[{variant.name}]({model_varian_url})"])
        mount_gradio_app(nicegui_app, variant.gradio_app, model_varian_url)
index_md = "\n".join(index_md_lines)


@ui.page("/")
def index():
    ui.markdown(index_md)


ui.run()

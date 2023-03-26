from flask import Flask, request, send_file
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from io import BytesIO

app = Flask(__name__)

# required parameters
model_id = "dreamlike-art/dreamlike-photoreal-2.0"
steps = 10

# optional parameters
use_xformers = False
use_2pass = False
use_unet = True

# setup pipes
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

if use_xformers:
    pipe.enable_xformers_memory_efficient_attention()

if use_unet:
    pipe.unet = torch.compile(pipe.unet)

if use_2pass:
    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe_img2img = pipe_img2img.to("cuda")
    if use_xformers:
        pipe_img2img.enable_xformers_memory_efficient_attention()

    if use_unet:
        pipe_img2img.unet = torch.compile(pipe_img2img.unet)


@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form.get('prompt', '')

    if not prompt:
        return "No prompt provided.", 400

    image = pipe(
        prompt,
        width=1024, #1024
        height=576, #576
        num_inference_steps=steps,
        negative_prompt=""
    ).images[0]

    # 2pass for more detail
    if use_2pass:
        image = image.resize((2048, 1152))
        image = pipe_img2img(
            prompt,
            image=image,
            strength=0.75,
            num_inference_steps=steps,
            negative_prompt=""
        ).images[0]

    image = image.resize((1920, 1080))
    img_io = BytesIO()
    image.save(img_io, format='JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run()
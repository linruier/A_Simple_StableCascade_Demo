import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
import gradio as gr

# Disable memory-efficient and flash SDP for compatibility with certain hardware
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

device = "cuda"
torch_dtype = torch.float16  # Use bfloat16 for Prior and float16 for Decoder

prior = StableCascadePriorPipeline.from_pretrained(
    "stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16).to(device)
decoder = StableCascadeDecoderPipeline.from_pretrained(
    "stabilityai/stable-cascade", torch_dtype=torch_dtype).to(device)


def generate_image(prompt, negative_prompt="", num_inference_steps=20, guidance_scale=4.0, width=1024, height=1024):
    prior_output = prior(
        prompt=prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
        num_inference_steps=num_inference_steps
    )
    decoder_output = decoder(
        image_embeddings=prior_output.image_embeddings.to(torch_dtype),
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=0.0,
        output_type="pil",
        num_inference_steps=10
    ).images

    return decoder_output[0]  # Return the first image in the list


with gr.Blocks() as demo:
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="Enter a prompt")
        negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Optional negative prompt")
        num_inference_steps = gr.Slider(minimum=10, maximum=30, step=1, value=20, label="Inference Steps")
        guidance_scale = gr.Slider(minimum=0, maximum=20, step=0.1, value=4.0, label="Guidance Scale")
        width = gr.Slider(minimum=512, maximum=2048, step=64, value=1024, label="Width")
        height = gr.Slider(minimum=512, maximum=2048, step=64, value=1024, label="Height")
        generate_button = gr.Button("Generate Image")

    image = gr.Image(label="Generated Image")

    generate_button.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, num_inference_steps, guidance_scale, width, height],
        outputs=image
    )

demo.launch()

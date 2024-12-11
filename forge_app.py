import copy
import spaces
import gradio as gr
import torch
from diffusers import DiffusionPipeline, LCMScheduler, AutoencoderKL
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download


class TimestepShiftLCMScheduler(LCMScheduler):
    def __init__(self, *args, shifted_timestep=250, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_to_config(shifted_timestep=shifted_timestep)

    def set_timesteps(self, *args, **kwargs):
        super().set_timesteps(*args, **kwargs)
        self.origin_timesteps = self.timesteps.clone()
        self.shifted_timesteps = (self.timesteps * self.config.shifted_timestep / self.config.num_train_timesteps).long()
        self.timesteps = self.shifted_timesteps

    def step(self, model_output, timestep, sample, generator=None, return_dict=True):
        if self.step_index is None:
            self._init_step_index(timestep)
        self.timesteps = self.origin_timesteps
        output = super().step(model_output, timestep, sample, generator, return_dict)
        self.timesteps = self.shifted_timesteps
        return output


base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(
    base_model_id,
    vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16),
    image_encoder=None,
    feature_extractor=None,

    torch_dtype=torch.float16,
    variant="fp16",
)

repo = "ChenDY/NitroFusion"

unet_realism = pipe.unet
unet_realism.load_state_dict(load_file(hf_hub_download(repo, "nitrosd-realism_unet.safetensors"), device="cpu"))
scheduler_realism = TimestepShiftLCMScheduler.from_pretrained(base_model_id, subfolder="scheduler", shifted_timestep=250)
scheduler_realism.config.original_inference_steps = 4

unet_vibrant = copy.deepcopy(pipe.unet)
unet_vibrant.load_state_dict(load_file(hf_hub_download(repo, "nitrosd-vibrant_unet.safetensors"), device="cpu"))
scheduler_vibrant = TimestepShiftLCMScheduler.from_pretrained(base_model_id, subfolder="scheduler", shifted_timestep=500)
scheduler_vibrant.config.original_inference_steps = 4

del pipe.unet
pipe.to(torch.float16)
pipe.enable_model_cpu_offload()
spaces.automatically_move_pipeline_components(pipe)


def process_image(model_choice, num_images, height, width, prompt, seed, inference_steps):
    global pipe

    pipe.vae.to('cpu')

    # Switch to the selected model
    if model_choice == "NitroSD-Realism":
        pipe.unet = unet_realism.to('cuda')
        pipe.scheduler = scheduler_realism
    elif model_choice == "NitroSD-Vibrant":
        pipe.unet = unet_vibrant.to('cuda')
        pipe.scheduler = scheduler_vibrant
    else:
        raise ValueError("Invalid model choice.")

    # Generate the image
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = pipe(
            prompt              = [prompt] * num_images,
            generator           = torch.manual_seed(int(seed)),
            num_inference_steps = inference_steps,
            guidance_scale      = 0.0,
            height              = int(height),
            width               = int(width),
            output_type         = "latent",
        ).images

        pipe.unet.to('cpu')
        pipe.vae.to('cuda')

        result = []
        total = len(output)
        for i in range (total):
            print (f'NitroFusion: VAE: {i+1} of {total}', end='\r', flush=True)

            latent = (output[i:i+1]) / pipe.vae.config.scaling_factor
            image = pipe.vae.decode(latent, return_dict=False)[0]
            image = pipe.image_processor.postprocess(image, output_type='pil', do_denormalize=[True])[0]

            result.append(image)
        print ('NitroFusion: VAE: done  ')

    return result

# Gradio UI
with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                    ### NitroFusion Single-Step Text-To-Image
                """)
                model_choice = gr.Dropdown(
                    label="Choose Model",
                    choices=["NitroSD-Realism", "NitroSD-Vibrant"],
                    value="NitroSD-Realism",
                    interactive=True,
                )
                prompt = gr.Text(label="Prompt", value="a photo of a cat", interactive=True)
                with gr.Row():
                    width = gr.Slider(
                        label="Image Width", minimum=768, maximum=1024, step=8, value=1024, interactive=True
                    )
                    height = gr.Slider(
                        label="Image Height", minimum=768, maximum=1024, step=8, value=1024, interactive=True
                    )
                with gr.Row():
                    inference_steps = gr.Slider(
                        label="Inference Steps", minimum=1, maximum=4, step=1, value=1, interactive=True,
                    )
                    num_images = gr.Slider(
                        label="Number of Images", minimum=1, maximum=4, step=1, value=1, interactive=True
                    )
                seed = gr.Number(label="Seed", value=2024, interactive=True)
                btn = gr.Button(value="Generate Image")
            with gr.Column():
                output = gr.Gallery(height=1024)

            btn.click(
                process_image,
                inputs=[model_choice, num_images, height, width, prompt, seed, inference_steps],
                outputs=[output],
            )

if __name__ == "__main__":
    demo.queue.launch()
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, AnimateDiffPipeline, MotionAdapter, DDIMScheduler

def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

class TADPoLe(nn.Module):
    def __init__(self, device, text_cond='', sd_version='2.1', fp16=False):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading tadpole...')

        if self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)
        pipe.to(device)

        self.eval()
        for param in self.parameters():
            param.requires_grad = False

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        conditional_text = self.get_text_embeds([text_cond])
        unconditional_text = self.get_text_embeds([""])
        self.c_in = torch.cat([unconditional_text, conditional_text])
        self.c_in_condition_only = conditional_text

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.precision_t)

        del pipe
        del self.tokenizer
        del self.text_encoder

        print(f'[INFO] loaded tadpole!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):

        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings

    @torch.no_grad()
    def encode_imgs(self, imgs):

        imgs = F.interpolate(imgs, (512, 512), mode='bilinear', align_corners=False)

        normalized = 2.0 * imgs - 1.0

        latents = self.vae.encode(normalized).latent_dist.sample()
        latents *= self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def get_reward(self, latent, t, align_scale=2000, recon_scale=200, noise=None):

        if noise is None:
            noise = torch.randn_like(latent)

        # make unconditional and conditional predictions
        latents_noisy = self.scheduler.add_noise(latent, noise, t)
        latents_repeat = latents_noisy.repeat_interleave(repeats=2, dim=0)

        pred_uncond, pred_text = self.unet(
            latents_repeat,
            t,
            encoder_hidden_states=self.c_in,
        ).sample.chunk(2)

        # make alignment and reconstruction term predictions
        alignment_pred = ((pred_text - pred_uncond)**2).mean([1,2,3])
        text_natural_pred = ((pred_text - noise)**2).mean([1,2,3])
        uncond_natural_pred = ((pred_uncond - noise)**2).mean([1,2,3])
        recon_pred = uncond_natural_pred - text_natural_pred

        reward = symlog(align_scale*alignment_pred) + symlog(recon_scale*recon_pred)

        return reward


class VideoTADPoLe(nn.Module):
    def __init__(self, device, text_cond='', neg_text_cond='bad quality, worse quality', fp16=False):
        super().__init__()

        self.device = device

        print(f'[INFO] loading video tadpole...')

        self.precision_t = torch.float16 if fp16 else torch.float32

        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=self.precision_t)
        # load SD 1.5 based finetuned model
        model_key = "sd-legacy/stable-diffusion-v1-5"
        pipe = AnimateDiffPipeline.from_pretrained(model_key, motion_adapter=adapter, torch_dtype=self.precision_t)
        pipe.scheduler = DDIMScheduler.from_pretrained(
            model_key,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )
        # enable memory savings
        pipe.enable_vae_slicing()
        pipe.to(device)

        self.vae = pipe.vae
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler

        prompt_embeds, neg_prompt_embeds = pipe.encode_prompt(
            text_cond,
            device,
            1,
            True,
            neg_text_cond,
        )
        self.c_in = torch.cat([neg_prompt_embeds, prompt_embeds])

        print(f'[INFO] loaded video tadpole!')

    @torch.no_grad()
    def encode_imgs(self, imgs):

        imgs = F.interpolate(imgs, (512, 512), mode='bilinear', align_corners=False)

        normalized = 2.0 * imgs/255. - 1.0

        latents = self.vae.encode(normalized).latent_dist.sample()
        latents *= self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def get_reward(self, latents, t, align_scale=2000, recon_scale=200, noise=None):

        if noise is None:
            noise = torch.randn_like(latents)

        # make unconditional and conditional predictions
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        batch_input = latents_noisy.permute(1,0,2,3).unsqueeze(0)
        latents_repeat = torch.cat([batch_input] * 2)

        pred_uncond, pred_text = self.unet(
            latents_repeat,
            t,
            encoder_hidden_states=self.c_in,
        ).sample.chunk(2)

        source_noise = noise.permute(1,0,2,3).unsqueeze(0)

        # make alignment and reconstruction term predictions
        alignment_pred = ((pred_text - pred_uncond)**2).mean([0,1,3,4])
        text_natural_pred = ((pred_text - source_noise)**2).mean([0,1,3,4])
        uncond_natural_pred = ((pred_uncond - source_noise)**2).mean([0,1,3,4])
        recon_pred = uncond_natural_pred - text_natural_pred

        rewards = symlog(align_scale*alignment_pred) + symlog(recon_scale*recon_pred)

        return rewards

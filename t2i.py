#!/usr/bin/env python

# Stable Diffusion txt2img for Hugging Face diffusers pipelines
# Based on the suggested sample script

from datetime import datetime
import sys
from diffusers import StableDiffusionPipeline

if len(sys.argv) != 2:
    raise ValueError('Usage: ' + sys.argv[0] + ' "<prompt>"')

# Use date time as output filename
dt = datetime.now()
filename = dt.strftime('%Y-%m-%d_%H%M%S') + '.png'

# Configure generation model and device
model_id = "CompVis/stable-diffusion-v1-4"
device = "mps"
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

##
## Image generation options
##
prompt = sys.argv[1]

# Either height or width must be 512; don't change both at once
height = 512
width = 512

# Higher number yields higher quality, but less diversity
num_inference_steps = 50

# How strictly to follow prompt. Values between 7 and 8.5 are best
guidance_scale = 7.5

image = pipe(prompt, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)["sample"][0]
image.save(filename)

flog = open("log.txt", "a+")
flog.write("\"{0}\",\"{1}\",\"{2}\",\"{3}\"\n".format(filename, prompt, num_inference_steps, guidance_scale))
flog.close()

# SD - Landscape

## sd-landscape.bat
Batch script launches Stable Diffusion, configures it via the API, and generates a couple desktop backgrounds through a multi-step process (generating at a lower resolution, then upscaling to roughly 1920 by 1080). Outputs them in my backgrounds directory.

## Config
Prompts and config options are all constants in sd-landscape.py

## Goals
Main idea is to have the original txt2img prompt be random or procedurally generated, or even suggested by GPT-3 API based on some info relevant to today (maybe my local weather or something). So I can have a fun on-topic background every morning.

## References/Sources
### Stable Diffusion webui AMD fork
https://github.com/lshqqytiger/stable-diffusion-webui-directml
### The SD models and embeddings I tried are taken from
https://civitai.com/

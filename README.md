# SD - Landscape

## sd-landscape.bat
Batch script launches Stable Diffusion locally, configures it via the API, does some extra work to come up with a text prompt for txt2img generation, and finally generates a couple desktop backgrounds through a multi-step process (generating at a lower resolution, then upscaling to roughly 1920 by 1080). Outputs them in my backgrounds directory.

## Config
Stable Diffusion prompts and config options are set in config.toml (not tracked, contains API key and location to use for weather API). Defaults in sd-landscape.py.

## Goals
The dream was to generate a txt2img prompt with some random or contextual elements, or even suggested by GPT-3 API, based on some info relevant to today. For now, I started with my local weather. So, I can have a fun on-topic background every morning. If it's raining, my background will be raining.

Also want to build this out into a tidy app, so I can run it in the background daily.

## References/Sources
### Stable Diffusion webui AMD fork
https://github.com/lshqqytiger/stable-diffusion-webui-directml
### The SD models and embeddings I tried are taken from
https://civitai.com/

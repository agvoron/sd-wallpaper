import base64
import glob
import io
import os
import requests
import subprocess
import time

from datetime import datetime as dt
from PIL import Image, PngImagePlugin
from requests import ConnectionError, Timeout, HTTPError

APP_NAME = "sd-landscape"
SD_URL = "http://127.0.0.1:7860"
ARCHIVE_PATH = "output/"
DEPLOY_PATH = "backgrounds/"
LOG_PATH = "logs/"

TXT2IMG_PROMPT = "(4k absurdres best quality landscape photo), (night), mystical ominous eerie gloomy dark fantasy (forest), draped with vines"
SD_UPSCALE_PROMPT = "cropped 4k absurdres best quality landscape, forest"
NEG_PROMPT = "easynegative (worst quality:1.5), (low quality:1.5), lowres, pixelated, blurred, cropped, jpeg artifacts, text, artist name, signature, logo, watermark"
SAMPLER = "DPM++ 2M Karras"

NUM_IMGS_TO_GENERATE = 6

MODELS = [
    ["aZovyaRPGArtistTools_v2.safetensors [da5224a242]", "vae-ft-ema-560000-ema-pruned.safetensors"],
    ["AOM3.safetensors [eb4099ba9c]", "kl-f8-anime2.ckpt"]
]


def retry_post_request(url, json):
    """
    Could use HTTPAdapter and Retry API from requests instead, but they don't have the options I want.
    I want to retry if:
    - the server hasn't started up yet ("Connection actively refused") (ConnectionError),
    - when I get a NaN check error (image gen fails) (HTTPError with response code 500),
    - or when the server hangs (Timeout)
    """
    print(f"Sending POST request to: {url}")
    response = None
    retries = 3
    while retries > 0:
        try:
            response = requests.post(url=url, json=json) # timeout=60 probably needs to be much much more; TODO test some timings
            response.raise_for_status()
            break
        except ConnectionError as e:
            retries -= 1
            if retries <= 0:
                print(f"ConnectionError raised, ran out of retries. {APP_NAME} aborting...")
                raise
            print(f"ConnectionError raised when posting request. Retrying... ({retries} tries left)")
            time.sleep(5)
        except HTTPError as e:
            retries -= 1
            if retries <= 0:
                print(f"HTTPError raised, ran out of retries. {APP_NAME} aborting...")
                raise
            print(f"HTTPError raised when posting request. Retrying... ({retries} tries left)")
            time.sleep(5)
        # except Timeout as e:
        #     print(f"Request timed out. The server is probably hanging. {APP_NAME} aborting...")
        #     # TODO when the server hangs, can't just retry.
        #     # Try sending an Interrupt request? Or a signal to the subprocess? Or just restart?
        #     raise

    with open(f"{LOG_PATH}last_response.log", "w") as f:
        f.write(str(response))
    
    return response


def save_image_with_png_info(img_json, save_paths):
    """
    Query PNG info for the generated image before saving it.
    Here and everywhere else, assume img_json contains only one image.
    Reasons:
    - I need to do separate requests per image because any individual image can error out from the NaN check,
    - batches come through weird in the API and I don't like it,
    - and my GPU can't handle parallel batches anyway, so might as well run in series.
    """
    print(f"Saving image to {len(save_paths)} locations.")
    img = img_json["images"][0]
    image = Image.open(io.BytesIO(base64.b64decode(img.split(",",1)[0])))

    print("Sending png info request.")
    image_payload = {
        "image": "data:image/png;base64," + img
    }
    info_response = retry_post_request(url=f"{SD_URL}/sdapi/v1/png-info", json=image_payload)

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", info_response.json().get("info"))

    now = dt.now()
    datetime_str = now.strftime("%Y%m%d%H%M%S") # e.g. 20230101182732
    for path in save_paths:
        img_filename = f"{path}{APP_NAME}_{datetime_str}.png"
        print(f"Saving to {img_filename}.")
        image.save(img_filename, pnginfo=pnginfo)


def main():
    print(f"Starting main {APP_NAME} script.")
    # TODO add some logging output
    # TODO set the checkpoint and VAE from a random list of choices
    # TODO generate a random prompt based on some factors
    # ideas: today's weather, news headlines, random keywords, text from language model

    # set payloads with options for each request
    options_payload = {
        "sd_model_checkpoint": MODELS[1][0],
        "sd_vae": MODELS[1][1]
    }

    txt2img_payload = {
        "prompt": TXT2IMG_PROMPT,
        "negative_prompt": NEG_PROMPT,
        "seed": -1,
        "sampler_name": SAMPLER,
        "batch_size": 1,
        "steps": 16,
        "cfg_scale": 9,
        "width": 768,
        "height": 432,

        "enable_hr": True,
        "denoising_strength": 0.6,
        "firstphase_width": 768,
        "firstphase_height": 432,
        "hr_scale": 1.5, # yields 1152x648 (16:9)
        "hr_upscaler": "Latent (bicubic antialiased)",
        "hr_second_pass_steps": 28,

        "save_images": True,
        # "override_settings": {
        #     "sd_model_checkpoint": MODELS[0][0],
        #     "sd_vae": MODELS[0][1]
        # }
    }

    sd_upscale_payload = {
        "prompt": SD_UPSCALE_PROMPT,
        "negative_prompt": NEG_PROMPT,
        "seed": -1,
        "sampler_name": SAMPLER,
        "batch_size": 1,
        "steps": 22,
        "cfg_scale": 10,
        "width": 1080, # 16x9 tiles; with 192 overlap, results in a 2x3 tiling. Tested with 1152x648 and 128 for 2x2, but I'm getting OOM for some reason (despite doing the same res above)
        "height": 608,

        "denoising_strength": 0.36,

        "save_images": True,
        # "override_settings": {
        #     "sd_model_checkpoint": MODELS[0][0],
        #     "sd_vae": MODELS[0][1]
        # },
        "script_name": "SD upscale",
        "script_args": [
            None, # first arg is ignored in script
            192, # SD upscale tile overlap
            5, # "4x_foolhardy_Remacri", # SD upscale index of upscaler (in list) TODO maybe do a lookup to ensure I grab the right one...
            1.7 # SD upscale scale factor; yields 1958x1101 (16:9)
        ]
    }

    print("Configuring SD options.")
    retry_post_request(url=f"{SD_URL}/sdapi/v1/options", json=options_payload)

    imgs_for_upscale = []
    for _ in range(NUM_IMGS_TO_GENERATE):
        print("Sending txt2img request.")
        response = retry_post_request(url=f"{SD_URL}/sdapi/v1/txt2img", json=txt2img_payload)
        img_json = response.json()
        save_image_with_png_info(img_json, [ARCHIVE_PATH])
        img = img_json["images"][0]
        imgs_for_upscale.append("data:image/png;base64," + img)

    # delete from output path TODO only do this if new images are ready
    print(f"Deleting the previous batch of images from {DEPLOY_PATH}.")
    old_images = glob.glob(f"{DEPLOY_PATH}{APP_NAME}_*.png")
    for old in old_images:
        os.remove(old)

    for img in imgs_for_upscale:
        print("Sending img2img SD upscale request.")
        # put an image in the payload, again, one at a time
        sd_upscale_payload.update({"init_images": [img]}) 

        sd_upscale_response = retry_post_request(url=f"{SD_URL}/sdapi/v1/img2img", json=sd_upscale_payload)
        sd_upscale_json = sd_upscale_response.json()
        save_image_with_png_info(sd_upscale_json, [ARCHIVE_PATH, DEPLOY_PATH])


if __name__ == "__main__":
    # TODO schedule to run on boot

    print("Launching Stable Diffusion server...")
    launch_server_bat = f"{os.path.abspath(os.path.dirname(__file__))}\\launch-server.bat"
    # creationflags is a bit field. Use | (bitwise or) to set multiple flags if needed
    subprocess.Popen(launch_server_bat, creationflags=subprocess.CREATE_NEW_CONSOLE)
    time.sleep(30) # wait for SD server to start up

    main()
    # TODO close SD server subprocess
    print(f"{APP_NAME} is done.")

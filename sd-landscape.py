import base64
import glob
import io
import os
import random
import requests
import subprocess
import time
import tomllib

from datetime import datetime as dt
from PIL import Image, PngImagePlugin
from requests import ConnectionError, Timeout, HTTPError, RequestException

APP_NAME = "sd-landscape"

# global defaults
DEFAULT_SD_URL = "http://127.0.0.1:7860"
DEFAULT_ARCHIVE_PATH = "output/"
DEFAULT_DEPLOY_PATH = "backgrounds/"
DEFAULT_LOG_PATH = "logs/"

DEFAULT_TXT2IMG_PROMPT = "4k absurdres best quality fantasy paradise landscape, exotic jungle rainforest"
DEFAULT_SD_UPSCALE_PROMPT = "4k 100mm absurdres best quality photo, extremely detailed foliage, leaves"
DEFAULT_NEG_PROMPT = "easynegative, bad-artist-neg, (worst quality:1.5), (low quality:1.5), lowres, pixelated, blurred, cropped, jpeg artifacts, text, artist name, signature, logo, watermark"
DEFAULT_SAMPLER = "DPM++ 2M Karras"

DEFAULT_NUM_IMGS_TO_GENERATE = 2

DEFAULT_MODELS = [
    ["aZovyaRPGArtistTools_v2.safetensors [da5224a242]", "vae-ft-ema-560000-ema-pruned.safetensors"],
    ["dreamshaper_4BakedVaeFp16.safetensors [db2c51c333]", "vae-ft-ema-560000-ema-pruned.safetensors"],
    ["landscapes-cheeseDaddys_41.safetensors [7ed3c68f22]", "vae-ft-ema-560000-ema-pruned.safetensors"]
]


def configure(config):
    """
    Configure global variables from dict.
    """
    print("Configuring...")
    global config_sd_url
    global config_archive_path
    global config_deploy_path
    global config_log_path

    global config_txt2img_prompt
    global config_sd_upscale_prompt
    global config_neg_prompt
    global config_sampler

    global config_num_imgs_to_generate

    global config_models

    global weather_api_key
    global weather_query

    config_sd_url = config.get("sd_url", DEFAULT_SD_URL)
    config_archive_path = config.get("archive_path", DEFAULT_ARCHIVE_PATH)
    config_deploy_path = config.get("deploy_path", DEFAULT_DEPLOY_PATH)
    config_log_path = config.get("log_path", DEFAULT_LOG_PATH)

    config_txt2img_prompt = config.get("txt2img_prompt", DEFAULT_TXT2IMG_PROMPT)
    config_sd_upscale_prompt = config.get("sd_upscale_prompt", DEFAULT_SD_UPSCALE_PROMPT)
    config_neg_prompt = config.get("neg_prompt", DEFAULT_NEG_PROMPT)
    config_sampler = config.get("sampler", DEFAULT_SAMPLER)

    config_num_imgs_to_generate = config.get("num_imgs_to_generate", DEFAULT_NUM_IMGS_TO_GENERATE)

    config_models = config.get("models", DEFAULT_MODELS)

    weather_api_key = config.get("weather_api_key", None)
    weather_query = config.get("weather_query", None)


def start_stable_diffusion():
    print("Starting Stable Diffusion server...")
    launch_server_bat = f"{os.path.abspath(os.path.dirname(__file__))}\\launch-server.bat"
    # creationflags is a bit field. Use | (bitwise or) to set multiple flags if needed
    # the subprocess.CREATE_NEW_PROCESS_GROUP flag normally allows for os.kill() to be used
    # however, the flags are incompatible, so I can't kill the subproces from here
    # so, can't restart the server programatically
    subprocess.Popen(launch_server_bat, creationflags=subprocess.CREATE_NEW_CONSOLE) 
    time.sleep(30) # wait for SD server to start up


def generate_txt2img_prompt(txt2img_prompt):
    """
    Generate and append some extra text to the configured prompt.
    For now, request weather from API to provide a prompt theme
    TODO
    """
    # Set some defaults if API fails
    daytime_string = random.choice(["day", "night"])
    weather_string = "clear"
    try:
        print("Querying weather...")
        response = requests.get(f"https://api.weatherapi.com/v1/current.json?key={weather_api_key}&q={weather_query}")
        response.raise_for_status()

        weather_data = response.json()
        is_day = int(weather_data["current"]["is_day"])
        if is_day == 0:
            daytime_string = "night"
        else:
            daytime_string = "day"
        weather_string = weather_data["current"]["condition"]["text"].lower()

    # For any exception, couldn't get weather, just give up and use the default
    except RequestException as err:
        print(f"Failed to query weather data. Error: {err}")
    except (KeyError, IndexError) as err:
        print(f"Error parsing weather API response: {err}")

    # TODO randomly choose a scenery string from a configured preset list
    # TODO add other features to the prompt
    # ideas: news headlines, random keywords, text from language model

    # Edit the prompt string
    append_string = f" {weather_string} {daytime_string}" # e.g. "clear night"; "party cloudy day";
    print(f"Appending to prompt:{append_string}")
    prompt = f"{txt2img_prompt}{append_string}"

    return prompt


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
    max_tries = 3 # Including the first one
    tries = max_tries
    while tries > 0:
        tries -= 1
        retry_delay = pow(5, (max_tries - tries))
        try:
            response = requests.post(url=url, json=json, timeout=1200) # 20 minutes
            response.raise_for_status()
            break
        except ConnectionError as e:
            if tries <= 0:
                print(f"ConnectionError raised, ran out of retries.")
                raise
            print(f"ConnectionError raised when posting request. Sleeping for {retry_delay} seconds and retrying... ({tries} tries left)")
            time.sleep(retry_delay)
        except HTTPError as e:
            if tries <= 0:
                print(f"HTTPError raised, ran out of retries.")
                raise
            print(f"HTTPError raised when posting request. Sleeping for {retry_delay} seconds and retrying... ({tries} tries left)")
            time.sleep(retry_delay)
        except Timeout as e:
            print(f"Request timed out. The server is probably hanging.")
            raise
    
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
    info_response = retry_post_request(url=f"{config_sd_url}/sdapi/v1/png-info", json=image_payload)

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
    start_stable_diffusion()

    try:
        with open("config.toml", "rb") as f:
            configure(tomllib.load(f))
    except:
        configure({})

    # TODO add some logging output

    # set payloads with options for each request; use a random model
    random_model = random.choice(config_models)
    options_payload = {
        "sd_model_checkpoint": random_model[0],
        "sd_vae": random_model[1]
    }

    txt2img_payload = {
        "prompt": generate_txt2img_prompt(config_txt2img_prompt), # Edit the prompt string with some extra data based on today's context
        "negative_prompt": config_neg_prompt,
        "seed": -1,
        "sampler_name": config_sampler,
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
        "prompt": config_sd_upscale_prompt,
        "negative_prompt": config_neg_prompt,
        "seed": -1,
        "sampler_name": config_sampler,
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

    print(f"Configuring SD options: {random_model[0]}")
    retry_post_request(url=f"{config_sd_url}/sdapi/v1/options", json=options_payload)

    imgs_for_upscale = []
    for _ in range(config_num_imgs_to_generate):
        print("Sending txt2img request.")
        response = retry_post_request(url=f"{config_sd_url}/sdapi/v1/txt2img", json=txt2img_payload)
        img_json = response.json()
        save_image_with_png_info(img_json, [config_archive_path])
        img = img_json["images"][0]
        imgs_for_upscale.append("data:image/png;base64," + img)

    # delete from output path TODO only do this if new images are ready
    print(f"Deleting the previous batch of images from {config_deploy_path}.")
    old_images = glob.glob(f"{config_deploy_path}{APP_NAME}_*.png")
    for old in old_images:
        os.remove(old)

    for img in imgs_for_upscale:
        print("Sending img2img SD upscale request.")
        # put an image in the payload, again, one at a time
        sd_upscale_payload.update({"init_images": [img]}) 

        sd_upscale_response = retry_post_request(url=f"{config_sd_url}/sdapi/v1/img2img", json=sd_upscale_payload)
        sd_upscale_json = sd_upscale_response.json()
        save_image_with_png_info(sd_upscale_json, [config_archive_path, config_deploy_path])
    
    print(f"{APP_NAME} finished succesfully.")


if __name__ == "__main__":
    main()

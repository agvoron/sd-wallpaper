@ECHO OFF

REM Hardcoded SD webui install location - DirectML fork
CD %USERPROFILE%\Desktop\StableDiffusionIshqqyFork\stable-diffusion-webui-directml

SET PYTHON=
SET GIT=
SET VENV_DIR=
SET COMMANDLINE_ARGS= --api --medvram --no-half --no-half-vae --precision full --opt-sub-quad-attention --opt-split-attention-v1
REM Changes from CLI args used for webui:
REM OMIT --disable-nan-check --autolaunch 
REM INCLUDE --api

CALL webui.bat

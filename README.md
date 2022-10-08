# prompt-morph
Generate morph sequences with Stable Diffusion. Interpolate between two or more prompts and create an image at each step.

# Installation
1. Copy `prompt_morph.py` into the `scripts` folder in [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
1. Add `moviepy==1.0.3` to `requirements_versions.txt` or install it manually

Enter at least two prompts in the text area in the script interface (one per line). If you use a negative prompt, it will apply to the whole sequence.
The script creates a `morphs` directory in your output directory and saves each sequence in its own folder here.  

# How does it work?
An explanation of multi-cond guidance from Birch-san can be found [here](https://www.reddit.com/r/StableDiffusion/comments/xr7wwf/sequential_token_weighting_invented_by/iqdm5ya/) but in summary, multiple prompts are used to guide the sampling process. Each prompt has a weight, which allows us to animate a transition from A to B by going from 100% A and 0% B to 0% A and 100% B.

# Example
[morbin_time.webm](https://user-images.githubusercontent.com/114889020/193788624-872bc76c-d045-458f-8e9c-8a13815017e8.webm)

# Credits
- Old multi-cond guidance code borrowed from [Birch-san](https://github.com/Birch-san/stable-diffusion/blob/birch-mps-waifu/scripts/txt2img_fork.py)

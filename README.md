# prompt-morph
Generate morph sequences with Stable Diffusion using multi-cond guidance.

# Usage
This is a script for AUTOMATIC1111/stable-diffusion-webui. Install by copying it into the `scripts` directory.  
The script creates a `morphs` directory in your output directory and saves sequences in their own folder here.

# How does it work?
An explanation of multi-cond guidance from Birch-san can be found [here](https://www.reddit.com/r/StableDiffusion/comments/xr7wwf/sequential_token_weighting_invented_by/iqdm5ya/) but in summary, multiple prompts are used to guide the sampling process. Each prompt has a weight, which allows us to animate a transition from A to B by going from 100% A and 0% B to 0% A and 100% B.

# Example
[morbin_time.webm](https://user-images.githubusercontent.com/114889020/193682082-0a6665d0-8a91-445a-87d9-370bcd81d407.webm)

# Credits
- Multi-cond guidance code borrowed from [Birch-san](https://github.com/Birch-san/stable-diffusion/blob/birch-mps-waifu/scripts/txt2img_fork.py)

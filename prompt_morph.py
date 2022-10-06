import math
import os

import gradio as gr
import torch

from modules import images, processing, prompt_parser, scripts, shared
from modules.processing import Processed, process_images
from modules.shared import cmd_opts, opts, state


"""
Interpolate from one prompt to another to create a morph sequence.
"""
class Script(scripts.Script):
    def title(self):
        return "Prompt morph"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        target_prompt = gr.Textbox(label="Target prompt")
        n_images = gr.Slider(minimum=2, maximum=256, value=9, step=1, label="Number of Images")

        return [target_prompt, n_images]

    def n_evenly_spaced(self, a, n):
        k, m = divmod(len(a), n)
        res = [a[i*k+min(i, m)] for i in range(n)]
        # ensure last image is included
        res[-1] = a[-1]
        return res

    # build prompt with weights scaled by t in [0.0, 1.0]
    def prompt_at_t(self, weight_indexes, prompt_list, t):
        return " AND ".join(
            [
                ":".join((prompt_list[index], str(weight * t)))
                for index, weight in weight_indexes
            ]
        )

    def run(self, p, target_prompt, n_images):
        state.job_count = n_images

        # override batch count and size
        p.batch_size = 1
        p.n_iter = 1

        # fix seed because we'll be reusing it
        processing.fix_seed(p)

        # write images to a numbered folder in morphs
        morph_path = os.path.join(p.outpath_samples, "morphs")
        os.makedirs(morph_path, exist_ok=True)
        morph_number = images.get_next_sequence_number(morph_path, "")
        morph_path = os.path.join(morph_path, f"{morph_number:05}")
        p.outpath_samples = morph_path

        # parsed prompts
        start_prompt = p.prompt
        res_indexes, prompt_flat_list, prompt_indexes = prompt_parser.get_multicond_prompt_list([p.prompt, target_prompt])
        prompt_weights, target_weights = res_indexes

        # TODO: interpolate between multiple prompts like keyframes
            # TODO: generate video directly with moviepy
        # TODO: integrate seed travel so end prompt can use different seed
        # one image for each interpolation step (including start and end)
        all_images = []
        for i in range(n_images):
            state.job = f"{i+1} out of {n_images}"

            # update prompt weights
            t = i / (n_images - 1)
            scaled_prompt = self.prompt_at_t(prompt_weights, prompt_flat_list, 1.0 - t)
            scaled_target = self.prompt_at_t(target_weights, prompt_flat_list, t)
            p.prompt = f'{scaled_prompt} AND {scaled_target}'

            processed = process_images(p)
            if not state.interrupted:
                all_images.append(processed.images[0])

        # limit max images shown to avoid lagging out the interface
        if len(all_images) > 25:
            all_images = self.n_evenly_spaced(all_images, 25)
        if opts.return_grid:
            grid = images.image_grid(all_images)
            all_images  = [grid] + all_images
            if opts.grid_save:
                prompt = f"interpolate: {start_prompt} to {target_prompt}"
                images.save_image(grid, p.outpath_grids, "grid", processed.all_seeds[0], prompt, opts.grid_format, info=processed.infotext(p, 0), short_filename=not opts.grid_extended_filename, p=p, grid=True)
        processed.images = all_images

        return processed

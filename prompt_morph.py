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
        prompts = gr.TextArea(label="Prompt list", placeholder="Enter one prompt per line. Blanks lines will be ignored.")
        n_images = gr.Slider(minimum=2, maximum=256, value=9, step=1, label="Number of images per transition")

        return [prompts, n_images]

    def n_evenly_spaced(self, a, n):
        res = [a[math.ceil(i/(n-1) * (len(a)-1))] for i in range(n)]
        return res

    # build prompt with weights scaled by t in [0.0, 1.0]
    def prompt_at_t(self, weight_indexes, prompt_list, t):
        return " AND ".join(
            [
                ":".join((prompt_list[index], str(weight * t)))
                for index, weight in weight_indexes
            ]
        )

    def run(self, p, prompts, n_images):
        prompts = [line.strip() for line in prompts.splitlines()]
        prompts = [line for line in prompts if line != ""]

        if len(prompts) < 2:
            print("prompt_morph: at least 2 prompts required")
            return Processed(p, [], p.seed)

        state.job_count = 1 + (n_images - 1) * len(prompts)

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

        all_images = []
        for n in range(1, len(prompts)):
            # parsed prompts
            start_prompt = prompts[n-1]
            target_prompt = prompts[n]
            res_indexes, prompt_flat_list, prompt_indexes = prompt_parser.get_multicond_prompt_list([start_prompt, target_prompt])
            prompt_weights, target_weights = res_indexes

            # TODO: generate video directly with moviepy
            # TODO: integrate seed travel so end prompt can use different seed
            # one image for each interpolation step (including start and end)
            for i in range(n_images):
                # first image is same as last of previous morph
                if i == 0 and n > 1:
                    continue
                state.job = f"Morph {n}/{len(prompts)-1}, image {i+1}/{n_images}"

                # update prompt weights
                t = i / (n_images - 1)
                scaled_prompt = self.prompt_at_t(prompt_weights, prompt_flat_list, 1.0 - t)
                scaled_target = self.prompt_at_t(target_weights, prompt_flat_list, t)
                p.prompt = f'{scaled_prompt} AND {scaled_target}'

                processed = process_images(p)
                if not state.interrupted:
                    all_images.append(processed.images[0])

        prompt = f"interpolate: {' | '.join([prompt for prompt in prompts])}"
        processed.all_prompts = [prompt]
        processed.prompt = prompt
        processed.info = processed.infotext(p, 0)

        # limit max images shown to avoid lagging out the interface
        if len(all_images) > 25:
            all_images = self.n_evenly_spaced(all_images, 25)
        processed.images = all_images
        if opts.return_grid:
            grid = images.image_grid(all_images)
            processed.images  = [grid] + all_images
            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", processed.all_seeds[0], processed.prompt, opts.grid_format, info=processed.infotext(p, 0), short_filename=not opts.grid_extended_filename, p=p, grid=True)

        return processed

import math
import os

import gradio as gr
import torch

from modules import images, processing, prompt_parser, scripts, shared
from modules.processing import Processed, process_images
from modules.shared import cmd_opts, opts, state


def n_evenly_spaced(a, n):
    res = [a[math.ceil(i/(n-1) * (len(a)-1))] for i in range(n)]
    return res

# build prompt with weights scaled by t in [0.0, 1.0]
def prompt_at_t(weight_indexes, prompt_list, t):
    return " AND ".join(
        [
            ":".join((prompt_list[index], str(weight * t)))
            for index, weight in weight_indexes
        ]
    )


"""
Interpolate between two (or more) prompts and create an image at each step.
"""
class Script(scripts.Script):
    def title(self):
        return "Prompt morph"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        prompts = gr.TextArea(label="Prompt list", placeholder="Enter one prompt per line. Blanks lines will be ignored.")
        n_images = gr.Slider(minimum=2, maximum=256, value=25, step=1, label="Number of images per transition")
        save_video = gr.Checkbox(label='Save results as video', value=True)
        video_fps = gr.Number(label='Frames per second', value=5)

        return [prompts, n_images, save_video, video_fps]

    def run(self, p, prompts, n_images, save_video, video_fps):
        prompts = [line.strip() for line in prompts.splitlines()]
        prompts = [line for line in prompts if line != ""]

        if len(prompts) < 2:
            print("prompt_morph: at least 2 prompts required")
            return Processed(p, [], p.seed)

        state.job_count = 1 + (n_images - 1) * (len(prompts) - 1)

        # override batch count and size
        p.batch_size = 1
        p.n_iter = 1

        # fix seed because we'll be reusing it
        processing.fix_seed(p)

        if save_video:
            import numpy as np
            try:
                import moviepy.video.io.ImageSequenceClip as ImageSequenceClip
            except ImportError:
                print(f"moviepy python module not installed. Will not be able to generate video.")
                return Processed(p, [], p.seed)

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

            # TODO: integrate seed travel so end prompt can use different seed
            # one image for each interpolation step (including start and end)
            for i in range(n_images):
                # first image is same as last of previous morph
                if i == 0 and n > 1:
                    continue
                state.job = f"Morph {n}/{len(prompts)-1}, image {i+1}/{n_images}"

                # update prompt weights
                t = i / (n_images - 1)
                scaled_prompt = prompt_at_t(prompt_weights, prompt_flat_list, 1.0 - t)
                scaled_target = prompt_at_t(target_weights, prompt_flat_list, t)
                p.prompt = f'{scaled_prompt} AND {scaled_target}'

                processed = process_images(p)
                if not state.interrupted:
                    all_images.append(processed.images[0])

        if save_video:
            clip = ImageSequenceClip.ImageSequenceClip([np.asarray(t) for t in all_images], fps=video_fps)
            clip.write_videofile(os.path.join(morph_path, f"morph-{morph_number:05}.webm"), codec='libvpx-vp9', ffmpeg_params=['-pix_fmt', 'yuv420p', '-crf', '32', '-b:v', '0'], logger=None)

        prompt = f"interpolate: {' | '.join([prompt for prompt in prompts])}"
        # TODO: instantiate new Processed instead of overwriting one from the loop
        processed.all_prompts = [prompt]
        processed.prompt = prompt
        processed.info = processed.infotext(p, 0)

        processed.images = all_images
        # limit max images shown to avoid lagging out the interface
        if len(processed.images) > 25:
            processed.images = n_evenly_spaced(processed.images, 25)

        if opts.return_grid:
            grid = images.image_grid(processed.images)
            processed.images.insert(0, grid)
            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", processed.all_seeds[0], processed.prompt, opts.grid_format, info=processed.infotext(p, 0), short_filename=not opts.grid_extended_filename, p=p, grid=True)

        return processed

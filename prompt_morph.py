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

# build prompt with weights scaled by t
def prompt_at_t(weight_indexes, prompt_list, t):
    return " AND ".join(
        [
            ":".join((prompt_list[index], str(weight * t)))
            for index, weight in weight_indexes
        ]
    )

F_LINEAR = "Linear"
F_SINE = "Sine (slow, fast, slow)"
F_HALF_PARABOLIC = "S-Parabola (fast, slow, fast)"
F_PARABOLIC = "Parabolic (slow, fast, faster)"
F_PARABOLIC_BOUNCE = "Parabolic Bounce (parabola, reverse every other keyframe)"

MORPH_FUNCTIONS = [
    F_LINEAR,
    F_SINE,
    F_HALF_PARABOLIC,
    F_PARABOLIC,
    F_PARABOLIC_BOUNCE,
]

"""
Interpolate between two (or more) prompts and create an image at each step.
"""
class Script(scripts.Script):
    def title(self):
        return "Prompt morph"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        i1 = gr.HTML("<p style=\"margin-bottom:0.75em\">Keyframe Format: <br>Seed | Prompt or just Prompt</p>")
        prompt_list = gr.TextArea(label="Prompt list", placeholder="Enter one prompt per line. Blank lines will be ignored.")
        n_images = gr.Slider(minimum=2, maximum=256, value=25, step=1, label="Number of images between keyframes")
        save_video = gr.Checkbox(label='Save results as video', value=True)
        video_fps = gr.Number(label='Frames per second', value=5)

        morph_func = gr.Dropdown(label="Morph Function", choices=MORPH_FUNCTIONS, value=F_LINEAR, type="value", elem_id="morph_func")
        return [i1, prompt_list, morph_func, n_images, save_video, video_fps]

    def run(self, p, i1, prompt_list, morph_func, n_images, save_video, video_fps):
        # override batch count and size
        p.batch_size = 1
        p.n_iter = 1

        prompts = []
        for line in prompt_list.splitlines():
            line = line.strip()
            if line == '' or line.startswith("#"):
                continue

            # TODO: This breaks the :| facial expression used by WaifuDiffusion.
            prompt_args = line.split('|')
            if len(prompt_args) == 1:  # no args
                seed, prompt = '', prompt_args[0]
            else:
                seed, prompt = prompt_args
            prompts.append((seed.strip(), prompt.strip()))

        if len(prompts) < 2:
            msg = "prompt_morph: at least 2 prompts required"
            print(msg)
            return Processed(p, [], p.seed, info=msg)

        state.job_count = 1 + (n_images - 1) * (len(prompts) - 1)

        if save_video:
            import numpy as np
            try:
                import moviepy.video.io.ImageSequenceClip as ImageSequenceClip
            except ImportError:
                msg = "moviepy python module not installed. Will not be able to generate video."
                print(msg)
                return Processed(p, [], p.seed, info=msg)

        # TODO: use a timestamp instead
        # write images to a numbered folder in morphs
        morph_path = os.path.join(p.outpath_samples, "morphs")
        os.makedirs(morph_path, exist_ok=True)
        morph_number = images.get_next_sequence_number(morph_path, "")
        morph_path = os.path.join(morph_path, f"{morph_number:05}")
        p.outpath_samples = morph_path

        all_images = []
        for n in range(1, len(prompts)):
            # parsed prompts
            start_seed, start_prompt = prompts[n-1]
            target_seed, target_prompt = prompts[n]
            res_indexes, prompt_flat_list, prompt_indexes = prompt_parser.get_multicond_prompt_list([start_prompt, target_prompt])
            prompt_weights, target_weights = res_indexes

            # fix seeds. interpret '' as use previous seed
            if start_seed != '':
                if start_seed == '-1':
                    start_seed = -1
                p.seed = start_seed
            processing.fix_seed(p)

            if target_seed == '':
                p.subseed = p.seed
            else:
                if target_seed == '-1':
                    target_seed = -1
                p.subseed = target_seed
            processing.fix_seed(p)
            p.subseed_strength = 0

            # one image for each interpolation step (including start and end)
            for i in range(n_images):
                # first image is same as last of previous morph
                if i == 0 and n > 1:
                    continue
                state.job = f"Morph {n}/{len(prompts)-1}, image {i+1}/{n_images}"

                # TODO: optimize when weight is zero
                # update prompt weights and subseed strength
                x = i / (n_images - 1)
                t = self.calculate_prompt_weight(morph_func, n, x)
                #print ("MORPH FUNC IS " + morph_func + " at step " + str(i) + "/" + str(n_images) +", x=" + str(x) + ", t=" + str(t))
                scaled_prompt = prompt_at_t(prompt_weights, prompt_flat_list, 1.0 - t)
                scaled_target = prompt_at_t(target_weights, prompt_flat_list, t)
                p.prompt = f'{scaled_prompt} AND {scaled_target}'
                if p.seed != p.subseed:
                    p.subseed_strength = t

                processed = process_images(p)
                if not state.interrupted:
                    all_images.append(processed.images[0])

        if save_video:
            clip = ImageSequenceClip.ImageSequenceClip([np.asarray(t) for t in all_images], fps=video_fps)
            clip.write_videofile(os.path.join(morph_path, f"morph-{morph_number:05}.webm"), codec='libvpx-vp9', ffmpeg_params=['-pix_fmt', 'yuv420p', '-crf', '32', '-b:v', '0'], logger=None)

        prompt = "\n".join([f"{seed} | {prompt}" for seed, prompt in prompts])
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

    def calculate_prompt_weight(self, morph_func, n, x):

        if (morph_func == F_LINEAR):
            # 0 to 1
            t = x
        elif (morph_func == F_SINE):
            # 0 is 1 and pi is -1 - sort of an s-shape
            x_pi = math.pi * x
            t = 0.5 - (0.5*math.cos(x_pi))
        elif (morph_func == F_HALF_PARABOLIC):
            # a parabola where the left half is flipped down
            t = ((((2 * x) - 1) * abs((2 * x) - 1)) / 2) + 0.5
        elif (morph_func == F_PARABOLIC):
            # accelerate
            t = x**2
        elif (morph_func == F_PARABOLIC_BOUNCE):
            # Alternate between accelerating and decelerating
            if (n % 2 == 1):
                t = x**2
            else:
                t = 1 - ((1 - x)**2)
        else:
            # default to linear
            print ("Morph Function " + morph_func + " not recognized. Using " + F_LINEAR + " instead.")
            t = x0
        return t

import math
import os

from typing import Callable, Optional, Iterable

import gradio as gr
import torch

from modules import images, processing, scripts, sd_samplers, shared, prompt_parser
from modules.processing import process_images, slerp, Processed
from modules.shared import opts, cmd_opts, state


def k_forward_multiconditionable(
    inner_model: Callable,
    x: torch.Tensor,
    sigma: torch.Tensor,
    uncond: torch.Tensor,
    cond: torch.Tensor,
    cond_scale: float,
    cond_arities: Optional[Iterable[int]],
    cond_weights: Optional[Iterable[float]],
    use_half: bool=False,
) -> torch.Tensor:
    '''
    Magicool k-sampler prompt positive/negative weighting from birch-san.
    https://github.com/Birch-san/stable-diffusion/blob/birch-mps-waifu/scripts/txt2img_fork.py
    '''
    uncond_count = uncond.size(dim=0)
    cond_count = cond.size(dim=0)
    cond_in = torch.cat((uncond, cond)).to(x.device)
    del uncond, cond
    cond_arities_tensor = torch.tensor(cond_arities, device=cond_in.device)
    if use_half and (x.dtype == torch.float32 or x.dtype == torch.float64):
        x = x.half()
    x_in = cat_self_with_repeat_interleaved(t=x,
        factors_tensor=cond_arities_tensor, factors=cond_arities,
        output_size=cond_count)
    del x
    sigma_in = cat_self_with_repeat_interleaved(t=sigma,
        factors_tensor=cond_arities_tensor, factors=cond_arities,
        output_size=cond_count)
    del sigma
    uncond_out, conds_out = inner_model(x_in, sigma_in, cond=cond_in) \
        .split([uncond_count, cond_count])
    del x_in, sigma_in, cond_in
    unconds = repeat_interleave_along_dim_0(t=uncond_out,
        factors_tensor=cond_arities_tensor, factors=cond_arities,
        output_size=cond_count)
    del cond_arities_tensor
    # transform
    #   tensor([0.5, 0.1])
    # into:
    #   tensor([[[[0.5000]]],
    #           [[[0.1000]]]])
    weight_tensor = torch.tensor(list(cond_weights),
        device=uncond_out.device, dtype=uncond_out.dtype) * cond_scale
    weight_tensor = weight_tensor.reshape(len(list(cond_weights)), 1, 1, 1)
    deltas: torch.Tensor = (conds_out-unconds) * weight_tensor
    del conds_out, unconds, weight_tensor
    cond = sum_along_slices_of_dim_0(deltas, arities=cond_arities)
    del deltas
    return uncond_out + cond


def cat_self_with_repeat_interleaved(
    t: torch.Tensor,
    factors: Iterable[int],
    factors_tensor: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """
    Fast-paths for a pattern which in its worst-case looks like:
    t=torch.tensor([[0,1],[2,3]])
    factors=(2,3)
    torch.cat((t, t.repeat_interleave(factors, dim=0)))
    tensor([[0, 1],
            [2, 3],
            [0, 1],
            [0, 1],
            [2, 3],
            [2, 3],
            [2, 3]])
    Fast-path:
      `len(factors) == 1`
      it's just a normal repeat
    t=torch.tensor([[0,1]])
    factors=(2)
    tensor([[0, 1],
            [0, 1],
            [0, 1]])
    
    t=torch.tensor([[0,1],[2,3]])
    factors=(2)
    tensor([[0, 1],
            [2, 3],
            [0, 1],
            [2, 3],
            [0, 1],
            [2, 3]])
    """
    if len(factors) == 1:
        return repeat_along_dim_0(t, factors[0]+1)
    return torch.cat((t, repeat_interleave_along_dim_0(t=t,
        factors_tensor=factors_tensor,
        factors=factors,
        output_size=output_size,
    ))).to(t.device)


def repeat_along_dim_0(t: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Repeats a tensor's contents along its 0th dim `factor` times.
    repeat_along_dim_0(torch.tensor([[0,1]]), 2)
    tensor([[0, 1],
            [0, 1]])
    # shape changes from (1, 2)
    #                 to (2, 2)
    
    repeat_along_dim_0(torch.tensor([[0,1],[2,3]]), 2)
    tensor([[0, 1],
            [2, 3],
            [0, 1],
            [2, 3]])
    # shape changes from (2, 2)
    #                 to (4, 2)
    """
    assert factor >= 1
    if factor == 1:
        return t
    if t.size(dim=0) == 1:
        # prefer expand() whenever we can, since doesn't copy
        return t.expand(factor * t.size(dim=0), *(-1,)*(t.ndim-1))
    return t.repeat((factor, *(1,)*(t.ndim-1)))


def repeat_interleave_along_dim_0(
    t: torch.Tensor,
    factors: Iterable[int],
    factors_tensor: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """
    repeat_interleave()s a tensor's contents along its 0th dim.
    factors=(2,3)
    factors_tensor = torch.tensor(factors)
    output_size=factors_tensor.sum().item() # 5
    t=torch.tensor([[0,1],[2,3]])
    repeat_interleave_along_dim_0(t=t, factors=factors, factors_tensor=factors_tensor, output_size=output_size)
    tensor([[0, 1],
            [0, 1],
            [2, 3],
            [2, 3],
            [2, 3]])
    """
    factors_len = len(factors)
    assert factors_len >= 1
    if len(factors) == 1:
        # prefer repeat() whenever we can, because MPS doesn't support repeat_interleave()
        return repeat_along_dim_0(t, factors[0])
    if t.device.type != 'mps':
        return t.repeat_interleave(factors_tensor, dim=0, output_size=output_size)
    return torch.cat([repeat_along_dim_0(split, factor)
        for split, factor in zip(t.split(1, dim=0), factors)]).to(t.device)


def sum_along_slices_of_dim_0(t: torch.Tensor, arities: Iterable[int]) -> torch.Tensor:
    """
    Implements fast-path for a pattern which in the worst-case looks like this:
    t=torch.tensor([[1],[2],[3]])
    arities=(2,1)
    torch.cat([torch.sum(split, dim=0, keepdim=True) for split in t.split(arities)])
    tensor([[3],
            [3]])
    Fast-path:
      `len(arities) == 1`
      it's just a normal sum(t, dim=0, keepdim=True)
    t=torch.tensor([[1],[2]])
    arities=(2)
    t.sum(dim=0, keepdim=True)
    tensor([[3]])
    """
    if len(arities) == 1:
        if t.size(dim=0) == 1:
            return t
        return t.sum(dim=0, keepdim=True)
    splits: List[torch.Tensor] = t.split(arities)
    device = t.device
    del t
    sums: List[torch.Tensor] = [torch.sum(split, dim=0, keepdim=True)
        for split in splits]
    del splits
    return torch.cat(sums).to(device)


def n_evenly_spaced(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m)] for i in range(n)]


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

        # back up CFGDenoiser.forward
        orig = sd_samplers.CFGDenoiser.forward

        # prompt weights
        cond_weights = [1.0, 0.0]

        # replacement forward function
        def forward(self, x, sigma, uncond, cond, cond_scale):
            if not hasattr(self, 'target_latent'):
                self.target_latent = shared.sd_model.get_learned_conditioning([target_prompt])
            cond = torch.cat(
                [prompt_parser.reconstruct_cond_batch(cond, self.step), self.target_latent],
                dim=0
            )
            uncond = prompt_parser.reconstruct_cond_batch(uncond, self.step)

            denoised = k_forward_multiconditionable(
                self.inner_model,
                x,
                sigma,
                uncond,
                cond,
                cond_scale,
                cond_arities=[2], #cond_arities,
                cond_weights=cond_weights,
                use_half=False, #use_half,
            )
            self.step += 1
            return denoised
        sd_samplers.CFGDenoiser.forward = forward

        # TODO: interpolate between multiple prompts like keyframes
            # TODO: generate video directly with moviepy
        # TODO: integrate seed travel so end prompt can use different seed
        # one image for each interpolation step (including start and end)
        all_images = []
        p.prompt = [p.prompt, target_prompt]
        for i in range(n_images):
            state.job = f"{i+1} out of {n_images}"

            # update prompt weights
            t = i / (n_images - 1)
            cond_weights[0] = 1.0 - t
            cond_weights[1] = t

            processed = process_images(p)
            all_images.append(processed.images[0])
        # limit max images shown to avoid lagging out the interface
        if len(all_images) > 25:
            all_images = n_evenly_spaced(all_images, 25)
        if opts.return_grid:
            grid = images.image_grid(all_images)
            all_images  = [grid] + all_images
            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", processed.all_seeds[0], processed.all_prompts[0], opts.grid_format, info=processed.infotext(p, 0), short_filename=not opts.grid_extended_filename, p=p, grid=True)
        processed.images = all_images

        # restore original CFGDenoiser.forward
        sd_samplers.CFGDenoiser.forward = orig

        return processed

import logging

import numpy as np
import torch.nn.functional as F

import comfy
from .patch_util import PatchKeys, add_model_patch_option, set_model_patch, set_model_patch_replace, \
    is_hunyuan_video_model, is_flux_model, is_ltxv_video_model, is_mochi_video_model, is_wan_video_model

tea_cache_key_attrs = "tea_cache_attr"
coefficients_obj = {
    'Flux': [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01],
    'HunYuanVideo': [7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02],
    'LTXVideo': [2.14700694e+01, -1.28016453e+01, 2.31279151e+00, 7.92487521e-01, 9.69274326e-03],
    'MochiVideo': [-3.51241319e+03,  8.11675948e+02, -6.09400215e+01,  2.42429681e+00, 3.05291719e-03],
    # 100 prompts + 30 steps(29 diff) + 33 frames in WanVideo_t2v_1.3B
    'WanVideo': [2337.2725448820943, -1115.0859732839863, 181.95799689350622, -11.506484961515016, 0.6124379402853622],
}

def get_teacache_global_cache(transformer_options, timesteps):
    diffusion_model = transformer_options.get(PatchKeys.running_net_model)
    if hasattr(diffusion_model, "flux_tea_cache"):
        tea_cache = getattr(diffusion_model, "flux_tea_cache", {})
        transformer_options[tea_cache_key_attrs] = tea_cache
    attrs = transformer_options.get(tea_cache_key_attrs, {})
    attrs['step_i'] = timesteps[0].detach().cpu().item()
    # print(str(attrs['step_i']))

def tea_cache_enter_for_wanvideo(x, timestep, context, transformer_options):
    get_teacache_global_cache(transformer_options, timestep)
    return x, timestep, context

def tea_cache_enter_for_mochivideo(x, timestep, context, attention_mask, num_tokens, transformer_options):
    get_teacache_global_cache(transformer_options, timestep)
    return x, timestep, context, attention_mask, num_tokens

def tea_cache_enter_for_ltxvideo(x, timestep, context, attention_mask, frame_rate, guiding_latent, guiding_latent_noise_scale, transformer_options):
    get_teacache_global_cache(transformer_options, timestep)
    return x, timestep, context, attention_mask, frame_rate, guiding_latent, guiding_latent_noise_scale

# For Flux and HunYuanVideo
def tea_cache_enter(img, img_ids, txt, txt_ids, timesteps, y, guidance, control, attn_mask, transformer_options):
    get_teacache_global_cache(transformer_options, timesteps)
    return img, img_ids, txt, txt_ids, timesteps, y, guidance, control, attn_mask

def tea_cache_patch_blocks_before(img, txt, vec, ids, pe, transformer_options):
    real_model = transformer_options[PatchKeys.running_net_model]
    attrs = transformer_options.get(tea_cache_key_attrs, {})
    step_i = attrs['step_i']
    timestep_start = attrs['timestep_start']
    timestep_end = attrs['timestep_end']
    in_step = timestep_end <= step_i <= timestep_start
    # print(str(timestep_end)+' '+ str(step_i)+' '+str(timestep_start))

    if attrs['rel_l1_thresh'] > 0 and in_step:
        inp = img.clone()
        vec_ = vec.clone()
        coefficient_type = 'Flux'
        if is_ltxv_video_model(real_model):
            coefficient_type = 'LTXVideo'
            modulated_inp = comfy.ldm.common_dit.rms_norm(inp)
            double_block_0 = real_model.transformer_blocks[0]
            num_ada_params = double_block_0.scale_shift_table.shape[0]
            ada_values = double_block_0.scale_shift_table[None, None] + vec_.reshape(img.shape[0], vec_.shape[1], num_ada_params, -1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
            modulated_inp = modulated_inp * (1 + scale_msa) + shift_msa
        elif is_mochi_video_model(real_model):
            coefficient_type = 'MochiVideo'
            double_block_0 = real_model.blocks[0]
            mod_x = double_block_0.mod_x(F.silu(vec_))
            scale_msa_x, gate_msa_x, scale_mlp_x, gate_mlp_x = mod_x.chunk(4, dim=1)
            # copied from comfy.ldm.genmo.joint_model.asymm_models_joint.modulated_rmsnorm
            modulated_inp = comfy.ldm.common_dit.rms_norm(inp)
            modulated_inp = modulated_inp * (1 + scale_msa_x.unsqueeze(1))
        elif is_wan_video_model(real_model):
            coefficient_type = 'WanVideo'
            block_0 = real_model.blocks[0]
            e_0 = (comfy.model_management.cast_to(block_0.modulation, dtype=inp.dtype, device=inp.device) + vec_).chunk(6, dim=1)

            modulated_inp = block_0.norm1(inp) * (1 + e_0[1]) + e_0[0]
        else:
            double_block_0 = real_model.double_blocks[0]
            img_mod1, img_mod2 = double_block_0.img_mod(vec_)
            modulated_inp = double_block_0.img_norm1(inp)
            if is_hunyuan_video_model(real_model):
                coefficient_type = 'HunYuanVideo'
                # if img_mod1.scale is None and img_mod1.shift is None:
                #     pass
                # elif img_mod1.shift is None:
                #     modulated_inp = modulated_inp * (1 + img_mod1.scale)
                # elif img_mod1.scale is None:
                #     modulated_inp =  modulated_inp + img_mod1.shift
                # else:
                #     modulated_inp = modulated_inp * (1 + img_mod1.scale) + img_mod1.shift
                if img_mod1.scale is not None:
                    modulated_inp = modulated_inp * (1 + img_mod1.scale)
                if img_mod1.shift is not None:
                    modulated_inp = modulated_inp + img_mod1.shift
            else:
                # Flux
                modulated_inp = (1 + img_mod1.scale) * modulated_inp + img_mod1.shift
        if attrs['cnt'] == 0 or attrs['cnt'] == attrs['total_steps'] - 1:
            should_calc = True
            attrs['accumulated_rel_l1_distance'] = 0
        else:
            coefficients = coefficients_obj[coefficient_type]
            rescale_func = np.poly1d(coefficients)
            attrs['accumulated_rel_l1_distance'] += rescale_func(((modulated_inp - attrs['previous_modulated_input']).abs().mean() / attrs['previous_modulated_input'].abs().mean()).cpu().item())

            if attrs['accumulated_rel_l1_distance'] < attrs['rel_l1_thresh']:
                should_calc = False
            else:
                should_calc = True
                attrs['accumulated_rel_l1_distance'] = 0
        attrs['previous_modulated_input'] = modulated_inp
        attrs['cnt'] += 1
        if attrs['cnt'] == attrs['total_steps']:
            attrs['cnt'] = 0
    else:
        should_calc = True

    attrs['should_calc'] = should_calc

    del real_model
    return img, txt, vec, ids, pe

def tea_cache_patch_double_blocks_replace(original_args, wrapper_options):
    img = original_args['img']
    txt = original_args['txt']
    transformer_options = wrapper_options.get('transformer_options', {})
    attrs = transformer_options.get(tea_cache_key_attrs, {})
    should_calc = attrs.get('should_calc', True)
    if not should_calc:
        img += attrs['previous_residual']
    else:
        # (b, seq_len, _)
        attrs['ori_img'] = img.clone()
        img, txt = wrapper_options.get('original_blocks')(**original_args, transformer_options=transformer_options)
    return img, txt

def tea_cache_patch_blocks_transition_replace(original_args, wrapper_options):
    img = original_args['img']
    transformer_options = wrapper_options.get('transformer_options', {})
    attrs = transformer_options.get(tea_cache_key_attrs, {})
    should_calc = attrs.get('should_calc', True)
    if should_calc:
        img = wrapper_options.get('original_func')(**original_args, transformer_options=transformer_options)
    return img

def tea_cache_patch_single_blocks_replace(original_args, wrapper_options):
    img = original_args['img']
    txt = original_args['txt']
    transformer_options = wrapper_options.get('transformer_options', {})
    attrs = transformer_options.get(tea_cache_key_attrs, {})
    should_calc = attrs.get('should_calc', True)
    if should_calc:
        img = wrapper_options.get('original_blocks')(**original_args, transformer_options=transformer_options)
    return img, txt

def tea_cache_patch_blocks_after_replace(original_args, wrapper_options):
    img = original_args['img']
    transformer_options = wrapper_options.get('transformer_options', {})
    attrs = transformer_options.get(tea_cache_key_attrs, {})
    should_calc = attrs.get('should_calc', True)
    if should_calc:
        img = wrapper_options.get('original_func')(**original_args)
    return img

def tea_cache_patch_final_transition_after(img, txt, transformer_options):
    attrs = transformer_options.get(tea_cache_key_attrs, {})
    should_calc = attrs.get('should_calc', True)
    if should_calc:
        attrs['previous_residual'] = img - attrs['ori_img']
    return img

def tea_cache_patch_dit_exit(img, transformer_options):
    tea_cache = transformer_options.get(tea_cache_key_attrs, {})
    setattr(transformer_options.get(PatchKeys.running_net_model), "flux_tea_cache", tea_cache)
    return img

def tea_cache_prepare_wrapper(wrapper_executor, noise, latent_image, sampler, sigmas, denoise_mask=None,
                                  callback=None, disable_pbar=False, seed=None):
    cfg_guider = wrapper_executor.class_obj

    # Use cfd_guider.model_options, which is copied from modelPatcher.model_options and will be restored after execution without any unexpected contamination
    temp_options = add_model_patch_option(cfg_guider, tea_cache_key_attrs)
    temp_options['total_steps'] = len(sigmas) - 1
    temp_options['cnt'] = 0
    try:
        out = wrapper_executor(noise, latent_image, sampler, sigmas, denoise_mask=denoise_mask, callback=callback,
                               disable_pbar=disable_pbar, seed=seed)
    finally:
        diffusion_model = cfg_guider.model_patcher.model.diffusion_model
        if hasattr(diffusion_model, "flux_tea_cache"):
            del diffusion_model.flux_tea_cache

    return out

class ApplyTeaCachePatchAdvanced:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rel_l1_thresh": ("FLOAT",
                                  {
                                      "default": 0.25,
                                      "min": 0.0,
                                      "max": 5.0,
                                      "step": 0.01,
                                      "tooltip": "Flux: 0 (original), 0.25 (1.5x speedup), 0.4 (1.8x speedup), 0.6 (2.0x speedup), and 0.8 (2.25x speedup).\n"
                                                 "HunYuanVideo: 0 (original), 0.1 (1.6x speedup), 0.15 (2.1x speedup).\n"
                                                 "LTXVideo: 0 (original), 0.03 (1.6x speedup), 0.05 (2.1x speedup).\n"
                                                 "MochiVideo: 0 (original), 0.06 (1.5x speedup), 0.09 (2.1x speedup).\n"
                                                 "WanVideo: 0 (original), 0.42 (1.5x speedup), 0.45 (1.85x speedup)."
                                  }),
                "start_at": ("FLOAT",
                             {
                                 "default": 0.0,
                                 "step": 0.01,
                                 "max": 1.0,
                                 "min": 0.0,
                             },
                             ),
                "end_at": ("FLOAT", {
                    "default": 1.0,
                    "step": 0.01,
                    "max": 1.0,
                    "min": 0.0,
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_patch_advanced"
    CATEGORY = "patches/speed"
    DESCRIPTION = ("Apply the TeaCache patch to accelerate the model. Use it together with nodes that have the suffix ForwardOverrider."
                   "\nThis is effective only for Flux, HunYuanVideo, LTXVideo, WanVideo and MochiVideo.")

    def apply_patch_advanced(self, model, rel_l1_thresh, start_at=0.0, end_at=1.0):

        model = model.clone()
        patch_key = "tea_cache_wrapper"
        if rel_l1_thresh == 0 or len(model.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, patch_key)) > 0:
            return (model,)

        diffusion_model = model.get_model_object('diffusion_model')
        if not is_flux_model(diffusion_model) and not is_hunyuan_video_model(diffusion_model) and not is_ltxv_video_model(diffusion_model)\
                and not is_mochi_video_model(diffusion_model) and not is_wan_video_model(diffusion_model):
            logging.warning("TeaCache patch is not applied because the model is not supported.")
            return (model,)

        tea_cache_attrs = add_model_patch_option(model, tea_cache_key_attrs)

        tea_cache_attrs['rel_l1_thresh'] = rel_l1_thresh
        model_sampling = model.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(start_at)
        sigma_end = model_sampling.percent_to_sigma(end_at)
        tea_cache_attrs['timestep_start'] = model_sampling.timestep(sigma_start)
        tea_cache_attrs['timestep_end'] = model_sampling.timestep(sigma_end)

        if is_ltxv_video_model(diffusion_model):
            set_model_patch(model, PatchKeys.options_key, tea_cache_enter_for_ltxvideo, PatchKeys.dit_enter)
        elif is_mochi_video_model(diffusion_model):
            set_model_patch(model, PatchKeys.options_key, tea_cache_enter_for_mochivideo, PatchKeys.dit_enter)
        elif is_wan_video_model(diffusion_model):
            set_model_patch(model, PatchKeys.options_key, tea_cache_enter_for_wanvideo, PatchKeys.dit_enter)
        else:
            set_model_patch(model, PatchKeys.options_key, tea_cache_enter, PatchKeys.dit_enter)

        set_model_patch(model, PatchKeys.options_key, tea_cache_patch_blocks_before, PatchKeys.dit_blocks_before)

        set_model_patch_replace(model, PatchKeys.options_key, tea_cache_patch_double_blocks_replace, PatchKeys.dit_double_blocks_replace)
        set_model_patch_replace(model, PatchKeys.options_key, tea_cache_patch_blocks_transition_replace, PatchKeys.dit_blocks_transition_replace)
        set_model_patch_replace(model, PatchKeys.options_key, tea_cache_patch_single_blocks_replace, PatchKeys.dit_single_blocks_replace)
        set_model_patch_replace(model, PatchKeys.options_key, tea_cache_patch_blocks_after_replace, PatchKeys.dit_blocks_after_transition_replace)

        set_model_patch(model, PatchKeys.options_key, tea_cache_patch_final_transition_after, PatchKeys.dit_final_layer_before)
        set_model_patch(model, PatchKeys.options_key, tea_cache_patch_dit_exit, PatchKeys.dit_exit)

        # Just add it once when connecting in series
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                   patch_key,
                                   tea_cache_prepare_wrapper
                                   )
        return (model, )

class ApplyTeaCachePatch(ApplyTeaCachePatchAdvanced):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rel_l1_thresh": ("FLOAT",
                                  {
                                      "default": 0.25,
                                      "min": 0.0,
                                      "max": 5.0,
                                      "step": 0.01,
                                      "tooltip": "Flux: 0 (original), 0.25 (1.5x speedup), 0.4 (1.8x speedup), 0.6 (2.0x speedup), and 0.8 (2.25x speedup).\n"
                                                 "HunYuanVideo: 0 (original), 0.1 (1.6x speedup), 0.15 (2.1x speedup).\n"
                                                 "LTXVideo: 0 (original), 0.03 (1.6x speedup), 0.05 (2.1x speedup).\n"
                                                 "MochiVideo: 0 (original), 0.06 (1.5x speedup), 0.09 (2.1x speedup).\n"
                                                 "WanVideo: 0 (original), 0.42 (1.5x speedup), 0.45 (1.85x speedup)."
                                  }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_patch"
    CATEGORY = "patches/speed"
    DESCRIPTION = ("Apply the TeaCache patch to accelerate the model. Use it together with nodes that have the suffix ForwardOverrider."
                   "\nThis is effective only for Flux, HunYuanVideo, LTXVideo, WanVideo and MochiVideo.")

    def apply_patch(self, model, rel_l1_thresh):

        return super().apply_patch_advanced(model, rel_l1_thresh, start_at=0.0, end_at=1.0)

NODE_CLASS_MAPPINGS = {
    "ApplyTeaCachePatch": ApplyTeaCachePatch,
    "ApplyTeaCachePatchAdvanced": ApplyTeaCachePatchAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyTeaCachePatch": "ApplyTeaCachePatch",
    "ApplyTeaCachePatchAdvanced": "ApplyTeaCachePatchAdvanced",
}

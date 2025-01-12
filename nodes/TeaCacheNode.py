import numpy as np

import comfy
from .patch_util import PatchKeys, add_model_patch_option, set_model_patch, set_model_patch_replace

tea_cache_key_attrs = "tea_cache_attr"
coefficients_obj = {
    'Flux': [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01],
    'HunYuanVideo': [7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02]
}

def tea_cache_enter(img, img_ids, txt, txt_ids, timesteps, y, guidance, control, attn_mask, transformer_options):
    diffusion_model = transformer_options.get(PatchKeys.running_net_model)
    if hasattr(diffusion_model, "flux_tea_cache"):
        tea_cache = getattr(diffusion_model, "flux_tea_cache", {})
        transformer_options[tea_cache_key_attrs] = tea_cache
    return img, img_ids, txt, txt_ids, timesteps, y, guidance, control, attn_mask

def tea_cache_patch_blocks_before(img, txt, vec, ids, pe, transformer_options):
    real_model = transformer_options[PatchKeys.running_net_model]
    attrs = transformer_options.get(tea_cache_key_attrs, {})

    # tea cache src code
    # if self.emb is not None:
    #     emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
    # emb = self.linear(self.silu(emb))
    # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
    # x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
    # x, gate_msa, shift_mlp, scale_mlp, gate_mlp
    inp = img.clone()
    vec_ = vec.clone()
    double_block_0 = real_model.double_blocks[0]
    img_mod1, img_mod2 = double_block_0.img_mod(vec_)
    modulated_inp = double_block_0.img_norm1(inp)
    modulated_inp = (1 + img_mod1.scale) * modulated_inp + img_mod1.shift
    if attrs['cnt'] == 0 or attrs['cnt'] == attrs['total_steps'] - 1:
        should_calc = True
        attrs['accumulated_rel_l1_distance'] = 0
    else:
        coefficients = coefficients_obj[attrs['coefficient_type']]
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

    attrs['should_calc'] = should_calc

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

class ApplyTeaCachePatch:

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
                                                 "HunYuanVideo: 0 (original), 0.1 (1.6x speedup), 0.15 (2.1x speedup)"
                                  }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_patch"
    CATEGORY = "patches/speed"

    def apply_patch(self, model, rel_l1_thresh):

        model = model.clone()
        diffusion_model = model.get_model_object('diffusion_model')
        diffusion_model = diffusion_model
        if not isinstance(diffusion_model, comfy.ldm.flux.model.Flux) and not isinstance(diffusion_model, comfy.ldm.hunyuan_video.model.HunyuanVideo):
            return model,

        set_model_patch(model, PatchKeys.options_key, tea_cache_enter, PatchKeys.dit_enter)
        set_model_patch(model, PatchKeys.options_key, tea_cache_patch_blocks_before, PatchKeys.dit_blocks_before)

        set_model_patch_replace(model, PatchKeys.options_key, tea_cache_patch_double_blocks_replace, PatchKeys.dit_double_blocks_replace)
        set_model_patch_replace(model, PatchKeys.options_key, tea_cache_patch_blocks_transition_replace, PatchKeys.dit_blocks_transition_replace)
        set_model_patch_replace(model, PatchKeys.options_key, tea_cache_patch_single_blocks_replace, PatchKeys.dit_single_blocks_replace)
        set_model_patch_replace(model, PatchKeys.options_key, tea_cache_patch_blocks_after_replace, PatchKeys.dit_blocks_after_transition_replace)

        set_model_patch(model, PatchKeys.options_key, tea_cache_patch_final_transition_after, PatchKeys.dit_final_layer_before)
        set_model_patch(model, PatchKeys.options_key, tea_cache_patch_dit_exit, PatchKeys.dit_exit)


        tea_cache_attrs = add_model_patch_option(model, tea_cache_key_attrs)
        tea_cache_attrs['rel_l1_thresh'] = rel_l1_thresh
        if isinstance(diffusion_model, comfy.ldm.flux.model.Flux):
            tea_cache_attrs['coefficient_type'] = 'Flux'

        elif isinstance(diffusion_model, comfy.ldm.hunyuan_video.model.HunyuanVideo):
                tea_cache_attrs['coefficient_type'] = 'HunYuanVideo'

        patch_key = "tea_cache_wrapper"
        if len(model.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, patch_key)) == 0:
            # Just add it once when connecting in series
            model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                       patch_key,
                                       tea_cache_prepare_wrapper
                                       )
        return (model, )

NODE_CLASS_MAPPINGS = {
    "ApplyTeaCachePatch": ApplyTeaCachePatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyTeaCachePatch": "ApplyTeaCachePatch",
}

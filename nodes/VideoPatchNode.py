import types

import torch
from torch import Tensor

import comfy
from .patch_util import PatchKeys
from comfy.ldm.flux.layers import timestep_embedding


def hunyuan_forward_orig(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    txt_mask: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    control=None,
    transformer_options={},
) -> Tensor:
    patches_replace = transformer_options.get("patches_replace", {})
    patches_point = transformer_options.get(PatchKeys.options_key, {})

    transformer_options[PatchKeys.running_net_model] = self

    patches_enter = patches_point.get(PatchKeys.dit_enter, [])
    if patches_enter is not None and len(patches_enter) > 0:
        for patch_enter in patches_enter:
            img, img_ids, txt, txt_ids, timesteps, y, guidance, control, txt_mask = patch_enter(img,
                                                                                                 img_ids,
                                                                                                 txt,
                                                                                                 txt_ids,
                                                                                                 timesteps,
                                                                                                 y,
                                                                                                 guidance,
                                                                                                 control,
                                                                                                 attn_mask = txt_mask,
                                                                                                 transformer_options=transformer_options
                                                                                                 )

    initial_shape = list(img.shape)
    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256, time_factor=1.0).to(img.dtype))

    vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])

    if self.params.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    if txt_mask is not None and not torch.is_floating_point(txt_mask):
        txt_mask = (txt_mask - 1).to(img.dtype) * torch.finfo(img.dtype).max

    txt = self.txt_in(txt, timesteps, txt_mask)

    ids = torch.cat((img_ids, txt_ids), dim=1)
    pe = self.pe_embedder(ids)

    img_len = img.shape[1]
    if txt_mask is not None:
        attn_mask_len = img_len + txt.shape[1]
        attn_mask = torch.zeros((1, 1, attn_mask_len), dtype=img.dtype, device=img.device)
        attn_mask[:, 0, img_len:] = txt_mask
    else:
        attn_mask = None

    blocks_replace = patches_replace.get("dit", {})

    patch_blocks_before = patches_point.get(PatchKeys.dit_blocks_before, [])
    if patch_blocks_before is not None and len(patch_blocks_before) > 0:
        for blocks_before in patch_blocks_before:
            img, txt, vec, ids, pe = blocks_before(img, txt, vec, ids, pe, transformer_options)

    def double_blocks_wrap(img, txt, vec, pe, control=None, attn_mask=None, transformer_options={}):
        running_net_model = transformer_options[PatchKeys.running_net_model]
        for i, block in enumerate(running_net_model.double_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"],
                                                   txt=args["txt"],
                                                   vec=args["vec"],
                                                   pe=args["pe"],
                                                   attn_mask=args.get("attention_mask"))
                    return out

                out = blocks_replace[("double_block", i)]({"img": img,
                                                           "txt": txt,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           "attention_mask": attn_mask
                                                           },
                                                          {
                                                              "original_block": block_wrap,
                                                              "transformer_options": transformer_options
                                                          })
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask)

            if control is not None:  # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

        return img, txt

    patch_double_blocks_replace = patches_point.get(PatchKeys.dit_double_blocks_replace)

    if patch_double_blocks_replace is not None:
        img, txt = patch_double_blocks_replace({"img": img,
                                                "txt": txt,
                                                "vec": vec,
                                                "pe": pe,
                                                "control": control,
                                                "attn_mask": attn_mask,
                                                },
                                               {
                                                   "original_blocks": double_blocks_wrap,
                                                   "transformer_options": transformer_options
                                               })
    else:
        img, txt = double_blocks_wrap(img=img,
                                      txt=txt,
                                      vec=vec,
                                      pe=pe,
                                      control=control,
                                      attn_mask=attn_mask,
                                      transformer_options=transformer_options
                                      )

    patches_double_blocks_after = patches_point.get(PatchKeys.dit_double_blocks_after, [])
    if patches_double_blocks_after is not None and len(patches_double_blocks_after) > 0:
        for patch_double_blocks_after in patches_double_blocks_after:
            img, txt = patch_double_blocks_after(img, txt, transformer_options)

    patch_blocks_transition = patches_point.get(PatchKeys.dit_blocks_transition_replace)

    def blocks_transition_wrap(**kwargs):
        txt = kwargs["txt"]
        img = kwargs["img"]
        return torch.cat((img, txt), 1)

    if patch_blocks_transition is not None:
        img = patch_blocks_transition({"img": img, "txt": txt, "vec": vec, "pe": pe},
                                      {
                                          "original_func": blocks_transition_wrap,
                                          "transformer_options": transformer_options
                                      })
    else:
        img = blocks_transition_wrap(img=img, txt=txt)

    patches_single_blocks_before = patches_point.get(PatchKeys.dit_single_blocks_before, [])
    if patches_single_blocks_before is not None and len(patches_single_blocks_before) > 0:
        for patch_single_blocks_before in patches_single_blocks_before:
            img, txt = patch_single_blocks_before(img, txt, transformer_options)

    def single_blocks_wrap(img, txt, vec, pe, control=None, attn_mask=None, transformer_options={}):
        running_net_model = transformer_options[PatchKeys.running_net_model]
        for i, block in enumerate(running_net_model.single_blocks):
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"],
                                       vec=args["vec"],
                                       pe=args["pe"],
                                       attn_mask=args.get("attention_mask"))
                    return out

                out = blocks_replace[("single_block", i)]({"img": img,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           "attention_mask": attn_mask},
                                                          {
                                                              "original_block": block_wrap,
                                                              "transformer_options": transformer_options
                                                          })
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

            if control is not None:  # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, : img_len] += add

        return img

    patch_single_blocks_replace = patches_point.get(PatchKeys.dit_single_blocks_replace)

    if patch_single_blocks_replace is not None:
        img, txt = patch_single_blocks_replace({"img": img,
                                                "txt": txt,
                                                "vec": vec,
                                                "pe": pe,
                                                "control": control,
                                                "attn_mask": attn_mask
                                                },
                                               {
                                                   "original_blocks": single_blocks_wrap,
                                                   "transformer_options": transformer_options
                                               })
    else:
        img = single_blocks_wrap(img=img,
                                 txt=txt,
                                 vec=vec,
                                 pe=pe,
                                 control=control,
                                 attn_mask=attn_mask,
                                 transformer_options=transformer_options
                                 )

    patch_blocks_exit = patches_point.get(PatchKeys.dit_blocks_after, [])
    if patch_blocks_exit is not None and len(patch_blocks_exit) > 0:
        for blocks_after in patch_blocks_exit:
            img, txt = blocks_after(img, txt, transformer_options)

    def final_transition_wrap(**kwargs):
        img = kwargs["img"]
        img_len = kwargs["img_len"]
        return img[:, : img_len]

    patch_blocks_after_transition_replace = patches_point.get(PatchKeys.dit_blocks_after_transition_replace)
    if patch_blocks_after_transition_replace is not None:
        img = patch_blocks_after_transition_replace({"img": img, "txt": txt, "vec": vec, "pe": pe, "img_len": img_len},
                                                    {
                                                        "original_func": final_transition_wrap,
                                                        "transformer_options": transformer_options
                                                    })
    else:
        img = final_transition_wrap(img=img, img_len=img_len)

    patches_final_layer_before = patches_point.get(PatchKeys.dit_final_layer_before, [])
    if patches_final_layer_before is not None and len(patches_final_layer_before) > 0:
        for patch_final_layer_before in patches_final_layer_before:
            img = patch_final_layer_before(img, txt, transformer_options)

    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

    shape = initial_shape[-3:]
    for i in range(len(shape)):
        shape[i] = shape[i] // self.patch_size[i]
    img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
    img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
    img = img.reshape(initial_shape)

    patches_exit = patches_point.get(PatchKeys.dit_exit, [])
    if patches_exit is not None and len(patches_exit) > 0:
        for patch_exit in patches_exit:
            img = patch_exit(img, transformer_options)

    del transformer_options[PatchKeys.running_net_model]

    return img


def hunyuan_outer_sample_function_wrapper(wrapper_executor, noise, latent_image, sampler, sigmas, denoise_mask=None,
                                  callback=None, disable_pbar=False, seed=None):
    cfg_guider = wrapper_executor.class_obj
    diffusion_model = cfg_guider.model_patcher.model.diffusion_model
    # set hook
    set_hook(diffusion_model, hunyuan_forward_orig)

    try:
        out = wrapper_executor(noise, latent_image, sampler, sigmas, denoise_mask=denoise_mask, callback=callback,
                               disable_pbar=disable_pbar, seed=seed)
    finally:
        # cleanup hook
        clean_hook(diffusion_model)
    return out


def set_hook(diffusion_model, target_forward_orig):
    # comfy.ldm.hunyuan_video.model.HunyuanVideo.video_old_forward_orig = comfy.ldm.hunyuan_video.model.HunyuanVideo.forward_orig
    # comfy.ldm.hunyuan_video.model.HunyuanVideo.forward_orig = hunyuan_forward_orig
    diffusion_model.video_old_forward_orig = types.MethodType(diffusion_model.forward_orig, diffusion_model)
    diffusion_model.forward_orig = types.MethodType(target_forward_orig, diffusion_model)



def clean_hook(diffusion_model):
    # if hasattr(comfy.ldm.hunyuan_video.model.HunyuanVideo, 'video_old_forward_orig'):
    #     comfy.ldm.hunyuan_video.model.HunyuanVideo.forward_orig = comfy.ldm.hunyuan_video.model.HunyuanVideo.video_old_forward_orig
    #     del comfy.ldm.hunyuan_video.model.HunyuanVideo.video_old_forward_orig
    if hasattr(diffusion_model, 'video_old_forward_orig'):
        diffusion_model.forward_orig = types.MethodType(diffusion_model.video_old_forward_orig, diffusion_model)
        del diffusion_model.video_old_forward_orig


class VideoForwardOverrider:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_patch"
    CATEGORY = "patches/dit"
    DESCRIPTION = "Support HunYuanVideo"

    def apply_patch(self, model):
        model = model.clone()
        if isinstance(model.get_model_object('diffusion_model'), comfy.ldm.hunyuan_video.model.HunyuanVideo):
            patch_key = "video_hunyuan_forward_override_wrapper"
            if len(model.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, patch_key)) == 0:
                # Just add it once when connecting in series
                model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                           patch_key,
                                           hunyuan_outer_sample_function_wrapper
                                           )
        return (model,)


NODE_CLASS_MAPPINGS = {
    "VideoForwardOverrider": VideoForwardOverrider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoForwardOverrider": "VideoForwardOverrider",
}

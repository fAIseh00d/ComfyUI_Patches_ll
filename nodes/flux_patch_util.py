import torch
from torch import Tensor
from comfy.ldm.flux.layers import timestep_embedding

class PatchKeys:
    options_key = "patches_point"
    dit_enter = "patch_dit_enter"
    dit_blocks_before = "patch_dit_blocks_before"
    dit_double_blocks_replace = "patch_dit_double_blocks_replace"
    dit_blocks_transition_replace = "patch_dit_blocks_transition_replace"
    dit_single_blocks_replace = "patch_dit_single_blocks_replace"
    dit_blocks_after = "patch_dit_blocks_after"
    dit_blocks_after_transition_replace = "patch_dit_final_layer_before_replace"
    dit_final_layer_before = "patch_dit_final_layer_before"
    dit_exit = "patch_dit_exit"


def flux_forward_orig(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    control = None,
    transformer_options={},
    attn_mask: Tensor = None,
) -> Tensor:
    patches_replace = transformer_options.get("patches_replace", {})
    patches_insert = transformer_options.get(PatchKeys.options_key, {})

    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    transformer_options["running_net_model"] = self

    patches_enter = patches_insert.get(PatchKeys.dit_enter, [])
    if patches_enter is not None and len(patches_enter) > 0:
        for patch_enter in patches_enter:
            img, img_ids, txt, txt_ids, timesteps, y, guidance, control, attn_mask = patch_enter(img,
                                                                                                 img_ids,
                                                                                                 txt,
                                                                                                 txt_ids,
                                                                                                 timesteps,
                                                                                                 y,
                                                                                                 guidance,
                                                                                                 control,
                                                                                                 attn_mask,
                                                                                                 transformer_options
                                                                                                 )

    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    vec = vec + self.vector_in(y)
    txt = self.txt_in(txt)

    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    blocks_replace = patches_replace.get("dit", {})

    patches_blocks_enter = patches_insert.get(PatchKeys.dit_blocks_before, [])
    if patches_blocks_enter is not None and len(patches_blocks_enter) > 0:
        for patch_blocks_enter in patches_blocks_enter:
            img, txt, vec, ids, pe = patch_blocks_enter(img, txt, vec, ids, pe, transformer_options)

    def double_blocks_wrap(img, txt, vec, pe, control=None, attn_mask=None, transformer_options={}):
        running_net_model = transformer_options["running_net_model"]
        for i, block in enumerate(running_net_model.double_blocks):
            # 0 -> 18
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"],
                                                   txt=args["txt"],
                                                   vec=args["vec"],
                                                   pe=args["pe"],
                                                   attn_mask=args.get("attn_mask"))
                    return out

                out = blocks_replace[("double_block", i)]({"img": img,
                                                           "txt": txt,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           # add timestep for pulid
                                                           # "timestep": timesteps,
                                                           # "attn_mask": attn_mask
                                                           },
                                                          {
                                                              "original_block": block_wrap,
                                                              "flux_model": running_net_model,
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

    patch_double_blocks_replace = patches_insert.get(PatchKeys.dit_double_blocks_replace)

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

    patch_blocks_transition = patches_insert.get(PatchKeys.dit_blocks_transition_replace)

    def blocks_transition_wrap(**kwargs):
        txt = kwargs["txt"]
        img = kwargs["img"]
        return torch.cat((txt, img), 1)

    if patch_blocks_transition is not None:
        img = patch_blocks_transition({"img": img, "txt": txt, "vec": vec, "pe": pe},
                                      {
                                          "original_func": blocks_transition_wrap,
                                          "transformer_options": transformer_options
                                      })
    else:
        img = blocks_transition_wrap(img=img, txt=txt)

    def single_blocks_wrap(img, txt, vec, pe, control=None, attn_mask=None, transformer_options={}):
        running_net_model = transformer_options["running_net_model"]
        for i, block in enumerate(running_net_model.single_blocks):
            # 0 -> 37
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"],
                                       vec=args["vec"],
                                       pe=args["pe"],
                                       attn_mask=args.get("attn_mask"))
                    return out

                out = blocks_replace[("single_block", i)]({"img": img,
                                                           "timestep": timesteps,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           "attn_mask": attn_mask},
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
                        img[:, txt.shape[1]:, ...] += add

        return img

    patches_single_blocks_replace = patches_insert.get(PatchKeys.dit_single_blocks_replace)

    if patches_single_blocks_replace is not None:
        img, txt = patches_single_blocks_replace({"img": img,
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

    patches_blocks_exit = patches_insert.get(PatchKeys.dit_blocks_after, [])
    if patches_blocks_exit is not None and len(patches_blocks_exit) > 0:
        for patch_blocks_exit in patches_blocks_exit:
            img, txt = patch_blocks_exit(img, txt, transformer_options)

    def final_transition_wrap(**kwargs):
        img = kwargs["img"]
        txt = kwargs["txt"]
        return img[:, txt.shape[1]:, ...]

    patch_blocks_after_transition_replace = patches_insert.get(PatchKeys.dit_blocks_after_transition_replace)
    if patch_blocks_after_transition_replace is not None:
        img = patch_blocks_after_transition_replace({"img": img, "txt": txt, "vec": vec, "pe": pe},
                                     {
                                         "original_func": final_transition_wrap,
                                         "transformer_options": transformer_options
                                     })
    else:
        img = final_transition_wrap(img=img, txt=txt)

    patches_final_layer_before = patches_insert.get(PatchKeys.dit_final_layer_before, [])
    if patches_final_layer_before is not None and len(patches_final_layer_before) > 0:
        for patch_final_layer_before in patches_final_layer_before:
            img = patch_final_layer_before(img, txt, transformer_options)

    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

    patches_exit = patches_insert.get(PatchKeys.dit_exit, [])
    if patches_exit is not None and len(patches_exit) > 0:
        for patch_exit in patches_exit:
            img = patch_exit(img, transformer_options)

    del transformer_options["running_net_model"]

    return img


def set_model_patch(model_patcher, options_key, patch, name):
    to = model_patcher.model_options["transformer_options"]
    if options_key not in to:
        to[options_key] = {}
    to[options_key][name] = to[options_key].get(name, []) + [patch]

def set_model_patch_replace(model_patcher, options_key, patch, name):
    to = model_patcher.model_options["transformer_options"]
    if options_key not in to:
        to[options_key] = {}
    to[options_key][name] = patch

def add_model_patch_option(model, patch_key):
    if 'transformer_options' not in model.model_options:
        model.model_options['transformer_options'] = {}
    to = model.model_options['transformer_options']
    if patch_key not in to:
        to[patch_key] = {}
    return to[patch_key]
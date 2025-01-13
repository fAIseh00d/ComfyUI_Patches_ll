from .patch_lib.FluxPatch import flux_forward_orig
from .patch_lib.HunYuanVideoPatch import hunyuan_forward_orig
from .patch_lib.LTXVideoPatch import ltx_forward_orig
from .patch_util import is_hunyuan_video_model, is_ltxv_video_model, is_flux_model


def get_new_forward_orig(diffusion_model):
    if is_hunyuan_video_model(diffusion_model):
        return hunyuan_forward_orig
    if is_ltxv_video_model(diffusion_model):
        return ltx_forward_orig
    if is_flux_model(diffusion_model):
        return flux_forward_orig
    return None

def get_old_method_name(diffusion_model):
    if is_hunyuan_video_model(diffusion_model):
        return 'forward_orig'
    if is_ltxv_video_model(diffusion_model):
        return 'forward'
    if is_flux_model(diffusion_model):
        return 'forward_orig'
    return None
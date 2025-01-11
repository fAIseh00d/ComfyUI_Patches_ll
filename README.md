[中文文档](README_CN.md)

Add some hooks method support. Such as `TeaCache`, `PuLID-Flux`.

## Preview (Image with WorkFlow)
![save api extended](example/workflow_base.png)

Working with `PuLID` (need my other custom nodes [ComfyUI_PuLID_Flux_ll](https://github.com/lldacing/ComfyUI_PuLID_Flux_ll))
![save api extended](example/PuLID_with_teacache.png)


## Install

- Manual
```shell
    cd custom_nodes
    git clone https://github.com/lldacing/ComfyUI_Patches_ll.git
    cd ComfyUI_Patches_ll
    # restart ComfyUI
```

## Nodes
- FluxForwardOverrider
  - Add some hooks method support to the `Flux` model
- ApplyTeaCachePatch
  - Use the `hooks` provided in `FluxForwardOverrider` to support `TeaCache` acceleration (currently only supports Flux, video related will be added in future)

## Thanks

[TeaCache](https://github.com/ali-vilab/TeaCache)

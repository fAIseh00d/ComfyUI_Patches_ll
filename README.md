[中文文档](README_CN.md)

Add some hooks method. Such as: `TeaCache` and `First Block Cache` for `PuLID-Flux` `Flux` `HunYuanVideo` `LTXVideo` `MochiVideo`.

## Preview (Image with WorkFlow)
![save api extended](example/workflow_base.png)

Working with `PuLID` (need my other custom nodes [ComfyUI_PuLID_Flux_ll](https://github.com/lldacing/ComfyUI_PuLID_Flux_ll))
![save api extended](example/PuLID_with_teacache.png)


## Install

- Manual
```shell
    cd custom_nodes
    git clone https://github.com/lldacing/ComfyUI_Patches_ll.git
    # restart ComfyUI
```

## Nodes
- FluxForwardOverrider
  - Add some hooks method support to the `Flux` model
- VideoForwardOverrider
  - Add some hooks method support to the video model. Support `HunYuanVideo`, `LTXVideo`, `MochiVideo`
- DitForwardOverrider
  - Auto add some hooks method for model (automatically identify model type). Support `Flux`, `HunYuanVideo`, `LTXVideo`, `MochiVideo`
- ApplyTeaCachePatch
  - Use the `hooks` provided in `*ForwardOverrider` to support `TeaCache` acceleration. Support `Flux`, `HunYuanVideo`, `LTXVideo`, `MochiVideo`
  - In my test results, the video quality is not good after acceleration for `MochiVideo`
- ApplyTeaCachePatchAdvanced
  - Support `start_at` and `end_at`
- ApplyFirstBlockCachePatch
  - Use the `hooks` provided in `*ForwardOverrider` to support `First Block Cache` acceleration. Support `Flux`, `HunYuanVideo`, `LTXVideo`, `MochiVideo`
  - In my test results, the video quality is not good after acceleration for `MochiVideo`
- ApplyFirstBlockCachePatchAdvanced
  - Support `start_at` and `end_at`

## Thanks

[TeaCache](https://github.com/ali-vilab/TeaCache)
[ParaAttention](https://github.com/chengzeyi/ParaAttention)
[Comfy-WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed)

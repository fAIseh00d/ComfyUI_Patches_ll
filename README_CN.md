[English](README.md)

添加一些钩子方法支持。例如支持`TeaCache`和`PulID-Flux`。

## 预览 (图片含工作流)
![save api extended](example/workflow_base.png)

Working with `PuLID` (need my other custom nodes [ComfyUI_PuLID_Flux_ll](https://github.com/lldacing/ComfyUI_PuLID_Flux_ll))
![save api extended](example/PuLID_with_teacache.png)


## 安装

- 手动安装
```shell
    cd custom_nodes
    git clone https://github.com/lldacing/ComfyUI_Patches_ll.git
    cd ComfyUI_Patches_ll
    # restart ComfyUI
```

## 节点
- FluxForwardOverrider
  - 为`Flux`模型增加一些`hook`方法支持
- ApplyTeaCachePatch
  - 使用`FluxForwardOverrider`中的预留的hook，支持`TeaCache`加速（目前支持`Flux`，后面会加视频相关)

## 感谢

[TeaCache](https://github.com/ali-vilab/TeaCache)


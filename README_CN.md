[English](README.md)

添加一些钩子方法。例如使用`TeaCache`加速`PulID-Flux`、`Flux`、`混元视频`。

## 预览 (图片含工作流)
![save api extended](example/workflow_base.png)

加速`PuLID` (需要配合我的另一插件 [ComfyUI_PuLID_Flux_ll](https://github.com/lldacing/ComfyUI_PuLID_Flux_ll)使用)
![save api extended](example/PuLID_with_teacache.png)


## 安装

- 手动安装
```shell
    cd custom_nodes
    git clone https://github.com/lldacing/ComfyUI_Patches_ll.git
    # restart ComfyUI
```

## 节点
- FluxForwardOverrider
  - 为`Flux`模型增加一些`hook`方法支持
- VideoForwardOverrider
  - 为视频模型添加一些`hook`方法支持. 支持 `HunYuanVideo`
- DitForwardOverrider
  - 为Dit架构模型增加一些`hook`方法支持(自动识别模型类型). 支持 `Flux`, `HunYuanVideo`
- ApplyTeaCachePatch
  - 使用`*ForwardOverrider`中支持的`hook`方法提供`TeaCache`加速，支持 `Flux`, `HunYuanVideo`

## 感谢

[TeaCache](https://github.com/ali-vilab/TeaCache)


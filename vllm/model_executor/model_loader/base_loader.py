# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.model_executor.model_loader.utils import (
    initialize_model, process_weights_after_loading, set_default_torch_dtype)


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows 
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError

    """
    调用顺序：
    BaseModelLoader.load_model()--> 
    (以下都是DefaultModelLoader中的函数)
    load_weights() 
        -> get_all_weights()
            -> _get_weights_iterator()
                -> _prepare_weights()
                    -> _maybe_download_from_modelscope() 
    """
    def load_model(self, vllm_config: VllmConfig,
                   model_config: ModelConfig) -> nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                # 初始化模型架构,还没有加载模型, 只是获得一个model class
                model = initialize_model(vllm_config=vllm_config,
                                         model_config=model_config)
            # Quantization does not happen in `load_weights` but after it
            # 加载模型权重
            self.load_weights(model, model_config)
            # 后处理
            process_weights_after_loading(model, model_config, target_device)
        # model.eval()是模型的评估模式
        # 与之相对的是model.train()，训练模式，dropout层会随机丢弃一些神经元，batchnorm层会使用当前batch的统计信息，有一些训练特定行为。
        # 而评估模式dropout层会停止随机丢弃，而是使用所有神经元；batchnorm层会使用训练时计算的全局统计信息。所以模型在这个模式下能使用最优的配置。    
        return model.eval()

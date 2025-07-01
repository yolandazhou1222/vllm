# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .utils import maybe_prefix

logger = init_logger(__name__)
'''
#eagle:推测解码speculative decoding技术中的草稿模型draft model.
推测解码是什么:
用一个小而快的模型(draft model)先生成几个候选token,
然后用大而准确的模型(target model)来验证这些候选token,
如果验证通过就采用,否则重新生成.
这样可以加速文本生成过程.

#transformers架构中的头和尾:
##头)input embedding输入嵌入:
把token id转换为embedding向量
    输入:一个整数序列(token IDs),每个整数代表词汇表中的一个token。
    输出:一个三维张量 (batch_size, sequence_length, hidden_size)。其中：
        batch_size:一次处理的样本数
        sequence_length:输入序列的长度
        hidden_size:嵌入向量的维度(模型隐藏层大小,如768)

##尾)lm head(language model head):
将transformers模型最后一层输出的的隐藏状态hidden_state,映射回词汇表上的概率分布(即转换为logits,即每个token的预测概率)
位置:lm head在解码器后面.
结构:
    linear层:将维度为hidden_size的hidden_state映射到维度为vocab_size的向量.
            (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, vocab_size)
    softmax层:将linear层的输出(维度为vocab_size的向量)应用softmax函数,转换为概率分布.这个分布表示在当前位置,词汇表中每个词作为下一个词(即目标词)出现的概率.
            output = Softmax(Linear(hidden_state))
输入:
    transformers模型最后一层输出的hidden_state张量(batch_size, sequence_length, hidden_size)
输出:
    logits张量(batch_size, sequence_length, vocab_size)
    vocab_size表示该位置对应目标词的预测概率分布
    
#当前代码结构:
EAGLE MODEL:
   

why:
EAGLE论文发现第一层的输入层归一化和最后的输出层归一化对性能提升不大
用这些"虚拟"层替换原有的归一化层，保持接口一致但跳过计算
这是一种性能优化策略


'''

class DummyInputLayerNorm(nn.Module):

    def __init__(self, weight=None, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight) if weight is not None else None
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x):
        return x


class DummyOutputNorm(nn.Module):

    def forward(self, x, residual):
        if residual is None:
            return x
        else:
            return x + residual, None


class EAGLE(nn.Module):
    """This class implements the EAGLE draft model from the paper: https://arxiv.org/pdf/2401.15077
    Reference implementation: https://github.com/SafeAILab/EAGLE
    
    Differences from reference implementation:
    1. In reference, LlamaDecoderLayer implementation doesn't have 
       input_layernorm for 1st decoder layer (https://github.com/SafeAILab/EAGLE/blob/7d065d084443fbfd386f88839efd7193c12be869/eagle/model/cnets.py#L427).
       Following this approach, our implementation also disables
       the input_layernorm for the first decoder layer.
    2. We allow any decoder layer to be used in EAGLE whereas in reference 
       decoder layer is fixed to be LlamaDecoderLayer.
    3. We have an optional token_map which reduces draft vocab to most 
       frequently used tokens to give some additional speed-up by reducing 
       sampling overhead. This is disabled unless the checkpoint file has 
       explicit token_map tensor and config has an optional attribute 
       truncated_vocab_size < vocab_size. To use this technique, one has to find
       the top-k most frequent tokens in target dataset and add that as a tensor
       in the draft checkpoint (using key token_map). Also, the draft config
       needs to have truncated_vocab_size (=k) as an attribute.
    4. We allow an enhanced EAGLE architecture similar to the DeepSeek MTP 
       module with regards to the use of additional RMS norms. The original 
       EAGLE architecture 1) skips the pre-attention norm in its first 
       transformer block, and 2) skips the final output norm, both of which we 
       found to be suboptimal. We also add the support for separate norms
       applying to both the token embedding and hidden states before projection
       as in DeepSeek MTP, which we found to improve performance as well.
    """
    '''
    self.model和self.model.model的区别:
    在EAGLE类的__init__方法中,self.model是通过ModelRegistry.resolve_model_cls(architectures)获取的模型类,创建的实例.
    所以architectures为deepseekmtp时,self.model是DeepSeekMTP类的实例.
    那么self.model.model呢?
    self.model是DeepSeekMTP类,
    DeepSeekMTP的__init__方法里的self.model是DeepSeekMultiTokenPredictor类的实例,
    所以self.model.model是DeepSeekMultiTokenPredictor类的实例.
    '''

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.dtype = vllm_config.model_config.dtype
        self.config = config

        architectures = getattr(self.config.model, "architectures", [])
        # 根据模型的architectures,初始化
        model_cls, _ = ModelRegistry.resolve_model_cls(architectures)

        self.model = model_cls(vllm_config=vllm_config,
                               prefix=maybe_prefix(prefix, "model"))
        # 特征融合
        self.fc = nn.Linear(config.model.hidden_size * 2,#输入维度
                            config.model.hidden_size,#输出维度
                            bias=getattr(self.config, "eagle_fc_bias", False))#是否使用偏置

        # Modify layer normalization and residual connections as suggested
        # in the EAGLE framework: https://github.com/SafeAILab/EAGLE
        # While weights and biases are generally not needed,
        # they are retained here to support certain unit tests
        # (e.g., spec_decode/e2e/test_eagle_correctness.py).
        # 替换掉第一层和最后一层的归一化层，保持结构不变但加速
        if not hasattr(self.config.model,
                       "skip_prenorm") or self.config.model.skip_prenorm:
            # !!!报错1:找不到layer0
            # 因为nextn模型在rank6,当前是rank0.
            # rank分配策略:load_model<--get_model(config)<--confg里的get_layers_start_end_indeces<--get_pp_indices()
            # 这个分配策略会尽量避免放到rank0和最后一个rank
            self.model.model.layers[0].input_layernorm = DummyInputLayerNorm(
                weight=self.model.model.layers[0].input_layernorm.weight)

        if not hasattr(
                self.config.model,
                "skip_output_norm") or self.config.model.skip_output_norm:
            self.model.model.norm = DummyOutputNorm()

        self.add_para_norm = False
        if hasattr(self.config.model,
                   "add_para_norm") and self.config.model.add_para_norm:
            self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.add_para_norm = True

        self.orig_vocab_size = config.vocab_size
        self.truncated_vocab_size = config.truncated_vocab_size
        # 可以截断词汇表,因为草稿模型只需要预测最常用的token,
        # 减小词汇表大小可以减少计算开销
        # 可以通过init最后的token_map把小词汇表映射回完整词汇表
        self.unpadded_vocab_size = self.truncated_vocab_size

        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=self.truncated_vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
        )

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                self.truncated_vocab_size,
                                                logit_scale)

        # Token map is a idx to token mapping to reduce the vocab size for
        # the draft model. Using smaller vocab size for draft, containing
        # only most frequent tokens reduces the speculation overhead. This
        # doesn't affect the acceptance rate much and thus gives more speed
        # -up. By default, this is disabled and is only used if the EAGLE
        # checkpoint file has token_map tensor.
        self.token_map = None

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        # !!!报错2: DeepSeekMultiTokenPredictor类里没有get_input_embeddings方法,
        # 但它有embed_tokens属性,尝试用 return self.model.model.embed_tokens(input_ids)
        return self.model.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)

        # Handle both empty previous_hidden_states
        # and mismatched batch size
        batch_size = inputs_embeds.size(0)
        if previous_hidden_states.size(0) == 0 or \
           previous_hidden_states.size(0) != batch_size:
            hidden_dim = self.config.model.hidden_size
            device = inputs_embeds.device
            # Create zero tensor with matching batch size
            previous_hidden_states = \
                torch.zeros(batch_size, hidden_dim, device=device)

        # 合并inputs_embeds和之前的hidden_states,作为新的inputs_embeds
        if self.add_para_norm:
            # enorm和hnorm是干啥的？
            inputs_embeds = torch.cat([
                self.enorm(inputs_embeds),
                self.hnorm(previous_hidden_states)
            ],
                                      dim=-1)
        else:
            inputs_embeds = torch.cat([inputs_embeds, previous_hidden_states],
                                      dim=-1)
        # 调了fc
        # inputs_embeds: (batch_size, sequence_length, hidden_size * 2)
        # fc后的inputs_embeds: (batch_size, sequence_length, hidden_size)
        inputs_embeds = self.fc(inputs_embeds)
        # 在位置0处mask掉inputs_embeds
        # 位置0一般是特殊的开始位置,要掩码防止它干扰正常的序列处理
        inputs_embeds[positions == 0] = 0  # masking inputs at position=0
        # base model的forward
        hidden_states = self.model.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
        )
        #!!!报错3: DeepSeekMultiTokenPredictor.forward()没有intermediate_tensors参数.
        # 但DeepSeekMTP.forward()有intermediate_tensors参数.要改吗？
        # 如果改的话会报新的错:DeepSeekMTP.forward()需要传previous_hidden_states
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)

        if self.token_map is not None:
            _logits = logits
            logits = -torch.inf * torch.ones(
                size=(*_logits.shape[:-1], self.orig_vocab_size),
                device=_logits.device,
                dtype=_logits.dtype)

            logits[..., self.token_map] = _logits

        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # This implementation is incompatible with https://huggingface.co/yuhuili/EAGLE-LLaMA3-Instruct-8B
        # due to missing lm_head weights and its config being that of a
        # Llama model. Here's a compatible version with the same weights:
        # https://huggingface.co/abhigoyal/EAGLE-LLaMA3-Instruct-8B-vllm
        # Also, here's an example script for converting trained EAGLE
        # checkpoint to vLLM compatible version: https://gist.github.com/abhigoyal1997/1e7a4109ccb7704fbc67f625e86b2d6d
        model_weights = {}
        for name, loaded_weight in weights:
            if name == "token_map":
                # 截断词汇表
                if self.config.truncated_vocab_size < self.config.vocab_size:
                    self.token_map = nn.Parameter(loaded_weight,
                                                  requires_grad=False)
            elif name.startswith("fc.weight"):
                weight_loader = getattr(self.fc.weight, "weight_loader",
                                        default_weight_loader)
                weight_loader(self.fc.weight, loaded_weight)
            elif name.startswith("fc.bias"):
                if self.fc.bias is not None:
                    weight_loader = getattr(self.fc.bias, "weight_loader",
                                            default_weight_loader)
                    weight_loader(self.fc.bias, loaded_weight)
                else:
                    logger.warning_once("Found bias in the loaded weights but "
                                        "the model config doesn't have bias.")
            elif name.startswith("enorm.weight"):
                weight_loader = getattr(self.enorm.weight, "weight_loader",
                                        default_weight_loader)
                weight_loader(self.enorm.weight, loaded_weight)
            elif name.startswith("hnorm.weight"):
                weight_loader = getattr(self.hnorm.weight, "weight_loader",
                                        default_weight_loader)
                weight_loader(self.hnorm.weight, loaded_weight)
            elif name.startswith("model.lm_head.") or name.startswith(
                    "model.model."):
                model_weights[name.split("model.", 1)[-1]] = loaded_weight
            elif name.startswith("lm_head.") or name.startswith("model."):
                model_weights[name] = loaded_weight
            else:
                model_weights[f"model.{name}"] = loaded_weight

        if "lm_head.weight" in model_weights:
            lm_head_weight = model_weights.pop("lm_head.weight")

            if self.token_map is not None and\
                lm_head_weight.shape[0] > self.token_map.shape[0]:

                lm_head_weight = lm_head_weight[self.token_map]

        else:
            # NOTE(Shangming): initialize the placeholder for lm_head weight.
            lm_head_weight = torch.zeros(
                self.lm_head.org_vocab_size,
                self.lm_head.embedding_dim,
                dtype=self.dtype,
            )

        weight_loader = getattr(self.lm_head.weight, "weight_loader",
                                default_weight_loader)
        weight_loader(self.lm_head.weight, lm_head_weight)

        self.model.load_weights(model_weights.items())

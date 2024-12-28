# -*- coding:utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from param import args


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
    (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-8):
        """Construct an RMSNorm module."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        x = x / rms
        return self.weight * x


class SelfAttention(nn.Module):
    """Self-attention module for SASRec."""

    def __init__(self):
        super().__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({args.hidden_size}) is not a multiple of the number of attention "
                f"heads ({args.num_attention_heads})"
            )
        self.num_attention_heads: int = args.num_attention_heads
        self.attention_head_size: int = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size: int = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.RMSNorm = RMSNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_multihead(self, x):
        """Transpose to multi head attention shape.

        e.g., [256, 50, 64] -> [256, 50, 2, 32] -> [256, 2, 50, 32]
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor: Tensor, attention_mask: Tensor):
        """Self-attention:

        1. multi head attention
            - input + W matrix -> Q K V
            - Q K V -> Multihead
            - Q^T * K / sqrt(d_k)
            - mask attention score
            - softmax
            - dropout
        * No: 2. Add & Norm
            - weight sum
            - Layer Norm
        3. FFN
            - linear
            - dropout
        4. Add & Norm
            - residual connection
            - Layer Norm
        5. return

        Args:
            input_tensor (Tensor): [256, 50, 64] -> [batch_size, seq_len, hidden_size]
            attention_mask (Tensor): [256, 1, 50, 50] -> [batch_size, 1, seq_len, seq_len]

        Returns:
            Tensor: shape: [batch_size, seq_len, hidden_size]
        """
        # * shape: [batch_size, seq_len, hidden_size]
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        # * shape: [batch_size, heads, seq_len, head_size]
        query_layer = self.transpose_multihead(mixed_query_layer)
        key_layer = self.transpose_multihead(mixed_key_layer)
        value_layer = self.transpose_multihead(mixed_value_layer)

        # * Take the dot product between "query" and "key" to get the raw attention scores.
        # * formula: Q^T * K / sqrt(d_k)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # * Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # * [batch_size heads seq_len seq_len] scores
        # * [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # * Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # * This is actually dropping out entire tokens to attend to, which might
        # * seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)

        # * formula: dropout(softmax(Q^T * K)) * V
        context_layer = torch.matmul(attention_probs, value_layer)

        # * shape: [batch_size, seq_len, heads, head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # * shape: [batch_size, seq_len, hidden_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # * Linear + dropout + residual + layer norm
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.RMSNorm(hidden_states + input_tensor)

        return hidden_states


class Intermediate(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):
        """FFN: Feed forward network.
        1. linear
        2. activation
        3. linear
        4. dropout
        5. residual connection
        6. layer norm
        7. return

        Args:
            input_tensor (Tensor): [256, 50, 64] -> [batch_size, seq_len, hidden_size]

        Returns:
            Tensor: shape: [batch_size, seq_len, hidden_size]
        """

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Layer(nn.Module):
    """Layer block in the transformer model."""

    def __init__(self):
        super().__init__()
        self.attention = SelfAttention()
        self.intermediate = Intermediate()

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        return self.intermediate(attention_output)


class Encoder(nn.Module):
    """Encoder: a stack of N layers."""

    def __init__(self, num_hidden_layer=args.num_hidden_layers):
        super().__init__()
        layer = Layer()
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_hidden_layer)])

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, output_all_encoded_layers=True) -> list[Tensor]:
        """

        Args:
            hidden_states (Tensor): input. [256, 50, 64] -> [batch_size, seq_len, hidden_size]
            attention_mask (Tensor): [256, 1, 50, 50] -> [batch_size, 1, seq_len, seq_len]
            output_all_encoded_layers (bool, optional): if True, return output of all layers. Else, return the last layer only. Defaults to True.

        Returns:
            list[Tensor]: list of hidden states of all layers or the last layer. SHAPE: [Layer_num, batch_size, seq_len, hidden_size] or [1, batch_size, seq_len, hidden_size
        """

        all_encoder_layers: list[Tensor] = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class LigthGCNLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, adj, embeds):
        return torch.sparse.mm(adj, embeds)


class NGCFLayer(nn.Module):

    def __init__(self, in_features=args.hidden_size, out_features=args.hidden_size):
        super().__init__()
        # self.dropout = nn.Dropout(args.gcn_dropout_prob)
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, adj, embeds):
        embeds = self.linear(embeds)
        # embeds = self.dropout(embeds)
        return F.leaky_relu(torch.sparse.mm(adj, embeds), negative_slope=0.2)


class ProbSparseSelfAttention(nn.Module):
    """Self-attention module for SASRec."""

    def __init__(self):
        super().__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({args.hidden_size}) is not a multiple of the number of attention "
                f"heads ({args.num_attention_heads})"
            )
        self.num_attention_heads: int = args.num_attention_heads
        self.attention_head_size: int = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size: int = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.RMSNorm = RMSNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

        self.factor = 5  # factor：缩放因子，用于计算采样的数量  factor * ln(Q长度)
        self.scale = None  # scale：决定是否缩放
        self.mask_flag = True  # mask_flag：决定是否使用掩码，默认为True

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        # K_expand.shape: torch.Size([32, 8, 96, 96, 64])
        # torch.Size([96, 25])
        # K_sample.shape: torch.Size([32, 8, 96, 25, 64])
        # Q_K_sample.shape: torch.Size([32, 8, 96, 25])

        # 通过在 K 上添加一个新的维度并扩展到 [B, H, L_Q, L_K, E] 的形状
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        # 根据 index_sample 对 K_expand 进行采样，得到一个形状为 [B, H, L_Q, sample_k, E] 的张量
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]

        # 计算查询和采样的键之间的点积，得到一个形状为 [B, H, L_Q, sample_k] 的张量
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # 找到具有稀疏度度量的Top-K查询
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # 使用减少的Q计算Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        # * M.shape: torch.Size([32, 8, 96])
        # * M_top.shape: torch.Size([32, 8, 25])
        # * Q_reduce.shape: torch.Size([32, 8, 25, 64])
        # * Q_K.shape: torch.Size([32, 8, 25, 96])
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape

        if not self.mask_flag:
            # 计算 V 在最后一个维度上的均值，形状为 [B, H, D]
            V_sum = V.mean(dim=-2)
            # 将 V_sum 在新添加的维度上扩展到 [B, H, L_Q, D]，并返回它的副本
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # 使用掩码
            assert L_Q == L_V
            # 断言查询长度 L_Q 和值的长度 L_V 相等，即只适用于自注意力（self-attention）
            contex = V.cumsum(dim=-2)

            # * V.shape: torch.Size([32, 8, 72, 64])
            # * contex.shape: torch.Size([32, 8, 72, 64])
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # 使用softmax函数计算注意力权重，形状为 [B, H, n_top, L_V]
        attn = torch.softmax(scores, dim=-1)
        # 使用注意力权重 attn 和值 V 进行加权求和
        # 并将结果赋值到 context_in 的对应位置，对应位置形状为 [B, H, n_top, D]
        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = torch.matmul(
            attn, V
        ).type_as(context_in)
        if self.output_attention:
            # 初始化一个全为1的张量，并将其除以 L_V，形成形状为 [B, H, L_V, L_V]
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            # 将计算得到的注意力权重 attn 赋值到 attns 的对应位置
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward_prob(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype("int").item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype("int").item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # 加入缩放系数
        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # 初始化上下文向量
        context = self._get_initial_context(values, L_Q)
        # 更新上下文向量
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        return context.transpose(2, 1).contiguous(), attn

    def forward(self, input_tensor: Tensor, attention_mask: Tensor):
        # * shape: [batch_size, seq_len, hidden_size]
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        # * shape: [batch_size, heads, seq_len, head_size]
        query_layer = self.transpose_multihead(mixed_query_layer)
        key_layer = self.transpose_multihead(mixed_key_layer)
        value_layer = self.transpose_multihead(mixed_value_layer)

        # * Take the dot product between "query" and "key" to get the raw attention scores.
        # * formula: Q^T * K / sqrt(d_k)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # * Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # * [batch_size heads seq_len seq_len] scores
        # * [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # * Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # * This is actually dropping out entire tokens to attend to, which might
        # * seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)

        # * formula: dropout(softmax(Q^T * K)) * V
        context_layer = torch.matmul(attention_probs, value_layer)

        # * shape: [batch_size, seq_len, heads, head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # * shape: [batch_size, seq_len, hidden_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # * Linear + dropout + residual + layer norm
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.RMSNorm(hidden_states + input_tensor)

        return hidden_states

    def transpose_multihead(self, x):
        """Transpose to multi head attention shape.

        e.g., [256, 50, 64] -> [256, 50, 2, 32] -> [256, 2, 50, 32]
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)


class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        """
        创建一个形状为 (L, scores.shape[-1]) 的全为 True 的布尔张量
        .to(device): 将张量移动到指定的设备（CPU 或 GPU）
        .triu(1): 获取该张量的上三角部分（不包括主对角线），下三角部分设为 False
        """
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        """
        _mask[None, None, :]: 在前两个维度上增加两个维度，使其形状变为 (1, 1, L, scores.shape[-1])。
        .expand(B, H, L, scores.shape[-1]): 扩展张量，使其形状变为 (B, H, L, scores.shape[-1])
        """
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        """
        torch.arange(B)[:, None, None]: 创建一个形状为 (B, 1, 1) 的张量，用于索引批次维度。
        torch.arange(H)[None, :, None]: 创建一个形状为 (1, H, 1) 的张量，用于索引注意力头维度。
        index: 索引序列维度
        """
        indicator = _mask_ex[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

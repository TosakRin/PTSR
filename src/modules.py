# -*- coding:utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import copy
import math

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


class Embeddings(nn.Module):
    """No use. Construct the embeddings from item, position."""

    def __init__(self):
        super().__init__()

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)  # 不要乱用padding_idx
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        items_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = items_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    """Self-attention module."""

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
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        """Transpose to multi head attention shape.

        e.g. [256, 50, 64] -> [256, 50, 2, 32] -> [256, 2, 50, 32]
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor: Tensor, attention_mask: Tensor):
        """Self-attention:

        1. multi head attention
        2. softmax
        3. dropout
        4. weighted sum
        5. linear
        6. dropout
        7. residual connection
        8. layer norm
        9. return

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
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

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
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

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
        """Feed forward network.
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

    def __init__(self):
        super().__init__()
        layer = Layer()
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, output_all_encoded_layers=True) -> list[Tensor]:
        """

        Args:
            hidden_states (Tensor): [256, 50, 64] -> [batch_size, seq_len, hidden_size]
            attention_mask (Tensor): [256, 1, 50, 50] -> [batch_size, 1, seq_len, seq_len]
            output_all_encoded_layers (bool, optional): if True, return all layers. Else, return the last layer. Defaults to True.

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
        return F.leaky_relu(torch.spmm(adj, embeds), negative_slope=0.2)


class GATLayer(nn.Module):
    """
    这个GATLayer实现包括以下关键步骤：

    - 线性变换：首先对每个节点的特征应用一个线性变换（通过权重矩阵W）。
    - 注意力机制：然后，计算节点对之间的注意力系数。这是通过首先计算一个得分（使用参数a），然后应用LeakyReLU非线性激活函数来完成的。注意力系数是通过对这些得分应用softmax函数来归一化的，确保每个节点的邻居的注意力系数之和为1。
    - 特征更新：最后，使用注意力系数加权邻居节点的特征，来更新每个节点的特征。

    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_features, out_features, dropout_rate=0.6, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate
        self.alpha = alpha  # LeakyReLU的负斜率

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout_rate)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # 应用线性变换
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # 仅在邻接矩阵中有边的地方计算注意力
        attention = F.softmax(attention, dim=1)  # 对每个节点的邻居应用softmax
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)  # 更新节点特征

        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # 节点数

        Wh1 = Wh.repeat_interleave(N, dim=0)
        Wh2 = Wh.repeat(N, 1)

        # 拼接所有节点对的特征
        all_combinations_matrix = torch.cat([Wh1, Wh2], dim=1)

        # 应用一个共享的线性变换到每一对节点特征上，并应用LeakyReLU
        e = self.leakyrelu(torch.matmul(all_combinations_matrix, self.a).squeeze(1))

        return e.view(N, N)  # 转换回N x N矩阵，其中每个元素代表节点对的注意力得分

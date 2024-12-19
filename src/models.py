# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


import torch
from torch import Tensor, nn

from modules import Encoder, LayerNorm, LigthGCNLayer, NGCFLayer
from param import args


class SASRecModel(nn.Module):
    def __init__(self):
        super().__init__()
        # * args.item_size: max_item + 2, 0 for padding.
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.subseq_embeddings = nn.Embedding(args.num_subseq_id, args.hidden_size)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        # self.subseqs_embeddings = nn.Embedding(args.num_subseq_id, args.hidden_size)
        self.all_subseq_emb: Tensor = torch.zeros(args.num_subseq_id, args.hidden_size)
        self.all_item_emb: Tensor = torch.zeros(args.item_size, args.hidden_size)
        self.adagrad_params = [self.item_embeddings.weight]
        self.adam_params = [p for n, p in self.named_parameters() if n != "item_embeddings.weight"]
        self.prefix_encoder = Encoder(1)
        self.item_encoder = Encoder()
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.criterion = nn.BCELoss(reduction="none")  # todo: BCELoss
        self.apply(self.init_weights)

    def add_position_embedding(self, sequence: Tensor, item_embeddings):
        """

        1. Add position embedding
        2. LayerNorm
        3. dropout

        Args:
            sequence (Tensor): [256, 50] -> [batch_size, seq_length]

        Returns:
            Tensor: _description_
        """
        position_ids = torch.arange(sequence.size(1), dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def forward(self, input_ids: Tensor):
        # * item embedding branch
        if args.gcn_mode in ["batch", "batch_gcn", "global"]:
            item_embeddings = self.all_item_emb[input_ids]
        else:
            item_embeddings = self.get_item_embeddings(input_ids)
        extended_attention_mask = self.get_transformer_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids, item_embeddings)
        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)

        # * only use the last layer, SHAPE: [batch_size, seq_length, hidden_size]
        return item_encoded_layers[-1]

    def forward_p(self, input_ids: Tensor, sp):
        """extend task 的 forward pass. 主要改动:
        1. 使用 subseq_embeddings 作为 token embeddings
        2. 使用 prefix_encoder 作为 Transformer encoder
        3. 使用 sp 作为 prefix 的第一个 token,即item_encoder得到的结果
        4. 不使用 position embedding + 取消 mask 中的对角线 mask

        Args:
            input_ids (Tensor): 已经 Padding 完成的输入序列,注意其中元素是图中的 subseq_id 而非 item_id. SHAPE: [batch_size, seq_length], e.g. [256, 50]
            sp (tensor): prefix 的第一个 token, 即item_encoder得到的结果, SHAPE: [batch_size, hidden_size], e.g. [256, 64]

        Returns:
            tensor: TRM 最后一层的输出. SHAPE: [batch_size, seq_length, hidden_size], e.g. [256, 50, 64]
        """
        prefix_embeddings = self.subseq_embeddings(input_ids)
        batch_size, seq_length = input_ids.shape
        first_zero_indices = (input_ids == 0).int().argmax(dim=1)
        mask = torch.arange(seq_length, device=input_ids.device).expand(
            batch_size, seq_length
        ).cuda() == first_zero_indices.unsqueeze(-1)
        mask = mask.unsqueeze(-1).expand_as(prefix_embeddings)
        prefix_embeddings[mask] = sp.unsqueeze(1).expand(-1, seq_length, -1)[mask]

        extended_attention_mask = self.get_extend_mask(input_ids)
        # prefix_embeddings = self.add_position_embedding(input_ids, prefix_embeddings)
        # prefix_embeddings = self.dropout(self.LayerNorm(prefix_embeddings))

        item_encoded_layers = self.prefix_encoder(
            prefix_embeddings, extended_attention_mask, output_all_encoded_layers=True
        )
        return item_encoded_layers[-1]

    def get_pad_mask(self, input_ids: Tensor):
        """不带对角线的 pad mask. 即无位置关系的 self-attention.

        Args:
            input_ids (Tensor): 已经 Padding 完成的输入序列. SHAPE: [batch_size, seq_length], e.g. [256, 50]

        Returns:
            tensor: pad mask. SHAPE: [batch_size, 1, 1, seq_length], e.g. [256, 1, 1, 50]
        """
        # * Shape: [batch_size, seq_length]
        pad_mask = (input_ids > 0).long()
        pad_mask: Tensor = pad_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len: int = pad_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        pad_mask = pad_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        pad_mask = (1.0 - pad_mask) * -10000.0
        return pad_mask

    def get_extend_mask(self, input_ids: Tensor):
        # * Shape: [batch_size, seq_length]
        batch_size, seq_length = input_ids.shape

        # 创建基础的attention mask，其中非零元素表示有效位置
        attention_mask = (input_ids > 0).long()

        # 找到每个序列中第一个0出现的位置
        first_zero_indices = (input_ids == 0).int().argmax(dim=1)
        # 如果没有0，则设置为seq_length，这样就不会影响mask
        first_zero_indices[first_zero_indices == 0] = seq_length

        # 根据第一个0的位置更新attention mask
        for i in range(batch_size):
            if first_zero_indices[i] < seq_length:
                attention_mask[i, first_zero_indices[i] :] = 0  # mask掉从第一个0开始及其之后的所有位置

        # * Shape: [batch_size, 1, 1, seq_length]
        extended_attention_mask: Tensor = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len: int = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)

        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        # * shape: [1, 1, seq_length, seq_length]
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        # * Hadamand product and boardcast
        # * mask SHAPE: [batch_size, 1, seq_length, seq_length]
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_item_embeddings(self, input_ids: Tensor):
        if args.gcn_mode != "None":
            gcn_embeddings = self.gcn_embeddings(input_ids)
            item_embeddings = self.item_embeddings(input_ids)
            item_embeddings += gcn_embeddings
        else:
            item_embeddings = self.item_embeddings(input_ids)
        return item_embeddings

    def get_transformer_mask(self, input_ids: Tensor):
        # * Shape: [batch_size, seq_length]
        attention_mask = (input_ids > 0).long()
        # * Shape: [batch_size, 1, 1, seq_length]
        extended_attention_mask: Tensor = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len: int = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)

        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        # * shape: [1, 1, seq_length, seq_length]
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        # * Hadamand product and boardcast
        # * mask SHAPE: [batch_size, 1, seq_length, seq_length]
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def predict_full(self, user_seq_emb: Tensor):
        """Predict: Rating = User_seq_emb * Item_emb^T.

        Args:
            user_seq_emb (Tensor): User sequence output. Use the last item output of last layer. SHAPE: [batch_size, hidden_size] -> [256, 64]

        Returns:
            Tensor: Rating prediction. SHAPE: [batch_size, item_size]
        """
        # * SHAPE: [Item_size, Hidden_size]
        if args.gcn_mode in ["batch", "batch_gcn", "global"]:
            test_item_emb = self.all_item_emb
        elif args.gcn_mode == "None":
            test_item_emb = self.item_embeddings.weight
        else:
            raise ValueError(f"Invalid gcn_mode: {args.gcn_mode}")
        return torch.matmul(user_seq_emb, test_item_emb.transpose(0, 1))


class GCN(nn.Module):
    """LightGCN model."""

    def __init__(self):
        super().__init__()
        # * LightGCN
        if args.gcn_mode in ["batch", "global"]:
            self.gcn_layers = nn.Sequential(*[LigthGCNLayer() for _ in range(args.gnn_layer)])
        elif args.gcn_mode in ["batch_gcn"]:
            self.gcn_layers = nn.Sequential(*[NGCFLayer() for _ in range(args.gnn_layer)])

    def forward(self, adj, subseq_emb: Tensor, target_emb: Tensor):
        """Forward pass of the GCN model.

        Args:
            adj (_type_): D^(-1/2) * A * D^(-1/2)
            subseq (Tensor): subseq item embedding (Transformer output)
            target (Tensor): target item embedding.

        Returns:
            tuple(Tensor, Tensor): aggregation of subseq and target item embeddings.
        """
        ini_emb = torch.concat([subseq_emb, target_emb], axis=0)
        layers_gcn_emb_list = [ini_emb]
        for gcn in self.gcn_layers:
            # * layers_gcn_emb_list[-1]: use the last layer's output as input
            gcn_emb = gcn(adj, layers_gcn_emb_list[-1])
            layers_gcn_emb_list.append(gcn_emb)
        sum_emb = sum(layers_gcn_emb_list) / len(layers_gcn_emb_list)
        return sum_emb[: args.num_subseq_id], sum_emb[args.num_subseq_id :]

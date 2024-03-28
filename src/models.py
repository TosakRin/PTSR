# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from typing import Union

import faiss  # type: ignore
import numpy as np
import torch
from torch import Tensor, nn

from cprint import pprint_color
from modules import Encoder, LayerNorm
from param import args


class KMeans:
    centroids: Tensor

    def __init__(
        self, num_cluster: int, seed: int, hidden_size: int, gpu_id: int = 0, device: Union[torch.device, str] = "cpu"
    ):
        """KMeans clustering.

        Args:
            num_cluster (int): number of clusters
            seed (int): random seed
            hidden_size (int): hidden size of embedding
        """
        pprint_color(">>> Initialize KMeans Clustering")
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = 0
        self.device = device
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)

    def __init_cluster(
        self,
        hidden_size: int,
        verbose: bool = False,
        niter: int = 20,
        nredo: int = 5,
        max_points_per_centroid: int = 4096,
        min_points_per_centroid: int = 0,
    ):
        """Initialize the clustering.

        Args:
            hidden_size (int): hidden size of embedding
            verbose (bool, optional): verbose during training?
            niter (int, optional): clustering iterations. Defaults to 20.
            nredo (int, optional): redo clustering this many times and keep best. Defaults to 5.
            max_points_per_centroid (int, optional): to limit size of dataset. Defaults to 4096.
            min_points_per_centroid (int, optional): otherwise you get a warning. Defaults to 0.

        Returns:
            tuple[Clustering, GpuIndexFlatL2]: clustering and index
        """
        pprint_color(
            f">>> cluster train iterations: {niter}",
        )
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose
        clus.niter = niter
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        res = faiss.StandardGpuResources()
        res.noTempMemory()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = self.gpu_id
        index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
        pprint_color(f">>> FAISS Device: {faiss.get_num_gpus()}")
        return clus, index

    def train(self, x: np.ndarray):
        """train to get centroids. Save to `self.centroids`.

        Args:
            x (np.ndarray): [131413, 64] -> [subseq_num, hidden_size]
        """
        if x.shape[0] > self.num_cluster:
            self.clus.train(x, self.index)
        # * get cluster centroids. Shape: [num_cluster, hidden_size] -> [256, 64]
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
        # * convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)

    def query(self, x):
        # self.index.add(x)
        D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
        seq2cluster = [int(n[0]) for n in I]
        # pprint_color("cluster number:", self.num_cluster,"cluster in batch:", len(set(seq2cluster)))
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        return seq2cluster, self.centroids[seq2cluster]


class SASRecModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.item_embeddings = nn.Embedding(
            num_embeddings=args.item_size, embedding_dim=args.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(num_embeddings=args.max_seq_length, embedding_dim=args.hidden_size)
        self.item_encoder = Encoder()
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    def add_position_embedding(self, sequence: Tensor):
        """Add positional embeddings to item embeddings.

        Args:
            sequence (Tensor): [256, 50] -> [batch_size, seq_length]

        Returns:
            _type_: _description_
        """
        seq_length: int = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)

        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def forward(self, input_ids: Tensor):

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
        # * shape: [batch_size, 1, seq_length, seq_length]
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb: Tensor = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)

        # * only use the last layer, SHAPE: [batch_size, seq_length, hidden_size]
        return item_encoded_layers[-1]

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


# GRU Encoder
class GRUEncoder(nn.Module):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self):
        super().__init__()

        # load parameters info
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)

        self.embedding_size = args.hidden_size  # 64
        self.hidden_size = args.hidden_size * 2  # 128
        self.num_layers = args.num_hidden_layers - 1  # 1
        self.dropout_prob = args.hidden_dropout_prob  # 0.3

        # define layers and loss
        self.emb_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)

    def forward(self, item_seq):
        item_seq_emb = self.item_embeddings(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        # seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        seq_output = gru_output
        return seq_output


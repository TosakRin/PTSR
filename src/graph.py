"""
/@
 @Author       : TosakRin sick_person@outlook.com
 @Date         : 2024-03-26 13:33:18
 @LastEditors  : TosakRin sick_person@outlook.com
 @LastEditTime : 2024-03-28 16:03:41
 @FilePath     : /ICSRec/src/graph.py
 @Description  :
 @/
"""

import os
import pickle

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import coo_matrix  # type: ignore

from cprint import pprint_color
from datasets import TargetSubseqs
from param import args
from utils import get_max_item


class Graph:
    def __init__(
        self,
        adj_path,
    ):
        self.adj_path = adj_path
        if not os.path.exists(self.adj_path):
            raise FileNotFoundError(f'adjacency matrix not found in "{self.adj_path}"')
        else:
            self.load_graph()

    def norm_adj(self, mat: sp.csr_matrix) -> sp.coo_matrix:
        """
        normalize adjacency matrix: D^-0.5 * A^ * D^-0.5`

        Args:
            mat (sp.csr_matrix): A^ (A^ = A + I)

        Returns:
            coo_matrix: D^-0.5 * A^ * D^-0.5
        """
        # * equal to axis=1. sum of each row and finnaly get a column vector.
        degree = np.array(mat.sum(axis=-1))
        # * ^0.5 and squeeze to 1 dimension.
        d_sqrt_inv = np.reshape(np.power(degree, -0.5), [-1])
        d_sqrt_inv[np.isinf(d_sqrt_inv)] = 0.0
        # * D^-0.5
        d_sqrt_inv_mat = sp.diags(d_sqrt_inv)
        # * D^-0.5 * A^ * D^-0.5
        return mat.dot(d_sqrt_inv_mat).transpose().dot(d_sqrt_inv_mat).tocoo()

    def get_torch_adj(self, mat: sp.csr_matrix) -> torch.Tensor:
        """
        1. transfer A to D^-0.5 * A^ * D^-0.5.
        2. make torch sparse tensor from scipy sparse matrix.

        Args:
            mat (sp.csr_matrix): raw input item-item transition matrix.

        Returns:
            torch.sparse.FloatTensor: D^-0.5 * A^ * D^-0.5 in torch sparse tensor format.
        """
        # make ui adj
        a = sp.csr_matrix((mat.shape[0], mat.shape[0]))
        b = sp.csr_matrix((mat.shape[1], mat.shape[1]))
        # * A = [0, A; A^T, 0]
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        # * A^ = A + I
        # todo: https://github.com/gusye1234/LightGCN-PyTorch/blob/master/code/dataloader.py
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        # * D^-0.5 * A^ * D^-0.5
        mat = self.norm_adj(mat)

        # * `sp.coo_matrix` to `torch.sparse.FloatTensor`
        # * (2, nnz), index of non-zero elements.
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse_coo_tensor(idxs, vals, shape).cuda()

    def load_graph(self):
        with open(self.adj_path, "rb") as fs:
            train_matrix = (pickle.load(fs) != 0).astype(np.float32)
        if not isinstance(train_matrix, coo_matrix):
            train_matrix = sp.coo_matrix(train_matrix)

        self.train_matrix = train_matrix
        args.user, args.item = train_matrix.shape
        self.torch_A = self.get_torch_adj(train_matrix)

    @staticmethod
    def build_graph(
        target_subseqs_dict: dict[int, list[list[int]]],
        subseq_id_map: dict[tuple[int], int],
        num_items: int,
        num_subseqs: int,
    ):
        """graph is a sparse matrix, shape: [num_subseqs, num_items].

        Args:
            subseq_target_set (dict[str, dict[int, list[list[int]]]]): _description_
            subseq_id_map (dict[tuple[int], int]): _description_
            num_items (int): _description_
            num_subseqs (int): _description_

        Returns:
            _type_: _description_
        """

        pprint_color(f"==>> num_items: {num_items}")
        pprint_color(f"==>> num_subseqs: {num_subseqs}")
        target_item_list = []
        sub_seq_list = []
        rating_list = []

        for target_item, subseqs in target_subseqs_dict.items():
            # pprint_color(f"==>> target_item: {target_item}")
            # pprint_color(f"==>> subseqs: {subseqs}")

            for subseq in subseqs:
                # pprint_color(f"==>> subseq: {subseq}")
                # pprint_color(f"==>> tuple(subseq[1:]): {tuple(subseq[1:])}")
                # pprint_color(f"==>> subseq_id_map[subseq[1:]]: {subseq_id_map[tuple(subseq[1:])]}")
                target_item_list.append(target_item)
                sub_seq_list.append(subseq_id_map[tuple(subseq[1:])])
                rating_list.append(1)

        target_item_array = np.array(target_item_list)
        subseq_array = np.array(sub_seq_list)
        rating_array = np.array(rating_list)
        pprint_color(f"==>> max target id: {np.max(target_item_array)}")
        pprint_color(f"==>> max subseq id: {np.max(subseq_array)}")

        return coo_matrix((rating_array, (subseq_array, target_item_array)), (num_subseqs, num_items + 1))

    @staticmethod
    def print_sparse_matrix_info(graph):
        # * nnz is the number of non-zero entries in the matrix.
        pprint_color(f"==>> graph.nnz: {graph.nnz}")
        pprint_color(f"==>> graph.shape: {graph.shape}")
        pprint_color(f"==>> graph.max(): {graph.max()}")
        pprint_color(f"==>> graph.min(): {graph.min()}")
        pprint_color(f"==>> graph.sum(): {graph.sum()}")
        pprint_color(f"==>> graph.data.shape: {graph.data.shape}")
        pprint_color(f"==>> graph.row.shape: {graph.row.shape}")
        pprint_color(f"==>> graph.col.shape: {graph.col.shape}")

    @staticmethod
    def save_sparse_matrix(save_path, graph):
        with open(save_path, "wb") as f:
            pickle.dump(graph, f)
            pprint_color(f">>> save graph to {save_path}")


if __name__ == "__main__":
    force_flag = True
    dataset_list = [
        "Beauty",
        "ml-1m",
        "Sports_and_Outdoors",
        "Toys_and_Games",
    ]
    for dataset in dataset_list:
        target_subseqs_dict_path = f"../data/{dataset}_1_t.pkl"
        subseqs_path = f"../data/{dataset}_1.txt"
        seqs_path = f"../data/{dataset}.txt"
        sparse_matrix_path = f"../data/{dataset}_graph.pkl"
        if os.path.exists(sparse_matrix_path) and not force_flag:
            pprint_color(f'>>> "{sparse_matrix_path}" exists, skip.')
            continue

        target_subseqs_dict = TargetSubseqs.load_target_subseqs_dict(target_subseqs_dict_path)
        subseq_id_map, id_subseq_map = TargetSubseqs.get_subseq_id_map(subseqs_path)
        max_item = get_max_item(seqs_path)
        graph = Graph.build_graph(target_subseqs_dict, subseq_id_map, max_item + 1, len(subseq_id_map))
        Graph.save_sparse_matrix(sparse_matrix_path, graph)

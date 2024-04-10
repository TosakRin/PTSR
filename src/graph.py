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
from utils import get_max_item


class TargetSubseqs:
    """the minimal length of the subsequence is 4, which can generate the target item for train (-3), valid (-2), and test (-1).

    # * ATTENTION: the subseq[0] is User_ID, and the subseq[1:] is the real subsequence.
    """

    target_subseqs_dict: dict[int, list[list[int]]]

    def __init__(self, data_dir: str, data_name: str, save_path: str) -> None:
        """

        Args:
            data_path (str): data dir, e.g., ../data
            data_name (str): dataset name, e.g., Beauty
            save_path (str): save dir, e.g., ../data
        """
        self.subseqs_path = f"{data_dir}/{data_name}_1.txt"
        self.target_subseqs_path = f"{save_path}/{data_name}_1_t.pkl"

    def generate_target_subseqs_dict(self) -> None:
        """Generate the target item for each subsequence, and save to pkl file."""
        # * data_f is the subsequences file
        train_dic: dict[int, list[list[int]]] = {}
        valid_dic: dict[int, list[list[int]]] = {}
        test_dic: dict[int, list[list[int]]] = {}
        with open(self.subseqs_path, "r", encoding="utf-8") as fr:
            subseq_list: list[str] = fr.readlines()
            for subseq in subseq_list:
                items: list[str] = subseq.split(" ")
                # * items[0] is User ID
                tag_train = int(items[-3])
                tag_valid = int(items[-2])
                tag_test = int(items[-1])
                train_temp: list[int] = list(map(int, items[:-3]))
                valid_temp: list[int] = list(map(int, items[:-2]))
                test_temp: list[int] = list(map(int, items[:-1]))
                if tag_train not in train_dic:
                    train_dic.setdefault(tag_train, [])
                train_dic[tag_train].append(train_temp)
                if tag_valid not in valid_dic:
                    valid_dic.setdefault(tag_valid, [])
                valid_dic[tag_valid].append(valid_temp)
                if tag_test not in test_dic:
                    test_dic.setdefault(tag_test, [])
                test_dic[tag_test].append(test_temp)

        total_dic: dict[str, dict[int, list[list[int]]]] = {"train": train_dic, "valid": valid_dic, "test": test_dic}
        pprint_color(f'>>> Saving target-item specific subsequence set to "{self.target_subseqs_path}"')
        with open(self.target_subseqs_path, "wb") as fw:
            pickle.dump(total_dic, fw)

    def _load_target_subseqs_dict(self, target_subseqs_path: str, mode="train"):
        """get the prefix subsequence set (dict).

        Args:
            data_path (str): pkl file path. Subseq in pkl file contains User_ID (subseq[0]) and the real subsequence (subseq[1:].
            mode (str, optional): target item set type. Only use "train" in this paper. Defaults to "train".

        Returns:
            _type_: _description_
        """
        if not target_subseqs_path:
            raise ValueError("invalid data path")
        if not os.path.exists(target_subseqs_path):
            pprint_color("The dict not exist, generating...")
            self.generate_target_subseqs_dict()
        with open(target_subseqs_path, "rb") as read_file:
            data_dict: dict[str, dict[int, list[list[int]]]] = pickle.load(read_file)
        self.target_subseqs_dict = data_dict[mode]
        return data_dict[mode]

    @staticmethod
    def load_target_subseqs_dict(target_subseqs_path: str, mode="train"):
        """get the prefix subsequence set (dict).

        Args:
            data_path (str): pkl file path. Subseq in pkl file contains User_ID (subseq[0]) and the real subsequence (subseq[1:].
            mode (str, optional): target item set type. Only use "train" in this paper. Defaults to "train".

        Returns:
            _type_: _description_
        """
        if not target_subseqs_path:
            raise ValueError("invalid data path")
        if not os.path.exists(target_subseqs_path):
            pprint_color("The dict not exist, generating...")
            self.generate_target_subseqs_dict()
        with open(target_subseqs_path, "rb") as read_file:
            data_dict: dict[str, dict[int, list[list[int]]]] = pickle.load(read_file)
        return data_dict[mode]

    @staticmethod
    def print_target_subseqs(target_subseqs_dict, target_id):
        """print the subsequence list for the given target"""
        subseq_list = target_subseqs_dict[target_id]
        pprint_color(f">>> subseq number: {len(subseq_list)}")
        pprint_color(subseq_list)

    def find_overlapping_target_items_with_count(self) -> dict[tuple[int], int]:
        """find all pairs of target items that have overlapping subseqs and count the number of overlapping subseqs for each pair."""
        subseqs_set = {}
        overlapping_pairs_with_count = {}

        for target_item, subseqs in self.target_subseqs_dict.items():
            for user_subseq in subseqs:
                # * Convert the inner list to a tuple for hashing, since lists are not hashable
                # * Calculate a hash for the subseq, drop user_id (user_subseq[0])
                subseq_hash = hash(tuple(user_subseq[1:]))
                subseq_len = len(user_subseq[1:])

                if subseq_hash in subseqs_set:
                    # * For each previously stored key with the same hash, add or update a pair with its count
                    for previous_target_item in subseqs_set[subseq_hash]:
                        # * skip if the pair is the same item which means a target item has multiple identical subseqs.
                        if target_item == previous_target_item:
                            continue
                        # * Ensure the pair is ordered to avoid duplicates like (1,2) and (2,1)
                        ordered_target_item_pair = tuple(sorted([target_item, previous_target_item]))
                        if ordered_target_item_pair in overlapping_pairs_with_count:
                            overlapping_pairs_with_count[ordered_target_item_pair]["count"] += 1
                            overlapping_pairs_with_count[ordered_target_item_pair]["subseq_len"].append(subseq_len)
                        else:
                            overlapping_pairs_with_count[ordered_target_item_pair] = {
                                "count": 1,
                                "subseq_len": [subseq_len],
                            }
                    # Add the current key to the list of keys for this hash value
                    subseqs_set[subseq_hash].append(target_item)
                else:
                    subseqs_set[subseq_hash] = [target_item]

        return overlapping_pairs_with_count

    def find_same_subseq_for_target(self, target_1, target_2):
        # Extract subsequences for each target, excluding the first element of each subsequence
        target_1_subseqs = {tuple(subseq[1:]) for subseq in self.target_subseqs_dict[target_1]}
        target_2_subseqs = {tuple(subseq[1:]) for subseq in self.target_subseqs_dict[target_2]}

        # Find the intersection of subsequences between target_1 and target_2
        overlapping_subseqs = target_1_subseqs.intersection(target_2_subseqs)
        pprint_color(f"==>> overlapping_subseqs: {overlapping_subseqs}")

        # Return the overlapping subsequences
        return overlapping_subseqs

    @staticmethod
    def print_subseq_map_info(num_subseqs, subseq_id_map, id_subseq_map):
        pprint_color(f"==>> num_subseqs{' '*9}: {num_subseqs:>6}")
        pprint_color(f"==>> num_hashmap{' '*9}: {len(subseq_id_map):>6}")
        pprint_color(f"==>> duplicate subseq num: {num_subseqs - len(subseq_id_map):>6}")
        pprint_color(f"==>> subseq to id hashmap exapmle: {list(subseq_id_map.items())[:10]}")
        pprint_color(f"==>> id to subseq hashmap exapmle: {list(id_subseq_map.items())[:10]}")

    @staticmethod
    def get_subseq_id_map(subseqs_file: str) -> tuple[dict[tuple[int], int], dict[int, tuple[int]]]:
        """get subseq <-> id hashmap.

        We use input subseq during training to get the corresponding id.

        e.g., the compele subseq is [1, 2, 3, 4] while during training. We only use [1] as input subseq and 2 as target item.
        So we only need to store the id of [1] rather than the complete subseq.

        Args:
            subseqs_file (str): subseqs file path. e.g., ../data/Beauty_1.txt

        Returns:
            tuple[dict[tuple[int], int], dict[int, tuple[int]]]: num_subseqs, subseq_id_map, id_subseq_map
        """
        subseq_id_map: dict[tuple[int], int] = {}
        id_subseq_map: dict[int, tuple[int]] = {}
        num_subseqs = 0
        i = 0
        with open(subseqs_file, encoding="utf-8") as f:
            for index, line in enumerate(f):
                subseq: tuple[int] = tuple(map(int, line.strip().split(" ")[1:-3]))
                if subseq not in subseq_id_map:
                    subseq_id_map.setdefault(subseq, i)
                    id_subseq_map.setdefault(i, subseq)
                    i += 1
                num_subseqs = index
        TargetSubseqs.print_subseq_map_info(num_subseqs, subseq_id_map, id_subseq_map)

        return subseq_id_map, id_subseq_map


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

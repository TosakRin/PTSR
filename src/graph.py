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
from ast import literal_eval

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

    def __init__(self, subseqs_path: str, target_subseqs_path: str, subseqs_target_path: str) -> None:
        """

        Args:
            data_path (str): data dir, e.g., ../data
            data_name (str): dataset name, e.g., Beauty
            save_path (str): save dir, e.g., ../data
        """
        self.subseqs_path = subseqs_path
        self.target_subseqs_path = target_subseqs_path
        self.subseqs_target_path = subseqs_target_path

    @staticmethod
    def generate_target_subseqs_dict(subseqs_path, target_subseqs_path) -> None:
        """Generate the target item for each subsequence, and save to pkl file."""
        # * data_f is the subsequences file
        train_dic: dict[int, list[list[int]]] = {}
        valid_dic: dict[int, list[list[int]]] = {}
        test_dic: dict[int, list[list[int]]] = {}
        with open(subseqs_path, "r", encoding="utf-8") as fr:
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
        pprint_color(f'>>> Saving target-item specific subsequence set to "{target_subseqs_path}"')
        with open(target_subseqs_path, "wb") as fw:
            pickle.dump(total_dic, fw)

    def generate_subseqs_target_dict(self):
        """subseqs_target_dict: dict[str, dict[tuple[int], list[int]]].

        * Remember:

        - subseqs_target_dict: the subseq (aka, key) is the real subsequence.
        - target_subseqs_dict: the subseq[1:] (aka, value) is the real subsequence. The subseq[0] is User_ID.
        """
        # * data_f is the subsequences file
        train_dict: dict[tuple[int], list[int]] = {}
        valid_dict: dict[tuple[int], list[int]] = {}
        test_dict: dict[tuple[int], list[int]] = {}
        with open(self.subseqs_path, "r", encoding="utf-8") as fr:
            subseq_list: list[str] = fr.readlines()
            for subseq in subseq_list:
                items: list[str] = subseq.split(" ")
                target_train = int(items[-3])
                target_valid = int(items[-2])
                target_test = int(items[-1])
                # * items[0] is User ID
                subseq_train: tuple[int] = tuple(map(int, items[1:-3]))
                subseq_valid: tuple[int] = tuple(map(int, items[1:-2]))
                subseq_test: tuple[int] = tuple(map(int, items[1:-1]))
                if subseq_train not in train_dict:
                    train_dict.setdefault(subseq_train, [])
                train_dict[subseq_train].append(target_train)
                if subseq_valid not in valid_dict:
                    valid_dict.setdefault(subseq_valid, [])
                valid_dict[subseq_valid].append(target_valid)
                if subseq_test not in test_dict:
                    test_dict.setdefault(subseq_test, [])
                test_dict[subseq_test].append(target_test)

        total_dic: dict[str, dict[tuple[int], list[int]]] = {
            "train": train_dict,
            "valid": valid_dict,
            "test": test_dict,
        }
        pprint_color(f'>>> Saving subsequence specific target-item set to "{self.subseqs_target_path}"')
        with open(self.subseqs_target_path, "wb") as f:
            pickle.dump(total_dic, f)
        return total_dic

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
            self.generate_target_subseqs_dict(self.subseqs_path, self.target_subseqs_path)
        with open(target_subseqs_path, "rb") as read_file:
            data_dict: dict[str, dict[int, list[list[int]]]] = pickle.load(read_file)
        self.target_subseqs_dict = data_dict[mode]
        return data_dict[mode]

    @staticmethod
    def load_target_subseqs_dict(subseqs_path, target_subseqs_path: str, mode="train"):
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
            TargetSubseqs.generate_target_subseqs_dict(subseqs_path, target_subseqs_path)
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
        pprint_color(f"有相同 Target Item 的 Subseq 数: {np.sum(graph.tocsr().sum(axis=1)>1)}")
        pprint_color(f"有相同 Subseq 的 Target Item 数: {np.sum(graph.tocsc().sum(axis=0)>1)}")

    @staticmethod
    def save_sparse_matrix(save_path, graph):
        with open(save_path, "wb") as f:
            pickle.dump(graph, f)
            pprint_color(f">>> save graph to {save_path}")


def DS(i_file: str, o_file: str, max_len: int = 50) -> None:
    """Dynamic Segmentation operations to generate subsequence.

    子序列基本逻辑: 做一个从长度 4 开始的窗口, 当窗口长度不满 53 时, 向右增长滑动窗口的大小, 当窗口长度到达 53 后, 长度不变, start, end 每次向右滑动, 直到 end 到达原序列末尾.

    1. 序列长度小于等于 `max_save_len`, 以 `[start, end+1]` 生成最小子序列, 不断增加 `end`, 直到序列结束.
    2. 序列长度大于 `max_save_len`:
        2.1 `start < 1`, 以 `[start, end]` 生成最小子序列, 不断增加 `end`, 直到 `end` 到达序列末尾, 或者 `end - start < max_len`
        2.2 `start >= 1`, 以 `[start, start+max_len]` 生成子序列, 不断增加 `start`, 直到 start 到达序列长度 - max_save_len

    对于一个长度为 n (n>max_save_len) 的序列:

    - 2.1 生成的子序列个数为 max_save_len - end, 比如 n=85, max_save_len=53, end=4, 生成的子序列个数为 53 - 4 = 49. 这 49 个子序列的开始都为 0, 结束为 3, 4, 5, ..., 51. 长度为 4, 5, 6, ..., 52.
    - 2.2 生成的子序列个数为 n - max_keep_len, 比如 n=85, max_keep_len=52, 生成的子序列个数为 33. 这 33 个子序列的开始为 0, 1, 2, 3, ..., 32, 结束为 52, 53, 54, ..., 84. 长度都为 53.

    Args:
        i_file (str): input file path
        o_file (str): output file path
        max_len (int): the max length of the sequence
    """
    pprint_color(">>> Using DS to generate subsequence ...")
    with open(i_file, "r+", encoding="utf-8") as fr:
        seq_list = fr.readlines()
    subseq_dict: dict[str, list] = {}
    max_save_len = max_len + 3
    max_keep_len = max_len + 2
    for data in seq_list:
        u_i, seq_str = data.split(" ", 1)
        seq = seq_str.split(" ")
        # ? str -> int -> str
        seq[-1] = str(literal_eval(seq[-1]))
        subseq_dict.setdefault(u_i, [])
        start = 0
        # minimal subsequence length
        end = 3
        if len(seq) > max_save_len:
            while start < len(seq) - max_keep_len:
                end = start + 4
                while end < len(seq):
                    if start < 1 and end - start < max_save_len:
                        subseq_dict[u_i].append(seq[start:end])
                        end += 1
                    else:
                        subseq_dict[u_i].append(seq[start : start + max_save_len])
                        break
                start += 1
        else:
            while end < len(seq):
                subseq_dict[u_i].append(seq[start : end + 1])
                end += 1

    with open(o_file, "w+", encoding="utf-8") as fw:
        for u_i, subseq_list in subseq_dict.items():
            for subseq in subseq_list:
                fw.write(f"{u_i} {' '.join(subseq)}\n")
    pprint_color(f">>> DS done, written to {o_file}")


if __name__ == "__main__":
    pprint_color(">>> subsequences and graph generation pipeline")
    pprint_color(">>> Start to generate subsequence and build graph ...")

    force_flag = True
    dataset_list = [
        "Beauty",
        "ml-1m",
        "Sports_and_Outdoors",
        "Toys_and_Games",
    ]
    for dataset in dataset_list:
        data_root = "../subseq"
        if not os.path.exists(data_root):
            os.makedirs(data_root)
        max_len = 1
        seqs_path = f"../data/{dataset}.txt"
        subseqs_path = f"{data_root}/{dataset}_subseq_{max_len}.txt"
        target_subseqs_dict_path = f"{data_root}/{dataset}_t_{max_len}.pkl"
        sparse_matrix_path = f"{data_root}/{dataset}_graph_{max_len}.pkl"

        subseqs_path = f"{data_root}/{dataset}_subseq_merged.txt"
        target_subseqs_dict_path = f"{data_root}/{dataset}_t_merged.pkl"
        sparse_matrix_path = f"{data_root}/{dataset}_graph_merged.pkl"

        if os.path.exists(sparse_matrix_path) and not force_flag:
            pprint_color(f'>>> "{sparse_matrix_path}" exists, skip.')
            continue

        # * 1. generate subseqs from original seqs file
        DS(seqs_path, subseqs_path, max_len)
        # * 2. generate Target-Subseqs Dict from subseqs file
        target_subseqs_dict = TargetSubseqs.load_target_subseqs_dict(subseqs_path, target_subseqs_dict_path)
        # * 3. generate subseq id map from subseqs file
        subseq_id_map, _ = TargetSubseqs.get_subseq_id_map(subseqs_path)
        max_item = get_max_item(seqs_path)
        # * 4. build graph from target subseqs dict
        graph = Graph.build_graph(target_subseqs_dict, subseq_id_map, max_item + 1, len(subseq_id_map))
        Graph.save_sparse_matrix(sparse_matrix_path, graph)

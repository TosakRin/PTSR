"""
/@
 @Author       : TosakRin sick_person@outlook.com
 @Date         : 2024-03-18 15:52:01
 @LastEditors  : TosakRin sick_person@outlook.com
 @LastEditTime : 2024-03-23 20:20:39
 @FilePath     : /ICSRec/src/datasets.py
 @Description  : Dataset: Including subsequence generation (DS), target-item | subsequence set generation (Generate_tag), and dataset class (RecWithContrastiveLearningDataset).
 @/
"""

# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


import copy
import os
import pickle
import random
from ast import literal_eval

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from cprint import pprint_color
from param import args


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

    def find_overlapping_target_items_with_count(self):
        """find all pairs of target items that have overlapping subseqs and count the number of overlapping subseqs for each pair."""
        # This dictionary will map the hash of each inner list to its first key occurrence
        hash_map = {}

        # This dictionary will store the unique pairs of overlapping keys along with their counts
        overlapping_pairs_with_count = {}

        for target_item, subseqs in self.target_subseqs_dict.items():
            for user_subseq in subseqs:
                # * Convert the inner list to a tuple for hashing, since lists are not hashable
                # * Calculate a hash for the subseq, drop user_id (user_subseq[0])
                subseq_hash = hash(tuple(user_subseq[1:]))

                if subseq_hash in hash_map:
                    # * For each previously stored key with the same hash, add or update a pair with its count
                    for previous_target_item in hash_map[subseq_hash]:
                        # * skip if the pair is the same item which means a target item has multiple identical subseqs.
                        if target_item == previous_target_item:
                            continue
                        # * Ensure the pair is ordered to avoid duplicates like (1,2) and (2,1)
                        ordered_pair = tuple(sorted([target_item, previous_target_item]))
                        if ordered_pair in overlapping_pairs_with_count:
                            overlapping_pairs_with_count[ordered_pair] += 1
                        else:
                            overlapping_pairs_with_count[ordered_pair] = 1
                    # Add the current key to the list of keys for this hash value
                    hash_map[subseq_hash].append(target_item)
                else:
                    hash_map[subseq_hash] = [target_item]

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


class RecWithContrastiveLearningDataset(Dataset):
    def __init__(
        self,
        user_seq: list[list[int]],
        test_neg_items=None,
        data_type: str = "train",
    ) -> None:
        """torch.utils.dataDataset

        Args:
            user_seq (list[list[int]]): subseq list in the training phase and original sequence in the validation and testing phase. *Not including User_ID*.
            test_neg_items (_type_, optional): negative sample in test for sample based ranking. This paper use other sample in the the same batch as negative sample. So always be None.
            data_type (str, optional): _description_. Defaults to "train".
        """

        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len: int = args.max_seq_length

        # create target item sets
        self.sem_tag = TargetSubseqs(args.data_dir, args.data_name, args.data_dir)
        self.train_tag: dict[int, list[list[int]]] = self.sem_tag._load_target_subseqs_dict(
            f"{args.data_dir}/{args.data_name}_1_t.pkl", mode="train"
        )

    def _data_sample_rec_task(
        self, user_id: int, items: list[int], input_ids: list[int], target_pos, answer: list[int]
    ):
        """post-processing the data sample to Tensor for the RecWithContrastiveLearningDataset from __getitem__:

        1. padding
        2. truncating
        3. assembling

        Args:
            user_id (int):
            items (list[int]):
            input_ids (list[int]): Transforer input sequence.
            target_pos (_type_): Transformer output target sequence.
            answer (list[int]): target item

        Returns:
            tuple:
        """
        copied_input_ids = copy.deepcopy(input_ids)

        # * input padding and truncating
        pad_len = self.max_len - len(copied_input_ids)
        copied_input_ids = [0] * pad_len + copied_input_ids
        copied_input_ids = copied_input_ids[-self.max_len :]  #
        assert len(copied_input_ids) == self.max_len
        if isinstance(target_pos, tuple):  # * train and cluster
            pad_len_1 = self.max_len - len(target_pos[1])
            target_pos_1 = [0] * pad_len + target_pos[0]
            target_pos_2 = [0] * pad_len_1 + target_pos[1]
            target_pos_1 = target_pos_1[-self.max_len :]
            target_pos_2 = target_pos_2[-self.max_len :]
            assert len(target_pos_1) == self.max_len
            assert len(target_pos_2) == self.max_len

            subseqs_id = args.subseq_id_map[tuple(input_ids)]

        else:  # * valid and test
            target_pos = [0] * pad_len + target_pos
            target_pos = target_pos[-self.max_len :]
            assert len(target_pos) == self.max_len

        # * assemble sequence
        if self.test_neg_items is None:
            return (
                (
                    torch.tensor(subseqs_id, dtype=torch.long),
                    torch.tensor(user_id, dtype=torch.long),
                    torch.tensor(copied_input_ids, dtype=torch.long),
                    torch.tensor(target_pos_1, dtype=torch.long),
                    torch.tensor(target_pos_2, dtype=torch.long),
                    torch.tensor(answer, dtype=torch.long),
                )
                if isinstance(target_pos, tuple)
                else (
                    torch.tensor(user_id, dtype=torch.long),
                    torch.tensor(copied_input_ids, dtype=torch.long),
                    torch.tensor(target_pos, dtype=torch.long),
                    torch.tensor(answer, dtype=torch.long),
                )
            )

    def _add_noise_interactions(self, items: list[int]):
        """Add negative interactions to the sequence for robustness analysis.

        Args:
            items (list[int]): _description_

        Returns:
            _type_: _description_
        """
        copied_sequence = copy.deepcopy(items)
        insert_nums = max(int(args.noise_ratio * len(copied_sequence)), 0)
        if insert_nums == 0:
            return copied_sequence
        insert_idx = random.choices(list(range(len(copied_sequence))), k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, args.item_size - 2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, args.item_size - 2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def __getitem__(self, index: int):
        """Get the data sample for the RecWithContrastiveLearningDataset.

        Example:

        ```
        [0, 1, 2, 3, 4, 5, 6]

        train:
        input_id [0, 1, 2, 3]
        target_pos [1, 2, 3, 4]
        target_pos_ sampled from the target item set
        answer [4]

        valid:
        input_id [0, 1, 2, 3, 4]
        target_pos [1, 2, 3, 4, 5]
        answer [5]

        test:
        input_id [0, 1, 2, 3, 4, 5]
        target_pos [1, 2, 3, 4, 5, 6]
        answer [6]
        ```

        Args:
            index (int): _description_

        Returns:
            _type_: _description_
        """
        user_id = index
        user_seq = self.user_seq[index]
        assert self.data_type in {"train", "valid", "test"}

        if self.data_type == "train":
            # * Remember that Training data (items) is subsequence
            input_ids: list[int] = user_seq[:-3]
            target_pos = user_seq[1:-2]
            # * target_prefix: prefix subsequence of the target item in training stage.
            target_prefix_list: list[list[int]] = self.train_tag[user_seq[-3]]

            # * `item` and `target_prefix` are both subsequence.
            # * But `items` not include User_ID while `target_prefix` include User_ID.
            # * Because `items` comes from `train_user_seq` while `target_prefix` comes from `train_tag` (aka )
            # * So the following code always use target_prefix[1:].

            # ? 下面这个 target_pos_ 是什么意思?
            flag = False
            # * sample another subseq from the target item set
            for target_prefix in target_prefix_list:
                # * skip the subseq same with the input subseq
                if target_prefix[1:] == user_seq[:-3]:
                    continue
                target_pos_ = target_prefix[1:]
                flag = True
            if not flag:
                target_pos_ = random.choice(target_prefix_list)[1:]  #
            answer = [0]  # no use, just for the same format
        elif self.data_type == "valid":
            input_ids = user_seq[:-2]
            target_pos = user_seq[1:-1]
            answer = [user_seq[-2]]
        else:
            items_with_noise = self._add_noise_interactions(user_seq)
            input_ids = items_with_noise[:-1]
            target_pos = items_with_noise[1:]
            answer = [items_with_noise[-1]]

        # * Sample the data
        if self.data_type == "train":
            train_target_pos = (target_pos, target_pos_)
            return self._data_sample_rec_task(user_id, user_seq, input_ids, train_target_pos, answer)
        if self.data_type == "valid":
            return self._data_sample_rec_task(user_id, user_seq, input_ids, target_pos, answer)
        return self._data_sample_rec_task(user_id, items_with_noise, input_ids, target_pos, answer)

    def __len__(self):
        """
        consider n_view of a single sequence as one sample
        """
        return len(self.user_seq)


def DS(i_file: str, o_file: str, max_len: int = 50) -> None:
    """Dynamic Segmentation operations to generate subsequence.

    子序列基本逻辑:

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
    # training, validation, and testing
    max_save_len = max_len + 3
    # save
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
            # training, validation, and testing
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


def build_dataloader(user_seq, loader_type):
    data_type = loader_type if loader_type != "cluster" else "train"
    sampler = RandomSampler if loader_type == "train" else SequentialSampler
    pprint_color(f">>> Building {loader_type} Dataloader")
    dataset = RecWithContrastiveLearningDataset(user_seq, data_type=data_type)
    return DataLoader(dataset, sampler=sampler(dataset), batch_size=args.batch_size)


if __name__ == "__main__":
    # * dynamic segmentation
    DS("../data/Beauty.txt", "../data/Beauty_1.txt", 10)
    # DS_default("../data/Beauty.txt", "../data/Beauty_1.txt")
    # * generate target item
    g = TargetSubseqs("../data", "Beauty", "../data")
    # * generate the dictionary
    data = g._load_target_subseqs_dict("../data/Beauty_1_t.pkl", "train")
    i = 0
    # * Only one sequence in the data dictionary in the training phase has the target item ID
    for d_ in data:
        if len(data[d_]) < 2:
            i += 1
            pprint_color(f"less is : {data[d_]} target_id : {d_}")
    pprint_color(i)

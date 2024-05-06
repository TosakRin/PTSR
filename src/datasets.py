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
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from cprint import pprint_color
from graph import TargetSubseqs
from param import args


class RecWithContrastiveLearningDataset(Dataset):
    def __init__(
        self,
        user_seq: list[list[int]],
        data_type: str = "train",
    ) -> None:
        """torch.utils.dataDataset

        Args:
            user_seq (list[list[int]]): subseq list in the training phase and original sequence in the validation and testing phase. *Not including User_ID*.
            test_neg_items (_type_, optional): negative sample in test for sample based ranking. This paper use other sample in the the same batch as negative sample. So always be None.
            data_type (str, optional): dataset type. Defaults to "train". Choice: {"train", "valid", "test"}.
        """

        self.user_seq = user_seq
        self.data_type = data_type
        self.max_len: int = args.max_seq_length

        # create target item sets
        target_item_subseq = TargetSubseqs(args.subseqs_path, args.target_subseqs_path, args.subseqs_target_path)
        self.train_tag: dict[int, list[list[int]]] = target_item_subseq._load_target_subseqs_dict(
            args.target_subseqs_path, mode="train"
        )
        self.get_pad_user_seq()

    def _data_sample_rec_task(
        self, user_id: int, items: list[int], input_ids: list[int], target_pos, answer: list[int]
    ):
        """post-processing the data sample to Tensor for the RecWithContrastiveLearningDataset from __getitem__:

        1. padding
        2. truncating, example: [0, 0, ..., 1, 2, 3]
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
        copied_input_ids = input_ids

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
            items (list[int]): negative items as noise

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
            tuple(Tensor):
        """
        user_id = index
        if args.loader == "old":
            user_seq = self.user_seq[index]
            assert self.data_type in {"train", "valid", "test", "cluster"}

            if self.data_type in ["train", "cluster"]:
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
                    target_pos_ = random.choice(target_prefix_list)[1:]
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
            if self.data_type in ["train", "cluster"]:
                train_target_pos = (target_pos, target_pos_)
                return self._data_sample_rec_task(user_id, user_seq, input_ids, train_target_pos, answer)
            if self.data_type == "valid":
                return self._data_sample_rec_task(user_id, user_seq, input_ids, target_pos, answer)
            return self._data_sample_rec_task(user_id, items_with_noise, input_ids, target_pos, answer)
        elif args.loader == "new":
            # * new loader: 1. use global pad sequence 2. drop target_pos sample 3. remove test noise interactions
            pad_user_seq = self.pad_user_seq[index]
            if self.data_type in ["train", "cluster"]:
                user_seq = self.user_seq[index]
                input_ids = pad_user_seq[:-3]
                target_pos = pad_user_seq[1:-2]
                answer = [0]
                if self.data_type == "cluster":
                    subseqs_id = args.subseq_id_map[self.pad_origin_map[pad_user_seq][:-3]]
                else:
                    subseqs_id = []
                return (
                    torch.tensor(subseqs_id, dtype=torch.long),
                    torch.tensor(user_id, dtype=torch.long),
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(target_pos, dtype=torch.long),
                    torch.tensor(target_pos, dtype=torch.long),
                    torch.tensor(answer, dtype=torch.long),
                )
            elif self.data_type == "valid":
                input_ids = pad_user_seq[1:-2]
                target_pos = pad_user_seq[2:-1]
                answer = [pad_user_seq[-2]]
            else:
                # items_with_noise = self._add_noise_interactions(pad_user_seq)
                items_with_noise = pad_user_seq
                input_ids = items_with_noise[2:-1]
                target_pos = items_with_noise[3:]
                answer = [items_with_noise[-1]]
            return (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )
        else:
            raise ValueError(f"Invalid loader mode: {args.loader_mode}")

    def __len__(self):
        """consider n_view of a single sequence as one sample"""
        return len(self.user_seq)

    def get_pad_user_seq(self):
        """提前把 pad 做好, 不用到 Dataloader 每次 __getitem__() 阶段再做"""
        max_len = self.max_len + 3
        padded_user_seq = np.zeros((len(self.user_seq), max_len), dtype=int)

        for i, seq in enumerate(self.user_seq):
            padded_user_seq[i, -min(len(seq), max_len) :] = seq[-max_len:]

        self.pad_user_seq = tuple(map(tuple, padded_user_seq))
        user_seq = tuple(map(tuple, self.user_seq))

        self.origin_pad_map = dict(zip(user_seq, self.pad_user_seq))
        self.pad_origin_map = dict(zip(self.pad_user_seq, user_seq))


def build_dataloader(user_seq, loader_type):
    # data_type = loader_type if loader_type != "cluster" else "train"
    data_type = loader_type
    sampler = RandomSampler if loader_type == "train" else SequentialSampler
    pprint_color(f">>> Building {loader_type} Dataloader")
    dataset = RecWithContrastiveLearningDataset(user_seq, data_type=data_type)
    return DataLoader(
        dataset,
        sampler=sampler(dataset),
        batch_size=args.batch_size,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=8,
    )

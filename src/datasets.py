# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from cprint import pprint_color
from graph import TargetSubseqs
from param import args


class SRDataset(Dataset):
    def __init__(
        self,
        user_seq: list[list[int]],
        data_type: str = "train",
    ) -> None:
        """torch.utils.dataDataset

        Args:
            user_seq (list[list[int]]): subseq list in the training phase and original sequence in the validation and testing phase. *Not including User_ID*.
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
        # * new loader_type: 1. use global pad sequence 2. drop target_pos sample 3. remove test noise interactions
        pad_user_seq = self.pad_user_seq_array[index]
        if self.data_type == "train":
            input_ids = pad_user_seq[:-3]
            target_pos = pad_user_seq[1:-2]
            return (
                torch.from_numpy(input_ids),
                torch.from_numpy(target_pos),
            )
        if self.data_type == "graph":
            subseqs_id = args.subseq_id_map[self.pad_origin_map[self.pad_user_seq[index]][:-3]]
            input_ids = pad_user_seq[:-3]
            return (torch.tensor(subseqs_id), torch.from_numpy(input_ids))
        elif self.data_type == "valid":
            input_ids = pad_user_seq[1:-2]
            answer = [pad_user_seq[-2]]
        else:
            input_ids = pad_user_seq[2:-1]
            answer = [pad_user_seq[-1]]
        return (
            torch.tensor(user_id),
            torch.from_numpy(input_ids),
            torch.tensor(answer),
        )
        raise ValueError(f"Invalid loader_type mode: {args.loader_mode}")

    def __len__(self):
        """consider n_view of a single sequence as one sample"""
        return len(self.user_seq)

    def get_pad_user_seq(self):
        """Prepare the padding in advance, so there's no need to do it again during each __getitem__()  of the Dataloader."""
        max_len = self.max_len + 3
        padded_user_seq = np.zeros((len(self.user_seq), max_len), dtype=int)

        for i, seq in enumerate(self.user_seq):
            padded_user_seq[i, -min(len(seq), max_len) :] = seq[-max_len:]

        self.pad_user_seq = tuple(map(tuple, padded_user_seq))
        self.pad_user_seq_array = np.array(self.pad_user_seq)

        user_seq = tuple(map(tuple, self.user_seq))
        self.origin_pad_map = dict(zip(user_seq, self.pad_user_seq))
        self.pad_origin_map = dict(zip(self.pad_user_seq, user_seq))


def build_dataloader(user_seq, loader_type):
    sampler = RandomSampler if loader_type == "train" else SequentialSampler
    pprint_color(f">>> Building {loader_type} Dataloader")
    dataset = SRDataset(user_seq, data_type=loader_type)
    return DataLoader(
        dataset,
        sampler=sampler(dataset),
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
        # persistent_workers=True
    )

# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


import os
import random
from typing import Literal

import numpy as np
import torch
from scipy.sparse import csr_matrix  # type: ignore
from torch import Tensor

from cprint import pprint_color
from param import args


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        pprint_color(f'>>>model weight output dir: "{os.path.abspath(path)}" not exist, just created')
    else:
        pprint_color(f'>>>model weight output dir: "{os.path.abspath(path)}" existed')


def get_rating_matrix(
    user_seq: list[list[int]],
    num_users: int,
    num_items: int,
    mode: Literal["valid", "test"],
) -> csr_matrix:
    """rating matrix: shape: [user_num, max_item + 2]

    Args:
        user_seq (list[list[int]]): user sequence list
        num_users (int): user number
        num_items (int): max_item + 2

    Returns:
        csr_matrix: rating matrix for valid or test.
    """
    # * three lists are used to construct sparse matrix
    r: list[int] = []
    c: list[int] = []
    d: list[int] = []
    end = -1 if mode == "test" else -2

    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:end]:  #
            r.append(user_id)
            c.append(item)
            d.append(1)

    row = np.array(r)
    col = np.array(c)
    data = np.array(d)
    return csr_matrix((data, (row, col)), shape=(num_users, num_items))


def get_user_seqs(data_file: str) -> list[list[int]]:
    """read data file, preprocess to 2 list (user_id, user_seq) and 2 matrix (valid_rating_matrix, test_rating_matrix)

    Args:
        data_file (str):
        - train: data file path after subsequences split (aka after DS).
        - valid/test: original data file path.

    Returns:
        tuple[list[int], list[list[int]], int, csr_matrix, csr_matrix]: user_id_list, user_seq_list, max_item ID, valid_rating_matrix, test_rating_matrix
    """
    with open(data_file, encoding="utf-8") as f:
        subseq_list: list[str] = f.readlines()
        user_seq: list[list[int]] = []
        for subseq in subseq_list:
            items_list: list[str] = subseq.strip().split()[1:]
            items: list[int] = [int(item) for item in items_list]
            user_seq.append(items)
        return user_seq


def get_num_users(data_file):
    """get number of users in original data file"""
    with open(data_file, encoding="utf-8") as f:
        return len(f.readlines())


def get_max_item(data_file):
    """get max item id in original data file"""
    max_item = 0
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            if numbers := [int(item) for item in line.split()[1:]]:
                max_item = max(max_item, max(numbers))
    return max_item


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    score_min: np.ndarray
    best_score: np.ndarray = np.array([])

    def __init__(self, checkpoint_path: str, latest_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.latest_path = latest_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        """
        ```
        for i in range(len(score)):
            # 有一个指标增加了就认为是还在涨
            if score[i] > self.best_score[i] + self.delta:
                return False
        return True
        ```

        Args:
            score (_type_): _description_

        Returns:
            _type_: _description_
        """
        return all(score[i] <= self.best_score[i] + self.delta for i in range(len(score)))

    def __call__(self, score: np.ndarray, model):
        # score HIT@10 NDCG@10
        if not self.best_score:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            args.logger.debug(f">>> EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            args.logger.debug(">>> Validation score increased.  Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def mask_correlated_samples(batch_size: int):
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=torch.bool)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    return mask


# * False Negative Mask
def mask_correlated_samples_(label: Tensor):
    """
    Judge if other subsequence (except our subsequence pair) in the same batch has the same target item. Mask them on a subseq-subseq matrix and true the masked position matrix.

    basic example: the index 1 and index 3 has the same target item. So mask the position (1, 3) and (3, 1) to 0.

    ```python
    >>> import torch
    >>> torch.eq(torch.Tensor([1,2,3,2]), torch.Tensor([[1],[2],[3],[2]]))
    tensor([[ True, False, False, False],
            [False,  True, False,  True],
            [False, False,  True, False],
            [False,  True, False,  True]])
    ```

    Args:
        label (Tensor): The label tensor of shape [1, batch_size].

    Returns:
        Tensor: The mask tensor of shape [2*batch_size, 2*batch_size], where correlated samples are masked with 0.

    """
    # * SHAPE: [1, batch_size] -> [2, batch_size] -> [1, 2*batch_size] -> [2*batch_size, 1]
    label = label.view(1, -1)
    label = label.expand((2, label.shape[-1])).reshape(1, -1)
    label = label.contiguous().view(-1, 1)

    # * label: two subsequences' target item. label[0, batch_size] is the target item of the first subsequence. label[1, batch_size] is the target item of the second subsequence.
    # * SHAPE: [2*batch_size, 2*batch_size]
    mask = torch.eq(label, label.t())
    return mask == 0


def write_cmd(cmd):
    with open("../cmd.sh", mode="a", encoding="utf-8") as f:
        f.write(cmd)
        f.write("\n")

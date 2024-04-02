# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import json
import math
import os
import random
from typing import Literal

import numpy as np
import torch
from scipy.sparse import csr_matrix  # type: ignore

from cprint import pprint_color


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
        pprint_color(f'>>> output path: "{os.path.abspath(path)}" not exist, just created')
    else:
        pprint_color(f'>>> output path: "{os.path.abspath(path)}" exist')


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
            pprint_color(f">>> EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            # ({self.score_min:.6f} --> {score:.6f}) # 这里如果是一个值的话输出才不会有问题
            pprint_color(">>> Validation score increased.  Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


# No Use in this project
def nCr(n, r):
    """
    Calculates the number of combinations (n choose r).

    Args:
        n (int): The total number of items.
        r (int): The number of items to choose.

    Returns:
        int: The number of combinations.

    """
    f = math.factorial
    return f(n) // f(r) // f(n - r)


# No Use in this project
def neg_sample(item_set, item_size):  # []
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


# No Use in this project
def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)


# No Use in this project
def avg_pooling(x, dim):
    return x.sum(dim=dim) / x.size(dim)


# No Use in this project
def get_user_seqs_long(data_file):
    lines = open(data_file, encoding="utf-8").readlines()
    user_seq = []
    long_sequence = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(" ", 1)
        items = items.split(" ")
        items = [int(item) for item in items]
        long_sequence.extend(items)
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    return user_seq, max_item, long_sequence


# No Use in this project
def get_user_seqs_and_sample(data_file, sample_file):
    lines = open(data_file, encoding="utf-8").readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(" ", 1)
        items = items.split(" ")
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    lines = open(sample_file, encoding="utf-8").readlines()
    sample_seq = []
    for line in lines:
        user, items = line.strip().split(" ", 1)
        items = items.split(" ")
        items = [int(item) for item in items]
        sample_seq.append(items)

    assert len(user_seq) == len(sample_seq)

    return user_seq, max_item, sample_seq


# No Use in this project
def get_item2attribute_json(data_file):
    item2attribute = json.loads(open(data_file, encoding="utf-8").readline())
    attribute_set = set()
    for item, attributes in item2attribute.items():
        attribute_set = attribute_set | set(attributes)
    attribute_size = max(attribute_set)  # 331
    return item2attribute, attribute_size

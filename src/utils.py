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
import torch.nn.functional as F
from scipy.sparse import csr_matrix  # ignore_missing_imports

from cprint import pprint_color


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n - r)


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        pprint_color(f'>>> output path: "{os.path.abspath(path)}" not exist, just created')
    else:
        pprint_color(f'>>> output path: "{os.path.abspath(path)}" exist')


def neg_sample(item_set, item_size):  # []
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    score_min: np.ndarray

    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
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
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
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

    def __call__(self, score, model):
        # score HIT@10 NDCG@10

        if self.best_score is None:
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


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)


def avg_pooling(x, dim):
    return x.sum(dim=dim) / x.size(dim)


def generate_rating_matrix(
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


def get_user_seqs(data_file: str) -> tuple[list[int], list[list[int]], int, csr_matrix, csr_matrix]:
    """read data file, preprocess to 2 list (user_id, user_seq) and 2 matrix (valid_rating_matrix, test_rating_matrix)

    Args:
        data_file (str):
        - train: data file path after subsequences split.
        - valid/test: original data file path.

    Returns:
        tuple[list[int], list[list[int]], int, csr_matrix, csr_matrix]: user_id_list, user_seq_list, max_item ID, valid_rating_matrix, test_rating_matrix
    """
    with open(data_file, encoding="utf-8") as f:
        subseq_list: list[str] = f.readlines()
        user_seq: list[list[int]] = []
        user_id: list[int] = []
        item_set: set[int] = set()
        for subseq in subseq_list:
            user, items_str = subseq.strip().split(" ", 1)
            items_list: list[str] = items_str.split(" ")
            items: list[int] = [int(item) for item in items_list]
            user_seq.append(items)
            user_id.append(int(user))
            item_set = item_set | set(items)
        max_item: int = max(item_set)
        num_users: int = len(subseq_list)
        num_items: int = max_item + 2
        valid_rating_matrix: csr_matrix = generate_rating_matrix(
            user_seq,
            num_users,
            num_items,
            "valid",
        )
        test_rating_matrix: csr_matrix = generate_rating_matrix(user_seq, num_users, num_items, "test")
        return user_id, user_seq, max_item, valid_rating_matrix, test_rating_matrix


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


def get_item2attribute_json(data_file):
    item2attribute = json.loads(open(data_file, encoding="utf-8").readline())
    attribute_set = set()
    for item, attributes in item2attribute.items():
        attribute_set = attribute_set | set(attributes)
    attribute_size = max(attribute_set)  # 331
    return item2attribute, attribute_size


def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = sum(place in actual for place in predicted)
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        if act_set := set(actual[i]):
            pred_set = set(predicted[i][:topk])
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
                A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k) if actual else 0.0


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
                A list of lists of elements that are to be predicted
                (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def ndcg_k(actual, predicted, topk):
    res = 0

    for user_id, user_ratings in enumerate(actual):
        k = min(topk, len(user_ratings))
        idcg = idcg_k(k)
        dcg_k = sum(int(predicted[user_id][j] in set(user_ratings)) / math.log(j + 2, 2) for j in range(topk))
        res += dcg_k / idcg

    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum(1.0 / math.log(i + 2, 2) for i in range(k))
    return res or 1.0

# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause



import math

import numpy as np


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


# No Use in this project
def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = sum(place in actual for place in predicted)
    return num_hits / (topk + 0.0)


# No Use in this project
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


# No Use in this project
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


# No Use in this project
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

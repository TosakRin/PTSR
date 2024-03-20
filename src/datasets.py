# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import copy
import os
import pickle
import random
from ast import literal_eval

import torch
from torch.utils.data import Dataset

from utils import get_user_seqs, neg_sample


class Generate_tag:
    def __init__(self, data_path, data_name, save_path):
        self.path = data_path
        self.data_name = f"{data_name}_1"
        self.save_path = save_path

    def generate(self):
        data_f = f"{self.path}/{self.data_name}.txt"
        train_dic = {}
        valid_dic = {}
        test_dic = {}
        with open(data_f, "r", encoding="utf-8") as fr:
            data = fr.readlines()
            for d_ in data:
                items = d_.split(" ")
                tag_train = int(items[-3])
                tag_valid = int(items[-2])
                tag_test = int(items[-1])
                train_temp = list(map(int, items[:-3]))
                valid_temp = list(map(int, items[:-2]))
                test_temp = list(map(int, items[:-1]))
                if tag_train not in train_dic:
                    train_dic.setdefault(tag_train, [])
                train_dic[tag_train].append(train_temp)
                if tag_valid not in valid_dic:
                    valid_dic.setdefault(tag_valid, [])
                valid_dic[tag_valid].append(valid_temp)
                if tag_test not in test_dic:
                    test_dic.setdefault(tag_test, [])
                test_dic[tag_test].append(test_temp)

        total_dic = {"train": train_dic, "valid": valid_dic, "test": test_dic}
        print("Saving data to ", self.save_path)
        with open(f"{self.save_path}/{self.data_name}_t.pkl", "wb") as fw:
            pickle.dump(total_dic, fw)

    def load_dict(self, data_path):
        if not data_path:
            raise ValueError("invalid data path")
        if not os.path.exists(data_path):
            print("The dict not exist, generating...")
            self.generate()
        with open(data_path, "rb") as read_file:
            data_dict = pickle.load(read_file)
        return data_dict

    def get_data(self, data_path, mode):
        data = self.load_dict(data_path)
        return data[mode]


class RecWithContrastiveLearningDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train", similarity_model_type="offline"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

        # create target item sets
        self.sem_tag = Generate_tag(self.args.data_dir, self.args.data_name, self.args.data_dir)
        self.train_tag = self.sem_tag.get_data(f"{self.args.data_dir}/{self.args.data_name}_1_t.pkl", "train")
        self.true_user_id, _, _, _, _ = get_user_seqs(args.train_data_file)

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        pad_len = self.max_len - len(copied_input_ids)
        copied_input_ids = [0] * pad_len + copied_input_ids
        copied_input_ids = copied_input_ids[-self.max_len :]

        if type(target_pos) == tuple:
            pad_len_1 = self.max_len - len(target_pos[1])
            target_pos_1 = [0] * pad_len + target_pos[0]
            target_pos_2 = [0] * pad_len_1 + target_pos[1]
            target_pos_1 = target_pos_1[-self.max_len :]
            target_pos_2 = target_pos_2[-self.max_len :]
            assert len(target_pos_1) == self.max_len
            assert len(target_pos_2) == self.max_len
        else:
            target_pos = [0] * pad_len + target_pos
            target_pos = target_pos[-self.max_len :]
            assert len(target_pos) == self.max_len

        assert len(copied_input_ids) == self.max_len
        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(copied_input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            if type(target_pos) == tuple:
                cur_rec_tensors = (
                    torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                    torch.tensor(copied_input_ids, dtype=torch.long),
                    torch.tensor(target_pos_1, dtype=torch.long),
                    torch.tensor(target_pos_2, dtype=torch.long),
                    torch.tensor(answer, dtype=torch.long),
                )

            else:
                cur_rec_tensors = (
                    torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                    torch.tensor(copied_input_ids, dtype=torch.long),
                    torch.tensor(target_pos, dtype=torch.long),
                    torch.tensor(answer, dtype=torch.long),
                )
        return cur_rec_tensors

    def _add_noise_interactions(self, items):
        copied_sequence = copy.deepcopy(items)
        insert_nums = max(int(self.args.noise_ratio * len(copied_sequence)), 0)
        if insert_nums == 0:
            return copied_sequence
        insert_idx = random.choices(list(range(len(copied_sequence))), k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.args.item_size - 2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size - 2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def __getitem__(self, index):
        user_id = index
        t_user_id = self.true_user_id[index]
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            temp = self.train_tag[items[-3]]
            flag = False
            for t_ in temp:
                if t_[1:] == items[:-3]:
                    continue
                target_pos_ = t_[1:]
                flag = True
            if not flag:
                target_pos_ = random.choice(temp)[1:]
            seq_label_signal = items[-2]  # no use
            answer = [0]  # no use
        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]
        else:
            items_with_noise = self._add_noise_interactions(items)
            input_ids = items_with_noise[:-1]
            target_pos = items_with_noise[1:]
            answer = [items_with_noise[-1]]
        if self.data_type == "train":
            target_pos = (target_pos, target_pos_)
            return self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
        elif self.data_type == "valid":
            return self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
        else:
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
    """2.1 生成的子序列个数为 max_save_len - end, 比如 n=85, max_save_len=53, end=4, 生成的子序列个数为 53 - 4 = 49. 这 49 个子序列的开始都为 0, 结束为 3, 4, 5, ..., 51.
    print(">>> Using DS to generate subsequence ...")
    # ========== read original sequence ==========
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
        #
        else:
            while end < len(seq):
                subseq_dict[u_i].append(seq[start : end + 1])
                end += 1
    # ========== write to file ==========
    with open(o_file, "w+", encoding="utf-8") as fw:
        for u_i, subseq_list in subseq_dict.items():
            for subseq in subseq_list:
                fw.write(f"{u_i}{' '.join(subseq)}\n")
    print(f">>> DS done, written to {o_file}")


class SASRecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len :]
        target_pos = target_pos[-self.max_len :]
        target_neg = target_neg[-self.max_len :]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_rec_tensors

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use

        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        return self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)

    def __len__(self):
        return len(self.user_seq)


if __name__ == "__main__":
    # dynamic segmentation
    DS("../data/Beauty.txt", "../data/Beauty_1.txt", 10)
    # DS_default("../data/Beauty.txt", "../data/Beauty_1.txt")
    # generate target item
    g = Generate_tag("../data", "Beauty", "../data")
    # generate the dictionary
    data = g.get_data("../data/Beauty_1_t.pkl", "train")
    i = 0
    # Only one sequence in the data dictionary in the training phase has the target item ID
    for d_ in data:
        if len(data[d_]) < 2:
            i += 1
            print("less is : ", data[d_], d_)
    print(i)

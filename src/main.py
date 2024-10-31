# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


import os
import sys
import time
from typing import Union

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from cprint import pprint_color
from datasets import build_dataloader
from graph import TargetSubseqs
from logger import set_logger
from models import SASRecModel
from param import args, print_args_info
from trainers import PTSRTrainer, do_train
from utils import (
    check_path,
    get_max_item,
    get_num_users,
    get_rating_matrix,
    get_user_seqs,
    set_seed,
    write_cmd,
)


def set_datapath() -> None:
    # * data path
    args.seqs_path = f"../data/{args.data_name}.txt"
    args.subseqs_target_path = f"{args.data_dir}{args.data_name}_1_s.pkl"
    # if "data" in args.data_dir:
    #     args.subseqs_path = f"{args.data_dir}{args.data_name}_1.txt"
    #     args.target_subseqs_path = f"{args.data_dir}{args.data_name}_1_t.pkl"
    #     args.graph_path = f"{args.data_dir}{args.data_name}_graph.pkl"
    # # * other data path for sen
    # else:
    args.subseqs_path = f"{args.data_dir}{args.data_name}_subseq_{args.subseq_len}.txt"
    args.target_subseqs_path = f"{args.data_dir}{args.data_name}_t_{args.subseq_len}.pkl"
    args.graph_path = f"{args.data_dir}{args.data_name}_graph_{args.subseq_len}.pkl"

    pprint_color(f'==>> args.seqs_path          : "{args.seqs_path}"')
    pprint_color(f'==>> args.subseqs_path       : "{args.subseqs_path}"')
    pprint_color(f'==>> args.target_subseqs_path: "{args.target_subseqs_path}"')
    pprint_color(f'==>> args.subseqs_target_path: "{args.subseqs_target_path}"')
    pprint_color(f'==>> args.graph_path         : "{args.graph_path}"')


def set_log() -> None:
    save_time = time.strftime("%m%d-%H%M%S")
    args.save_name = f"{save_time}-{args.data_name}-{args.msg}"
    args.checkpoint_path = os.path.join(args.output_dir, f"{args.save_name}.pt")
    if args.log_tb_seperate:
        log_path = os.path.join(args.log_root, args.log_dir, args.save_name)
        tb_path = os.path.join(args.log_root, args.tb_dir, args.save_name)
    else:
        log_path = os.path.join(args.log_root, args.save_name, args.save_name)
        tb_path =  os.path.join(args.log_root, args.save_name)
    max_item = get_max_item(args.seqs_path)
    num_users = get_num_users(args.seqs_path)
    pprint_color(f">>> Max item: {max_item}, Num users: {num_users}")
    args.item_size = max_item + 2
    args.mask_id = max_item + 1

    args_info = print_args_info(args)
    args.logger = set_logger(name="exp_log", save_flag=True, save_path=log_path, save_type="file", train_flag=True)
    args.logger.info(args.save_name)
    args.logger.info(" ".join(sys.argv))
    args.logger.info(args_info)
    args.tb = SummaryWriter(log_dir=tb_path)
    write_cmd(f"{args.save_name} | {' '.join(sys.argv)}\n")
    return num_users


def main() -> None:
    set_seed(args.seed)
    check_path(args.output_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    pprint_color(f">>> Cuda Available: {torch.cuda.is_available()}")
    set_datapath()
    num_users = set_log()

    # * training data: train_user_seq is a list of subsequences.
    train_user_seq = get_user_seqs(args.subseqs_path if not args.graph_split else f"../data/{args.data_name}.txt")
    # * valid and test data: test_user_seq is a list of original sequences.
    test_user_seq = get_user_seqs(args.seqs_path)
    graph_user_seq = get_user_seqs(args.subseqs_path)

    # * set item score in train set to `0` in validation
    valid_rating_matrix = get_rating_matrix(
        test_user_seq,
        num_users,
        args.item_size,
        "valid",
    )
    test_rating_matrix = get_rating_matrix(test_user_seq, num_users, args.item_size, "test")
    args.rating_matrix = valid_rating_matrix

    args.subseq_id_map, _ = TargetSubseqs.get_subseq_id_map(args.subseqs_path)
    args.num_subseq_id = len(args.subseq_id_map)

    # * graph -> GNN, train -> SASRec
    train_dataloader = build_dataloader(train_user_seq, "train")
    graph_dataloader = build_dataloader(graph_user_seq, "graph")
    eval_dataloader = build_dataloader(test_user_seq, "valid")
    test_dataloader = build_dataloader(test_user_seq, "test")

    if args.encoder == "SAS":
        model: Union[SASRecModel] = SASRecModel()
    else:
        raise ValueError("args.encoder invalid.")
    trainer = PTSRTrainer(model, train_dataloader, graph_dataloader, eval_dataloader, test_dataloader)

    do_train(trainer, valid_rating_matrix, test_rating_matrix)
    args.tb.close()


if __name__ == "__main__":
    main()

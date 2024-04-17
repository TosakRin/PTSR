import os
import sys
import time
from typing import Union

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from cprint import pprint_color
from datasets import DS, build_dataloader
from graph import TargetSubseqs
from logger import set_logger
from models import GRUEncoder, SASRecModel
from param import args, print_args_info
from trainers import ICSRecTrainer, do_eval, do_train
from utils import (
    EarlyStopping,
    check_path,
    get_max_item,
    get_num_users,
    get_rating_matrix,
    get_user_seqs,
    set_seed,
    write_cmd,
)


def main() -> None:
    set_seed(args.seed)
    check_path(args.output_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    pprint_color(f">>> Cuda Available: {torch.cuda.is_available()}")
    pprint_color(f'==>> args.seqs_path          : "{args.seqs_path}"')
    pprint_color(f'==>> args.subseqs_path       : "{args.subseqs_path}"')
    pprint_color(f'==>> args.target_subseqs_path: "{args.target_subseqs_path}"')
    pprint_color(f'==>> args.subseqs_target_path: "{args.subseqs_target_path}"')
    pprint_color(f'==>> args.graph_path         : "{args.graph_path}"')

    save_time = time.strftime("%m%d-%H%M%S")
    args.save_name = f"{save_time}-{args.data_name}-{args.msg}"
    log_path = os.path.join(args.log_dir, args.save_name)
    tb_path = os.path.join("runs", args.save_name)
    args.checkpoint_path = os.path.join(args.output_dir, f"{args.save_name}.pt")

    # * construct supervisory signals via DS(·) operation
    if not os.path.exists(args.train_data_file):
        DS(args.data_file, args.train_data_file, args.max_seq_length)
    else:
        pprint_color(f'>>> Subsequence data already exists in "{args.train_data_file}". Skip DS operation.')

    # * training data: train_user_seq is a list of subsequences.
    pprint_color(f'>>> Loading train_user_seq (subsequence) from "{args.train_data_file}"')
    train_user_seq = get_user_seqs(args.train_data_file)
    # * valid and test data: test_user_seq is a list of original sequences.
    pprint_color(f'>>> Loading valid and test data (user sequence & rating matrix) from "{args.data_file}"')
    test_user_seq = get_user_seqs(args.data_file)

    max_item = get_max_item(args.data_file)
    num_users = get_num_users(args.data_file)
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

    # * set item score in train set to `0` in validation
    valid_rating_matrix = get_rating_matrix(
        test_user_seq,
        num_users,
        args.item_size,
        "valid",
    )
    test_rating_matrix = get_rating_matrix(test_user_seq, num_users, args.item_size, "test")
    args.rating_matrix = valid_rating_matrix

    args.subseq_id_map, args.id_subseq_map = TargetSubseqs.get_subseq_id_map(args.train_data_file)
    args.num_subseq_id = len(args.subseq_id_map)

    train_dataloader = build_dataloader(train_user_seq, "train")
    cluster_dataloader = build_dataloader(train_user_seq, "cluster")
    eval_dataloader = build_dataloader(test_user_seq, "valid")
    test_dataloader = build_dataloader(test_user_seq, "test")

    if args.encoder == "SAS":
        model: Union[SASRecModel, GRUEncoder] = SASRecModel()
    elif args.encoder == "GRU":
        model = GRUEncoder()
    trainer = ICSRecTrainer(model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader)

    if args.do_eval:
        args.checkpoint_path = os.path.join(args.output_dir, f"{args.model_name}-SAS-{args.data_name}-latest.pt")
    else:
        do_train(trainer, valid_rating_matrix, test_rating_matrix)
    args.tb.close()


if __name__ == "__main__":
    main()

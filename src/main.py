import os
import sys
import time
from typing import Union

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from cprint import pprint_color
from datasets import build_dataloader
from graph import DS, TargetSubseqs
from logger import set_logger
from models import GRUEncoder, SASRecModel
from param import args, print_args_info
from trainers import ICSRecTrainer, do_eval, do_train
from utils import (
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

    # * data path
    args.seqs_path = f"../data/{args.data_name}.txt"
    args.subseqs_target_path = f"{args.data_dir}{args.data_name}_1_s.pkl"
    if "data" in args.data_dir:
        args.subseqs_path = f"{args.data_dir}{args.data_name}_1.txt"
        args.target_subseqs_path = f"{args.data_dir}{args.data_name}_1_t.pkl"
        args.graph_path = f"{args.data_dir}{args.data_name}_graph.pkl"
    # * other data path for sen
    else:
        args.subseqs_path = f"{args.data_dir}{args.data_name}_subseq_{args.subseq_len}.txt"
        args.target_subseqs_path = f"{args.data_dir}{args.data_name}_t_{args.subseq_len}.pkl"
        args.graph_path = f"{args.data_dir}{args.data_name}_graph_{args.subseq_len}.pkl"

    pprint_color(f'==>> args.seqs_path          : "{args.seqs_path}"')
    pprint_color(f'==>> args.subseqs_path       : "{args.subseqs_path}"')
    pprint_color(f'==>> args.target_subseqs_path: "{args.target_subseqs_path}"')
    pprint_color(f'==>> args.subseqs_target_path: "{args.subseqs_target_path}"')
    pprint_color(f'==>> args.graph_path         : "{args.graph_path}"')

    save_time = time.strftime("%m%d-%H%M%S")
    args.save_name = f"{save_time}-{args.data_name}-{args.msg}"
    log_path = os.path.join(args.log_root, args.log_dir, args.save_name)
    tb_path = os.path.join(args.log_root, args.tb_dir, args.save_name)
    args.checkpoint_path = os.path.join(args.output_dir, f"{args.save_name}.pt")

    # * construct supervisory signals via DS(·) operation
    if not os.path.exists(args.subseqs_path):
        DS(args.seqs_path, args.subseqs_path, args.max_seq_length)
    else:
        pprint_color(f'>>> Subsequence data already exists in "{args.subseqs_path}". Skip DS operation.')

    # * training data: train_user_seq is a list of subsequences.
    train_user_seq = get_user_seqs(args.subseqs_path if not args.graph_split else f"../data/{args.data_name}_1.txt")
    # * valid and test data: test_user_seq is a list of original sequences.
    test_user_seq = get_user_seqs(args.seqs_path)

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

    # * set item score in train set to `0` in validation
    valid_rating_matrix = get_rating_matrix(
        test_user_seq,
        num_users,
        args.item_size,
        "valid",
    )
    test_rating_matrix = get_rating_matrix(test_user_seq, num_users, args.item_size, "test")
    args.rating_matrix = valid_rating_matrix

    args.subseq_id_map, args.id_subseq_map = TargetSubseqs.get_subseq_id_map(args.subseqs_path)
    args.num_subseq_id = len(args.subseq_id_map)

    # * cluster -> GNN, train -> SASRec
    cluster_user_seq = get_user_seqs(args.subseqs_path)
    train_dataloader = build_dataloader(train_user_seq, "train")
    cluster_dataloader = build_dataloader(cluster_user_seq, "cluster")
    eval_dataloader = build_dataloader(test_user_seq, "valid")
    test_dataloader = build_dataloader(test_user_seq, "test")

    if args.encoder == "SAS":
        model: Union[SASRecModel, GRUEncoder] = SASRecModel()
    elif args.encoder == "GRU":
        model = GRUEncoder()
    trainer = ICSRecTrainer(model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader)

    do_train(trainer, valid_rating_matrix, test_rating_matrix)
    args.tb.close()


if __name__ == "__main__":
    main()

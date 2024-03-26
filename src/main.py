import argparse
import os
import time
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from cprint import pprint_color
from datasets import DS, RecWithContrastiveLearningDataset
from logger import set_logger
from models import GRUEncoder, SASRecModel
from param import parse_args, print_args_info
from trainers import ICSRecTrainer
from utils import EarlyStopping, check_path, get_user_seqs, set_seed


def build_dataloader(args, user_seq, loader_type):
    data_type = loader_type if loader_type != "cluster" else "train"
    sampler = RandomSampler if loader_type == "train" else SequentialSampler
    pprint_color(f">>> Building {loader_type} Dataloader")
    dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type=data_type)
    return DataLoader(dataset, sampler=sampler(dataset), batch_size=args.batch_size)


def main() -> None:
    args: argparse.Namespace = parse_args()
    set_seed(args.seed)
    check_path(args.output_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    pprint_color(f">>> Using Cuda: {torch.cuda.is_available()}")
    args.data_file = f"{args.data_dir}{args.data_name}.txt"
    args.train_data_file = f"{args.data_dir}{args.data_name}_1.txt"

    # * construct supervisory signals via DS(·) operation
    if not os.path.exists(args.train_data_file):
        DS(args.data_file, args.train_data_file, args.max_seq_length)
    else:
        pprint_color(f'>>> Subsequence data already exists in "{args.train_data_file}". Skip DS operation.')

    # * training data: train_user_seq is a list of subsequences.
    pprint_color(f'>>> Loading train_user_seq (subsequence) from "{args.train_data_file}"')
    _, train_user_seq, _, _, _ = get_user_seqs(args.train_data_file)
    # * valid and test data: user_seq is a list of original sequences.
    pprint_color(f'>>> Loading valid and test data (user sequence & rating matrix) from "{args.data_file}"')
    _, user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)

    # ? 为什么要加 2
    args.item_size = max_item + 2
    args.mask_id = max_item + 1

    print_args_info(args)

    save_time = time.strftime("%Y%m%d-%H%M%S")
    save_name = f"{save_time}-{args.model_name}-{args.data_name}"
    log_path = os.path.join(args.output_dir, save_name)
    args.logger = set_logger(name="exp_log", save_flag=True, save_path=log_path, save_type="file", train_flag=True)
    args.logger.info(save_name)

    checkpoint = f"{save_name}.pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # * set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    if not args.do_eval:
        train_dataloader = build_dataloader(args, train_user_seq, "train")
        cluster_dataloader = build_dataloader(args, train_user_seq, "cluster")
        eval_dataloader = build_dataloader(args, user_seq, "valid")
    else:
        train_dataloader = None
        cluster_dataloader = None
        eval_dataloader = None
    test_dataloader = build_dataloader(args, user_seq, "test")

    if args.encoder == "SAS":
        model: Union[SASRecModel, GRUEncoder] = SASRecModel(args=args)
    elif args.encoder == "GRU":
        model = GRUEncoder(args=args)
    trainer = ICSRecTrainer(model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args)

    if args.do_eval:
        args.checkpoint_path = os.path.join(args.output_dir, f"{args.model_name}-SAS-{args.data_name}-0.pt")
    else:
        pprint_color(">>> Train ICSRec Start")
        early_stopping = EarlyStopping(args.checkpoint_path, patience=40, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            # * evaluate on NDCG@20
            scores, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                pprint_color(">>> Early stopping")
                break
        pprint_color("---------------Change to test_rating_matrix-------------------")
    do_eval(trainer, args, test_rating_matrix)


def do_eval(trainer, args, test_rating_matrix):
    pprint_color(">>> Test ICSRec Start")
    pprint_color(f">>> Load model from {args.checkpoint_path} for test")
    trainer.args.train_matrix = test_rating_matrix
    trainer.load(args.checkpoint_path)
    scores, result_info = trainer.test(0, full_sort=True)
    pprint_color(result_info)
    args.logger.info(result_info)


if __name__ == "__main__":
    main()

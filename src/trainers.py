# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import gc
import math
import time
import warnings
from ast import literal_eval
from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adagrad, AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from cprint import pprint_color
from graph import Graph
from loss import EmbLoss, cicl_loss
from metric import get_metric, ndcg_k, recall_at_k
from models import GCN, GRUEncoder, SASRecModel
from param import args
from utils import EarlyStopping

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


def do_train(trainer, valid_rating_matrix, test_rating_matrix):
    pprint_color(">>> Train ICSRec Start")
    early_stopping = EarlyStopping(args.checkpoint_path, args.latest_path, patience=50)
    for epoch in range(args.epochs):
        args.rating_matrix = valid_rating_matrix
        trainer.train(epoch)
        # * evaluate on NDCG@20
        if args.do_eval:
            scores, _ = trainer.valid(epoch)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                pprint_color(">>> Early stopping")
                break

        # * test on while training
        if args.do_test and epoch >= args.min_test_epoch:
            args.rating_matrix = test_rating_matrix
            _, _ = trainer.test(epoch, full_sort=True)


def do_eval(trainer, test_rating_matrix):
    pprint_color(">>> Test ICSRec Start")
    pprint_color(f'>>> Load model from "{args.latest_path}" for test')
    args.rating_matrix = test_rating_matrix
    trainer.load(args.latest_path)
    _, _ = trainer.test(0, full_sort=True)


class Trainer:

    def __init__(
        self,
        model: Union[SASRecModel, GRUEncoder],
        train_dataloader: Optional[DataLoader],
        cluster_dataloader: Optional[DataLoader],
        eval_dataloader: Optional[DataLoader],
        test_dataloader: DataLoader,
    ) -> None:
        pprint_color(">>> Initialize Trainer")

        cuda_condition = torch.cuda.is_available() and not args.no_cuda
        self.device: torch.device = torch.device("cuda" if cuda_condition else "cpu")
        torch.set_float32_matmul_precision(args.precision)

        self.model = torch.compile(model) if args.compile else model
        self.gcn = GCN()
        # self.predictor = nn.Linear(args.hidden_size, args.hidden_size)
        # nn.init.xavier_normal_(self.predictor.weight)
        # self.reg_loss = EmbLoss()
        if cuda_condition:
            self.model.cuda()
            self.gcn.cuda()
            # self.predictor.cuda()
            # self.reg_loss.cuda()
        self.graph = Graph(args.graph_path)
        self.model.graph = self.graph

        self.train_dataloader, self.cluster_dataloader, self.eval_dataloader, self.test_dataloader = (
            train_dataloader,
            cluster_dataloader,
            eval_dataloader,
            test_dataloader,
        )

        self.optim_adam = AdamW(self.model.adam_params, lr=args.lr_adam, weight_decay=args.weight_decay)
        self.optim_adam = AdamW(self.model.parameters(), lr=args.lr_adam, weight_decay=args.weight_decay)
        self.optim_adagrad = Adagrad(self.model.adagrad_params, lr=args.lr_adagrad, weight_decay=args.weight_decay)
        self.scheduler = self.get_scheduler(self.optim_adam)

        # * prepare padding subseq for subseq embedding update
        self.all_subseq = self.get_all_pad_subseq(self.cluster_dataloader)
        # self.all_subseq = self.all_subseq.to(self.device)
        # self.pad_mask = (self.all_subseq > 0).to(self.device)  # todo: 显存
        self.pad_mask = self.all_subseq > 0

        self.num_non_pad = self.pad_mask.sum(dim=1, keepdim=True)  # todo: 可以抽出来

        self.best_scores = {
            "valid": {
                "Epoch": 0,
                "HIT@5": 0.0,
                "HIT@10": 0.0,
                "HIT@20": 0.0,
                "NDCG@5": 0.0,
                "NDCG@10": 0.0,
                "NDCG@20": 0.0,
            },
            "test": {
                "Epoch": 0,
                "HIT@5": 0.0,
                "HIT@10": 0.0,
                "HIT@20": 0.0,
                "NDCG@5": 0.0,
                "NDCG@10": 0.0,
                "NDCG@20": 0.0,
            },
        }

        # pprint_color(f">>> Total Parameters: {sum(p.nelement() for p in self.model.parameters())}")

    def get_full_sort_score(
        self, epoch: int, answers: np.ndarray, pred_list: np.ndarray, mode
    ) -> tuple[list[float], str]:
        """
        Calculate the full sort score for a given epoch.

        Args:
            epoch (int): The epoch number.
            answers (np.ndarray): The ground truth answers. SHAPE: [user_num, 1]
            pred_list (np.ndarray): The predicted list of items. SHAPE: [user_num, 20]

        Returns:
            list: A list containing the recall and NDCG scores at different values of k.
            str: A string representation of the scores.

        """
        recall, ndcg = [], []
        for k in [5, 10, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": round(recall[0], 4),
            "HIT@10": round(recall[1], 4),
            "HIT@20": round(recall[2], 4),
            "NDCG@5": round(ndcg[0], 4),
            "NDCG@10": round(ndcg[1], 4),
            "NDCG@20": round(ndcg[2], 4),
        }

        for key, value in post_fix.items():
            if key != "Epoch":
                args.tb.add_scalar(f"{mode}/{key}", value, epoch, new_style=True)
        args.logger.warning(post_fix)

        self.get_best_score(post_fix, mode)
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[2], ndcg[2]], str(post_fix)

    def get_best_score(self, scores, mode):
        for key, value in scores.items():
            if key in self.best_scores[mode]:
                self.best_scores[mode][key] = max(self.best_scores[mode][key], value)
            if key != "Epoch":
                args.tb.add_scalar(
                    f"best_{mode}/best_{key}",
                    self.best_scores[mode][key],
                    self.best_scores[mode]["Epoch"],
                    new_style=True,
                )

        (
            args.logger.critical(f"{self.best_scores[mode]}")
            if mode == "test"
            else args.logger.error(f"{self.best_scores[mode]}")
        )

    def save(self, file_name: str):
        """Save the model to the file_name"""
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name: str):
        """Load the model from the file_name"""
        self.model.load_state_dict(torch.load(file_name))

    def icsrec_loss(self, subsequence_1, subsequence_2, target_pos_1):
        # * intent representation learning task\
        cicl, ficl = 0.0, 0.0
        coarse_intent_1, coarse_intent_2 = self.model(subsequence_1), self.model(subsequence_2)
        subseq_pair = [coarse_intent_1, coarse_intent_2]
        if "c" in args.cl_mode:
            cicl = cicl_loss(subseq_pair, target_pos_1)
        if "f" in args.cl_mode:
            ficl = ficl_loss(subseq_pair, self.clusters_t[0])
        return cicl, ficl

    def recon_loss(self, subseq_emb, item_emb, subseq_id, item_id):
        with torch.no_grad():
            subseq_target, item_target = subseq_emb.clone(), item_emb.clone()
            subseq_target.detach()
            item_target.detach()
            subseq_target = F.dropout(subseq_target, 0.5)
            item_target = F.dropout(item_target, 0.5)
        subseq_online, item_online = self.predictor(subseq_emb), self.predictor(item_emb)
        subseq_online = subseq_online[subseq_id, :]
        item_online = item_online[item_id, :]
        subseq_target = subseq_target[subseq_id, :]
        item_target = item_target[item_id, :]
        loss_si = 1 - F.cosine_similarity(subseq_online, item_target.detach(), dim=-1).mean()
        loss_is = 1 - F.cosine_similarity(item_online, subseq_target.detach(), dim=-1).mean()
        reg_loss = self.reg_loss(subseq_emb, item_emb)
        # print(f"loss_si: {loss_si}, loss_is: {loss_is}, reg_loss: {reg_loss}")
        return (loss_si + loss_is).mean() + reg_loss.item()

    @staticmethod
    def get_scheduler(optimizer):
        if args.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        elif args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr)
        elif args.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=args.factor, patience=args.patience, verbose=True
            )
        elif args.scheduler == "multistep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=literal_eval(args.milestones), gamma=args.gamma
            )
            pprint_color(f">>> scheduler: {args.scheduler}, milestones: {args.milestones}, gamma: {args.gamma}")
        elif args.scheduler == "warmup+cosine":
            warm_up_with_cosine_lr = lambda epoch: (
                epoch / args.warm_up_epochs
                if epoch <= args.warm_up_epochs
                else 0.5 * (math.cos((epoch - args.warm_up_epochs) / (args.epochs - args.warm_up_epochs) * math.pi) + 1)
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
        elif args.scheduler == "warmup+multistep":
            warm_up_with_multistep_lr = lambda epoch: (
                epoch / args.warm_up_epochs
                if epoch <= args.warm_up_epochs
                else args.gamma ** len([m for m in literal_eval(args.milestones) if m <= epoch])
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)
        else:
            raise ValueError("Invalid scheduler")
        return scheduler

    def get_all_pad_subseq(self, gcn_dataloader: DataLoader) -> tuple[Tensor, Tensor]:
        """collect all padding subsequence index and subsequence for updating subseq embeddings.

        Returns:
            tuple[Tensor, Tensor]: all_subseq_ids is subseq id for all_subseq. all_subseq is padding subseq (index, not embedding)
        """
        all_subseq_ids = []
        all_subseq = []
        for _, (rec_batch) in tqdm(
            enumerate(gcn_dataloader),
            total=len(gcn_dataloader),
            desc=f"{args.save_name} | Device: {args.gpu_id} | get_all_pad_subseq",
            leave=False,
            dynamic_ncols=True,
        ):
            subseq_id, _, subsequence, _, _, _ = rec_batch
            all_subseq_ids.append(subseq_id)
            all_subseq.append(subsequence)
        all_subseq_ids = torch.cat(all_subseq_ids, dim=0)
        all_subseq = torch.cat(all_subseq, dim=0)

        # * remove duplicate subsequence
        tensor_np = all_subseq_ids.numpy()
        _, indices = np.unique(tensor_np, axis=0, return_index=True)
        sorted_indices = np.sort(indices)
        all_subseq_ids = all_subseq_ids[sorted_indices]
        all_subseq = all_subseq[sorted_indices]
        # * check if ID is always increasing
        # print(torch.all(torch.diff(all_subseq_ids) > 0))
        # id_padded_subseq_map = dict(zip(all_subseq_ids, all_subseq))
        return all_subseq

    def subseq_embed_update(self, epoch):
        self.model.item_embeddings.cpu()
        self.model.subseq_embeddings.cpu()
        subseq_emb = self.model.item_embeddings(self.all_subseq)
        subseq_emb_avg: Tensor = (
            torch.sum(subseq_emb * self.pad_mask.unsqueeze(-1), dim=1) / self.num_non_pad
        )  # todo: mean换linear
        # * Three subseq embed update methods: 1. nn.Parameter 2. nn.Embedding 3. model.subseq_embeddings
        # self.model.subseq_embeddings = nn.Parameter(subseq_emb_avg)
        # self.model.subseq_embeddings = subseq_emb_avg

        self.model.subseq_embeddings.weight.data = (
            subseq_emb_avg if epoch == 0 else (subseq_emb_avg + self.model.subseq_embeddings.weight.data) / 2
        )  # todo: 这样可以实现快速收敛

        self.model.item_embeddings.to(self.device)
        self.model.subseq_embeddings.to(self.device)


class ICSRecTrainer(Trainer):

    def train_epoch(self, epoch, cluster_dataloader, train_dataloader):
        self.model.train()
        if epoch == 0:
            train_matrix = self.graph.edge_random_dropout(self.graph.train_matrix, args.dropout_rate)
            self.graph.torch_A = self.graph.get_torch_adj(train_matrix)

        rec_avg_loss, joint_avg_loss, icl_losses = 0.0, 0.0, 0.0
        batch_num = len(train_dataloader)
        args.tb.add_scalar("train/LR", self.optim_adam.param_groups[0]["lr"], epoch, new_style=True)

        # * two different update: 1. update subseq embeddings 2. gcn update
        # * update subseq embeddings: 1. every epoch(√) 2. every batch 3. no update
        if args.gcn_mode != "None":
            self.subseq_embed_update(epoch)
        # * update gcn: 1. every epoch 2. every batch(√) 3. no update
        if args.gcn_mode == "global":
            # * call gcn every epoch
            _, self.model.all_item_emb = self.gcn(
                self.graph.torch_A, self.model.subseq_embeddings.weight, self.model.item_embeddings.weight
            )

        for batch_i, (rec_batch) in tqdm(
            enumerate(train_dataloader),
            total=batch_num,
            leave=False,
            desc=f"{args.save_name} | Device: {args.gpu_id} | Rec Training Epoch {epoch}",
            dynamic_ncols=True,
        ):
            # * rec_batch shape: key_name x batch_size x feature_dim
            rec_batch = tuple(t.to(self.device) for t in rec_batch)
            subseq_id, _, subsequence_1, target_pos_1, subsequence_2, target_id = rec_batch

            # * prediction task
            intent_output = self.model(subsequence_1)
            logits = self.model.predict_full(intent_output[:, -1, :])
            rec_loss = nn.CrossEntropyLoss()(logits, target_pos_1[:, -1])

            cicl_loss, ficl_loss = 0.0, 0.0
            if args.cl_mode in ["c", "f", "cf"]:
                cicl_loss, ficl_loss = self.icsrec_loss(subsequence_1, subsequence_2, target_pos_1)

            icl_loss = args.lambda_0 * cicl_loss + args.beta_0 * ficl_loss
            joint_loss = args.rec_weight * rec_loss + icl_loss
            if args.recon:
                joint_loss += self.recon_loss(self.model.all_subseq_emb, self.model.all_item_emb, subseq_id, target_id)

            # self.optim_adagrad.zero_grad()
            self.optim_adam.zero_grad()
            joint_loss.backward()
            # self.optim_adagrad.step()
            self.optim_adam.step()

            rec_avg_loss += rec_loss.item()
            if not isinstance(icl_loss, float):
                icl_losses += icl_loss.item()
            else:
                icl_losses += icl_loss
            joint_avg_loss += joint_loss.item()
            if args.batch_loss:
                args.tb.add_scalar("batch_loss/rec_loss", rec_loss.item(), epoch * batch_num + batch_i, new_style=True)
                # args.tb.add_scalar("batch_train/icl_loss", icl_loss.item(), epoch * batch_num + batch_i, new_style=True)
                args.tb.add_scalar(
                    "batch_loss/joint_loss", joint_loss.item(), epoch * batch_num + batch_i, new_style=True
                )

        self.scheduler.step()
        # * print & write log for each epoch
        # * post_fix: print the average loss of the epoch
        post_fix = {
            "Epoch": epoch,
            "lr_adam": round(self.optim_adam.param_groups[0]["lr"], 6),
            "lr_adagrad": round(self.optim_adagrad.param_groups[0]["lr"], 6),
            "rec_avg_loss": round(rec_avg_loss / batch_num, 4),
            "icl_avg_loss": round(icl_losses / batch_num, 4),
            "joint_avg_loss": round(joint_avg_loss / batch_num, 4),
        }

        for key, value in post_fix.items():
            if "loss" in key:
                args.tb.add_scalar(f"train/{key}", value, epoch, new_style=True)

        # metadata = [f"Item_{i}" for i in range(args.item_size)]
        # args.tb.add_embedding(self.model.item_embeddings.weight, metadata=metadata, tag="ItemEmbeddings", global_step=epoch)
        # args.tb.add_embedding(self.model.all_item_emb, metadata=metadata, tag="ItemEmbeddings", global_step=epoch)
        # metadata = [f"Subseq_{i}" for i in range(args.num_subseq_id)]
        # args.tb.add_embedding(self.model.all_subseq_emb, metadata=metadata, tag="SubseqEmbeddings", global_step=epoch)

        if (epoch + 1) % args.log_freq == 0:
            args.logger.info(str(post_fix))

    def full_test_epoch(self, epoch: int, dataloader: DataLoader, mode):
        with torch.no_grad():
            self.model.eval()
            # * gcn is fixed in the test phase. So it's unnecessary to call gcn() every batch.
            if args.gcn_mode != "None":
                _, self.model.all_item_emb = self.gcn(
                    self.graph.torch_A, self.model.subseq_embeddings.weight, self.model.item_embeddings.weight
                )
            for i, batch in tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                leave=False,
                desc=f"{args.save_name} | Device: {args.gpu_id} | Test Epoch {epoch}",
                dynamic_ncols=True,
            ):
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, _, answers = batch
                # * SHAPE: [Batch_size, Seq_len, Hidden_size] -> [256, 50, 64]
                recommend_output: Tensor = self.model(input_ids)  # [BxLxH]
                # * Use the last item output. SHAPE: [Batch_size, Hidden_size] -> [256, 64]
                recommend_output = recommend_output[:, -1, :]  # [BxH]

                # * recommendation results. SHAPE: [Batch_size, Item_size]
                rating_pred = self.model.predict_full(recommend_output)
                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()

                # * 将已经有评分的 item 的预测评分设置为 0, 防止推荐已经评分过的 item
                rating_pred[args.rating_matrix[batch_user_index].toarray() > 0] = 0
                # * argpartition T: O(n)  argsort O(nlogn) | reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                # * Get the *index* of the largest 20 items, but its order is not sorted. SHAPE: [Batch_size, 20]
                ind: np.ndarray = np.argpartition(rating_pred, -20)[:, -20:]

                # * np.arange(len(rating_pred): [0, 1, 2, ..., Batch_size-1]. SHAPE: [Batch_size]
                # * np.arange(len(rating_pred))[:, None]: [[0], [1], [2], ..., [Batch_size-1]]. SHAPE: [Batch_size, 1]

                # * Get the *value* of the largest 20 items. SHAPE: [Batch_size, 20]
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                # * Sort the largest 20 items's value to get the index. SHAPE: [Batch_size, 20]
                # * ATTENTION: `arr_ind_argsort` is the index of the `ind`, not the index of the `rating_pred`
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                # * Get the real Item ID of the largest 20 items. SHAPE: [Batch_size, 20]
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

            return self.get_full_sort_score(epoch, answer_list, pred_list, mode)

    def train(self, epoch) -> None:
        assert self.train_dataloader is not None
        args.mode = "train"
        self.train_epoch(epoch, self.cluster_dataloader, self.train_dataloader)

    def valid(self, epoch) -> tuple[list[float], str]:
        assert self.eval_dataloader is not None
        args.mode = "valid"
        return self.full_test_epoch(epoch, self.eval_dataloader, "valid")

    def test(self, epoch, full_sort=False) -> tuple[list[float], str]:
        args.mode = "test"
        if full_sort:
            return self.full_test_epoch(epoch, self.test_dataloader, "test")
        return self.sample_test_epoch(epoch, self.test_dataloader)

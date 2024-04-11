"""
/@
 @Author       : TosakRin sick_person@outlook.com
 @Date         : 2024-03-18 15:52:01
 @LastEditors  : TosakRin sick_person@outlook.com
 @LastEditTime : 2024-04-01 02:45:40
 @FilePath     : /ICSRec/src/trainers.py
 @Description  :
 @/
"""

# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#


import gc
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
from metric import get_metric, ndcg_k, recall_at_k
from models import GCN, GRUEncoder, KMeans, SASRecModel
from param import args

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


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
        if cuda_condition:
            self.model.cuda()

        self.batch_size: int = args.batch_size
        self.sim: str = args.sim  # * the calculate ways of the similarity.

        cluster = KMeans(
            num_cluster=args.intent_num,
            seed=1,
            hidden_size=64,
            gpu_id=args.gpu_id,
            device=self.device,
        )
        self.clusters: list[KMeans] = [cluster]
        self.clusters_t: list[list[KMeans]] = [self.clusters]

        self.train_dataloader = train_dataloader
        self.cluster_dataloader = cluster_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        graph_path = f"{args.data_dir}/{args.data_name}_graph.pkl"
        self.graph = Graph(graph_path)
        self.model.graph = self.graph
        self.optim_adam = AdamW(self.model.adam_params, lr=args.lr_adam, weight_decay=args.weight_decay)
        self.optim_adam = AdamW(self.model.parameters(), lr=args.lr_adam, weight_decay=args.weight_decay)
        self.optim_adagrad = Adagrad(self.model.adagrad_params, lr=args.lr_adagrad, weight_decay=args.weight_decay)
        self.scheduler = self.get_scheduler(self.optim_adam)
        self.all_subseq_id, self.all_subseq = self.get_all_pad_subseq(self.cluster_dataloader)

        self.best_scores = {
            "valid": {
                "Epoch": 0,
                "HIT@5": 0.0,
                "NDCG@5": 0.0,
                "HIT@10": 0.0,
                "NDCG@10": 0.0,
                "HIT@20": 0.0,
                "NDCG@20": 0.0,
            },
            "test": {
                "Epoch": 0,
                "HIT@5": 0.0,
                "NDCG@5": 0.0,
                "HIT@10": 0.0,
                "NDCG@10": 0.0,
                "HIT@20": 0.0,
                "NDCG@20": 0.0,
            },
        }

        pprint_color(f">>> Total Parameters: {sum(p.nelement() for p in self.model.parameters())}")

    # * Sample-based. NO USE in the paper
    def get_sample_scores(self, epoch: int, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)

        post_fix = {
            "Epoch": epoch,
            "HIT@1": round(HIT_1, 4),
            "NDCG@1": round(NDCG_1, 4),
            "HIT@5": round(HIT_5, 4),
            "NDCG@5": round(NDCG_5, 4),
            "HIT@10": round(HIT_10, 4),
            "NDCG@10": round(NDCG_10, 4),
            "MRR": round(MRR, 4),
        }
        pprint_color(post_fix)
        args.logger.info(str(post_fix))
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

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
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": round(recall[0], 4),
            "NDCG@5": round(ndcg[0], 4),
            "HIT@10": round(recall[1], 4),
            "NDCG@10": round(ndcg[1], 4),
            "HIT@20": round(recall[3], 4),
            "NDCG@20": round(ndcg[3], 4),
        }

        for key, value in post_fix.items():
            if key != "Epoch":
                args.tb.add_scalar(f"{mode}/{key}", value, epoch, new_style=True)

        # pprint_color(post_fix)
        args.logger.warning(post_fix)
        self.get_best_score(post_fix, mode)
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

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
        args.logger.critical(f"{self.best_scores[mode]}")

    def save(self, file_name: str):
        """Save the model to the file_name"""
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name: str):
        """Load the model from the file_name"""
        self.model.load_state_dict(torch.load(file_name))

    def mask_correlated_samples(self, batch_size: int):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    # * False Negative Mask
    def mask_correlated_samples_(self, label: Tensor):
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

    def info_nce(self, z_i: Tensor, z_j: Tensor, temp: float, batch_size: int, sim_way: str = "dot", intent_id=None):
        """
        Calculates the InfoNCE loss for positive and negative pairs.

        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.

        Args:
            z_i (Tensor): The embeddings of the first item in the positive pair. SHAPE: [batch_size, hidden_size]
            z_j (Tensor): The embeddings of the second item in the positive pair. SHAPE: [batch_size, hidden_size]
            temp (float): The temperature parameter for scaling the similarity scores.
            batch_size (int): The size of the batch.
            sim_way (str, optional): The similarity calculation method. Can be "dot" or "cos". Defaults to "dot".
            intent_id (optional): The intent ID for masking correlated samples. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: The logits [batch_size*2, batch_size*2 + 1] and labels [batch_size*2] for the InfoNCE loss.
        """
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)  # * SHAPE: [batch_size*2, hidden_size]
        if sim_way == "cos":
            sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim_way == "dot":
            sim = torch.mm(z, z.t()) / temp

        # * torch.diag: Returns the elements from the diagonal of a matrix. SHAPE: [batch_size]
        # * Positive: (0, 256), (1, 257) ... (255, 511) and (256, 0), (257, 1) ... (511, 255)
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        # * SHAPE: [batch_size*2, 1]
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        if args.f_neg:
            mask = self.mask_correlated_samples_(intent_id)
            # * SHAPE: [batch_size*2, batch_size*2]
            negative_samples = sim
            negative_samples[mask == 0] = float("-inf")
        else:
            mask = self.mask_correlated_samples(batch_size)
            negative_samples = sim[mask].reshape(N, -1)

        # * SHAPE: [batch_size*2]
        labels = torch.zeros(N).to(positive_samples.device).long()
        # * SHAPE: [batch_size*2, batch_size*2 + 1]
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def cicl_loss(self, coarse_intents: list[Tensor], target_item):
        """Coarse Intents: make 2 subsequence with the same target item closer by infoNCE.

        Args:
            coarse_intents (list[Tensor]): A list of coarse intents. Tensor SHAPE: [batch_size, seq_len, hidden_size]
            target_item (Tensor): The target item.

        Returns:
            Tensor: The calculated contrastive loss.
        """
        coarse_intent_1, coarse_intent_2 = coarse_intents[0], coarse_intents[1]
        sem_nce_logits, sem_nce_labels = self.info_nce(
            coarse_intent_1[:, -1, :],
            coarse_intent_2[:, -1, :],
            args.temperature,
            coarse_intent_1.shape[0],
            self.sim,
            target_item[:, -1],
        )
        return nn.CrossEntropyLoss()(sem_nce_logits, sem_nce_labels)

    def ficl_loss(self, subseq_pair: list[Tensor], clusters_t: list[KMeans]):
        """
        Calculates the FICL (Federated InfoNCE Contrastive Learning) loss.

        Args:
            sequences (list[Tensor]): subsequence pair with the same target item. subseq SHAPE: [batch_size, seq_len, hidden_size]
            clusters_t (list[KMeans]): A list of clusters.

        Returns:
            torch.Tensor: The FICL loss.

        """
        for i, subseq in enumerate(subseq_pair):
            coarse_intent = subseq[:, -1, :]
            intent_n = coarse_intent.view(-1, coarse_intent.shape[-1])
            intent_n = intent_n.detach().cpu().numpy()
            intent_id, fined_intent = clusters_t[0].query(intent_n)

            fined_intent = fined_intent.view(fined_intent.shape[0], -1)
            a, b = self.info_nce(
                coarse_intent.view(coarse_intent.shape[0], -1),
                fined_intent,
                args.temperature,
                coarse_intent.shape[0],
                sim_way=self.sim,
                intent_id=intent_id,
            )
            loss_n = nn.CrossEntropyLoss()(a, b)

            if i == 0:
                ficl_loss = loss_n
            else:
                ficl_loss += loss_n

        return ficl_loss

    def icsrec_loss(self, subsequence_1, subsequence_2, target_pos_1):
        # * intent representation learning task\
        cicl_loss, ficl_loss = 0.0, 0.0
        coarse_intent_1, coarse_intent_2 = self.model(subsequence_1), self.model(subsequence_2)
        subseq_pair = [coarse_intent_1, coarse_intent_2]
        if "c" in args.cl_mode:
            cicl_loss = self.cicl_loss(subseq_pair, target_pos_1)
        if "f" in args.cl_mode:
            ficl_loss = self.ficl_loss(subseq_pair, self.clusters_t[0])
        return cicl_loss, ficl_loss

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

        Args:
            gcn_dataloader (DataLoader): _description_

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
        print(torch.all(torch.diff(all_subseq_ids) > 0))

        id_padded_subseq_map = dict(zip(all_subseq_ids, all_subseq))
        return all_subseq_ids, all_subseq

class ICSRecTrainer(Trainer):

    def train_epoch(self, epoch, cluster_dataloader, train_dataloader):
        # * contrastive mode: {'cf':coarse-grain+fine-grain,'c':only coarse-grain,'f':only fine-grain}
        if args.cl_mode in ["cf", "f"]:
            with torch.no_grad():
                self.cluster_epoch(cluster_dataloader)

        if args.gcn_mode == "global":
            with torch.no_grad():
                self.gcn_epoch(cluster_dataloader)

        self.model.train()
        rec_avg_loss, joint_avg_loss, icl_losses = 0.0, 0.0, 0.0
        batch_num = len(train_dataloader)
        args.tb.add_scalar("train/LR", self.optim_adam.param_groups[0]["lr"], epoch, new_style=True)

        self.subseq_embed_init(self.cluster_dataloader)
        for batch_i, (rec_batch) in tqdm(
            enumerate(train_dataloader),
            total=batch_num,
            leave=False,
            desc=f"{args.save_name} | Device: {args.gpu_id} | Rec Training Epoch {epoch}",
            dynamic_ncols=True,
        ):
            # * rec_batch shape: key_name x batch_size x feature_dim
            rec_batch = tuple(t.to(self.device) for t in rec_batch)
            _, _, subsequence_1, target_pos_1, subsequence_2, _ = rec_batch

            # * prediction task
            intent_output = self.model(subsequence_1)
            logits = self.model.predict_full(intent_output[:, -1, :])
            rec_loss = nn.CrossEntropyLoss()(logits, target_pos_1[:, -1])

            cicl_loss, ficl_loss = 0.0, 0.0
            if args.cl_mode in ["c", "f", "cf"]:
                cicl_loss, ficl_loss = self.icsrec_loss(subsequence_1, subsequence_2, target_pos_1)

            icl_loss = args.lambda_0 * cicl_loss + args.beta_0 * ficl_loss
            joint_loss = args.rec_weight * rec_loss + icl_loss

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
            "epoch": epoch,
            "lr_adam": round(self.optim_adam.param_groups[0]["lr"], 6),
            "lr_adagrad": round(self.optim_adagrad.param_groups[0]["lr"], 6),
            "rec_avg_loss": round(rec_avg_loss / batch_num, 4),
            "icl_avg_loss": round(icl_losses / batch_num, 4),
            "joint_avg_loss": round(joint_avg_loss / batch_num, 4),
        }

        for key, value in post_fix.items():
            if "loss" in key:
                args.tb.add_scalar(f"train/{key}", value, epoch, new_style=True)

        if (epoch + 1) % args.log_freq == 0:
            # pprint_color(str(post_fix))
            args.logger.info(str(post_fix))

    def cluster_epoch(self, cluster_dataloader):
        """
        cluster datalooader contains
        5 tensors: user_id, input_id, target_pos_1, target_pos_2, anwser
        user_id, answer SHAPE: batch_size x 1
        input_id, target_pos_1, target_pos_2 SHAPE: batch_size x seq_len

        Args:
            cluster_dataloader (Dataloader):
        """
        assert cluster_dataloader is not None
        self.model.eval()
        kmeans_training_data = []

        subseq_embedding_dict = OrderedDict()

        for _, (rec_batch) in tqdm(
            enumerate(cluster_dataloader),
            total=len(cluster_dataloader),
            desc=f"{args.save_name} | Device: {args.gpu_id} | Cluster Training",
            leave=False,
            dynamic_ncols=True,
        ):
            rec_batch = tuple(t.to(self.device) for t in rec_batch)
            subseq_id, _, subsequence, _, _, _ = rec_batch
            # * SHAPE: [Batch_size, Seq_len, Hidden_size] -> [256, 50, 64]
            seq_output_last_layer = self.model(subsequence)
            # * SHAPE: [Batch_size, Hidden_size] -> [256, 64], use the last item as output.
            seq_output_last_item = seq_output_last_layer[:, -1, :]
            # * detach: Returns a new Tensor, detached from the current graph. The result will never require gradient.
            seq_output_last_item = seq_output_last_item.detach().cpu().numpy()
            kmeans_training_data.append(seq_output_last_item)

        # * SHAPE: [SubSeq_num, Hidden_size] -> [131413, 64]
        kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)
        kmeans_training_data_t = [kmeans_training_data]

        # * Cluster after encoding
        for i, clusters in enumerate(self.clusters_t):
            for j, cluster in enumerate(clusters):
                cluster.train(kmeans_training_data_t[i])
                self.clusters_t[i][j] = cluster

        del kmeans_training_data
        del kmeans_training_data_t
        gc.collect()

    def full_test_epoch(self, epoch: int, dataloader: DataLoader, mode):
        with torch.no_grad():
            self.model.eval()
            # * gcn is fixed in the test phase. So it's unnecessary to call gcn() every batch.
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
                recommend_output: Tensor = self.model.inference(input_ids)  # [BxLxH]
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
        self.train_epoch(epoch, self.cluster_dataloader, self.train_dataloader)

    def valid(self, epoch, full_sort=False) -> tuple[list[float], str]:
        assert self.eval_dataloader is not None
        return self.full_test_epoch(epoch, self.eval_dataloader, "valid")

    def test(self, epoch, full_sort=False) -> tuple[list[float], str]:
        if full_sort:
            return self.full_test_epoch(epoch, self.test_dataloader, "test")
        return self.sample_test_epoch(epoch, self.test_dataloader)

    def gcn_epoch(self, gcn_dataloader):
        """
        cluster datalooader contains
        5 tensors: user_id, input_id, target_pos_1, target_pos_2, anwser
        user_id, answer SHAPE: batch_size x 1
        input_id, target_pos_1, target_pos_2 SHAPE: batch_size x seq_len

        Args:
            cluster_dataloader (Dataloader):
        """
        self.model.eval()
        subseq_embedding_dict = OrderedDict()
        for _, (rec_batch) in tqdm(
            enumerate(gcn_dataloader),
            total=len(gcn_dataloader),
            desc=f"{args.save_name} | Device: {args.gpu_id} | GCN Training",
            leave=False,
            dynamic_ncols=True,
        ):
            rec_batch = tuple(t.to(self.device) for t in rec_batch)
            subseq_id, _, subsequence, _, _, _ = rec_batch
            # * subseq_emb: mean pooling of items in subsequence, ignore the padding item.
            pad_mask = (subsequence > 0).float()  # [batch_size, seq_len]
            subseq_emb = self.model.item_embeddings(subsequence)  # [batch_size, seq_len, hidden_size]
            num_non_pad = pad_mask.sum(dim=1, keepdim=True)  # [batch_size, 1]
            weighted_emb = subseq_emb * pad_mask.unsqueeze(-1)  # [batch_size, seq_len, hidden_size]
            subseq_emb_avg = weighted_emb.sum(dim=1) / num_non_pad  # [batch_size, hidden_size]
            for i in range(subseq_id.shape[0]):
                subseq_embedding_dict.setdefault(subseq_id[i].item(), subseq_emb_avg[i])

        subseq_emb_list = list(subseq_embedding_dict.values())
        subseq_emb = nn.Parameter(torch.stack(subseq_emb_list)).to(self.device)
        item_emb = nn.Parameter(self.model.item_embeddings.weight).to(self.device)
        gcn_subseq_emb, gcn_item_emb = self.gcn(self.graph.torch_A, subseq_emb, item_emb)

        # self.model.item_embeddings = nn.Embedding.from_pretrained(gcn_item_emb)
        # self.model.item_embeddings.weight.data.copy_(gcn_item_emb.detach())
        self.model.gcn_embeddings = nn.Embedding.from_pretrained(gcn_item_emb)

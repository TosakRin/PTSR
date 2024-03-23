# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#


import argparse
import gc
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from cprint import pprint_color
from models import GRUEncoder, KMeans, SASRecModel
from utils import get_metric, ndcg_k, recall_at_k


class Trainer:

    def __init__(
        self,
        model: Union[SASRecModel, GRUEncoder],
        train_dataloader: DataLoader,
        cluster_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        test_dataloader: DataLoader,
        args: argparse.Namespace,
    ) -> None:

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device: torch.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        self.batch_size: int = self.args.batch_size
        self.sim: str = self.args.sim  # * the calculate ways of the similarity.

        cluster = KMeans(
            num_cluster=args.intent_num,
            seed=1,
            hidden_size=64,
            gpu_id=args.gpu_id,
            device=self.device,
        )
        self.clusters: list[KMeans] = [cluster]
        self.clusters_t: list[list[KMeans]] = [self.clusters]

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.cluster_dataloader = cluster_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        self.optim = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        pprint_color(f">>> Total Parameters: {sum(p.nelement() for p in self.model.parameters())}")

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
        self.args.logger.info(str(post_fix))
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch: int, answers, pred_list):
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
        pprint_color(post_fix)
        self.args.logger.info(str(post_fix))
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name: str):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name: str):
        self.model.load_state_dict(torch.load(file_name))

    def mask_correlated_samples(self, batch_size: int):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    # False Negative Mask
    def mask_correlated_samples_(self, label):
        label = label.view(1, -1)
        label = label.expand((2, label.shape[-1])).reshape(1, -1)
        label = label.contiguous().view(-1, 1)
        mask = torch.eq(label, label.t())
        return mask == 0

    def info_nce(self, z_i, z_j, temp, batch_size: int, sim_way: str = "dot", intent_id=None):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        if sim_way == "cos":
            sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim_way == "dot":
            sim = torch.mm(z, z.t()) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        if self.args.f_neg:
            mask = self.mask_correlated_samples_(intent_id)
            negative_samples = sim
            negative_samples[mask == 0] = float("-inf")
        else:
            mask = self.mask_correlated_samples(batch_size)
            negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def predict_full(self, user_seq_emb: Tensor):
        """Predict: Rating = User_seq_emb * Item_emb^T

        Args:
            seq_out (Tensor): User sequence output. Use the last item output of last layer. SHAPE: [batch_size, hidden_size] -> [256, 64]

        Returns:
            Tensor: _description_
        """
        # * SHAPE: [Item_size, Hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # * SHAPE: [Batch_size, Item_size]
        rating_pred = torch.matmul(user_seq_emb, test_item_emb.transpose(0, 1))
        return rating_pred

    def cicl_loss(self, coarse_intents, target_item):
        coarse_intent_1, coarse_intent_2 = coarse_intents[0], coarse_intents[1]
        sem_nce_logits, sem_nce_labels = self.info_nce(
            coarse_intent_1[:, -1, :],
            coarse_intent_2[:, -1, :],
            self.args.temperature,
            coarse_intent_1.shape[0],
            self.sim,
            target_item[:, -1],
        )
        cicl_loss = nn.CrossEntropyLoss()(sem_nce_logits, sem_nce_labels)
        return cicl_loss

    def ficl_loss(self, sequences, clusters_t):
        output = sequences[0][:, -1, :]
        intent_n = output.view(-1, output.shape[-1])  # [BxH]
        intent_n = intent_n.detach().cpu().numpy()
        intent_id, seq_to_v = clusters_t[0].query(intent_n)

        seq_to_v = seq_to_v.view(seq_to_v.shape[0], -1)  # [BxH]
        a, b = self.info_nce(
            output.view(output.shape[0], -1),
            seq_to_v,
            self.args.temperature,
            output.shape[0],
            sim_way=self.sim,
            intent_id=intent_id,
        )
        loss_n_0 = nn.CrossEntropyLoss()(a, b)

        output_s = sequences[1][:, -1, :]
        intent_n = output_s.view(-1, output_s.shape[-1])
        intent_n = intent_n.detach().cpu().numpy()
        intent_id, seq_to_v_1 = clusters_t[0].query(intent_n)  # [BxH]
        seq_to_v_1 = seq_to_v_1.view(seq_to_v_1.shape[0], -1)  # [BxH]
        a, b = self.info_nce(
            output_s.view(output_s.shape[0], -1),
            seq_to_v_1,
            self.args.temperature,
            output_s.shape[0],
            sim_way=self.sim,
            intent_id=intent_id,
        )
        loss_n_1 = nn.CrossEntropyLoss()(a, b)
        ficl_loss = loss_n_0 + loss_n_1

        return ficl_loss


class ICSRecTrainer(Trainer):
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):
        super().__init__(model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args)

    def iteration(
        self,
        epoch: int,
        dataloader: DataLoader,
        cluster_dataloader: Optional[DataLoader] = None,
        full_sort=True,
        train=True,
    ):
        if train:
            pprint_color(f">>> Train Epoch: {epoch}")
            # * contrastive mode: {'cf':coarse-grain+fine-grain,'c':only coarse-grain,'f':only fine-grain}
            if self.args.cl_mode in ["cf", "f"]:
                assert cluster_dataloader is not None
                # ============== intentions clustering ================ #
                pprint_color(">>> Train Clustering in Train Epoch:")
                self.model.eval()
                # * save N
                kmeans_training_data = []

                # ============== Go through the encoder ================ #
                for i, (rec_batch) in tqdm(enumerate(cluster_dataloader), total=len(cluster_dataloader)):
                    rec_batch = tuple(t.to(self.device) for t in rec_batch)

                    # * 5 tensors: user_id, input_id, target_pos_1, target_pos_2, anwser
                    # * user_id, answer SHAPE: batch_size x 1
                    # * input_id, target_pos_1, target_pos_2 SHAPE: batch_size x seq_len

                    _, subsequence, _, _, _ = rec_batch
                    # * SHAPE: [Batch_size, Seq_len, Hidden_size] -> [256, 50, 64]
                    sequence_output_a = self.model(subsequence)
                    # * SHAPE: [Batch_size, Hidden_size] -> [256, 64], use the last item output.
                    sequence_output_b = sequence_output_a[:, -1, :]  # [BxH]
                    kmeans_training_data.append(sequence_output_b.detach().cpu().numpy())

                # * SHAPE: [SubSeq_num, Hidden_size] -> [131413, 64]
                kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)
                kmeans_training_data_t = [kmeans_training_data]

                # ================= Cluster after encoding ================== #
                for i, clusters in tqdm(enumerate(self.clusters_t), total=len(self.clusters_t)):
                    for j, cluster in enumerate(clusters):
                        cluster.train(kmeans_training_data_t[i])
                        self.clusters_t[i][j] = cluster

                # ================= clean memory ================= #
                del kmeans_training_data
                del kmeans_training_data_t
                gc.collect()

            # ================= model training ================== #
            pprint_color(">>> Performing Rec model Training:")
            self.model.train()
            rec_avg_loss = 0.0
            joint_avg_loss = 0.0
            icl_losses = 0.0

            batch_num = len(dataloader)
            pprint_color(f">>> rec dataset length (batch num): {batch_num}")

            for i, (rec_batch) in tqdm(enumerate(dataloader), total=batch_num):
                # * rec_batch shape: key_name x batch_size x feature_dim
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, subsequence_1, target_pos_1, subsequence_2, _ = rec_batch

                # ================= prediction task ================= #
                intent_output = self.model(subsequence_1)
                logits = self.predict_full(intent_output[:, -1, :])  #  [Bx|I|]
                rec_loss = nn.CrossEntropyLoss()(logits, target_pos_1[:, -1])

                # ================= intent representation learning task ================= #
                coarse_intent_1 = self.model(subsequence_1)
                coarse_intent_2 = self.model(subsequence_2)
                if self.args.cl_mode in ["c", "cf"]:
                    cicl_loss = self.cicl_loss([coarse_intent_1, coarse_intent_2], target_pos_1)
                else:
                    cicl_loss = 0.0
                if self.args.cl_mode in ["f", "cf"]:
                    ficl_loss = self.ficl_loss([coarse_intent_1, coarse_intent_2], self.clusters_t[0])
                else:
                    ficl_loss = 0.0
                icl_loss = self.args.lambda_0 * cicl_loss + self.args.beta_0 * ficl_loss

                # ================= multi-task learning =================#
                joint_loss = self.args.rec_weight * rec_loss + icl_loss

                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                rec_avg_loss += rec_loss.item()
                if not isinstance(icl_loss, float):
                    icl_losses += icl_loss.item()
                else:
                    icl_losses += icl_loss
                joint_avg_loss += joint_loss.item()

            # ================= print & write log for each epoch ================= #
            # * post_fix: print the average loss of the epoch
            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": round(rec_avg_loss / batch_num, 4),
                "icl_avg_loss": round(icl_losses / batch_num, 4),
                "joint_avg_loss": round(joint_avg_loss / batch_num, 4),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                pprint_color(str(post_fix))

            self.args.logger.info(str(post_fix))

        else:
            pprint_color(f">>> Valid Epoch: {epoch}")
            # ================= model evaluation ================== #
            self.model.eval()
            pred_list = None
            if full_sort:
                answer_list = None
                for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, _, answers = batch
                    # * SHAPE: [Batch_size, Seq_len, Hidden_size] -> [256, 50, 64]
                    recommend_output: Tensor = self.model(input_ids)  # [BxLxH]
                    # * Use the last item output. SHAPE: [Batch_size, Hidden_size] -> [256, 64]
                    recommend_output = recommend_output[:, -1, :]  # [BxH]

                    # * recommendation results. SHAPE: [Batch_size, Item_size]
                    rating_pred = self.predict_full(recommend_output)
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()

                    # * 将已经有评分的 item 的预测评分设置为 0, 防止推荐已经评分过的 item
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
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
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            # * sample-based sort: Not used in the paper
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                recommend_output = self.model.finetune(input_ids)
                test_neg_items = torch.cat((answers, sample_negs), -1)
                recommend_output = recommend_output[:, -1, :]

                test_logits = self.predict_sample(recommend_output, test_neg_items)
                test_logits = test_logits.cpu().detach().numpy().copy()
                if i == 0:
                    pred_list = test_logits
                else:
                    pred_list = np.append(pred_list, test_logits, axis=0)

            return self.get_sample_scores(epoch, pred_list)

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader, self.cluster_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

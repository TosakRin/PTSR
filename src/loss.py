import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models import KMeans
from param import args
from utils import mask_correlated_samples, mask_correlated_samples_


def info_nce(z_i: Tensor, z_j: Tensor, temp: float, batch_size: int, sim_way: str = "dot", intent_id=None):
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
        mask = mask_correlated_samples_(intent_id)
        # * SHAPE: [batch_size*2, batch_size*2]
        negative_samples = sim
        negative_samples[mask == 0] = float("-inf")
    else:
        mask = mask_correlated_samples(batch_size)
        negative_samples = sim[mask].reshape(N, -1)

    # * SHAPE: [batch_size*2]
    labels = torch.zeros(N).to(positive_samples.device).long()
    # * SHAPE: [batch_size*2, batch_size*2 + 1]
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    return logits, labels


def cicl_loss(coarse_intents: list[Tensor], target_item):
    """Coarse Intents: make 2 subsequence with the same target item closer by infoNCE.

    Args:
        coarse_intents (list[Tensor]): A list of coarse intents. Tensor SHAPE: [batch_size, seq_len, hidden_size]
        target_item (Tensor): The target item.

    Returns:
        Tensor: The calculated contrastive loss.
    """
    coarse_intent_1, coarse_intent_2 = coarse_intents[0], coarse_intents[1]
    sem_nce_logits, sem_nce_labels = info_nce(
        coarse_intent_1[:, -1, :],
        coarse_intent_2[:, -1, :],
        args.temperature,
        coarse_intent_1.shape[0],
        args.sim,
        target_item[:, -1],
    )
    return nn.CrossEntropyLoss()(sem_nce_logits, sem_nce_labels)


def ficl_loss(subseq_pair: list[Tensor], clusters_t: list[KMeans]):
    """ "
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
        a, b = info_nce(
            coarse_intent.view(coarse_intent.shape[0], -1),
            fined_intent,
            args.temperature,
            coarse_intent.shape[0],
            sim_way=args.sim,
            intent_id=intent_id,
        )
        loss_n = nn.CrossEntropyLoss()(a, b)

        if i == 0:
            ficl_loss = loss_n
        else:
            ficl_loss += loss_n

    return ficl_loss

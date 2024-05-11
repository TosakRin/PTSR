"""
/@
 @Author       : TosakRin sick_person@outlook.com
 @Date         : 2024-03-21 18:16:35
 @LastEditors  : TosakRin sick_person@outlook.com
 @LastEditTime : 2024-03-28 22:42:03
 @FilePath     : /ICSRec/src/param.py
 @Description  :
 @/
"""

import argparse

from cprint import pprint_color, print_color


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # * system args
    parser.add_argument("--data_dir", default="../data/", type=str)
    parser.add_argument("--output_dir", default="ckpt", type=str)
    parser.add_argument("--log_root", default=".", type=str)
    parser.add_argument("--log_dir", default="logs", type=str)
    parser.add_argument("--tb_dir", default="runs", type=str)
    parser.add_argument("--data_name", default="Sports_and_Outdoors", type=str)
    parser.add_argument("--encoder", default="SAS", type=str)  # * {"SAS":SASRec,"GRU":GRU4Rec}
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--model_idx", default=0, type=int, help="model idenfier 10, 20, 30...")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # * robustness experiments
    parser.add_argument(
        "--noise_ratio",
        default=0.0,
        type=float,
        help="percentage of negative interactions in a sequence - robustness analysis",
    )

    # * contrastive learning task args
    parser.add_argument(
        "--temperature", default=1.0, type=float, help="softmax temperature (default:  1.0) - not studied."
    )
    parser.add_argument("--intent_num", default=512, type=int, help="the multi intent nums!.")
    parser.add_argument("--sim", default="dot", type=str, help="the calculate ways of the similarity.")

    # * model args
    parser.add_argument("--model_name", default="ICSRec", type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--subseq_len", default=50, type=int)

    # * train args
    parser.add_argument("--lr_adam", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--lr_adagrad", type=float, default=0.01, help="learning rate of adagrad")

    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=2022, type=int)
    # * loss weight
    parser.add_argument("--rec_weight", type=float, default=1, help="weight of contrastive learning task")
    parser.add_argument(
        "--lambda_0", type=float, default=0.1, help="weight of coarse-grain intent contrastive learning task"
    )
    parser.add_argument("--beta_0", type=float, default=0.1, help="weight of fine-grain contrastive learning task")

    # * ablation experiments
    parser.add_argument("--cl_mode", type=str, default="", help="contrastive mode")
    # * {'cf':coarse-grain+fine-grain,'c':only coarse-grain,'f':only fine-grain}
    parser.add_argument(
        "--f_neg", action="store_true", help="delete the FNM (False Negative Mining) component (both in cicl and ficl)"
    )

    # * learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")

    # * GNN
    parser.add_argument("--gnn_layer", default=2, type=int, help="number of gnn layers")

    parser.add_argument("--warm_up_epochs", default=5, type=int, help="epoch number of warm up")
    parser.add_argument("--latest_path", default="output/ICSRec-SAS-Beauty-latest.pt", type=str)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--msg", type=str, default="", required=True)
    parser.add_argument("--precision", choices=["highest", "high", "medium"], default="highest")
    parser.add_argument(
        "--gcn_mode", type=str, default="None", help="gcn mode", choices=["None", "global", "batch", "batch_gcn"]
    )
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--min_test_epoch", default=0, type=int)
    parser.add_argument("--scheduler", default="None", type=str, help="scheduler")
    parser.add_argument("--milestones", nargs="?", default="[50,75,100]", help="milestones for MultiStepLR")
    parser.add_argument("--gamma", default=0.1, type=float, help="gamma for MultiStepLR")
    parser.add_argument("--batch_loss", action="store_true", help="Tensorboard record batch loss")
    parser.add_argument("--loader", default="None", type=str, help="dataloader mode")
    parser.add_argument("--recon", action="store_true", help="reconstruct the model")
    parser.add_argument("--graph_split", action="store_true", help="train use 50 subsequnce while graph use others")

    return parser.parse_args()


def print_args_info(args) -> None:
    """print the args info"""
    pprint_color("-------------------- Configure Info: -------------------- ")
    args_info = "".join(f"{arg:<30} : {getattr(args, arg):>35}\n" for arg in sorted(vars(args)))
    print_color("---------------------------------------------------------- ")
    return args_info


args: argparse.Namespace = parse_args()

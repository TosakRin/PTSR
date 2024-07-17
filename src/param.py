import argparse

from cprint import pprint_color, print_color


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # * system args
    parser.add_argument("--data_dir", default="../data/", type=str, help="data directory")
    parser.add_argument("--output_dir", default="../ckpt", type=str, help="checkpoint directory")
    parser.add_argument(
        "--log_root", default=".", type=str, help="log root directory, includding log files and tensorboard files"
    )
    parser.add_argument("--log_dir", default="logs", type=str, help="log directory in log_root")
    parser.add_argument("--tb_dir", default="runs", type=str, help="tensorboard directory in log_root")
    parser.add_argument("--data_name", default="Sports_and_Outdoors", type=str, help="dataset name")
    parser.add_argument("--encoder", default="SAS", type=str)
    parser.add_argument("--do_eval", action="store_true", help="do evaluation during training")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # * robustness experiments
    parser.add_argument(
        "--noise_ratio",
        default=0.0,
        type=float,
        help="percentage of negative interactions in a sequence - robustness analysis",
    )

    parser.add_argument("--sim", default="dot", type=str, help="the calculate ways of the similarity.")

    # * model args
    parser.add_argument("--model_name", default="PTSR", type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--subseq_len", default=50, type=int)

    # * train args
    parser.add_argument("--lr_adam", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument(
        "--f_neg", action="store_true", help="delete the FNM (False Negative Mining) component (both in cicl and ficl)"
    )

    # * learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--warm_up_epochs", default=5, type=int, help="epoch number of warm up")
    parser.add_argument("--do_test", action="store_true", help="do test simultaneously with training")
    parser.add_argument("--min_test_epoch", default=0, type=int, help="minimum epoch number to begin testing")
    parser.add_argument("--scheduler", default="None", type=str, help="scheduler")
    parser.add_argument("--gamma", default=0.1, type=float, help="gamma for MultiStepLR")
    parser.add_argument("--milestones", nargs="?", default="[50,75,100]", help="milestones for MultiStepLR")

    # * GNN
    parser.add_argument("--gnn_layer", default=2, type=int, help="number of gnn layers")
    parser.add_argument(
        "--gcn_mode", type=str, default="None", help="gcn mode", choices=["None", "global", "batch", "batch_gcn"]
    )
    parser.add_argument("--graph_split", action="store_true", help="train use 50 subsequnce while graph use others")
    parser.add_argument("--dropout_rate", type=float, default=0, help="dropout rate for Graph")

    #
    parser.add_argument("--latest_path", default="output/PTSR-SAS-Beauty-latest.pt", type=str)
    parser.add_argument("--compile", action="store_true", help="torch.compile to accelerate")
    parser.add_argument("--precision", choices=["highest", "high", "medium"], default="highest", help="torch precision")
    parser.add_argument("--msg", type=str, default="", required=True, help="msg for the run")
    parser.add_argument("--batch_loss", action="store_true", help="TensorBoard record batch loss")
    parser.add_argument(
        "--loader_type",
        default="None",
        choices=["new", "old"],
        type=str,
        help="dataloader mode, new for more efficient way and old for vanilla way.",
    )

    return parser.parse_args()


def print_args_info(args) -> None:
    """print the args info"""
    pprint_color("-------------------- Configure Info: -------------------- ")
    args_info = "".join(f"{arg:<30} : {getattr(args, arg):>35}\n" for arg in sorted(vars(args)))
    print_color("---------------------------------------------------------- ")
    return args_info


args: argparse.Namespace = parse_args()

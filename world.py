import os
import multiprocessing

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="DRO_SSM")
    # Learning   adv_lr  eta_epochs  warm_up_epochs alpha beta  clip_grad_norm reg_weight noise_ratio  learning_mode
    parser.add_argument("--activate_func", type=str, default = "exp")
    parser.add_argument("--valid_topks", nargs="?", default="[20]", help="@k test list")

    parser.add_argument("--num_quantile_negative_items", type = int)

    parser.add_argument("--diff_margin_and_topk", action="store_true")
    parser.add_argument("--lambda_k", type=int, default=10, help='The topk chosen')

    parser.add_argument("--shift_mode",type=str,default='others',help='shift mode')
    parser.add_argument("--learning_mode",type=str,default='personlized',help='noise ratio')

    parser.add_argument("--noise_ratio",type=float,default=0.125,help='noise ratio')

    parser.add_argument("--reg_weight",type=float,default=1e-6,help='llpauc alpha')
    parser.add_argument("--alpha",type=float,default=0.75,help='llpauc alpha')
    parser.add_argument("--beta",type=float,default=0.80,help='llpauc beta')
    parser.add_argument("--clip_grad_norm",type=float,default=0.80,help='llpauc clip_grad_norm')

    parser.add_argument("--adv_interval",type=int,default=5,help='the interval of adversarial training')
    parser.add_argument("--eta_epochs",type=int,default=5,help='eta_epochs')
    parser.add_argument("--warm_up_epochs",type=int,default=0,help='warm_up_epochs')

    parser.add_argument("--adv_lr",type=float,default=1e-3,help='adv lr rate')
    parser.add_argument("--w_emb_dim",type=int,default=64,help='adv embeding dim')
    parser.add_argument("--search_optimizer",type=str,default='SGD',help='[SGD,Adam]')
    parser.add_argument("--lr", type=float, default=1e-4, help="the learning rate")
    parser.add_argument("--lr2", type=float, default=1e-3, help="the maegin searching learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="the weight decay for l2 normalizaton")
    parser.add_argument("--train_batch", type=int, default=1024, help="the batch size for bpr loss training procedure")
    parser.add_argument("--testbatch", type=int, default=1024, help="the batch size of users for testing")
    parser.add_argument("--dataset", type=str, default="yelp2018_10")
    parser.add_argument("--datapath", type=str, default="IID_Data_Used")
    parser.add_argument("--topks", nargs="?", default="[20]", help="@k test list")
    parser.add_argument("--comment", type=str, default="", help="comment of running")
    parser.add_argument("--epochs", type=int, default=6001, help="total number of epochs")
    parser.add_argument("--multicore", type=int, default=0, help="whether we use multiprocessing or not in test")
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument("--recdim", type=int, default=64, help="the embedding size of lightGCN")
    parser.add_argument("--model", type=str, default="lgn", help="rec-model, support [mf, lgn]")
    parser.add_argument("--loss", type=str, default="softmax", help="loss function, support [bpr, softmax, renyi]")
    parser.add_argument("--norm_emb", type=int, default=1, help="whether normalize embeddings")
    parser.add_argument("--cuda", type=str, default="0", help="use which cuda")
    parser.add_argument("--full_batch", action="store_true")
    parser.add_argument("--resume_dir", type=str, default="")
    parser.add_argument("--sample_mode", type=str, default="uniform", help = "[uniform, no_sample]")
    parser.add_argument("--param_mode", type=str, default="artifical_set", help = "[artifical_set, adaptive]")
    parser.add_argument("--constant_protect", type=float, default=0, help = "protect nan gradient")


    parser.add_argument("--mode", type=str, default='lower bound learning')
    # SSM Loss
    parser.add_argument("--ssm_temp", type=float, default=0.1)
    parser.add_argument("--ssm_temp2", type=float, default=1.0)
    parser.add_argument("--num_negative_items", type=int, default=64)
    parser.add_argument("--neg_coefficient", type=float, default=1)

    """The neg_coefficient and num_negative_items is also used by renyi loss"""

    # renyi Loss
    parser.add_argument("--renyi_margin", type=float, default=0.7)
    parser.add_argument("--renyi_margin2", type=float, default=0.7)
    parser.add_argument("--renyi_omega", type=float, default=150)


    # LightGCN
    parser.add_argument("--layer", type=int, default=2, help="the layer num of lightGCN")
    parser.add_argument("--enable_dropout", type=int, default=0, help="using the dropout or not")
    parser.add_argument("--keepprob", type=float, default=0.0)
    
    # XSimGCL or SimGCL or LightGCL
    parser.add_argument("--cl_rate", type=float, default=0.001)         # the weight of InfoNCE
    parser.add_argument("--eps", type=float, default=0.1)               # the modulus of noise 
    parser.add_argument("--cl_temp", type=float, default=0.2)           # the contrastive learning temperature
    parser.add_argument("--cl_layer", type=int, default=0)              # the cl layer
    parser.add_argument("--q", type=int, default=5)

    #  cl_rate = [0.2]
    #      eps = [0.1, 0.2, 0.05]
    #  cl_temp = [0.2]
    # cl_layer = [0]

    #
    parser.add_argument("--sample_noise", type = int, default = 0)
    parser.add_argument("--sample_method", type = str, default = "random")

    return parser.parse_args()


args = parse_args()

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

all_models = ["lgn", "mf"]
config = {
    "activate_func": args.activate_func,
    "lambda_k": args.lambda_k,
    "shift_mode": args.shift_mode,
    "learning_mode": args.learning_mode,
    "noise_ratio": args.noise_ratio,
    "reg_weight": args.reg_weight,
    "clip_grad_norm": args.clip_grad_norm,
    "alpha": args.alpha,
    "beta":args.beta,
    
    "adv_interval": args.adv_interval,
    "warm_up_epochs": args.warm_up_epochs,
    "eta_epochs": args.eta_epochs,
    "adv_lr": args.adv_lr,
    "w_emb_dim":args.w_emb_dim,

    "search_optimizer":args.search_optimizer,
    "mode": args.mode,
    "constant_protect": args.constant_protect,
    "sample_mode": args.sample_mode,
    "param_mode": args.param_mode,
    "model": args.model,
    "dataset": args.dataset,
    "datapath": args.datapath,
    "train_batch": args.train_batch,
    "n_layers": args.layer,
    "latent_dim_rec": args.recdim,
    "enable_dropout": args.enable_dropout,
    "keep_prob": args.keepprob,
    "test_u_batch_size": args.testbatch,
    "multicore": args.multicore,
    "loss": args.loss,
    "lr": args.lr,
    "lr2": args.lr2,
    "weight_decay": args.weight_decay,
    "norm_emb": args.norm_emb,
    "full_batch": args.full_batch,
    "num_negative_items": args.num_negative_items,
    
    "cuda": args.cuda,
    "ssm_temp": args.ssm_temp,
    "ssm_temp2": args.ssm_temp2,
    "resume_dir": args.resume_dir,
    
    "neg_coefficient": args.neg_coefficient,
    
    "cl_rate": args.cl_rate,
    "eps": args.eps,
    "cl_temp": args.cl_temp,
    "cl_layer": args.cl_layer,
    "q": args.q,

    "renyi_margin": args.renyi_margin,
    "renyi_margin2": args.renyi_margin2,
    "renyi_omega": args.renyi_omega,



    "sample_noise": args.sample_noise,
    "sample_method": args.sample_method,


    "diff_margin_and_topk": args.diff_margin_and_topk,
    "num_quantile_negative_items": args.num_quantile_negative_items


}

CORES = multiprocessing.cpu_count() // 2
seed = args.seed
# dataset = args.dataset
model_name = args.model

TRAIN_epochs = args.epochs
topks = eval(args.topks)
valid_topks = None
if config["loss"] == "topk_loss":
    print(config["loss"])
    valid_topks = [args.lambda_k]
else:
    valid_topks = eval(args.valid_topks)


comment = args.comment

METHOD_CAT = None



def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")

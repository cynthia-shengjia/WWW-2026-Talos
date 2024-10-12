from optimizer.optim_Base import IROptimizer
from torch import nn
import torch



class AdvInfoNCEModel(IROptimizer):
    def __init__(self, model, config):
        super().__init__()

        # === Model ===
        self.model = model

        # === Hypter-parameter ===
        self.lr                 = config["lr"]
        self.weight_decay       = config["weight_decay"]
        self.temp               = config["ssm_temp"]
        self.adv_lr             = config["adv_lr"]
        self.adv_interval       = config["adv_interval"]
        self.eta_epoch          = config["eta_epoch"]

        self.w_emb_dim = config['w_emb_dim']

        # === Model Parameter ===
        self.embed_user_p = nn.Embedding(self.model.num_users, self.w_emb_dim)
        self.embed_item_p = nn.Embedding(self.model.num_items, self.w_emb_dim)
        nn.init.xavier_normal_(self.embed_user_p.weight)
        nn.init.xavier_normal_(self.embed_item_p.weight)

        # === Model Optimizer ===


    def cal_loss(self, y_pred):
        pass

    def regularize(self,users_emb, pos_emb, neg_emb):
        pass

    def cal_loss_graph(self, users, pos, neg):
        pass

    def step(self, user, pos, neg):
        pass




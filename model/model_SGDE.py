from model.model_Base import IRModel
from torch import nn
import torch.nn.functional as F
import torch


class SGDEModel(IRModel):
    def __init__(self, config: dict, num_users: int, num_items: int, SVD_User, SVD_Value, SVD_Item):
        super().__init__(config, num_users, num_items)


        self.std:float          = config["SGDE_std"]
        self.reg_vec:int        = config["SGDE_rec_vec"]
        self.beta:float         = config["SGDE_beta"]


        svd_filter          = torch.exp(self.beta * SVD_Value[:self.reg_vec])
        self.user_vector    = SVD_User[:,:self.reg_vec] * svd_filter
        self.item_vector    = SVD_Item[:,:self.reg_vec] * svd_filter
        self._init_weight()
        print(f"RSGDE is already to go(dropout:{config['enable_dropout']})")


    def _init_weight(self):
        self.FS = nn.Embedding(self.reg_vec, self.latent_dim)
        nn.init.xavier_uniform_(self.FS.weight)

    def compute(self):
        if self.training:
            users_emb = torch.normal(self.user_vector,  std =  self.std).mm(self.FS.weight)
            items_emb = torch.normal(self.item_vector,  std =  self.std).mm(self.FS.weight)
        else:
            users_emb =  self.user_vector.mm(self.FS.weight)
            items_emb =  self.item_vector.mm(self.FS.weight)


        if self.norm:
            users_emb = F.normalize(input = users_emb, p = 2, dim = 1)
            items_emb = F.normalize(input = items_emb, p = 2, dim = 1)
        return users_emb, items_emb
    
    
    def additional_loss(*args, **kwargs):
        return 0
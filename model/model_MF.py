# -------------------------------------------------------------------
# Copyright (c) 2025 Anonymous Authors of NeurIPS 2025 Submission 10239
# -------------------------------------------------------------------
# Module: Model - Matrix Factorization (MF)
# Description:
#  This module provides the Matrix Factorization (MF) model for ItemRec.
#  Reference:
#  - Y. Koren, R. Bell and C. Volinsky, "Matrix Factorization Techniques for Recommender Systems," 
#   in Computer, vol. 42, no. 8, pp. 30-37, Aug. 2009, doi: 10.1109/MC.2009.263.
# -------------------------------------------------------------------

from model.model_Base import IRModel
from torch import nn

import torch
import torch.nn.functional as F

class MFModel(IRModel):
    def __init__(self, config: dict, num_users: int, num_items: int, Graph = None):
        super().__init__(config, num_users, num_items)
        self._init_weight()
    def _init_weight(self):
        self.embedding_user = nn.Embedding(self.num_users,self.latent_dim)
        self.embedding_item = nn.Embedding(self.num_items,self.latent_dim)


    def compute(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        if self.norm:
            users_emb = F.normalize(input = users_emb, p = 2, dim = 1)
            items_emb = F.normalize(input = items_emb, p = 2, dim = 1)

        return users_emb, items_emb
    
    def additional_loss(*args, **kwargs):
        return 0
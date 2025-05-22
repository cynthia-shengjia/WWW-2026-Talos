# -------------------------------------------------------------------
# Copyright (C) 2025 Anonymous Authors of NeurIPS 2025
# -------------------------------------------------------------------
# Module: Model - Bilateral Softmax Loss
# Description:
#  This module provides the BSL (Bilateral Softmax Loss) Optimizer for 
#  ItemRec. BSL is a novel loss function for item recommendation, which
#  considers the bilateral robustness of both positive and negative items.
#  Reference:
#  - Wu, J., Chen, J., Wu, J., Shi, W., Zhang, J., & Wang, X. (2023). 
#   BSL: Understanding and Improving Softmax Loss for Recommendation. 
#   arXiv preprint arXiv:2312.12882.
# -------------------------------------------------------------------

from optimizer.optim_Base import IROptimizer
from torch import nn
import torch

class BSLOptimizer(IROptimizer):
    def __init__(self, model, config):
        super().__init__()

        # === Model ===
        self.model  = model

        # === Hyper-parameter ===
        self.lr             = config['lr']
        self.weight_decay   = config["weight_decay"]
        
        self.temp1           = config['ssm_temp']
        self.temp2           = config['ssm_temp2']

        # ====== Optimizer Oher Parameters ==========
        self.neg_weight     = self.temp2 / self.temp1

        # === Model Optimizer ===
        self.optimizer_descent = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)

    def cal_loss(self, y_pred, users):
        # clip parameter
        pos_logits = torch.exp(y_pred[:, 0] /  self.temp1 )                     # (B)
        neg_logits = torch.exp(y_pred[:, 1:] / self.temp2 )                     # (B,N)
        neg_logits = torch.pow(torch.sum(neg_logits, dim=-1), self.neg_weight)  # (B)


        loss = - torch.log(pos_logits / neg_logits).mean()

        return loss


    def cal_loss_graph(self, users, pos, neg):
        embedding_user, embedding_item = self.model.compute()

        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        loss            =  self.cal_loss(y_pred, users = users)
        additional_loss =  self.model.additional_loss(
                                usr_idx = users.long(), 
                                pos_idx = pos.long(), 
                                embedding_user = embedding_user, 
                                embedding_item = embedding_item
                            )
        return loss, additional_loss

    def step(self, user, pos, neg):
        ssm_loss,additional_loss = self.cal_loss_graph(user, pos, neg)
        loss = ssm_loss + additional_loss
        self.optimizer_descent.zero_grad()

        loss.backward()

        self.optimizer_descent.step()
        return ssm_loss.cpu().item()
    
    def save(self,path):
        all_states = self.model.state_dict()
        torch.save(obj = all_states, f = path)


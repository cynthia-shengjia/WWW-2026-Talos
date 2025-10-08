# -------------------------------------------------------------------
# Copyright (c) 2025 Anonymous Authors of WWW 2026 Submission 3321
# -------------------------------------------------------------------
# Module: Model - RS@K Loss
# Description:
#  This module provides the RS@K Optimizer for ItemRec.
#  RS@K is surrogate loss for Recall@K.
#  Reference:
#  Patel, Y., Tolias, G., & Matas, J. (2022). Recall@ k surrogate loss with large batches and similarity mixup. 
#  In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 7502-7511).
# -------------------------------------------------------------------




from optimizer.optim_Base import IROptimizer
from torch import nn
import torch

class CVPROptimizer(IROptimizer):
    def __init__(self, model, config):
        super().__init__()

        # === Model ===
        self.model  = model

        # === Hyper-parameter ===
        self.lr             = config['lr']
        self.weight_decay   = config["weight_decay"]
        self.temp           = config['ssm_temp']
        self.temp2          = config["rank_temp"]
        self.lambda_k       = config["lambda_k"]


        # === Model Optimizer ===

        self.activation        = lambda x: torch.sigmoid(x)
        self.optimizer_descent = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)


    def cal_loss(self, y_pred):
        # clip parameter
        d       = (y_pred[:, 1:] - y_pred[:, 0].unsqueeze(dim = 1)) 
        ranking = self.activation(d / self.temp).sum(dim = 1)
        loss    =  1 - self.activation( (self.lambda_k - 1 - ranking) /self.temp2 )         

        return loss.mean()


    def cal_loss_graph(self, users, pos, neg):
        embedding_user, embedding_item = self.model.compute()

        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        loss            =  self.cal_loss(y_pred)
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


# -------------------------------------------------------------------
# Copyright (c) 2025 Anonymous Authors of NeurIPS 2025 Submission 10239
# -------------------------------------------------------------------
# Module: Model - TLatK Optimizer
# Description:
#  This module provides the NeurIPS 2025 submisstion TL@K (Top-K Loss) Optimizer for ItemRec.
# -------------------------------------------------------------------

from torch import nn
from optimizer.optim_Base import IROptimizer
import torch

class ExpOptimizer(IROptimizer):
    def __init__(self, model, config):
        super().__init__()

        # === Model ===
        self.model = model

        # === Hyper-parameter ===
        self.lr             = config["lr"]
        self.weight_decay   = config["weight_decay"]
        self.temp           = config["ssm_temp"]
        self.temp_beta      = config["ssm_temp_beta"]
        self.lambda_k       = config["lambda_k"]
        self.activation  = lambda x:   torch.sigmoid(x / self.temp_beta)  

        # === Model Optimizer ===

        self.quantile =  torch.zeros((self.model.num_users, 1)).cuda()  

        self.optimizer_descent = torch.optim.Adam([
            {'params': self.model.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
        ])


    def cal_quantile(self, users, user_all_pos, users_emb, all_pos_emb,  neg_emb):

        pos_scores      = torch.bmm(users_emb.unsqueeze(1), all_pos_emb.transpose(1, 2)).squeeze(1)
        mask            = (pos_scores ==0) * -1
        pos_scores     += mask
        neg_scores      = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        all_scores      = torch.cat( (pos_scores, neg_scores), dim = 1)

        with torch.no_grad():
            quantile = torch.topk(all_scores, self.lambda_k, dim=1)[0][:, -1]
            self.quantile[users] = quantile.unsqueeze(dim=1)

        return quantile

    def cal_loss(self, users, y_pred: torch.Tensor, quantile: torch.Tensor):
        trunc_pos    = y_pred[:,0] - quantile
        d            = y_pred[:,1:] - y_pred[:,0].unsqueeze(dim = 1)
        softmax_loss = torch.logsumexp(d / self.temp, dim = 1)
        weight       = self.activation(trunc_pos)
        loss         = weight * softmax_loss
        return loss.mean()


    def cal_loss_graph(self,users, pos, user_all_pos, neg):
        embedding_user, embedding_item = self.model.compute()

        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]
        
        # all_pos_embedding
        embedding_item_add = torch.cat( ( embedding_item, torch.zeros(1, self.model.latent_dim).cuda()) )   # padding
        all_pos_emb        = embedding_item_add[user_all_pos]

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        
        quantile = self.cal_quantile(users, user_all_pos, users_emb.detach(), all_pos_emb.detach(), neg_emb.detach())
        loss = self.cal_loss(users, y_pred, quantile)
        additional_loss =  self.model.additional_loss(
                        usr_idx = users.long(), 
                        pos_idx = pos.long(), 
                        embedding_user = embedding_user, 
                        embedding_item = embedding_item
                    )
        return loss, additional_loss

    def regularize(self,users_emb, pos_emb, neg_emb):
        regularize = (torch.norm(users_emb[:, :]) ** 2
                      + torch.norm(pos_emb[:, :]) ** 2
                      + torch.norm(neg_emb[:, :]) ** 2) / 2  # take hop=0
        return regularize

    def step(self, user, pos, user_all_pos, neg):
        
        # Second stage, compute the loss 
        ssm_loss,additional_loss = self.cal_loss_graph(user, pos, user_all_pos, neg)
        loss = ssm_loss + additional_loss
        self.optimizer_descent.zero_grad()

        loss.backward()

        self.optimizer_descent.step()
        return ssm_loss.cpu().item()

    def save(self,path):
        all_states = self.model.state_dict()
        all_states.update({
            "quantile": self.quantile.detach()
        })
        torch.save(obj = all_states, f = path)
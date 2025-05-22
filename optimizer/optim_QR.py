# -------------------------------------------------------------------
# Copyright (C) 2025 Anonymous Authors of NeurIPS 2025
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

        self.lr2            = config["lr2"]


        self.lambda_k       = config["lambda_k"]

        self.lambda_t       = self.lambda_k / self.model.num_items

        self.model_items    = self.model.num_items
        self.sampled_neg    = config["num_negative_items"]
        

        # activation function choice
        act_dict = {
            "sigmoid":  lambda x: torch.log( torch.sigmoid(x)  ),
        }

        self.activation  = act_dict[config["activate_func"]]

        # mode: <mutli> or <single>
        self.mode           = config["mode"]



        # === Model Optimizer ===

        self.quantile =nn.Parameter( (torch.zeros((self.model.num_users, 1))).cuda() )

        self.optimizer_descent = torch.optim.Adam([
            {'params': self.model.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
            {'params': self.quantile, "lr": self.lr2}
        ])


    def quantile_regression(self, users, user_all_pos, users_emb, all_pos_emb,  neg_emb):

        pos_scores      = torch.bmm(users_emb.unsqueeze(1), all_pos_emb.transpose(1, 2)).squeeze(1)
        neg_scores      = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        user_quantile   = self.quantile[users.long()]
        check_sum       = (user_all_pos != self.model_items)

        trunc_pos       = (  (1 - self.lambda_t) * torch.relu(pos_scores - user_quantile) + self.lambda_t * torch.relu(user_quantile - pos_scores)   ) * check_sum
        trunc_neg       = (1 - self.lambda_t) * torch.relu(neg_scores - user_quantile) + self.lambda_t * torch.relu(user_quantile - neg_scores)
        
        weight =  (self.model_items - check_sum.sum(dim = 1).unsqueeze(dim = 1)) / self.sampled_neg

        all_scores      = (torch.cat((trunc_pos, weight * trunc_neg), dim = 1)).sum(dim = 1) / self.model_items


        return all_scores.mean()

    def cal_loss(self, users, y_pred: torch.Tensor, quantile: torch.Tensor):
        trunc_pos = y_pred[:,0] - quantile.detach().squeeze()
        trunc_neg = y_pred[:,1:] - quantile.detach()


        pos_logits = torch.log( torch.exp( self.activation(trunc_pos)   / self.temp)        )
        neg_logits = torch.logsumexp( self.activation(trunc_neg )  / self.temp, dim = 1     )

        loss = neg_logits - pos_logits

        return loss.mean()


    def cal_loss_graph(self,users, pos, user_all_pos, neg, quantile):
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

        loss = self.cal_loss(users, y_pred,quantile)
        quantile_loss = self.quantile_regression(users, user_all_pos, users_emb.detach(), all_pos_emb.detach(), neg_emb.detach())
        additional_loss =  self.model.additional_loss(
                        usr_idx = users.long(), 
                        pos_idx = pos.long(), 
                        embedding_user = embedding_user, 
                        embedding_item = embedding_item
                    )
        return loss, additional_loss + quantile_loss

    def regularize(self,users_emb, pos_emb, neg_emb):
        regularize = (torch.norm(users_emb[:, :]) ** 2
                      + torch.norm(pos_emb[:, :]) ** 2
                      + torch.norm(neg_emb[:, :]) ** 2) / 2  # take hop=0
        return regularize

    def step(self, user, pos, user_all_pos, neg):
        
        # First stage,  compute the Top-K quantile.
        topk_quantile = self.quantile[user.long()]

        # Second stage, compute the loss 
        ssm_loss,additional_loss = self.cal_loss_graph(user, pos, user_all_pos, neg, topk_quantile)
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
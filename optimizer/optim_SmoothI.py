# -------------------------------------------------------------------
# Copyright (c) 2025 Anonymous Authors of WWW 2026 Submission 3321
# -------------------------------------------------------------------
# Module: Model - SmoothI@K Loss
# Description:
#  This module provides the SmoothI@K Optimizer for ItemRec.
#  SmoothI is surrogate for Top-K indicator. The SmoothI@K loss that we implementd is from:
#  https://github.com/ygcinar/SmoothI/blob/main/src/losses.py
#  It employs the precision@K surrogate form.
#  Reference:
#  Thonet T, Cinar Y G, Gaussier E, et al. Listwise learning to rank based on approximate rank indicators
#  Proceedings of the AAAI Conference on Artificial Intelligence. 2022, 36(8): 8494-8502.
# -------------------------------------------------------------------


from optimizer.optim_Base import IROptimizer
from torch import nn
import torch


class ListwiseSmoothIOptimizer(IROptimizer):
    def __init__(self, model, config):
        super().__init__()

        # === Model ===
        self.model  = model

        # === Hyper-parameter ===
        self.lr             = config['lr']
        self.weight_decay   = config["weight_decay"]
        self.alpha          = float(1 / config['ssm_temp'])
        self.delta          = config["SmoothI_delta"]
        self.lambda_k       = config["lambda_k"]

        # === Model Optimizer ===
        self.optimizer_descent = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)


    def make_pos_by_subtraction(self, scores):
        """
        Shift the scores to ensure positivity
        Args:
            scores: torch tensor of shape (batch_size, max.#ofdocs for a query) corresponds to scores
                                          (output of [batch_size,listLenght] logits)
        """
        min_vals, _ = torch.min(scores, -1)                     # min. logit values   [1,batchsize]
        return scores -  min_vals.unsqueeze(dim=1)              # .shape (bs, ll)

    def cal_loss(self, y_pred):
        # clip parameter

        batch_size, listlength = y_pred.shape                           # s.shape: (batchsize, listlength)
        shiftScores           = self.make_pos_by_subtraction(y_pred)    # shift the scores to postive
        prod                  = torch.ones(1).cuda()                    # I_jˆ{1,alpha} = 1
        loss                  = torch.zeros((batch_size)).cuda()        
        for k in range(self.lambda_k):
            logits       =   shiftScores * self.alpha * prod
            approx_inds  =   torch.softmax(logits, -1)[:,0]                                      # I_jˆ{k,alpha}, (1,B)
            prod         =   prod * (1 - approx_inds.unsqueeze(dim=1) - self.delta).detach()     # recursive compute the smooth indicator
            loss        +=  approx_inds

        pk  =  1 - loss / self.lambda_k
        return pk.mean()


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


# -------------------------------------------------------------------
# Copyright (c) 2025 Anonymous Authors of NeurIPS 2025 Submission 10239
# -------------------------------------------------------------------
# Module: Model - LLPAUC (Lower-Left Partial AUC)
# Description:
#  This module provides the LLPAUC Optimizer for ItemRec. LLPAUC is a partial AUC
#  loss function for item recommendation, which focuses on the lower-left part of
#  the ROC curve so as to optimize Top-K metrics.
#  Reference:
#  - Shi, W., Wang, C., Feng, F., Zhang, Y., Wang, W., Wu, J., & He, X. (2024). 
#   Lower-Left Partial AUC: An Effective and Efficient Optimization Metric for Recommendation. 
#   arXiv preprint arXiv:2403.00844.
# -------------------------------------------------------------------

from optimizer.optim_Base import IROptimizer
from torch import nn
import torch

class LLPAUCOptimizer(IROptimizer):
    def __init__(self, model, config):
        super().__init__()

        # === Model ===
        self.model  = model

        # === Hyper-parameter ===
        self.lr             = config['lr']
        self.weight_decay   = config["weight_decay"]
        self.alpha          = config['alpha']
        self.beta           = config['beta']

        # === Model parameter ===
        self.a          = nn.Parameter((torch.tensor([1.0])).cuda())
        self.b          = nn.Parameter((torch.tensor([0.0])).cuda())
        self.gamma      = nn.Parameter((torch.tensor([0.0])).cuda())
        self.sn         = nn.Parameter((torch.tensor([0.5])).cuda())
        self.sp         = nn.Parameter((torch.tensor([0.5])).cuda())
        self.theta_b    = nn.Parameter((torch.tensor([0.5])).cuda())
        self.theta_a    = nn.Parameter((torch.tensor([0.5])).cuda())

        # === Model Optimizer ===
        self.optimizer_descent = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.lr},
            {'params': [self.a, self.b], 'name': 'ab', 'lr': self.lr},
            {'params': self.sn, 'name': 'sn', 'lr': self.lr * 2},
            {'params': self.sp, 'name': 'sp', 'lr': self.lr * 2},
            {'params': self.theta_b, 'name': 'lamn', 'lr': self.lr},
            {'params': self.theta_a, 'name': 'lamp', 'lr': self.lr},
            {'params': self.gamma, 'name': 'g', 'lr': self.lr * 2}
        ], weight_decay = self.weight_decay)

    def _clip(self) -> None:
        self.a.data.clamp_(min = 0, max = 1)
        self.b.data.clamp_(min = 0, max = 1)
        self.gamma.data.clamp_(min = -1, max = 1)
        self.theta_a.data.clamp_(min = 0, max = 1e9)
        self.theta_b.data.clamp_(min = 0, max = 1e9)
        self.sp.data.clamp_(min = -1, max = 4)
        self.sn.data.clamp_(min = 0, max = 5)


    def cal_loss(self, y_pred):
        # clip parameter
        self._clip()

        pos_score = torch.sigmoid(y_pred[:, 0])
        neg_score = torch.sigmoid(y_pred[:, 1:])
        max_val_p = torch.log(1 + torch.exp(5 * (-torch.square(pos_score - self.a) + 2 * (1 + self.gamma) * pos_score - self.sp))) / 5
        max_val_n = torch.log(1 + torch.exp(5 * (torch.square(neg_score - self.b) + 2 * (1 + self.gamma) * neg_score - self.sn))) / 5
        loss = (
                -self.sp - torch.mean(max_val_p) / self.alpha + self.sn + torch.mean(max_val_n) / self.beta + -self.gamma ** 2 - self.theta_b * (self.b - 1 - self.gamma) +
                self.theta_a * (self.a + self.gamma)
        )
        return loss


    def cal_loss_graph(self, users, pos, neg):
        embedding_user, embedding_item = self.model.compute()

        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]
        batch_size = users_emb.shape[0]

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
        llpauc_loss,additional_loss = self.cal_loss_graph(user, pos, neg)
        loss = llpauc_loss + additional_loss
        self.optimizer_descent.zero_grad()

        loss.backward()
        self.gamma.grad = -self.gamma.grad
        self.sp.grad    = -self.sp.grad

        self.optimizer_descent.step()
        return llpauc_loss.cpu().item()

    def save(self, path):
        all_states = self.model.state_dict()

        all_states.update({
            "a":        self.a.detach(),
            "b":        self.b.detach(),
            "gamma":    self.gamma.detach(),
            "sn":       self.sn.detach(),
            "theta_b":  self.theta_b.detach(),
            "theta_a":  self.theta_a.detach()
        })
        
        torch.save(obj = all_states, f = path)

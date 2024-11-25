from optimizer.optim_Base import IROptimizer
from torch import nn
import torch
import math

class DrRLOptimizer(IROptimizer):
    def __init__(self, model, config):
        super().__init__()

        # === Model ===
        self.model  = model

        # === Hyper-parameter ===
        self.lr             = config['lr']
        self.weight_decay   = config["weight_decay"]

        self.lr2            = config["lr2"]
        self.margin         = config["renyi_margin"]
        self.omega          = config["renyi_omega"]

        #   Modell Additional Parameter
        self.margin_vector  = torch.full(
            size = (self.model.num_users,1), 
            fill_value = self.margin, 
            dtype = torch.float32,
            device = torch.device("cuda")
        )
        self.margin_vector  = nn.Parameter(self.margin_vector)
        
        
        self.omega_star     = self.omega / (self.omega - 1)
        self.eps            = 1e-1
        self.negative_items = config["num_negative_items"]
        self.neg_weight     = config["neg_coefficient"]

        # === Model Optimizer ===
        self.optimizer_descent = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.lr}
        ], weight_decay = self.weight_decay)
        self.optimizer_margin  = torch.optim.SGD([
            {'params': self.margin_vector, 'lr': self.lr2}
        ])

    def cal_loss(self, y_pred, margin_vec):
        pos_logits = y_pred[:, 0]
        neg_logits = y_pred[:, 1:]
        constant = math.pow(self.negative_items, -1 / self.omega_star)
        loss_search_margin = (constant * torch.norm( self.eps + torch.relu(neg_logits.detach() - margin_vec) * self.neg_weight, self.omega_star,dim = 1) + margin_vec.squeeze()).sum()

        self.optimizer_margin.zero_grad()
        loss_search_margin.backward()
        self.optimizer_margin.step()

        pos_loss = torch.relu(1 - pos_logits)
        neg_loss = constant * torch.norm( self.eps + torch.relu(neg_logits - margin_vec.detach())  * self.neg_weight, self.omega_star,dim = 1) + (margin_vec.detach()).squeeze()


        loss = (pos_loss + neg_loss).mean()

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

        margin = self.margin_vector[users.long()]

        loss            =  self.cal_loss(y_pred = y_pred, margin_vec = margin)
        additional_loss =  self.model.additional_loss(
                                usr_idx = users.long(), 
                                pos_idx = pos.long(), 
                                embedding_user = embedding_user, 
                                embedding_item = embedding_item
                            )
        return loss, additional_loss

    def step(self, user, pos, neg):     
        DrRL_loss,additional_loss = self.cal_loss_graph(user, pos, neg)
        loss = DrRL_loss + additional_loss
        self.optimizer_descent.zero_grad()
        loss.backward()
        self.optimizer_descent.step()
        return DrRL_loss.cpu().item()
    
    def save(self,path):
        all_states = self.model.state_dict()
        all_states.update({
            "renyi_margin": self.margin_vector.detach()
        })
        torch.save(obj = all_states, f = path)


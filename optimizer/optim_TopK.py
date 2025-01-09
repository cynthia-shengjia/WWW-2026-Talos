from torch import nn
from optimizer.optim_Base import IROptimizer
import torch

class TopKOptimizer(IROptimizer):
    def __init__(self, model, config):
        super().__init__()

        # === Model ===
        self.model = model

        # === Hyper-parameter ===
        self.lr             = config["lr"]
        self.weight_decay   = config["weight_decay"]
        self.temp           = config["ssm_temp"]

        self.lambda_k       = config["lambda_k"]

        self.mode           = config["mode"]


        self.activation  = lambda x: torch.log(torch.sigmoid(x))


        # === Model Optimizer ===
        self.optimizer_descent = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)



    def cal_loss(self, users, y_pred: torch.Tensor, quantile: torch.Tensor):
        trunc_pos = y_pred[:,0] - quantile.squeeze()
        trunc_neg = y_pred[:,1:] - quantile

        pos_logits = torch.log( torch.exp( self.activation(trunc_pos)   / self.temp)        )
        neg_logits = torch.logsumexp( self.activation(trunc_neg )  / self.temp, dim = 1     )

        loss = neg_logits - pos_logits

        return loss.mean()


    def cal_loss_graph(self,users, pos, neg):
        embedding_user, embedding_item = self.model.compute()

        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]

        with torch.no_grad():
            scores = torch.mm(users_emb, embedding_item.t())
            topk_quantile = torch.topk(input = scores, k = self.lambda_k, dim = -1)[0][:,-1].unsqueeze(1).detach()



        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        loss = self.cal_loss(users, y_pred,topk_quantile)
        additional_loss =  self.model.additional_loss(
                        usr_idx = users.long(), 
                        pos_idx = pos.long(), 
                        embedding_user = embedding_user, 
                        embedding_item = embedding_item
                    )
        return loss, additional_loss


    def step(self, user, pos, neg):
        

        # Second stage, compute the loss 
        ssm_loss,additional_loss = self.cal_loss_graph(user, pos, neg)
        loss = ssm_loss + additional_loss
        self.optimizer_descent.zero_grad()

        loss.backward()

        self.optimizer_descent.step()
        return ssm_loss.cpu().item()

     


    def save(self,path):
        all_states = self.model.state_dict()
        torch.save(obj = all_states, f = path)
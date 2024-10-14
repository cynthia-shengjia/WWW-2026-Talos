from torch import nn
from optimizer.optim_Base import IROptimizer
import torch

class PreAtKOptimizer(IROptimizer):
    def __init__(self, model, config):
        super().__init__()

        # === Model ===
        self.model = model

        # === Hyper-parameter ===
        self.lr             = config["lr"]
        self.weight_decay   = config["weight_decay"]
        self.temp           = config["ssm_temp"]

        self.lambda_k       = config["lambda_k"]

        # === Model Optimizer ===
        self.optimizer_descent = torch.optim.Adam(self.model.parameters(), lr = self.lr)


    def cal_loss(self, y_pred: torch.Tensor, quantile: torch.Tensor):
        trunc_pos = y_pred[:,0] - quantile.squeeze()
        trunc_neg = y_pred[:,1:] - quantile

        pos_logits = torch.sigmoid(trunc_pos  / self.temp)
        neg_logits = torch.sigmoid(trunc_neg  / self.temp)

        loss = -torch.log(pos_logits /  neg_logits.sum(dim = 1))

        return loss.mean()

    def regularize(self,users_emb, pos_emb, neg_emb):
        regularize = (torch.norm(users_emb[:, :]) ** 2
                      + torch.norm(pos_emb[:, :]) ** 2
                      + torch.norm(neg_emb[:, :]) ** 2) / 2  # take hop=0
        return regularize
    def cal_loss_graph(self,users, pos, neg, quantile):
        embedding_user, embedding_item = self.model.compute()

        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]
        batch_size = users_emb.shape[0]

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        loss = self.cal_loss(y_pred,quantile)
        emb_loss = self.weight_decay * self.regularize(users_emb, pos_emb, neg_emb) / batch_size
        return loss, emb_loss


    def step(self, user, pos, neg, epoch = None):
        
        # First stage,  compute the Top-K quantile.
        topk_quantile = None
        with torch.no_grad():
            topk_quantile = self.compute_topks(users = user)

        # Second stage, compute the loss 
        ssm_loss,emb_loss = self.cal_loss_graph(user, pos, neg, topk_quantile)
        loss = ssm_loss + emb_loss
        self.optimizer_descent.zero_grad()

        loss.backward()

        self.optimizer_descent.step()
        return ssm_loss.cpu().item()

    def compute_topks(self, users):
        """
            For Ablation Study, Computing the Top-K Quantile of All items
        """

        embedding_user, embedding_item = self.model.compute()
        users_emb = embedding_user[users.long()]
        scores = users_emb @ embedding_item.T
        return (torch.topk(input = scores, dim = 1, k = self.lambda_k)[0][:,-1]).unsqueeze(dim = 1)

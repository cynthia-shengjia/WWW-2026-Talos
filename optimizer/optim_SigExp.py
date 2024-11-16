from torch import nn
from optimizer.optim_Base import IROptimizer
import torch

class SigExpOptimizer(IROptimizer):
    def __init__(self, model, config):
        super().__init__()

        # === Model ===
        self.model = model

        # === Hyper-parameter ===
        self.lr             = config["lr"]
        self.weight_decay   = config["weight_decay"]
        self.temp           = config["ssm_temp"]
        self.temp2          = config["ssm_temp2"]
        self.mode           = config["mode"]

        self.lambda_k       = config["lambda_k"]

        self.activation     = lambda x: torch.log( torch.sigmoid(x)  )
        self.activation2    = lambda x: torch.log( torch.exp(x)      )

        # === Model Optimizer ===
        self.optimizer_descent = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        self.quantile = (torch.zeros((self.model.num_users, 1))).cuda()



    def cal_loss(self, y_pred: torch.Tensor, quantile: torch.Tensor):
        trunc_pos = y_pred[:,0] - quantile.squeeze()
        trunc_neg = y_pred[:,1:] - quantile
        if self.mode == "outside":
            pos_logits = torch.log( torch.exp( self.activation2(trunc_pos)   / self.temp2)        )
            neg_logits = torch.logsumexp( self.activation(trunc_neg )  / self.temp, dim = 1     )
        elif self.mode == "inner":
            pos_logits = torch.log( torch.exp( self.activation2(trunc_pos )  / self.temp2 )       )
            neg_logits = torch.logsumexp( self.activation(trunc_neg / self.temp )  , dim = 1     )

        loss = neg_logits - pos_logits

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
        additional_loss =  self.model.additional_loss(
                        usr_idx = users.long(), 
                        pos_idx = pos.long(), 
                        embedding_user = embedding_user, 
                        embedding_item = embedding_item
                    )
        return loss, emb_loss + additional_loss


    def step(self, user, pos, neg):
        
        # First stage,  compute the Top-K quantile.
        topk_quantile = self.quantile[user.long()]

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
        self.quantile[users.long()] = (torch.topk(input = scores, dim = 1, k = self.lambda_k)[0][:,-1]).unsqueeze(dim = 1)

    def estimate_topks(self, users, pos, neg):
        """
            Using All Positive Items and Sampled Negative Items to Estimate
        """
        embedding_user, embedding_item = self.model.compute()                                               # user and item embeddings

        embedding_item_add = torch.cat( ( embedding_item, torch.zeros(1, self.model.latent_dim).cuda()) )   # padding

        users_emb = embedding_user[users.long()]        # (B,dim)
        pos_emb = embedding_item_add[pos]               # (B,PadSize,dim)
        neg_emb = embedding_item[neg.long()]            # (B,PadSize,dim)



        pos_scores = torch.bmm(users_emb.unsqueeze(1), pos_emb.transpose(1, 2)).squeeze(1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)


        pos_check   =  torch.not_equal(pos, torch.full_like(pos, self.model.num_items).to(torch.int64))

        
        padding_all_scores  =  torch.cat( (self.model.f(pos_scores) * pos_check, self.model.f(neg_scores)), dim = 1 )
        all_scores          =  torch.cat( (pos_scores, neg_scores)                            , dim = 1 )


        _,topk_index        =  torch.topk( input = padding_all_scores, dim = 1, k = self.lambda_k       )

        self.quantile[users.long()] = torch.gather(input = all_scores, dim = 1, index = topk_index[:, -1].unsqueeze(dim = 1)          )
        


    def save(self,path):
        all_states = self.model.state_dict()
        all_states.update({
            "quantile": self.quantile.detach()
        })
        torch.save(obj = all_states, f = path)
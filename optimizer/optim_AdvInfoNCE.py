from optimizer.optim_Base import IROptimizer
from torch import nn
import torch



class AdvInfoNCEOptimizer(IROptimizer):
    def __init__(self, model, config):
        super().__init__()

        # === Model ===
        self.model = model

        # === Hypter-parameter ===
        self.lr                 = config["lr"]
        self.weight_decay       = config["weight_decay"]


        self.temp               = config["ssm_temp"]
        self.adv_lr             = config["adv_lr"]
        self.adv_interval       = config["adv_interval"]        # It would not be used in step, but used in mainAdvInfoNCE
        self.eta_epoch          = config["eta_epochs"]          # It would not be used in step, but used in mainAdvInfoNCE
        self.neg_weight         = config["neg_coefficient"]

        self.w_emb_dim          = 64

        # === Model Parameter ===
        self.embed_user_p = nn.Embedding(self.model.num_users, self.w_emb_dim)
        self.embed_item_p = nn.Embedding(self.model.num_items, self.w_emb_dim)
        nn.init.xavier_normal_(self.embed_user_p.weight)
        nn.init.xavier_normal_(self.embed_item_p.weight)

        self.embed_user_p = (self.embed_user_p).cuda()
        self.embed_item_p = (self.embed_item_p).cuda()

        # === Model Optimizer ===
        self.opt_model  =   torch.optim.Adam([ {    "params": self.model.parameters(), "lr": self.lr } ])
        self.opt_adv    =   torch.optim.Adam([ 
            {    "params": self.embed_user_p.parameters(), "lr": self.adv_lr }, 
            {    "params": self.embed_item_p.parameters(), "lr": self.adv_lr }  
        ])      


    def cal_loss(self, y_pred, p_negative):
        pos_logits = torch.exp(y_pred[:, 0] / self.temp)

        neg_logits = pos_logits + self.neg_weight * int(p_negative.shape[1]) * torch.sum(torch.exp(y_pred[:, 1:] / self.temp) * p_negative, dim=1)  # @ multiply with N

        ssm_loss = -torch.log(pos_logits / neg_logits)

        return ssm_loss.mean()

    def regularize(self,users_emb, pos_emb, neg_emb, users_p_emb, neg_p_emb):
        model_regularize = (
                    torch.norm(users_emb[:, :]) ** 2
                +   torch.norm(pos_emb[:, :])   ** 2
                +   torch.norm(neg_emb[:, :])   ** 2
            ) / 2
        loss_regularize  = (
                    torch.norm(users_p_emb) ** 2
                +   torch.norm(neg_p_emb)   ** 2
        ) / 2
        return model_regularize, loss_regularize

    def cal_loss_graph(self, users, pos, neg):
        embedding_user, embedding_item = self.model.compute()
        users_emb   = embedding_user[users.long()]
        pos_emb     = embedding_item[pos.long()]
        neg_emb     = embedding_item[neg.long()]
        batch_size  = users_emb.shape[0]

        users_p_emb = self.embed_user_p(users)
        neg_p_emb   = self.embed_item_p(neg)


        s_negative = torch.bmm(users_p_emb.unsqueeze(1), neg_p_emb.transpose(1, 2)).squeeze(1)

        p_negative = torch.softmax(s_negative, dim=1)

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        loss = self.cal_loss(y_pred, p_negative)

        emb_loss, reg_loss_prob = self.regularize(users_emb, pos_emb, neg_emb, users_p_emb, neg_p_emb)
        emb_loss                = self.weight_decay * emb_loss      / batch_size
        reg_loss_prob           = self.weight_decay * reg_loss_prob / batch_size 
        additional_loss         =  self.model.additional_loss(
                                    usr_idx = users.long(), 
                                    pos_idx = pos.long(), 
                                    embedding_user = embedding_user, 
                                    embedding_item = embedding_item
                                )
        return loss, emb_loss + additional_loss, reg_loss_prob + additional_loss


    def step(self, user, pos, neg, epoch, adv_training_flag):
        
        advInfoNCE_loss, emb_loss, reg_loss_prob = self.cal_loss_graph(users = user, pos = pos, neg = neg)
        
        if   adv_training_flag is True:
            loss = reg_loss_prob - advInfoNCE_loss
            self.opt_adv.zero_grad()
            loss.backward()
            self.opt_adv.step()
        elif adv_training_flag is False:
            loss = emb_loss + advInfoNCE_loss
            self.opt_model.zero_grad()
            loss.backward()
            self.opt_model.step()
        
        return advInfoNCE_loss.cpu().item()

    def save(self,path):
        all_states = self.model.state_dict()

        all_states.update({
            "embed_user_p": self.embed_user_p.weight.detach(),
            "embed_item_p": self.embed_item_p.weight.detach()
        })

        torch.save(obj = all_states, f = path)

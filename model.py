import os
import pdb
import random
import time
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

from tqdm import tqdm
import world
import torch
from dataloader import Loader
import dataloader
from torch import nn, optim,autograd
import numpy as np
import torch.nn.functional as F
# from torch_scatter import scatter_mean
from torch.distributions.beta import Beta
import math



class BasicModel(nn.Module):
    """ This is for loss function generalized """
    def __init__(self, config, num_users, num_items):
        super(BasicModel, self).__init__()
        self.config = config
        self.num_users = num_users
        self.num_items = num_items
        self.__init_weight()
        
    def __init_weight(self):
        self.omega = self.config['renyi_omega']
        self.margin = self.config["renyi_margin"]

        self.neg_weight = self.config["neg_coefficient"]
        self.param_mode = self.config['param_mode']  

        self.w_emb_dim  = self.config['w_emb_dim']  

        self.alpha = self.config['alpha']  
        self.beta = self.config['beta']



        self.lambda_t = 1 - float(self.config['lambda_k'] / self.config["num_negative_items"])
        self.dynamic_t = self.config["lambda_k"]

        if self.config['loss'] == 'renyi':
            if self.config['learning_mode'] == 'single':
                self.margin_vector = nn.Parameter(    (torch.tensor([self.margin])).cuda()           )
            else:
                self.margin_vector = (self.margin * torch.ones((self.num_users, 1),requires_grad=True)).cuda()
                self.margin_vector = nn.Parameter(self.margin_vector)
        elif self.config['loss'] == 'topk_loss':
            self.margin_vector = (self.margin * torch.ones((self.num_users, 1), requires_grad=True)).cuda()
            self.margin_vector = nn.Parameter(self.margin_vector)
        elif self.config['loss'] == 'llpauc':
            self.a= nn.Parameter(      (torch.tensor([1.0])).cuda()    )
            self.b= nn.Parameter(      (torch.tensor([0.0])).cuda()    )
            self.gamma=nn.Parameter(   (torch.tensor([0.0])).cuda()    )
            self.sn=nn.Parameter(      (torch.tensor([0.5])).cuda()    )
            self.sp = nn.Parameter(    (torch.tensor([0.5])).cuda()    )
            self.theta_b=nn.Parameter( (torch.tensor([0.5])).cuda()    )
            self.theta_a=nn.Parameter( (torch.tensor([0.5])).cuda()    )
        elif self.config['loss'] == 'advinfonce':
            # self.embed_user_p = nn.Embedding(self.num_users, self.w_emb_dim)
            # self.embed_item_p = nn.Embedding(self.num_items, self.w_emb_dim)
            # nn.init.xavier_normal_(self.embed_user_p.weight)
            # nn.init.xavier_normal_(self.embed_item_p.weight)

            self.embed_user_p = nn.Embedding(self.num_users, self.w_emb_dim)
            self.embed_item_p = nn.Embedding(self.num_items, self.w_emb_dim)
            nn.init.xavier_normal_(self.embed_user_p.weight)
            nn.init.xavier_normal_(self.embed_item_p.weight)

            # self.embed_user_p = nn.init.xavier_uniform_(torch.empty(self.num_users,self.w_emb_dim))
            # self.embed_item_p = nn.init.xavier_uniform_(torch.empty(self.num_items,self.w_emb_dim))

            # self.embed_user_p = nn.Parameter(self.embed_user_p)
            # self.embed_item_p = nn.Parameter(self.embed_item_p)

            self.embed_user_p.requires_grad_(False)
            self.embed_item_p.requires_grad_(False)
            

        # elif self.config['loss'] == 'softmax':
        #     self.temp_vector = (self.config["ssm_temp"] * torch.ones((self.num_users, 1),requires_grad=True)).cuda()
        #     self.temp_vector = nn.Parameter(self.temp_vector)

    def freeze_prob(self, flag):
        if flag:
            # if self.config['loss'] == 'advinfonce':
            self.embed_user_p.requires_grad_(False)
            self.embed_item_p.requires_grad_(False)
            self.embedding_user.requires_grad_(True)
            self.embedding_item.requires_grad_(True)
        else:
            # if self.config['loss'] == 'advinfonce':
            self.embed_user_p.requires_grad_(True)
            self.embed_item_p.requires_grad_(True)
            self.embedding_user.requires_grad_(False)
            self.embedding_item.requires_grad_(False)

    def compute_precision_topks(self, user):
        embedding_user, embedding_item = self.compute()
        users_emb = embedding_user[user.long()]
        scores = users_emb @ embedding_item.T
        return torch.topk(input = scores, dim = 1, k = self.dynamic_t)[0][:,-1]

    def compute_sort_quantile(self, user_pos, pos_scores, neg_scores):
        pos_check_tensor = torch.not_equal(user_pos, torch.full_like(user_pos, self.num_items).to(torch.int64))

        all_scores = torch.cat((self.f(pos_scores) * pos_check_tensor, self.f(neg_scores)), dim = 1)
        origin_all_scores = torch.cat((pos_scores, neg_scores), dim = 1)
        _,topk_index = torch.topk(input = all_scores, dim = 1, k = self.dynamic_t)

        return torch.gather(input = origin_all_scores, dim = 1, index=(topk_index[:,-1]).unsqueeze(dim = 1))

    def sort_to_get_quantile(self, users, user_pos, neg):
        embedding_user, embedding_item = self.compute()
        embedding_item_add = torch.cat( ( embedding_item, torch.zeros(1, self.latent_dim).cuda()) )
        users_emb = embedding_user[users.long()]        # (B,dim)
        pos_emb = embedding_item_add[user_pos]          # (B,PadSize,dim)
        neg_emb = embedding_item[neg.long()]            # (B,PadSize,dim)


        pos_scores = torch.bmm(users_emb.unsqueeze(1), pos_emb.transpose(1, 2)).squeeze(1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)

        margin = self.compute_sort_quantile(user_pos, pos_scores, neg_scores)

        return margin



    def precision_sort_topk_loss(self,users, pos, neg, margin_vec, epoch = None, batch_idx = None):
        embedding_user, embedding_item = self.compute()

        users_emb = embedding_user[users.long()]  # (Batch, Latent_dim)
        pos_emb = embedding_item[pos.long()]  # (Batch, Latent_dim)
        neg_emb = embedding_item[neg.long()]  # (Batch, Negative_Num, Latent_dim)

        batch_size = users_emb.shape[0]

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        topk_loss = self.compute_precision_topk_loss(y_pred, margin_vec)

        "L_2 regulization"
        regularize = (torch.norm(users_emb[:, :]) ** 2
                      + torch.norm(pos_emb[:, :]) ** 2
                      + torch.norm(neg_emb[:, :]) ** 2) / 2  # take hop=0
        emb_loss = self.weight_decay * regularize / batch_size


        return topk_loss, emb_loss

    """Util Functions to Compute the Difference Between Margin Vector and True TopK Scores"""
    def diff_margin_and_topk(self, users):
        user_embedding, item_embedding = self.compute()

        user_margin = self.margin_vector[users.long()]

        users_emb = user_embedding[users.long()]
        items_emb = item_embedding

        scores = torch.matmul(users_emb, items_emb.t())
        topk_scores, topk_indices = torch.topk(input = scores, dim = 1, k = self.dynamic_t)

        loss = torch.abs(user_margin.squeeze() - topk_scores[:,-1]).sum()

        return loss

    def getUsersRating(self, users):
        embedding_user, embedding_item = self.compute()
        users_emb = embedding_user[users.long()]
        items_emb = embedding_item

        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)



    """ Double Epoch Loss Update Strategies """
    def compute_quantile_loss(self, user_pos, pos_scores, neg_scores, margin_vec):

        pos_check_tensor = torch.not_equal(user_pos, torch.full_like(user_pos, self.num_items).to(torch.int64))
        # pos_check_tensor = (pos_scores != 0)
        pos_checksum = (torch.sum(pos_check_tensor, dim = 1) + neg_scores.shape[1]).to(torch.float32)

        lambda_t = self.dynamic_t / pos_checksum

        pos_quantile_one = (1 - lambda_t) * torch.sum(torch.relu(pos_scores.detach() - margin_vec) * pos_check_tensor.detach(), dim = 1)
        pos_quantile_two = lambda_t * torch.sum(torch.relu(margin_vec - pos_scores.detach()) * pos_check_tensor.detach(), dim = 1)


        neg_quantile_one = (1 - lambda_t) * torch.sum(torch.relu(neg_scores.detach() - margin_vec), dim = 1)
        neg_quantile_two = lambda_t * torch.sum(torch.relu(margin_vec - neg_scores.detach()), dim = 1)

        loss = (pos_quantile_one + pos_quantile_two + neg_quantile_one + neg_quantile_two) / pos_checksum


        return loss.sum()

    def quantile_loss(self, users, user_pos, neg):
        embedding_user, embedding_item = self.compute()
        embedding_item_add = torch.cat( ( embedding_item, torch.zeros(1, self.latent_dim).cuda()) )
        users_emb = embedding_user[users.long()]        # (B,dim)
        pos_emb = embedding_item_add[user_pos]          # (B,PadSize,dim)
        neg_emb = embedding_item[neg.long()]            # (B,PadSize,dim)


        pos_scores = torch.bmm(users_emb.unsqueeze(1), pos_emb.transpose(1, 2)).squeeze(1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)

        margin_vec = self.margin_vector[users.long()]

        loss = self.compute_quantile_loss(user_pos, pos_scores, neg_scores, margin_vec)

        return loss


    def compute_single_precision_topk_loss(self,y_pred):
        """ Loss Part """
        trunc_pos = y_pred[:, 0] 
        trunc_neg = y_pred[:, 1:]

        pos_logits = torch.sigmoid(trunc_pos / self.config["ssm_temp"])
        neg_logits = torch.sigmoid(trunc_neg / self.config['ssm_temp'])  # neg_logits = torch.sigmoid(  torch.cat( (trunc_neg, trunc_pos.unsqueeze(1)), 1 )  / self.config["ssm_temp"])
        topk_loss = -torch.log(pos_logits / neg_logits.sum(dim=1))



        return topk_loss.mean()

    def precision_single_topk_loss(self,users, pos, neg, epoch = None, batch_idx = None):
        embedding_user, embedding_item = self.compute()

        users_emb = embedding_user[users.long()]  # (Batch, Latent_dim)
        pos_emb = embedding_item[pos.long()]  # (Batch, Latent_dim)
        neg_emb = embedding_item[neg.long()]  # (Batch, Negative_Num, Latent_dim)

        batch_size = users_emb.shape[0]

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        topk_loss = self.compute_single_precision_topk_loss(y_pred)

        "L_2 regulization"
        regularize = (torch.norm(users_emb[:, :]) ** 2
                      + torch.norm(pos_emb[:, :]) ** 2
                      + torch.norm(neg_emb[:, :]) ** 2) / 2  # take hop=0
        emb_loss = self.weight_decay * regularize / batch_size


        return topk_loss, emb_loss

    
    def compute_precision_topk_loss(self,y_pred, margin_vec):
        """ Loss Part """
        trunc_pos = y_pred[:, 0] - (margin_vec.squeeze()).detach()
        trunc_neg = y_pred[:, 1:] - margin_vec.detach()

        pos_logits = torch.sigmoid(trunc_pos / self.config["ssm_temp"])
        neg_logits = torch.sigmoid(trunc_neg / self.config['ssm_temp'])  # neg_logits = torch.sigmoid(  torch.cat( (trunc_neg, trunc_pos.unsqueeze(1)), 1 )  / self.config["ssm_temp"])
        topk_loss = -torch.log(pos_logits / neg_logits.sum(dim=1))



        return topk_loss.mean()

    def precision_topk_loss(self,users, pos, neg, epoch = None, batch_idx = None):
        embedding_user, embedding_item = self.compute()

        users_emb = embedding_user[users.long()]  # (Batch, Latent_dim)
        pos_emb = embedding_item[pos.long()]  # (Batch, Latent_dim)
        neg_emb = embedding_item[neg.long()]  # (Batch, Negative_Num, Latent_dim)

        batch_size = users_emb.shape[0]

        margin_vec = self.margin_vector[users.long()]

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        topk_loss = self.compute_precision_topk_loss(y_pred, margin_vec)

        "L_2 regulization"
        regularize = (torch.norm(users_emb[:, :]) ** 2
                      + torch.norm(pos_emb[:, :]) ** 2
                      + torch.norm(neg_emb[:, :]) ** 2) / 2  # take hop=0
        emb_loss = self.weight_decay * regularize / batch_size


        return topk_loss, emb_loss



    """Softmax Loss"""
    def compute_ssm_loss(self, y_pred,user):
        """The softmax loss"""
        """ The following are original codes """
        if self.config['mode'] == 'reweight':
            pos_logits = torch.exp(y_pred[:, 0] / self.config["ssm_temp"])
            neg_logits = torch.exp(y_pred[:, 1:] / self.config["ssm_temp"])
        elif self.config['mode'] == 'multi':
            # pos_logits = torch.exp(self.config["neg_coefficient"] * y_pred[:, 0] / self.config["ssm_temp2"])
            pos_logits = torch.exp(self.config["neg_coefficient"] * y_pred[:, 0] / self.config["ssm_temp2"])
            neg_logits = torch.exp(y_pred[:, 1:] / self.config["ssm_temp2"])

        if self.config['mode'] == 'reweight':
            neg_logits = torch.sum(neg_logits, dim=-1)
            neg_logits = torch.pow(neg_logits, self.config["neg_coefficient"])
        elif self.config['mode'] == 'multi':
            user = user.contiguous().view(-1, 1)
            mask = torch.eq(user, user.T).float()
            pos_logits = (pos_logits.unsqueeze(0) * mask).sum(1) / mask.sum(1)
            neg_logits = torch.pow(torch.mean(neg_logits, dim=-1), self.config["neg_coefficient"])

        loss = - torch.log(pos_logits / neg_logits).mean()
        return loss

    def ssm_loss(self, users, pos, neg, epoch=None, batch_idx=None):
        embedding_user, embedding_item = self.compute()

        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]
        batch_size = users_emb.shape[0]

        if self.sample_mode == "uniform":
            pos_scores = torch.sum(users_emb * pos_emb, dim=1)
            neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
            y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        elif self.sample_mode == "no_sample":
            row_swap = torch.cat([torch.arange(batch_size).long(), torch.arange(batch_size).long()])
            col_before = torch.cat([torch.arange(batch_size).long(), torch.zeros(batch_size).long()])
            col_after = torch.cat([torch.zeros(batch_size).long(), torch.arange(batch_size).long()])
            y_pred = torch.mm(users_emb, pos_emb.t().contiguous())
            y_pred[row_swap, col_before] = y_pred[row_swap, col_after]

        loss = self.compute_ssm_loss(y_pred,users)

        regularize = (torch.norm(users_emb[:, :]) ** 2
                       + torch.norm(pos_emb[:, :]) ** 2
                        + torch.norm(neg_emb[:,:]) ** 2 ) / 2  # take hop=0
        emb_loss = self.weight_decay * regularize / batch_size
        
        return loss + emb_loss
        # return loss


    """BCE Loss"""
    def bce_loss(self, users, pos, neg):
        embedding_user, embedding_item = self.compute()
        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.squeeze().long()]
        
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)

        scores = torch.cat([pos_scores, neg_scores], dim=0)
        label = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)

        criterion = torch.nn.BCEWithLogitsLoss()

        regularize = (torch.norm(users_emb[:, :]) ** 2
                       + torch.norm(pos_emb[:, :]) ** 2
                       + torch.norm(neg_emb[:,:]) ** 2 ) / 2  # take hop=0

        emb_loss = self.weight_decay * regularize/ users_emb.shape[0]


        loss = criterion(scores, label)

        return loss + emb_loss

    """MSE Loss"""
    def rmse_loss(self, users, pos, neg):
        embedding_user, embedding_item = self.compute()
        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.squeeze().long()]
        
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)

        scores = torch.cat([self.f(pos_scores), self.f(neg_scores)], dim=0)
        label = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
        
        criterion = torch.nn.MSELoss()
        loss = criterion(scores, label)


        regularize = (torch.norm(users_emb[:, :]) ** 2
                       + torch.norm(pos_emb[:, :]) ** 2
                       + torch.norm(neg_emb[:,:]) ** 2 ) / 2  # take hop=0

        emb_loss = self.weight_decay * regularize/ users_emb.shape[0]

        return loss + emb_loss

    """BPR Loss"""
    def bpr_loss(self, users, pos, neg):
        embedding_user, embedding_item = self.compute()
        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.squeeze().long()]
        batch_size = users_emb.shape[0]

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)

        regularize = (torch.norm(users_emb[:, :]) ** 2
                       + torch.norm(pos_emb[:, :]) ** 2
                        + torch.norm(neg_emb[:,:]) ** 2 ) / 2  # take hop=0
        emb_loss = self.weight_decay * regularize / batch_size


        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))

        return loss + emb_loss

    """LLPAUC Loss"""
    def compute_llpauc(self, y_pred, a=None, b=None, gamma=None, theta_a=None, theta_b=None, sp=None, sn=None):
        a = torch.clip(a, 0, 1)
        b = torch.clip(b, 0, 1)
        gamma = torch.clip(gamma, -1, 1)
        theta_a = torch.clip(theta_a, 0, 1e9)
        theta_b = torch.clip(theta_b, 0, 1e9)
        sp = torch.clip(sp, -1, 4)
        sn = torch.clip(sn, 0, 5)
        pos_score = torch.sigmoid(y_pred[:, 0])
        neg_score = torch.sigmoid(y_pred[:, 1:])
        max_val_p = torch.log(1 + torch.exp(5 * (-torch.square(pos_score - a) + \
                                                 2 * (1 + gamma) * pos_score - sp))) / 5
        max_val_n = torch.log(1 + torch.exp(5 * (torch.square(neg_score - b) + \
                                                 2 * (1 + gamma) * neg_score - sn))) / 5
        loss = -sp - torch.mean(max_val_p) / self.alpha + \
               sn + torch.mean(max_val_n) / self.beta + \
               -gamma ** 2 - theta_b * (b - 1 - gamma) + theta_a * (a + gamma)

        return loss

    def llpauc(self, users, pos, neg, epoch=None, batch_idx=None):
        embedding_user, embedding_item = self.compute()

        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]
        batch_size = users_emb.shape[0]

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        loss = self.compute_llpauc(y_pred, a=self.a, b=self.b, gamma=self.gamma, theta_a=self.theta_a,
                                   theta_b=self.theta_b, sp=self.sp, sn=self.sn)

        regularize = (torch.norm(users_emb[:, :]) ** 2
                      + torch.norm(pos_emb[:, :]) ** 2
                      + torch.norm(neg_emb[:, :]) ** 2
                      + torch.norm(self.a) ** 2
                      + torch.norm(self.b) ** 2
                      + torch.norm(self.gamma) ** 2
                      + torch.norm(self.theta_a) ** 2
                      + torch.norm(self.theta_b) ** 2
                      + torch.norm(self.sp) ** 2
                      + torch.norm(self.sn) ** 2) / 2  # take hop=0
        emb_loss = self.weight_decay * regularize / batch_size

        return loss, emb_loss

    """AdvInfoNCE Loss"""
    def compute_advInfoNCE(self, y_pred, p_negative, k_neg, temp):
        pos_logits = torch.exp(y_pred[:, 0] / temp)

        neg_logits = pos_logits + k_neg * int(p_negative.shape[1]) * torch.sum(
            torch.exp(y_pred[:, 1:] / temp) * p_negative, dim=1)  # @ multiply with N

        ssm_loss = torch.mean(torch.negative(torch.log(pos_logits / neg_logits)))

        return ssm_loss

    def advInfoNCE(self, users, pos, neg, epoch=None, batch_idx=None):
        embedding_user, embedding_item = self.compute()
        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]

        users_p_emb = self.embed_user_p(users)
        neg_p_emb = self.embed_item_p(neg)

        # s_negative = torch.matmul(torch.unsqueeze(users_p_emb, 1),
        #                           neg_p_emb.permute(0, 2, 1)).squeeze(dim=1)

        # s_negative = torch.matmul(torch.unsqueeze(users_p_emb, 1),
        #                           neg_p_emb.permute(0, 2, 1)).squeeze(dim=1)
        s_negative = torch.bmm(users_p_emb.unsqueeze(1), neg_p_emb.transpose(1, 2)).squeeze(1)

        p_negative = torch.softmax(s_negative, dim=1)
        batch_size = users_emb.shape[0]

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        loss = self.compute_advInfoNCE(y_pred, p_negative, self.config['neg_coefficient'], self.config["ssm_temp"])

        regularize = (torch.norm(users_emb[:, :]) ** 2
                      + torch.norm(pos_emb[:, :]) ** 2
                      + + torch.norm(neg_emb[:, :]) ** 2) / 2  # take hop=0
        emb_loss = self.weight_decay * regularize / batch_size

        reg_neg_prob = 0.5 * torch.norm(users_p_emb) ** 2 + 0.5 * torch.norm(neg_p_emb) ** 2
        reg_neg_prob = reg_neg_prob / batch_size
        reg_loss_prob = self.weight_decay * reg_neg_prob

        return loss, emb_loss, reg_loss_prob


class PureMF(BasicModel):
    def __init__(self, config: dict, dataset: Loader):
        super(PureMF, self).__init__(config = config, num_users = dataset.n_user, num_items = dataset.m_items)
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.dataset = dataset
        self.latent_dim = config["latent_dim_rec"]
        self.config = config
        self.f = nn.Sigmoid()
        self.__init_weight()

        self.weight_decay = self.config["weight_decay"]
        self.sample_mode = self.config["sample_mode"]



    def __init_weight(self):
        self.embedding_user = nn.init.xavier_uniform_(torch.empty(self.num_users,self.latent_dim))
        self.embedding_item = nn.init.xavier_uniform_(torch.empty(self.num_items,self.latent_dim))

        self.embedding_user = nn.Parameter(self.embedding_user)
        self.embedding_item = nn.Parameter(self.embedding_item)
    
    

    def compute(self):
        users_emb = self.embedding_user
        items_emb = self.embedding_item

        if self.config["norm_emb"]:
            users_emb = F.normalize(users_emb, p=2, dim=1)
            items_emb = F.normalize(items_emb, p=2, dim=1)
        return users_emb, items_emb



class LightGCN(BasicModel):
    def __init__(self, config: dict, dataset: Loader):
        super(LightGCN, self).__init__(config = config, num_users = dataset.n_user, num_items=dataset.m_items)
        self.config = config
        self.dataset: dataloader.Loader = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config["latent_dim_rec"]
        self.n_layers = self.config["n_layers"]
        self.keep_prob = self.config["keep_prob"]

        self.weight_decay = self.config["weight_decay"]
        self.sample_mode = self.config["sample_mode"]


        self.embedding_user = nn.init.xavier_uniform_(torch.empty(self.num_users,self.latent_dim))
        self.embedding_item = nn.init.xavier_uniform_(torch.empty(self.num_items,self.latent_dim))

        self.embedding_user = nn.Parameter(self.embedding_user)
        self.embedding_item = nn.Parameter(self.embedding_item)


        world.cprint("use UNIFORMAL distribution initilizer")

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.Graph
        print(f"lgn is already to go(dropout:{self.config['enable_dropout']})")

    """Dropout is the edge_dropout in BSL"""
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    """Dropout is the edge_dropout in BSL"""


    def compute(self):
        """
        propagate methods for lightGCN；
        """
        users_emb = self.embedding_user
        items_emb = self.embedding_item
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config["enable_dropout"]:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph
            # g_droped is the interact_mat in BSL

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        
        if self.config["norm_emb"]:
            users = F.normalize(users, p=2, dim=1)
            items = F.normalize(items, p=2, dim=1)


        return users, items


class LightGCL(BasicModel):
    def __init__(self, config: dict, dataset: Loader):
        super(LightGCL, self).__init__()

        self.config = config
        self.dataset: dataloader.Loader = dataset

        self.num_users = dataset.n_users
        self.num_items = dataset.m_items

        self.E_u_0 = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.dataset.n_user, self.config["latent_dim_rec"]))
        )
        self.E_i_0 = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.dataset.m_item, self.config["latent_dim_rec"]))
        )

        self.train_csr = dataset.train_csr  # user_num x item_num的CSR矩阵
        self.adj_norm = dataset.adj_norm  # 邻接矩阵 torch.sparse.FloatTensor
        self.l = self.config["n_layers"]  # 层数
        self.E_u_list = [None] * (self.l + 1)  # 存储每一层聚合得到的embedding
        self.E_i_list = [None] * (self.l + 1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (self.l + 1)  # 存储每一层聚合得到的embedding 感觉没啥用
        self.Z_i_list = [None] * (self.l + 1)
        self.G_u_list = [None] * (self.l + 1)  # 存储每一层用SVD邻接矩阵聚合得到的embedding
        self.G_i_list = [None] * (self.l + 1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0
        self.temp = self.config["cl_temp"]
        self.lambda_1 = self.config["cl_rate"]
        # self.lambda_2 = lambda_2
        self.dropout = self.config["keep_prob"]
        self.act = nn.LeakyReLU(0.5)

        self.E_u = None
        self.E_i = None

        self.u_mul_s = dataset.u_mul_s
        self.v_mul_s = dataset.v_mul_s

        self.ut = dataset.ut
        self.vt = dataset.vt

        self.f = nn.Sigmoid()

    def compute(self):
        def sparse_dropout(mat, dropout):
            if dropout == 0.0:
                return mat
            indices = mat.indices()
            values = nn.functional.dropout(mat.values(), p=dropout)
            size = mat.size()
            return torch.sparse.FloatTensor(indices, values, size)

        for layer in range(1, self.l + 1):
            # GNN propagation
            self.Z_u_list[layer] = torch.spmm(sparse_dropout(self.adj_norm, self.dropout), self.E_i_list[layer - 1])
            self.Z_i_list[layer] = torch.spmm(
                sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1), self.E_u_list[layer - 1]
            )

            # svd_adj propagation
            vt_ei = self.vt @ self.E_i_list[layer - 1]
            self.G_u_list[layer] = self.u_mul_s @ vt_ei
            ut_eu = self.ut @ self.E_u_list[layer - 1]
            self.G_i_list[layer] = self.v_mul_s @ ut_eu

            # aggregate
            self.E_u_list[layer] = self.Z_u_list[layer]
            self.E_i_list[layer] = self.Z_i_list[layer]

        self.G_u = sum(self.G_u_list)
        self.G_i = sum(self.G_i_list)

        # aggregate across layers
        # 原来的邻接矩阵获得的embedding
        self.E_u = sum(self.E_u_list)
        self.E_i = sum(self.E_i_list)

        user_embedding = self.E_u / len(self.E_u_list)
        item_embedding = self.E_i / len(self.E_i_list)

        if self.config["norm_emb"]:
            # self.E_u = F.normalize(self.E_u, p=2, dim=1)
            # self.E_i = F.normalize(self.E_i, p=2, dim=1)
            # self.G_u = F.normalize(self.G_u, p=2, dim=1)
            # self.G_i = F.normalize(self.G_i, p=2, dim=1)
            user_embedding = F.normalize(user_embedding, p=2, dim=1)
            item_embedding = F.normalize(item_embedding, p=2, dim=1)

        return user_embedding, item_embedding

    def cal_cl_loss(self, uids, iids):
        self.compute()

        G_u_norm = self.G_u
        E_u_norm = self.E_u
        G_i_norm = self.G_i
        E_i_norm = self.E_i

        neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
        neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
        pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp, -5.0, 5.0)).mean() + (
            torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp, -5.0, 5.0)
        ).mean()
        loss = -pos_score + neg_score

        return loss

    def ssm_loss(self, users, pos, neg, epoch=None, batch_idx=None):
        embedding_user, embedding_item = self.compute()
        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]

        cl_loss = self.config["cl_rate"] * self.cal_cl_loss(users, pos)

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        loss = self.compute_ssm_loss(y_pred, self.config["ssm_temp"], epoch)

        loss = loss + cl_loss

        return loss


class XSimGCL(LightGCN):
    def __init__(self, config: dict, dataset: Loader):
        super(XSimGCL, self).__init__(config, dataset)

    def getUsersRating(self, users):
        embedding_user, embedding_item, _, _ = self.compute()
        users_emb = embedding_user[users.long()]
        items_emb = embedding_item

        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def compute(self):
        users_emb = self.embedding_user
        items_emb = self.embedding_item
        all_emb = torch.cat([users_emb, items_emb])
        # embs = [all_emb]
        embs = []
        if self.config["enable_dropout"]:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for k in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            random_noise = torch.rand_like(all_emb, device="cuda")
            all_emb += torch.sign(all_emb) * F.normalize(random_noise, dim=-1) * self.config["eps"]
            embs.append(all_emb)

            if k == self.config["cl_layer"]:
                all_embeddings_cl = all_emb

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(
            all_embeddings_cl, [self.num_users, self.num_items]
        )

        if self.config["norm_emb"]:
            users = F.normalize(users, p=2, dim=1)
            items = F.normalize(items, p=2, dim=1)
            user_all_embeddings_cl = F.normalize(user_all_embeddings_cl, p=2, dim=1)
            item_all_embeddings_cl = F.normalize(item_all_embeddings_cl, p=2, dim=1)

        return users, items, user_all_embeddings_cl, item_all_embeddings_cl

    def InfoNCE(self, view1, view2, temperature: float):
        pos_score = (view1 @ view2.T) / temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()
        # pos_score = (view1 * view2).sum(dim=-1)
        # pos_score = torch.exp(pos_score / temperature)
        # ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        # ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        # cl_loss = -torch.log(pos_score / ttl_score)
        # return torch.mean(cl_loss)

    def cal_cl_loss(self, idx, user_view1, user_view2, item_view1, item_view2):
        u_idx = torch.unique(  (idx[0].type(torch.long)).clone().detach()   )
        i_idx = torch.unique(  (idx[1].type(torch.long)).clone().detach()   )
        user_cl_loss = self.InfoNCE(user_view1[u_idx], user_view2[u_idx], self.config["cl_temp"])
        item_cl_loss = self.InfoNCE(item_view1[i_idx], item_view2[i_idx], self.config["cl_temp"])

        return user_cl_loss + item_cl_loss

    def bpr_loss(self, users, pos, neg):
        # print(neg.shape)
        neg = neg.squeeze()

        embedding_user, embedding_item, cl_user_emb, cl_item_emb = self.compute()
        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]

        batch_size = batch_size = users_emb.shape[0]

        cl_loss = self.config["cl_rate"] * self.cal_cl_loss(
            [users, pos], embedding_user, cl_user_emb, embedding_item, cl_item_emb
        )

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)

        regularize = (torch.norm(users_emb[:, :]) ** 2
                      + torch.norm(pos_emb[:, :]) ** 2  
                      + + torch.norm(neg_emb[:,:]) ** 2 ) / 2  # take hop=0
        emb_loss = self.weight_decay * regularize / batch_size

        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))

        loss = loss + cl_loss + emb_loss

        return loss


    

    def ssm_loss(self, users, pos, neg, epoch=None, batch_idx=None):       
        embedding_user, embedding_item, cl_user_emb, cl_item_emb = self.compute()
        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]

        batch_size = users_emb.shape[0]

        cl_loss = self.config["cl_rate"] * self.cal_cl_loss(
            [users, pos], embedding_user, cl_user_emb, embedding_item, cl_item_emb
        )

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        loss = self.compute_ssm_loss(y_pred, users)

        regularize = (torch.norm(users_emb[:, :]) ** 2
                       + torch.norm(pos_emb[:, :]) ** 2
                       + torch.norm(neg_emb[:,:]) ** 2 ) / 2  # take hop=0

        emb_loss = self.weight_decay * regularize/ users_emb.shape[0]

        loss = loss + cl_loss + emb_loss

        return loss
    
    def renyi_loss(self,users, pos, neg, opt_margin, epoch=None, batch_idx=None):
        embedding_user, embedding_item, cl_user_emb, cl_item_emb = self.compute()
        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]
        # margin_vec = self.margin_vector[users.long()]
        if self.config['learning_mode'] == 'single':
            margin_vec = self.margin_vector
        else:
            margin_vec = self.margin_vector[users.long()]
        batch_size = users_emb.shape[0]

        cl_loss = self.config["cl_rate"] * self.cal_cl_loss(
            [users, pos], embedding_user, cl_user_emb, embedding_item, cl_item_emb
        )

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        loss = self.compute_renyi_loss(y_pred,margin_vec,opt_margin)

        regularize = (torch.norm(users_emb[:, :]) ** 2
                       + torch.norm(pos_emb[:, :]) ** 2
                       + torch.norm(neg_emb[:,:]) ** 2 ) / 2  # take hop=0

        emb_loss = self.weight_decay * regularize/ users_emb.shape[0]

        loss = loss + cl_loss + emb_loss
        return loss
    def rmse_loss(self, users, pos, neg):

        embedding_user, embedding_item, cl_user_emb, cl_item_emb = self.compute()
        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.squeeze().long()]

        cl_loss = self.config["cl_rate"] * self.cal_cl_loss(
            [users, pos], embedding_user, cl_user_emb, embedding_item, cl_item_emb
        )
        
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)

        scores = torch.cat([self.f(pos_scores), self.f(neg_scores)], dim=0)
        label = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
        
        criterion = torch.nn.MSELoss()
        loss = criterion(scores, label)


        regularize = (torch.norm(users_emb[:, :]) ** 2
                       + torch.norm(pos_emb[:, :]) ** 2
                       + torch.norm(neg_emb[:,:]) ** 2 ) / 2  # take hop=0

        emb_loss = self.weight_decay * regularize/ users_emb.shape[0]

        return loss + emb_loss + cl_loss


    def quantile_loss(self, users, user_pos, neg):
        embedding_user, embedding_item, _ , _ = self.compute()
        embedding_item_add = torch.cat( ( embedding_item, torch.zeros(1, self.latent_dim).cuda()) )
        users_emb = embedding_user[users.long()]        # (B,dim)
        pos_emb = embedding_item_add[user_pos]          # (B,PadSize,dim)
        neg_emb = embedding_item[neg.long()]            # (B,PadSize,dim)


        pos_scores = torch.bmm(users_emb.unsqueeze(1), pos_emb.transpose(1, 2)).squeeze(1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)

        margin_vec = self.margin_vector[users.long()]

        loss = self.compute_quantile_loss(user_pos, pos_scores, neg_scores, margin_vec)

        return loss

    def precision_topk_loss(self,users, pos, neg, epoch = None, batch_idx = None):
        embedding_user, embedding_item, cl_user_emb, cl_item_emb = self.compute()

        users_emb = embedding_user[users.long()]    # (Batch, Latent_dim)
        pos_emb = embedding_item[pos.long()]        # (Batch, Latent_dim)
        neg_emb = embedding_item[neg.long()]        # (Batch, Negative_Num, Latent_dim)

        cl_loss = self.config["cl_rate"] * self.cal_cl_loss(
            [users, pos], embedding_user, cl_user_emb, embedding_item, cl_item_emb
        )


        batch_size = users_emb.shape[0]

        margin_vec = self.margin_vector[users.long()]

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        topk_loss = self.compute_precision_topk_loss(y_pred, margin_vec)

        "L_2 regulization"
        regularize = (torch.norm(users_emb[:, :]) ** 2
                      + torch.norm(pos_emb[:, :]) ** 2
                      + torch.norm(neg_emb[:, :]) ** 2) / 2  # take hop=0
        emb_loss = self.weight_decay * regularize / batch_size


        return topk_loss + cl_loss, emb_loss


    def bce_loss(self, users, pos, neg):
        embedding_user, embedding_item, cl_user_emb, cl_item_emb = self.compute()
        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.squeeze().long()]

        

        cl_loss = self.config["cl_rate"] * self.cal_cl_loss(
            [users, pos], embedding_user, cl_user_emb, embedding_item, cl_item_emb
        )

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)

        scores = torch.cat([pos_scores, neg_scores], dim=0)
        label = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)

        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(scores, label)

        
        regularize = (torch.norm(users_emb[:, :]) ** 2
                       + torch.norm(pos_emb[:, :]) ** 2
                       + torch.norm(neg_emb[:,:]) ** 2 ) / 2  # take hop=0

        emb_loss = self.weight_decay * regularize/ users_emb.shape[0]

        loss = loss + cl_loss + emb_loss

        return loss

    def advInfoNCE(self, users, pos, neg, epoch=None, batch_idx=None):
        embedding_user, embedding_item, cl_user_emb, cl_item_emb = self.compute()
        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]

        batch_size = users_emb.shape[0]


        cl_loss = self.config["cl_rate"] * self.cal_cl_loss(
            [users, pos], embedding_user, cl_user_emb, embedding_item, cl_item_emb
        )

        users_p_emb = self.embed_user_p(users)
        neg_p_emb = self.embed_item_p(neg)

        s_negative = torch.bmm(users_p_emb.unsqueeze(1), neg_p_emb.transpose(1, 2)).squeeze(1)


        p_negative = torch.softmax(s_negative, dim=1)


        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)


        loss = self.compute_advInfoNCE(y_pred, p_negative, self.config['neg_coefficient'], self.config["ssm_temp"])

        regularize = (torch.norm(users_emb[:, :]) ** 2
                      + torch.norm(pos_emb[:, :]) ** 2  
                      + + torch.norm(neg_emb[:,:]) ** 2 ) / 2  # take hop=0
        emb_loss = self.weight_decay * regularize / batch_size

        reg_neg_prob = 0.5 * torch.norm(users_p_emb) ** 2 + 0.5 * torch.norm(neg_p_emb) ** 2
        reg_neg_prob = reg_neg_prob / batch_size
        reg_loss_prob = self.weight_decay * reg_neg_prob

        return loss , emb_loss + cl_loss, reg_loss_prob + cl_loss
    
    def llpauc(self, users, pos, neg, epoch=None, batch_idx=None):
        embedding_user, embedding_item, cl_user_emb, cl_item_emb = self.compute()

        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]
        batch_size = users_emb.shape[0]

        cl_loss = self.config["cl_rate"] * self.cal_cl_loss(
            [users, pos], embedding_user, cl_user_emb, embedding_item, cl_item_emb
        )

        
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        
        loss = self.compute_llpauc(y_pred,a = self.a, b = self.b, gamma=self.gamma, theta_a=self.theta_a,
        theta_b=self.theta_b,sp=self.sp,sn=self.sn)

        regularize = (torch.norm(users_emb[:, :]) ** 2
                       + torch.norm(pos_emb[:, :]) ** 2
                        + torch.norm(neg_emb[:,:]) ** 2 
                        + torch.norm(self.a) ** 2 
                        + torch.norm(self.b) ** 2 
                        + torch.norm(self.gamma) ** 2 
                        + torch.norm(self.theta_a) ** 2
                        + torch.norm(self.theta_b) ** 2 
                        + torch.norm(self.sp) ** 2 
                        + torch.norm(self.sn) ** 2 ) / 2  # take hop=0
        emb_loss = self.weight_decay * regularize / batch_size
        
        return loss + cl_loss, emb_loss 


class SimGCL(LightGCN):
    def __init__(self, config: dict, dataset: Loader):
        super(SimGCL, self).__init__(config, dataset)

    def compute(self, perturbed=False):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config["enable_dropout"]:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            if perturbed:
                random_noise = torch.rand_like(all_emb, device="cuda")
                all_emb += torch.sign(all_emb) * F.normalize(random_noise, dim=-1) * self.config["eps"]
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        if self.config["norm_emb"]:
            users = F.normalize(users, p=2, dim=1)
            items = F.normalize(items, p=2, dim=1)

        return users, items

    def InfoNCE(self, view1, view2, temperature: float):
        pos_score = (view1 @ view2.T) / temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()

    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.compute(perturbed=True)
        user_view_2, item_view_2 = self.compute(perturbed=True)
        user_cl_loss = self.InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.config["cl_temp"])
        item_cl_loss = self.InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.config["cl_temp"])
        return user_cl_loss + item_cl_loss

    def ssm_loss(self, users, pos, neg, epoch=None, batch_idx=None):
        embedding_user, embedding_item = self.compute()
        users_emb = embedding_user[users.long()]
        pos_emb = embedding_item[pos.long()]
        neg_emb = embedding_item[neg.long()]

        cl_loss = self.config["cl_rate"] * self.cal_cl_loss([users, pos])

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        loss = self.compute_ssm_loss(y_pred, self.config["ssm_temp"], epoch)

        loss = loss + cl_loss

        return loss

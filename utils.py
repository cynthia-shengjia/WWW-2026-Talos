import world
import torch
from torch import nn, optim
from torch.optim import SGD
import numpy as np
from torch import log
from time import time
from sklearn.metrics import roc_auc_score
import os
import numpy as np

from torch.nn.utils.clip_grad import clip_grad_norm_


class LossFunc:
    def __init__(self, recmodel, config: dict, dataset=None):
        self.model = recmodel

        self.config = config

        self.weight_decay = config["weight_decay"]
        self.batch_size = config["train_batch"]

        print('\nAttention: this is the batch_size of model\n',self.batch_size)

        self.lr = config["lr"]
        print('\nAttention: this is the learning rate of model\n', self.lr)
        self.lr2 = config["lr2"]
        self.search_optimizer = config["search_optimizer"]
        

        if config['loss'] == 'advinfonce':
            self.adv_lr = config["adv_lr"]
            self.opt = torch.optim.Adam([param for param in recmodel.parameters() if param.requires_grad == True],
                                     lr=self.lr)
            self.adv_opt = torch.optim.Adam([param for param in recmodel.parameters() if param.requires_grad == False],
                                         lr=self.adv_lr)
        elif config['loss'] == 'llpauc':
            self.param_name_list=['a','b','theta_b','sn','gamma','sp','theta_a']
            self.opt = self._build_optimizer()
            self.clip_grad_norm = config["clip_grad_norm"]
        elif config['loss'] == 'topk_loss':
            self.opt_model = optim.Adam(recmodel.parameters(), lr = self.lr)
            self.opt_quant = optim.Adam([recmodel.margin_vector], lr = self.lr2)
        else:
            self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)
            if config['loss'] == 'renyi':
                if self.search_optimizer == "SGD":
                    self.opt_margin = optim.SGD([recmodel.margin_vector], lr = self.lr2)
                elif self.search_optimizer == "Adam":
                    self.opt_margin = optim.Adam([recmodel.margin_vector], lr = self.lr2)
            # elif config['loss'] == 'softmax':
            #     self.opt_temp = optim.SGD([recmodel.temp_vector], lr = self.lr2)


        self.dataset = dataset

    def check_diff_margin_and_topk(self, users) -> float:
        loss =  self.model.diff_margin_and_topk(users) / users.shape[0]
        return loss.cpu().item()

    def quantile_setp(self, users, user_pos, neg, epoch: int = None, batch_idx: int = None):
        loss = self.model.quantile_loss(users, user_pos, neg)

        self.opt_quant.zero_grad()
        loss.backward()
        self.opt_quant.step()

    def precision_step(self,users, pos, neg, epoch: int = None, batch_id: int = None) -> float:
        """

        Args:
            users: users in (u,i) interactions of a batch
            pos:   positive items in (u,i) interactions of a batch
            neg:   negative items sampled for user u

        Returns:
            float: The return value. The average loss value
        """
        quantile_loss,emb_loss = self.model.precision_topk_loss(users,pos,neg,epoch,batch_id)
        loss = quantile_loss + emb_loss

        self.opt_model.zero_grad()
        loss.backward()
        self.opt_model.step()

        return quantile_loss.cpu().item()

    def step(self, users, pos, neg, epoch: int = None, batch_id: int = None, flag: bool = True):
        if world.config["loss"] == "bpr":
            loss = self.model.bpr_loss(users, pos, neg)
        elif world.config["loss"] == "softmax":
            loss = self.model.ssm_loss(users, pos, neg, epoch, batch_id)
        elif world.config["loss"] == "bce":
            loss = self.model.bce_loss(users, pos, neg)
        elif world.config["loss"] == "rmse":
            loss = self.model.rmse_loss(users, pos, neg)
        elif world.config["loss"] == "renyi":
            loss = self.model.renyi_loss(users, pos, neg, self.opt_margin, epoch, batch_id)
        elif world.config["loss"] == "advinfonce":
            ssm_loss, emb_loss, reg_loss_prob = self.model.advInfoNCE(users, pos, neg, epoch, batch_id)
            if flag:
                loss = ssm_loss + emb_loss
            else:
                loss = reg_loss_prob - ssm_loss
        elif world.config["loss"] == 'llpauc':
            ssm_loss, emb_loss = self.model.llpauc(users, pos, neg, epoch, batch_id)
            loss = ssm_loss + emb_loss
        elif world.config["loss"] == "topk_loss":
            quantile_loss, topk_loss, emb_loss = self.model.PrecisionAtK(users,pos,neg,epoch,batch_id)
            # topk_loss, emb_loss = self.model.precision_topk_loss(users, pos, neg, epoch, batch_id)
            loss = topk_loss + emb_loss
        else:
            raise NotImplementedError
        
        if flag:
            if world.config["loss"] == 'llpauc':
                self.opt.zero_grad()
                loss.backward()
                if self.search_optimizer=='minmax_adam':
                    self.model.gamma.grad= -self.model.gamma.grad
                    self.model.sp.grad= -self.model.sp.grad
                # if self.clip_grad_norm:
                #     clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
                self.opt.step()
            elif world.config["loss"] == "topk_loss":

                # self.opt_quant.zero_grad()
                # quantile_loss.backward()
                # self.opt_quant.step()

                self.opt_model.zero_grad()
                loss.backward()
                self.opt_model.step()

            else:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        else:
            self.adv_opt.zero_grad()
            loss.backward()
            self.adv_opt.step()
    
        return loss.cpu().item()

    def _build_optimizer(self, **kwargs):
        r"""Init the Optimizer

        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            learner (str, optional): The name of used optimizer. Defaults to ``self.learner``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.learning_rate``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.

        Returns:
            torch.optim: the optimizer
        """
        params = kwargs.pop("params", self.model.parameters())
        learner = kwargs.pop("learner", self.search_optimizer)
        learning_rate = kwargs.pop("learning_rate", self.lr)
        weight_decay = kwargs.pop("weight_decay", self.weight_decay)

        if (
            self.config["reg_weight"]
            and weight_decay
            and weight_decay * self.config["reg_weight"] > 0
        ):
            print(
                "The parameters [weight_decay] and [reg_weight] are specified simultaneously, "
                "which may lead to double regularization."
            )

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate)
        elif learner.lower()=='minmax_adam':
            optimizer = optim.Adam([{'params': [param for name, param in self.model.named_parameters() if
                                            name not in self.param_name_list], 'name': 'net','lr':learning_rate},
                                {'params': [self.model.a, self.model.b],  'name': 'ab','lr':learning_rate},
                                {'params': self.model.sn,  'name': 'sn','lr':learning_rate*2},
                                    {'params': self.model.sp,  'name': 'sp','lr':learning_rate*2},
                                {'params': self.model.theta_b, 'name': 'lamn','lr':learning_rate},
                                    {'params': self.model.theta_a, 'name': 'lamp','lr':learning_rate},
                                {'params': self.model.gamma, 'name': 'g','lr':learning_rate*2}])
        elif learner.lower() == 'minmax_adam_ll':
            optimizer = optim.Adam([{'params': [param for name, param in self.model.named_parameters() if
                                                name not in self.param_name_list], 'name': 'net', 'lr': learning_rate},
                                    {'params': [self.model.a, self.model.b], 'name': 'ab', 'lr': learning_rate},
                                    {'params': self.model.sn, 'name': 'sn', 'lr': learning_rate*0.1 },
                                    {'params': self.model.sp, 'name': 'sp', 'lr': learning_rate*0.1 },
                                    {'params': self.model.theta_b, 'name': 'lamn', 'lr': learning_rate},
                                    {'params': self.model.theta_a, 'name': 'lamp', 'lr': learning_rate},
                                    {'params': self.model.gamma, 'name': 'g', 'lr': learning_rate * 2}],
                                    weight_decay=weight_decay)
        elif learner.lower()=='minmax':
            optimizer=MinMax([{'params': [param for name,param in self.model.named_parameters() if name not in self.param_name_list], 'name':'net'},
            {'params': [self.model.a, self.model.b], 'clip':(0, 1), 'name':'ab'},
            {'params': self.model.s, 'clip':(0, 5), 'name':'sn'},
            {'params': self.model.theta_b, 'clip':(0, 1e9), 'name':'lamn'},
            {'params': self.model.gamma, 'clip':(-1, 1), 'name':'g'}],
                                hparams=self.optim_hparams,weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == "sparse_adam":
            optimizer = optim.SparseAdam(params, lr=learning_rate)
            if weight_decay > 0:
                self.logger.warning(
                    "Sparse Adam cannot argument received argument [{weight_decay}]"
                )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

def UniformSample_original(dataset, num_neg):
    if world.config["loss"] == "bpr":
        users, pos_items, neg_items = dataset.get_train_neg_items(1)
        neg_items = neg_items.view(-1)
        assert neg_items.shape[0] == len(users)
        assert len(pos_items) == len(users)
        S = users, pos_items, neg_items
    elif world.config["loss"] == "softmax":
        users, pos_items, neg_items = dataset.get_train_neg_items(num_neg)
        neg_items = neg_items.view(-1, num_neg)
        assert neg_items.shape[0] == len(users)
        assert len(pos_items) == len(users)
        S = users, pos_items, neg_items
    else:
        raise NotImplementedError
    return S


# ===================end samplers==========================
# =====================utils====================================


def set_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get("batch_size")

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i : i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i : i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get("indices", False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError("All inputs to shuffle must have " "the same length.")

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """

    from time import time

    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get("name"):
            timer.NAMED_TAPE[kwargs["name"]] = (
                timer.NAMED_TAPE[kwargs["name"]] if timer.NAMED_TAPE.get(kwargs["name"]) else 0.0
            )
            self.named = kwargs["name"]
            if kwargs.get("group"):
                # TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {"recall": recall, "precision": precis}


def HitRatio(r):
    return np.sum(r)


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1.0 / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1.0 / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1.0 / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.0] = 1.0
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.0
    return np.sum(ndcg)


def AUC(all_item_scores, dataset, test_data):
    """
    design for a single user
    """
    dataset
    r_all = np.zeros((dataset.m_items,))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))  # 对于 predictTopK 中的每一个item，检查它是否在真实的test item中
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype("float")


# ====================end Metrics=============================
# =========================================================

def topK_target(batch_score,batch_user,batch_target,max_k):
    user_rank = np.zeros( (len(batch_user),max_k) )
    for idx,user in enumerate(batch_user):
        target_scores = batch_score[idx,batch_target[idx]]
        if max_k > len(target_scores):
            _,rank = torch.topk(target_scores,k=len(target_scores))
            padding_length = max_k - len(rank)
            last_element = rank[-1]
            padding = (torch.full((padding_length,), last_element)).cuda()  # 创建填充张量
            padded_tensor = torch.cat((rank, padding))
            
        else:
            _,padded_tensor = torch.topk(target_scores,k=max_k)


        user_rank[idx,:] = np.array(batch_target[idx])[padded_tensor.cpu().tolist()] 
    return user_rank
        




# =================== LLPAUC Optimizer =============

class MinMax(SGD):
    def __init__(self, params=None, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, init_lr=0.01, hparams=None):
        super(MinMax, self).__init__(params, momentum, dampening,
                                     weight_decay, nesterov)
        print('use minmax')
        self.nu = torch.tensor(hparams['nu']).cuda()
        self.lam = torch.tensor(hparams['lam']).cuda()
        self.k = torch.tensor(hparams['k']).cuda()
        self.m = torch.tensor(hparams['m']).cuda()
        self.c1 = torch.tensor(hparams['c1']).cuda()
        self.c2 = torch.tensor(hparams['c2']).cuda()
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['old_param'] = torch.zeros(p.shape).cuda()
                state['old_grad'] = torch.zeros(p.shape).cuda()

    @torch.no_grad()
    def step(self, closure=None, pre=False, t=0):
        """Performs a single optimization step.
        Args:
          closure (callable, optional): A closure that reevaluates the model
              and returns the loss.
        """

        eta = self.k*1.0/torch.pow(self.m+t, 0.333)

        # eta = torch.Tensor([1e-4]).cuda()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            old_grad_list = []
            old_param_list = []
            for p in group['params']:
                if p.grad is not None:
                    state = self.state[p]
                    params_with_grad.append(p)
                    if pre:
                        state['old_grad'].zero_()
                        state['old_grad'].add_(p.grad, alpha=1)
                        if 'clip' in group.keys():
                            # if group['name'] == 'g':
                            #     group['clip'][0] = max(
                            #         self.param_groups[1]['params'][1] - 1, -self.param_groups[1]['params'][0])
                            #     d_p_list.append(torch.clip(
                            #         p + self.lam * state['old_param'], group['clip'][0], group['clip'][1]))
                            # else:
                            d_p_list.append(torch.clip(
                            p - self.nu * state['old_param'], group['clip'][0], group['clip'][1]))
                        else:
                            d_p_list.append(p - self.nu * state['old_param'])
                    old_grad_list.append(state['old_grad'])
                    old_param_list.append(state['old_param'])
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])
            if pre:
                sgd(params_with_grad,
                    d_p_list,
                    momentum_buffer_list,
                    eta,
                    weight_decay=group['weight_decay'],
                    momentum=group['momentum'],
                    dampening=group['dampening'],
                    nesterov=group['nesterov'])
                # update momentum_buffers in state
                for p, momentum_buffer, o_p, o_g in zip(params_with_grad, momentum_buffer_list, old_param_list, old_grad_list):
                    state = self.state[p]
                    state['momentum_buffer'] = momentum_buffer
                    state['old_param'] = o_p
                    state['old_grad'] = o_g
            else:
                rho = self.c1 * torch.square(eta)
                xi = self.c2 * torch.square(eta)
                for p, momentum_buffer, o_p, o_g in zip(params_with_grad, momentum_buffer_list, old_param_list, old_grad_list):
                    state = self.state[p]
                    state['momentum_buffer'] = momentum_buffer
                    state['old_grad'] = o_g
                    if group['name'] == 'g':
                        all_grad = p.grad + (1 - xi) * \
                            (state['old_param'] - o_g)
                    else:
                        all_grad = p.grad + (1 - rho) * \
                            (state['old_param'] - o_g)
                    state['old_param'].zero_()
                    state['old_param'].add_(all_grad, alpha=1)

        return loss


def sgd(params,
        d_p_list,
        momentum_buffer_list,
        lr,
        weight_decay,
        momentum,
        dampening,
        nesterov):
    r"""Functional API that performs SGD algorithm computation.
    See :class:`~torch.optim.SGD` for details.
    """
    print('lr',lr)
    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.mul_(1-lr)
        param.add_(lr*d_p, alpha=1)


# =================== end LLPAUC Optimizer =================

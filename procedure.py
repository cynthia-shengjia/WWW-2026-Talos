"""
Design training and test process
"""
import random
import time
from tqdm import tqdm
import world
import numpy as np
import torch
import utils
from utils import minibatch
from utils import timer
import model
import multiprocessing
import pdb
import dataloader
from world import cprint


CORES = multiprocessing.cpu_count() // 2  # 4090服务器上有256个核心


def train_double_epoch(dataset: dataloader.Loader, recommend_model, loss_class, epoch, config, w=None, flag = True):
    Recmodel = recommend_model
    Recmodel.train()
    loss: utils.LossFunc = loss_class

    start = time.time()

    # minibatch data load
    users, posItems = dataset.trainUser_tensor, dataset.trainItem_tensor
    users, posItems = utils.shuffle(users, posItems)

    user_pos_items = dataset.user_pos_items

    # minibatch data training
    batch_size = config["train_batch"]
    total_batch = len(users) // batch_size + 1
    aver_loss = 0.0

    # 1. the padding positive items
    # 2. the users arrange to make minibatch run

    iter_num = epoch * total_batch
    for batch_id, (batch_users, batch_pos) in enumerate(utils.minibatch(users, posItems, batch_size=batch_size)):
        batch_users = batch_users.cuda(non_blocking=True)
        batch_pos = batch_pos.cuda(non_blocking=True)


        batch_not_interaction_tensor = (~dataset.interaction_tensor[batch_users]).float()
        batch_neg = torch.multinomial(batch_not_interaction_tensor, config["num_negative_items"], replacement=True)

        cri = loss.precision_step(batch_users, batch_pos, batch_neg, epoch, batch_id)
        w.add_scalar("Loss", cri, iter_num + batch_id)
        aver_loss += cri
        # iter_num += 1

    for batch_id, batch_users in enumerate(utils.minibatch(utils.shuffle( torch.arange(0, dataset.n_users) ), batch_size=batch_size)):
        batch_user_pos = user_pos_items[batch_users]

        batch_not_interaction_tensor = (~dataset.interaction_tensor[batch_users]).float()
        batch_neg = torch.multinomial(batch_not_interaction_tensor, config["num_quantile_negative_items"], replacement=True)

        loss.quantile_setp(users = batch_users, user_pos=batch_user_pos, neg = batch_neg)

    aver_loss = aver_loss / total_batch
    # w.add_scalar("Loss", aver_loss, epoch)

    aver_diff_loss = 0
    if config["diff_margin_and_topk"]:
        with torch.no_grad():
            for batch_id, batch_users in enumerate(utils.minibatch( torch.arange(0, dataset.n_users) , batch_size=batch_size) ):
                cri = loss.check_diff_margin_and_topk(batch_users)
                aver_diff_loss += cri
                w.add_scalar("diff_margin_topk", cri, iter_num + batch_id)

    # aver_diff_loss = aver_diff_loss / dataset.n_users

    time_one_epoch = int(time.time() - start)
    return f"Loss{aver_loss:.3f}-Time{time_one_epoch}"

def train_double_batch_epoch_diff_neg(dataset: dataloader.Loader, recommend_model, loss_class, epoch, config, w=None, flag = True, device = "cpu",seed = 2024):
    Recmodel = recommend_model
    Recmodel.train()
    loss: utils.LossFunc = loss_class

    start = time.time()

    # minibatch data load
    users, posItems = dataset.trainUser_tensor, dataset.trainItem_tensor
    users, posItems = utils.shuffle(users, posItems)

    user_pos_items = dataset.user_pos_items

    # minibatch data training
    batch_size = config["train_batch"]
    total_batch = len(users) // batch_size + 1
    aver_loss = 0.0

    # 1. the padding positive items
    # 2. the users arrange to make minibatch run
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    iter_num = epoch * total_batch
    for batch_id, (batch_users, batch_pos) in enumerate(utils.minibatch(users, posItems, batch_size=batch_size)):
        batch_users = batch_users.cuda(non_blocking=True)
        batch_pos = batch_pos.cuda(non_blocking=True)


        batch_user_pos = user_pos_items[batch_users]

        batch_not_interaction_tensor = (~dataset.interaction_tensor[batch_users]).float()
        batch_neg = torch.multinomial(batch_not_interaction_tensor, config["num_negative_items"], replacement=True)
        batch_quantile_neg = torch.multinomial(batch_not_interaction_tensor, 
        config["num_quantile_negative_items"], replacement=True,generator=generator)

        loss.quantile_setp(users = batch_users, user_pos=batch_user_pos, neg = batch_quantile_neg)
        cri = loss.precision_step(batch_users, batch_pos, batch_neg, epoch, batch_id)
        w.add_scalar("Loss", cri, iter_num + batch_id)
        aver_loss += cri
        # iter_num += 1     
        

    aver_loss = aver_loss / total_batch
    # w.add_scalar("Loss", aver_loss, epoch)

    aver_diff_loss = 0
    if config["diff_margin_and_topk"]:
        with torch.no_grad():
            for batch_id, batch_users in enumerate(utils.minibatch( torch.arange(0, dataset.n_users) , batch_size=batch_size) ):
                cri = loss.check_diff_margin_and_topk(batch_users)
                aver_diff_loss += cri
                w.add_scalar("diff_margin_topk", cri, iter_num + batch_id)

    # aver_diff_loss = aver_diff_loss / dataset.n_users

    time_one_epoch = int(time.time() - start)
    return f"Loss{aver_loss:.3f}-Time{time_one_epoch}"


def train_double_batch_epoch(dataset: dataloader.Loader, recommend_model, loss_class, epoch, config, w=None, flag = True, device = "cpu",seed = 2024):
    Recmodel = recommend_model
    Recmodel.train()
    loss: utils.LossFunc = loss_class

    start = time.time()

    # minibatch data load
    users, posItems = dataset.trainUser_tensor, dataset.trainItem_tensor
    users, posItems = utils.shuffle(users, posItems)

    user_pos_items = dataset.user_pos_items

    # minibatch data training
    batch_size = config["train_batch"]
    total_batch = len(users) // batch_size + 1
    aver_loss = 0.0

    # 1. the padding positive items
    # 2. the users arrange to make minibatch run

    iter_num = epoch * total_batch
    for batch_id, (batch_users, batch_pos) in enumerate(utils.minibatch(users, posItems, batch_size=batch_size)):
        batch_users = batch_users.cuda(non_blocking=True)
        batch_pos = batch_pos.cuda(non_blocking=True)


        batch_user_pos = user_pos_items[batch_users]

        batch_not_interaction_tensor = (~dataset.interaction_tensor[batch_users]).float()
        batch_neg = torch.multinomial(batch_not_interaction_tensor, config["num_negative_items"], replacement=True)

        loss.quantile_setp(users = batch_users, user_pos=batch_user_pos, neg = batch_neg)
        cri = loss.precision_step(batch_users, batch_pos, batch_neg, epoch, batch_id)
        w.add_scalar("Loss", cri, iter_num + batch_id)
        aver_loss += cri
        # iter_num += 1     
        

    aver_loss = aver_loss / total_batch
    # w.add_scalar("Loss", aver_loss, epoch)

    aver_diff_loss = 0
    if config["diff_margin_and_topk"]:
        with torch.no_grad():
            for batch_id, batch_users in enumerate(utils.minibatch( torch.arange(0, dataset.n_users) , batch_size=batch_size) ):
                cri = loss.check_diff_margin_and_topk(batch_users)
                aver_diff_loss += cri
                w.add_scalar("diff_margin_topk", cri, iter_num + batch_id)

    # aver_diff_loss = aver_diff_loss / dataset.n_users

    time_one_epoch = int(time.time() - start)
    return f"Loss{aver_loss:.3f}-Time{time_one_epoch}"



def Train_original(dataset: dataloader.Loader, recommend_model, loss_class, epoch, config, w=None, flag = True):
    Recmodel = recommend_model
    Recmodel.train()
    loss: utils.LossFunc = loss_class

    start = time.time()
    if config["full_batch"]:
        # This means set batch size equals to interaction size
        # This can make the training process fast, however the memory of GPU should be enough
        S = utils.UniformSample_original(dataset, config["num_negative_items"])
        users, posItems, negItems = S
        users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    else:
        users, posItems = dataset.trainUser_tensor, dataset.trainItem_tensor
        users, posItems = utils.shuffle(users, posItems)

    if config["full_batch"]:
        users = users.cuda(non_blocking=True)
        posItems = posItems.cuda(non_blocking=True)
        negItems = negItems.cuda(non_blocking=True)

        batch_size = len(users)
        aver_loss = loss.step(users, posItems, negItems, epoch)
        w.add_scalar(f"BPRLoss/BPR", aver_loss, epoch)
    else:
        batch_size = config["train_batch"]
        total_batch = len(users) // batch_size + 1
        aver_loss = 0.0

        iter_num = epoch * total_batch
        for batch_id, (batch_users, batch_pos) in enumerate(utils.minibatch(users, posItems, batch_size=batch_size)):
            batch_users = batch_users.cuda(non_blocking=True)
            batch_pos = batch_pos.cuda(non_blocking=True)

            batch_not_interaction_tensor = (~dataset.interaction_tensor[batch_users]).float()
            batch_neg = torch.multinomial(batch_not_interaction_tensor, config["num_negative_items"], replacement=True)
            
            cri = loss.step(batch_users, batch_pos, batch_neg, epoch, batch_id, flag)
            w.add_scalar("Loss", cri, iter_num + batch_id)
            aver_loss += cri
            # iter_num += 1

        aver_loss = aver_loss / total_batch
        # w.add_scalar("Loss", aver_loss, epoch)
    time_one_epoch = int(time.time() - start)
    return f"Loss{aver_loss:.3f}-Time{time_one_epoch}"

def Train_noise_original(dataset: dataloader.Loader, recommend_model, loss_class, epoch, config, w=None, flag = True):
    Recmodel = recommend_model
    Recmodel.train()
    loss: utils.LossFunc = loss_class

    start = time.time()
    if config["full_batch"]:
        S = utils.UniformSample_original(dataset, config["num_negative_items"])
        users, posItems, negItems = S
        users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    else:
        users, posItems = dataset.trainUser_tensor, dataset.trainItem_tensor
        users, posItems = utils.shuffle(users, posItems)

    if config["full_batch"]:
        users = users.cuda(non_blocking=True)
        posItems = posItems.cuda(non_blocking=True)
        negItems = negItems.cuda(non_blocking=True)

        batch_size = len(users)
        aver_loss = loss.step(users, posItems, negItems, epoch)
        w.add_scalar(f"BPRLoss/BPR", aver_loss, epoch)
    else:
        batch_size = config["train_batch"]
        total_batch = len(users) // batch_size + 1
        aver_loss = 0.0

        iter_num = epoch * total_batch
        for batch_id, (batch_users, batch_pos) in enumerate(utils.minibatch(users, posItems, batch_size=batch_size)):
            batch_users = batch_users.cuda(non_blocking=True)
            batch_pos = batch_pos.cuda(non_blocking=True)

            batch_is_interaction_tensor = (dataset.interaction_tensor[batch_users]).float()
            batch_not_interaction_tensor = (~dataset.interaction_tensor[batch_users]).float()
                    
            noise_data_size = int(config["num_negative_items"] * config["noise_ratio"])
            normal_data_size = int(config["num_negative_items"] - noise_data_size)

            batch_neg_noise = torch.multinomial(batch_is_interaction_tensor, noise_data_size, replacement=True)
            batch_neg_normal = torch.multinomial(batch_not_interaction_tensor, normal_data_size, replacement=True)
            


            batch_neg = torch.cat((batch_neg_normal,batch_neg_noise),dim = 1)
            
            # print(batch_neg)
            # input()

            
            cri = loss.step(batch_users, batch_pos, batch_neg, epoch, batch_id, flag)
            w.add_scalar("Loss", cri, iter_num + batch_id)
            aver_loss += cri
            # iter_num += 1

        aver_loss = aver_loss / total_batch
        # w.add_scalar("Loss", aver_loss, epoch)
    time_one_epoch = int(time.time() - start)
    return f"Loss{aver_loss:.3f}-Time{time_one_epoch}"

def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)  # 一个包含batch个元素的list，每个元素是一个np数组
    pre, recall, ndcg, hitratio = [], [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret["precision"])
        recall.append(ret["recall"])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
        hitratio.append(utils.HitRatio(r))
    return {
        "recall": np.array(recall),
        "precision": np.array(pre),
        "ndcg": np.array(ndcg),
        "hitratio": np.array(hitratio),
    }

def test_one_batch_exposure(X):
    sorted_items = X[0]
    groundTrue = X[1]

    r = utils.getLabel(groundTrue, sorted_items)  # 一个包含batch个元素的list，每个元素是一个np数组
    pre, recall, ndcg, hitratio = [], [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret["precision"])
        recall.append(ret["recall"])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
        hitratio.append(utils.HitRatio(r))
    return {
        "recall": np.array(recall),
        "precision": np.array(pre),
        "ndcg": np.array(ndcg),
        "hitratio": np.array(hitratio),
    }


def exposureValid(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config["test_u_batch_size"]  # 默认是100，多少个user一起test

    dataset: utils.BasicDataset
    # testDict: dict = dataset.testDict
    validDict: dict = dataset.validDict 
    # Recmodel: model.LightGCN

    Recmodel = Recmodel.eval()
    max_K = max(world.topks)

    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {
        "precision": np.zeros(len(world.topks)),
        "recall": np.zeros(len(world.topks)),
        "ndcg": np.zeros(len(world.topks)),
        "hitratio": np.zeros(len(world.topks)),
    }

    with torch.no_grad():
        # users = list(testDict.keys())
        users = list(validDict.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1

        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            # batch_users是一个tuple，里面是user的id
            allPos = dataset.getUserPosItems(batch_users)  # train positive items
            validSeries = dataset.getTestSeries(batch_users) # testSeries

            # groundTrue = [testDict[u] for u in batch_users]  # test positive items
            groundTrue = [validDict[u] for u in batch_users]  # valid positive items

            batch_users_gpu = torch.Tensor(batch_users).long().cuda()

            rating = Recmodel.getUsersRating(batch_users_gpu)  # 给出users和所有item的评分，返回二维tensor
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)


            rating_K = utils.topK_target(rating,batch_users,validSeries,max_k=max_K)

            # _, rating_K = torch.topk(rating, k=max_K)

            users_list.append(batch_users)
            rating_list.append(rating_K)  # 每个元素是一个二维tensor，表示每个user的topk item
            groundTrue_list.append(groundTrue)  # 每个元素是一个两层list，表示每个user的test positive items

        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch_exposure, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch_exposure(x))

        for result in pre_results:
            results["recall"] += result["recall"]
            results["precision"] += result["precision"]
            results["ndcg"] += result["ndcg"]
            results["hitratio"] += result["hitratio"]
        results["recall"] /= float(len(users))
        results["precision"] /= float(len(users))
        results["ndcg"] /= float(len(users))
        results["hitratio"] /= float(dataset.testDataSize)

        # for i in range(len(world.topks)):
        #     w.add_scalar(f"Test/Recall_{world.topks[i]}", results["recall"][i], epoch)
        #     w.add_scalar(f"Test/Precision_{world.topks[i]}", results["precision"][i], epoch)
        #     w.add_scalar(f"Test/NDCG_{world.topks[i]}", results["ndcg"][i], epoch)
        #     w.add_scalar(f"Test/HitRatio_{world.topks[i]}", results["hitratio"][i], epoch)
        if multicore == 1:
            pool.close()

        return results

def exposureTest(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config["test_u_batch_size"]  # 默认是100，多少个user一起test

    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    # Recmodel: model.LightGCN

    Recmodel = Recmodel.eval()
    max_K = max(world.topks)

    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {
        "precision": np.zeros(len(world.topks)),
        "recall": np.zeros(len(world.topks)),
        "ndcg": np.zeros(len(world.topks)),
        "hitratio": np.zeros(len(world.topks)),
    }

    with torch.no_grad():
        users = list(testDict.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1

        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            # batch_users是一个tuple，里面是user的id
            allPos = dataset.getUserPosItems(batch_users)  # train positive items
            testSeries = dataset.getTestSeries(batch_users) # testSeries

            groundTrue = [testDict[u] for u in batch_users]  # test positive items
            batch_users_gpu = torch.Tensor(batch_users).long().cuda()

            rating = Recmodel.getUsersRating(batch_users_gpu)  # 给出users和所有item的评分，返回二维tensor
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)


            rating_K = utils.topK_target(rating,batch_users,testSeries,max_k=max_K)

            # _, rating_K = torch.topk(rating, k=max_K)

            users_list.append(batch_users)
            rating_list.append(rating_K)  # 每个元素是一个二维tensor，表示每个user的topk item
            groundTrue_list.append(groundTrue)  # 每个元素是一个两层list，表示每个user的test positive items

        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch_exposure, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch_exposure(x))

        for result in pre_results:
            results["recall"] += result["recall"]
            results["precision"] += result["precision"]
            results["ndcg"] += result["ndcg"]
            results["hitratio"] += result["hitratio"]
        results["recall"] /= float(len(users))
        results["precision"] /= float(len(users))
        results["ndcg"] /= float(len(users))
        results["hitratio"] /= float(dataset.testDataSize)

        # for i in range(len(world.topks)):
        #     w.add_scalar(f"Test/Recall_{world.topks[i]}", results["recall"][i], epoch)
        #     w.add_scalar(f"Test/Precision_{world.topks[i]}", results["precision"][i], epoch)
        #     w.add_scalar(f"Test/NDCG_{world.topks[i]}", results["ndcg"][i], epoch)
        #     w.add_scalar(f"Test/HitRatio_{world.topks[i]}", results["hitratio"][i], epoch)
        if multicore == 1:
            pool.close()

        return results


def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config["test_u_batch_size"]  # 默认是100，多少个user一起test

    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    # Recmodel: model.LightGCN

    Recmodel = Recmodel.eval()
    max_K = max(world.topks)

    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {
        "precision": np.zeros(len(world.topks)),
        "recall": np.zeros(len(world.topks)),
        "ndcg": np.zeros(len(world.topks)),
        "hitratio": np.zeros(len(world.topks)),
    }

    with torch.no_grad():
        users = list(testDict.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1

        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            # batch_users是一个tuple，里面是user的id
            allPos = dataset.getUserPosItems(batch_users)  # train positive items
            groundTrue = [testDict[u] for u in batch_users]  # test positive items
            batch_users_gpu = torch.Tensor(batch_users).long().cuda()

            rating = Recmodel.getUsersRating(batch_users_gpu)  # 给出users和所有item的评分，返回二维tensor
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)

            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())  # 每个元素是一个二维tensor，表示每个user的topk item
            groundTrue_list.append(groundTrue)  # 每个元素是一个两层list，表示每个user的test positive items

        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))

        for result in pre_results:
            results["recall"] += result["recall"]
            results["precision"] += result["precision"]
            results["ndcg"] += result["ndcg"]
            results["hitratio"] += result["hitratio"]
        results["recall"] /= float(len(users))
        results["precision"] /= float(len(users))
        results["ndcg"] /= float(len(users))
        results["hitratio"] /= float(dataset.testDataSize)

        for i in range(len(world.topks)):
            w.add_scalar(f"Test/Recall_{world.topks[i]}", results["recall"][i], epoch)
            w.add_scalar(f"Test/Precision_{world.topks[i]}", results["precision"][i], epoch)
            w.add_scalar(f"Test/NDCG_{world.topks[i]}", results["ndcg"][i], epoch)
            w.add_scalar(f"Test/HitRatio_{world.topks[i]}", results["hitratio"][i], epoch)
        if multicore == 1:
            pool.close()

        return results



def Valid(dataset, Recmodel, epoch, w=None, multicore=0):
    cprint("[Valid]")
    u_batch_size = world.config["test_u_batch_size"]  # 默认是100，多少个user一起test

    dataset: utils.BasicDataset
    # testDict: dict = dataset.testDict
    validDict: dict = dataset.validDict
    # Recmodel: model.LightGCN

    Recmodel = Recmodel.eval()
    max_K = max(world.topks)

    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {
        "precision": np.zeros(len(world.topks)),
        "recall": np.zeros(len(world.topks)),
        "ndcg": np.zeros(len(world.topks)),
        "hitratio": np.zeros(len(world.topks)),
    }

    with torch.no_grad():
        # validDict
        # users = list(testDict.keys())
        users = list(validDict.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1

        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            # batch_users是一个tuple，里面是user的id
            allPos = dataset.getUserPosItems(batch_users)  # train positive items
            # groundTrue = [testDict[u] for u in batch_users]  # test positive items
            groundTrue = [validDict[u] for u in batch_users]  # valid positive items
            batch_users_gpu = torch.Tensor(batch_users).long().cuda()

            rating = Recmodel.getUsersRating(batch_users_gpu)  # 给出users和所有item的评分，返回二维tensor
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            # 排除所有的训练样本，不参与计算metric
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)      

            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())  # 每个元素是一个二维tensor，表示每个user的topk item
            groundTrue_list.append(groundTrue)  # 每个元素是一个两层list，表示每个user的valid positive items

        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))

        for result in pre_results:
            results["recall"] += result["recall"]
            results["precision"] += result["precision"]
            results["ndcg"] += result["ndcg"]
            results["hitratio"] += result["hitratio"]
        results["recall"] /= float(len(users))
        results["precision"] /= float(len(users))
        results["ndcg"] /= float(len(users))
        results["hitratio"] /= float(dataset.validDataSize)

        # for i in range(len(world.topks)):
        #     w.add_scalar(f"Test/Recall_{world.topks[i]}", results["recall"][i], epoch)
        #     w.add_scalar(f"Test/Precision_{world.topks[i]}", results["precision"][i], epoch)
        #     w.add_scalar(f"Test/NDCG_{world.topks[i]}", results["ndcg"][i], epoch)
        #     w.add_scalar(f"Test/HitRatio_{world.topks[i]}", results["hitratio"][i], epoch)
        if multicore == 1:
            pool.close()

        return results

# Ref: https://github.com/donalee/BUIR/blob/main/Utils/evaluation.py

import copy
import time

import numpy as np
import pandas as pd
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# random.seed(123)
np.random.seed(456)
# tf.compat.v1.set_random_seed(123)

torch.manual_seed(456)
torch.cuda.manual_seed_all(456)

cuda_device=torch.device('cuda:0')

def loadKGData():
    print('Loading TransE data...')
    # gene_id = np.load('./data/sl_data/gene_id.npy')
    gene_id = pd.read_csv('../data/preprocessed_data/meta_table_9845.csv')
    gene_id = np.asarray(gene_id['unified_id'])
    # kgemb_data = np.load('./data/kg_embed/kg_TransE_l2_entity.npy') # kg id
    kgemb_data = np.load('../data/preprocessed_data/kg_TransE_l2_entity.npy') # kg id
    return kgemb_data, gene_id

# map genes to continuous ids
def map_genes():
    # sl_df = pd.read_csv('./data/sl_data/sl_pair_processed.csv', header=None)
    sl_df = pd.read_csv('../data/preprocessed_data/human_sl_9845.csv')
    sl_df = sl_df[['unified_id_A', 'unified_id_B']]
    set_IDa = set(sl_df['unified_id_A'])
    set_IDb = set(sl_df['unified_id_B'])
    list_all = sorted(list(set_IDa | set_IDb))
    id2orig = {}
    orig2id = {}
    for i in range(len(list_all)):
        id2orig[i] = int(list_all[i])
        orig2id[list_all[i]] = int(i)
    return id2orig, orig2id


def build_comp_score_mat(model, all_feature, data_loader, orig2id, gpu):
    data_loader = np.asarray(data_loader)
    g1_list = set(data_loader[:, 0])  # origin id
    g2_list = set(data_loader[:, 1])
    g1_g2_set = sorted(list(g1_list | g2_list))

    local_ind = {}  # local continuous: global continuous
    sl_ind = {}  # global continuous: local continuous
    for i in range(len(g1_g2_set)):
        local_ind[i] = orig2id[g1_g2_set[i]]
        sl_ind[orig2id[g1_g2_set[i]]] = i

    local_feature = []
    for i in local_ind.keys():
        local_feature.append(all_feature[local_ind[i], :])
    local_feature = np.asarray(local_feature)

    local_feature = torch.tensor(local_feature).to(cuda_device)
    # local_feature = torch.tensor(local_feature).cuda()
    # local_feature = torch.tensor(local_feature).to(torch.device('cuda:%d' % gpu))

    g1_target, g1_online = model.get_embedding(local_feature)

    score_mat1 = torch.matmul(g1_target, g1_online.transpose(0, 1))
    score_mat2 = torch.matmul(g1_online, g1_target.transpose(0, 1))
    score_mat = score_mat1 + score_mat2
    score_mat = score_mat.cpu()
    score_mat = np.asarray(score_mat)
    score_mat[range(score_mat.shape[0]),range(score_mat.shape[0])] = 0

    return score_mat, local_ind, sl_ind


def to_dict(model, data_loader, orig2id, score_mat, sl_ind):
    data_dict = {}
    gt_data_dict = {}

    for gene1_id, gene2_id in data_loader:

        id1 = sl_ind[orig2id[int(gene1_id)]]  # local id
        id2 = sl_ind[orig2id[int(gene2_id)]]

        if id1 not in data_dict.keys():
            data_dict[id1] = {}
            gt_data_dict[id1] = {}
        if id2 not in data_dict.keys():
            data_dict[id2] = {}
            gt_data_dict[id2] = {}

        data_dict[id1][id2] = [1, float(score_mat[id1, id2])]
        data_dict[id2][id1] = [1, float(score_mat[id1, id2])]
        gt_data_dict[id1][id2] = 1
        gt_data_dict[id2][id1] = 1

    return data_dict, gt_data_dict


def cal_score_mat(model, all_gene_feature, gpu):

    local_feature = torch.tensor(all_gene_feature).to(cuda_device)
    # local_feature = torch.tensor(all_gene_feature).cuda()
    # local_feature = torch.tensor(all_gene_feature).to(torch.device('cuda:%d' % gpu))

    g1_target, g1_online = model.get_embedding(local_feature)

    score_mat1 = torch.matmul(g1_target, g1_online.transpose(0, 1))
    score_mat2 = torch.matmul(g1_online, g1_target.transpose(0, 1))
    score_mat = score_mat1 + score_mat2
    score_mat = score_mat.cpu()
    score_mat = np.asarray(score_mat)
    score_mat[range(score_mat.shape[0]), range(score_mat.shape[0])] = 0

    return score_mat


def evaluate(model, train_loader, test_loader, all_gene_feature, epoch, fold_num, gpu):
    metrics = {
        'P10': [], 'P20': [], 'P50': [], 'P100': [],
        'R10': [], 'R20': [], 'R50': [], 'R100': [],
        'N10': [], 'N20': [], 'N50': [], 'N100': []
    }
    tv_ind=list(range(len(train_loader)))
    train_loader=train_loader[tv_ind[:int(len(train_loader)*0.875)]]
    valid_loader=train_loader[tv_ind[int(len(train_loader)*0.875)]:]

    results = {'valid': copy.deepcopy(metrics), 'test': copy.deepcopy(metrics)}

    id2orig, orig2id = map_genes()

    # all_score_mat = cal_score_mat(model, all_gene_feature, gpu)

    train_local_score_mat, train_local_ind, train_sl_ind = build_comp_score_mat(model, all_gene_feature, train_loader,
                                                                                orig2id, gpu)
    valid_local_score_mat, valid_local_ind, valid_sl_ind = build_comp_score_mat(model, all_gene_feature, valid_loader,
                                                                                orig2id, gpu)
    test_local_score_mat, test_local_ind, test_sl_ind = build_comp_score_mat(model, all_gene_feature, test_loader,
                                                                             orig2id, gpu)

    print(train_local_score_mat.shape)
    print(valid_local_score_mat.shape)
    print(test_local_score_mat.shape)

    # Building train dicts
    train_dict, gt_train_dict = to_dict(model, train_loader, orig2id, train_local_score_mat, train_sl_ind)
    # Building valid dicts
    valid_dict, gt_valid_dict = to_dict(model, valid_loader, orig2id, valid_local_score_mat, valid_sl_ind)
    # Building test dicts
    test_dict, gt_test_dict = to_dict(model, test_loader, orig2id, test_local_score_mat, test_sl_ind)

    print(f"{len(valid_dict)} will be validating")
    print(f"{len(test_dict)} will be testing")

    for mode in ['valid', 'test']:

        if mode == 'valid':
            sorted_mat = np.argsort(valid_local_score_mat, axis=0, kind='stable')[::-1, :]
            data_dict = valid_dict
            gt_mat = gt_valid_dict
        elif mode == 'test':
            sorted_mat = np.argsort(test_local_score_mat, axis=0, kind='stable')[::-1, :]
            data_dict = test_dict
            gt_mat = gt_test_dict

        print(f"=============== now {mode} =================")
        tic = time.time()
        cnt = 0

        for test_gene in data_dict.keys():
            if len(gt_mat[test_gene].keys()) > 0:
                cnt += 1

                sorted_list = list(sorted_mat[:, test_gene])
                sorted_list_tmp = []

                already_seen_items = []

                if mode == 'valid':
                    global_ind = valid_local_ind[test_gene]  # global continuous id

                    if global_ind in train_sl_ind.keys():
                        test_in_local_train_ind = train_sl_ind[global_ind]
                    else:
                        test_in_local_train_ind = -1
                    if global_ind in test_sl_ind.keys():
                        test_in_local_test_ind = test_sl_ind[global_ind]
                    else:
                        test_in_local_test_ind = -1

                    # Find the overlap between valid and train genes
                    if test_in_local_train_ind in gt_train_dict.keys():
                        already_seen_train_items = set(gt_train_dict[test_in_local_train_ind].keys())  # train local
                        already_seen_train_items = [
                            valid_sl_ind[train_local_ind[i]] if train_local_ind[i] in valid_sl_ind else -1 for i in
                            already_seen_train_items]
                        already_seen_items += already_seen_train_items
                    if test_in_local_test_ind in gt_test_dict.keys():
                        already_seen_test_items = set(gt_test_dict[test_in_local_test_ind].keys())
                        already_seen_test_items = [
                            valid_sl_ind[test_local_ind[i]] if test_local_ind[i] in valid_sl_ind else -1 for i in
                            already_seen_test_items]
                        already_seen_items += already_seen_test_items

                elif mode == 'test':
                    global_ind = test_local_ind[test_gene]  # global continuous id

                    if global_ind in train_sl_ind.keys():
                        test_in_local_train_ind = train_sl_ind[global_ind]
                    else:
                        test_in_local_train_ind = -1
                    if global_ind in valid_sl_ind.keys():
                        test_in_local_valid_ind = valid_sl_ind[global_ind]
                    else:
                        test_in_local_valid_ind = -1

                    if test_in_local_train_ind in gt_train_dict.keys():
                        already_seen_train_items = set(gt_train_dict[test_in_local_train_ind].keys())  # local id
                        already_seen_train_items = [
                            test_sl_ind[train_local_ind[i]] if train_local_ind[i] in test_sl_ind.keys() else -1 for i in
                            already_seen_train_items]
                        already_seen_items += already_seen_train_items
                    if test_in_local_valid_ind in gt_valid_dict.keys():
                        already_seen_valid_items = set(gt_valid_dict[test_in_local_valid_ind].keys())
                        already_seen_valid_items = [
                            test_sl_ind[valid_local_ind[i]] if valid_local_ind[i] in test_sl_ind.keys() else -1 for i in
                            already_seen_valid_items]
                        already_seen_items += already_seen_valid_items

                for item in sorted_list:
                    if item not in already_seen_items:
                        sorted_list_tmp.append(item)
                    if len(sorted_list_tmp) == 100: break

                for topk in [10, 20, 50, 100]:
                    # hit topk
                    hit_topk = len(set(sorted_list_tmp[:topk]) & set(gt_mat[test_gene].keys()))

                    # ndcg topk
                    denom = np.log2(np.arange(2, topk + 2))
                    dcg_topk = np.sum(np.in1d(sorted_list_tmp[:topk], list(gt_mat[test_gene].keys())) / denom)
                    idcg_topk = np.sum((1 / denom)[:min(len(list(gt_mat[test_gene].keys())), topk)])

                    results[mode][f'P{topk}'].append(
                        0 if hit_topk == 0 or len(gt_mat[test_gene].keys()) == 0 
                        else hit_topk / min(topk, len(gt_mat[test_gene].keys()))
                        )
                    results[mode][f'R{topk}'].append(
                        0 if hit_topk == 0 or len(gt_mat[test_gene].keys()) == 0 
                        else hit_topk / len(gt_mat[test_gene].keys())
                        )
                    results[mode][f'N{topk}'].append(
                        0 if dcg_topk == 0 or idcg_topk == 0 
                        else dcg_topk / idcg_topk)
        toc = time.time()
        print(cnt)

    for mode in ['test', 'valid']:
        for topk in [10, 20, 50, 100]:
            results[mode]['P' + str(topk)] = round(np.asarray(results[mode]['P' + str(topk)]).mean(), 4)
            results[mode]['R' + str(topk)] = round(np.asarray(results[mode]['R' + str(topk)]).mean(), 4)
            results[mode]['N' + str(topk)] = round(np.asarray(results[mode]['N' + str(topk)]).mean(), 4)

    score_label_mat = []
    return results, score_label_mat


def print_eval_results(eval_results):
    for mode in ['valid', 'test']:
        for topk in [10, 20, 50, 100]:
            p = eval_results[mode]['P' + str(topk)]
            r = eval_results[mode]['R' + str(topk)]
            n = eval_results[mode]['N' + str(topk)]

            print('{:5s} P@{}: {:.4f}, R@{}: {:.4f}, N@{}: {:.4f}'.format(mode.upper(), topk, p, topk, r, topk, n))
        print()

def cal_final_result(result_all):
    for m in ['P', 'R', 'N']:
        for top in [10, 20, 50, 100]:
            tmp = []
            for i in [1, 2, 3, 4, 5]:
                tmp.append(result_all['result' + str(i)]['test'][f'{m}{top}'])
            print(f'{m}{top}:{round(np.mean(tmp), 4)},{round(np.std(tmp), 4)}')
            # logger.info(f'{m}{top}:{round(np.mean(tmp), 4)},{round(np.std(tmp), 4)}')
#!/usr/bin/python
# -*- coding:utf-8 -*-
# import pdb
# import csv
# import os
import numpy as np
from collections import defaultdict
# from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import auc
from scipy import sparse
import scipy.io as sio

# random.seed(123)
np.random.seed(456)
# tf.compat.v1.set_random_seed(123)

# torch.manual_seed(123)
# torch.cuda.manual_seed_all(123)

def load_ppi_data():
    with open("data/List_Proteins_in_SL.txt", "r") as inf:
        ppis = [line.rstrip() for line in inf]
        id_mapping = dict(zip(ppis, range(len(set(ppis)))))
    num = len(ppis)
    inter_pairs, inter_scores = [], []
    with open("data/SL_Human_Approved.txt", "r") as inf:
        for line in inf:
            id1, id2, s = line.rstrip().split()
            inter_pairs.append((id_mapping[id1], id_mapping[id2]))
            inter_scores.append(float(s))
    inter_pairs = np.array(inter_pairs, dtype=np.int32)
    inter_scores = np.array(inter_scores)
    go_pairs, go_sim = [], []
    with open("data/Human_GOsim.txt", "r") as inf:
        for i, line in enumerate(inf):
            data = line.rstrip().split()
            for j, s in enumerate(data):
                go_pairs.append((i, num - len(data) + j))
                go_sim.append(float(s))
    go_pairs = np.array(go_pairs, dtype=np.int32)
    go_sim_mat = sparse.coo_matrix((go_sim, (go_pairs[:, 0], go_pairs[:, 1])), shape=(num, num))
    return inter_pairs, inter_scores, go_sim_mat, id_mapping


def load_ppi_data_long(flag):
    if flag == 0:
        with open("data/List_Proteins_in_SL.txt", "r") as inf:
            ppis = [line.rstrip() for line in inf]
            id_mapping = dict(zip(ppis, range(len(set(ppis)))))
        num = len(ppis)
        inter_pairs, inter_scores = [], []
        with open("data/SL_Human_Approved.txt", "r") as inf:
            for line in inf:
                id1, id2, s = line.rstrip().split()
                inter_pairs.append((id_mapping[id1], id_mapping[id2]))
                inter_scores.append(float(s))
        inter_pairs = np.array(inter_pairs, dtype=np.int32)
        inter_scores = np.array(inter_scores)
        go_pairs, go_sim = [], []
        with open("data/Human_GOsim.txt", "r") as inf:
            for i, line in enumerate(inf):
                data = line.rstrip().split()
                for j, s in enumerate(data):
                    go_pairs.append((i, num - len(data) + j))
                    go_sim.append(float(s))
        go_pairs = np.array(go_pairs, dtype=np.int32)
        go_sim_mat = sparse.coo_matrix((go_sim, (go_pairs[:, 0], go_pairs[:, 1])), shape=(num, num))

        GOsim_CC = sio.loadmat('data/Human_GOsim_CC.mat')
        go_sim_cc_mat = GOsim_CC['Human_GOsim_CC']

        ppi_sparse = sio.loadmat('data/gene_ppi_sparse.mat')
        ppi_sparse_mat = ppi_sparse['gene_ppi_sparse']

        co_pathway = sio.loadmat('data/gene_co_pathway.mat')
        co_pathway_mat = co_pathway['gene_co_pathway']

    elif flag == 1:
        with open("SynlethDB_extension/List_Proteins_in_SL.txt", "r") as inf:
            ppis = [line.rstrip() for line in inf]
            id_mapping = dict(zip(ppis, range(len(set(ppis)))))
        num = len(ppis)
        inter_pairs, inter_scores = [], []
        with open("SynlethDB_extension/SL_Human_FinalCheck.txt", "r") as inf:
            for line in inf:
                id1, id2, s = line.rstrip().split()
                if id1 in id_mapping and id2 in id_mapping:
                    if id_mapping[id1] > id_mapping[id2]:
                        inter_pairs.append((id_mapping[id2], id_mapping[id1]))
                    elif id_mapping[id1] < id_mapping[id2]:
                        inter_pairs.append((id_mapping[id1], id_mapping[id2]))
                # inter_scores.append(float(s))
        inter_pairs = np.array(inter_pairs, dtype=np.int32)
        inter_scores = np.array(inter_scores)

        go_sim_mat = sio.loadmat('SynlethDB_extension/gene_similarity_BP.mat')
        go_sim_mat = go_sim_mat['gene_similarity_BP']

        go_sim_cc_mat = sio.loadmat('SynlethDB_extension/gene_similarity_CC.mat')
        go_sim_cc_mat = go_sim_cc_mat['gene_similarity_CC']

        ppi_sparse_mat = sio.loadmat('SynlethDB_extension/gene_ppi_sparse.mat')
        ppi_sparse_mat = ppi_sparse_mat['gene_ppi_sparse']

        co_pathway_mat = sio.loadmat('SynlethDB_extension/gene_ppi_pathway.mat')
        co_pathway_mat = co_pathway_mat['gene_ppi_pathway']

        return inter_pairs, inter_scores, go_sim_mat, go_sim_cc_mat, ppi_sparse_mat, co_pathway_mat, id_mapping


def cross_validation(intMat, seeds, cv=0, num=10):
    cv_data = defaultdict(list)
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cv == 0:
            index = prng.permutation(num_drugs)
        if cv == 1:
            index = prng.permutation(intMat.size)
        step = index.size / num
        for i in range(num):
            if i < num - 1:
                ii = index[i * step:(i + 1) * step]
            else:
                ii = index[i * step:]
            if cv == 0:
                test_data = np.array([[k, j] for k in ii for j in range(num_targets)], dtype=np.int32)
            elif cv == 1:
                test_data = np.array([[k / num_targets, k % num_targets] for k in ii], dtype=np.int32)
            x, y = test_data[:, 0], test_data[:, 1]
            test_label = intMat[x, y]
            W = np.ones(intMat.shape)
            W[x, y] = 0
            cv_data[seed].append((W, test_data, test_label))
    return cv_data


def kfold_cv(intMat, seed, cvs=3, num_folds=10, nest_cv=True):
    m, n = intMat.shape
    prng = np.random.RandomState(seed)
    cv_data = []
    if cvs == 1:
        kf = KFold(intMat.size, n_folds=num_folds, shuffle=True, random_state=prng)
    elif cvs == 2:
        kf = KFold(m, n_folds=num_folds, shuffle=True, random_state=prng)
    elif cvs == 3:
        kf = KFold(n, n_folds=num_folds, shuffle=True, random_state=prng)
    for train, test in kf:
        W = np.ones(intMat.shape)
        if cvs == 1:
            x, y = test / n, test % n
        elif cvs == 2:
            x = np.repeat(test, n)
            y = np.tile(np.arange(n), test.size)
        elif cvs == 3:
            x = np.tile(np.arange(m), test.size)
            y = np.repeat(test, m)
        W[x, y] = 0
        if nest_cv:
            inner_kf = KFold(train.size, n_folds=3, shuffle=True, random_state=prng)
            inner_cv_data = []
            for inner_train, inner_test in inner_kf:
                if cvs == 1:
                    x1, y1 = train[inner_test] / n, train[inner_test] % n
                elif cvs == 2:
                    x1 = np.repeat(train[inner_test], n)
                    y1 = np.tile(np.arange(n), inner_test.size)
                elif cvs == 3:
                    x1 = np.tile(np.arange(m), inner_test.size)
                    y1 = np.repeat(train[inner_test], m)
                W1 = W.copy()
                W1[x1, y1] = 0
                inner_cv_data.append((W1, x1, y1, intMat[x1, y1]))
            cv_data.append((inner_cv_data, W, x, y, intMat[x, y]))
        else:
            cv_data.append((W, x, y, intMat[x, y]))
    return cv_data


def svd_init(M, num_factors):
    from scipy.linalg import svd
    U, s, V = svd(M, full_matrices=False)
    ii = np.argsort(s)[::-1][:num_factors]
    s1 = np.sqrt(np.diag(s[ii]))
    U0, V0 = U[:, ii].dot(s1), s1.dot(V[ii, :])
    return U0, V0.T


def mean_confidence_interval(data, confidence=0.95):
    import scipy as sp
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h


def write_metric_vector_to_file(auc_vec, file_name):
    np.savetxt(file_name, auc_vec, fmt='%.6f')


def load_metric_vector(file_name):
    return np.loadtxt(file_name, dtype=np.float64)


def evaluation_two(train_pairs, test_pairs, predictR):
    num = predictR.shape[0]
    x, y = np.triu_indices(num, k=1)
    c_set = set(zip(x, y)) - set(zip(train_pairs[:, 0], train_pairs[:, 1])) - set(
        zip(train_pairs[:, 1], train_pairs[:, 0]))
    inx = np.array(list(c_set))
    Y = np.zeros((num, num))
    Y[test_pairs[:, 0], test_pairs[:, 1]] = 1
    Y[test_pairs[:, 1], test_pairs[:, 0]] = 1
    labels = Y[inx[:, 0], inx[:, 1]]
    val = predictR[inx[:, 0], inx[:, 1]]
    auc_val = roc_auc_score(labels, val)
    prec, rec, thr = precision_recall_curve(labels, val)
    aupr_val = auc(rec, prec)
    return auc_val, aupr_val


def generate_test_neg_data(intMat, num_pos_train):
    x_train, y_train = np.where(intMat == 1)
    x_test, y_test = np.where(intMat == 0)

    reorder = np.arange(x_test.size)
    np.random.shuffle(reorder)

    x_test_sampled, y_test_sampled = x_test[reorder[0:num_pos_train]], y_test[reorder[0:num_pos_train]]

    test_edges_neg = np.concatenate(
        (x_test_sampled.reshape([num_pos_train, -1]), y_test_sampled.reshape([num_pos_train, -1])), axis=1)

    return test_edges_neg


def evaluation_bal(adj_rec, edges_pos, edges_neg):
    # Predict on test set of edges
    preds = []
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]])

    preds_neg = []

    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]])
        if len(preds_neg) == len(preds):
            break

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    fpr, tpr, th = roc_curve(labels_all, preds_all)
    roc_score = auc(fpr, tpr)
    prec, rec, th = precision_recall_curve(labels_all, preds_all)
    aupr_score = auc(rec, prec)

    return roc_score, aupr_score



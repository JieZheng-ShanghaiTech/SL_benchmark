#!/usr/bin/python
# coding:utf-8
from numpy import *
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
# from sklearn import preprocessing
# import math
# import getopt
from scipy.stats import *

import utils.grsmf_utils as fun
import os

# random.seed(123)
np.random.seed(456)
# tf.compat.v1.set_random_seed(123)

# torch.manual_seed(123)
# torch.cuda.manual_seed_all(123)


class GRSMF:

    def __init__(self, lambda_d=2 ** (2), beta=2 ** (2), max_iter=150, seed=123, wandb_run=None):
        self.lambda_d = lambda_d  # default 100
        self.beta = beta
        self.max_iter = max_iter
        self.seed = seed
        self.wandb_run=wandb_run

    def fix_model(self, W, intMat, simMat, simMat_cc, simMat_sparse):
        size = intMat.shape
        X = np.multiply(W, intMat)
        S = simMat
        dd = np.array(S.sum(axis=1))
        D = np.diag(dd)
        L = D - S

        # S_c = simMat_cc
        dd_c = np.array(simMat_cc.sum(axis=1))
        # D_c = np.diag(dd_c)

        # S_p = simMat_sparse
        dd_p = np.array(simMat_sparse.sum(axis=1))
        # D_p = np.diag(dd_p)

        S = S + simMat_cc + simMat_sparse
        D = D + np.diag(dd_c) + np.diag(dd_p)

        lowest_score = Inf
        n = size[0]
        U_old = simMat
        U_old[range(n), range(n)] = U_old[range(n), range(n)] + 0.5
        # for i in range(n):
        #     U_old[i, i] = U_old[i, i] + 0.5
        yy = (U_old.sum(axis=1))
        yy1 = yy.reshape(n, 1)
        U_old = U_old / np.tile(yy1, (1, n))

        for j in range(self.max_iter):
            print("iteration: %d" % j)
            U_old,X, W = np.mat(U_old),np.mat(X), np.mat(W)
            Dan = 4 * (X * U_old) * np.multiply(W, X) + 2 * self.beta * S * U_old
            Dap = 4 * (X * U_old) * (np.multiply(W, U_old.T * X * U_old)) + 2 * self.lambda_d * U_old + 2 * self.beta * (D * U_old)
            Dapeps = np.finfo(float).eps * np.ones([n, n])
            cc = np.array((U_old / (Dap + Dapeps)).sum(axis=1).T)
            aa = np.diag(cc[0])
            ba = (np.multiply(U_old, Dan / (Dap + Dapeps))).sum(axis=1)
            self.U = np.multiply(U_old, aa * Dan + np.ones([n, n])) / (aa * Dap + np.tile(ba, (1, n)) + Dapeps)
            err = (abs(self.U - U_old)).sum(axis=0).sum(axis=1)
            self.wandb_run.log({
                'Mat_U_err':double(err)
            })
            if err < 1e-5:
                break
            U_old = self.U
        score = np.linalg.norm(np.multiply(W, X - U_old.T * X * U_old), 'fro') ** 2 + self.lambda_d * np.linalg.norm(U_old, 'fro') ** 2 + self.beta * np.trace(U_old.T * L * U_old)
        self.wandb_run.log({
                'train_loss':score
        })
        if score < lowest_score:
            self.U = (self.U + self.U.T) / 2
            lowest_score = score

        X_pre = self.U.T * X * self.U
        self.predictR = np.array(X_pre)

    def predict_scores(self, test_data, N):
        # X_pre = self.U.T * X * self.U
        # self.predictR = np.array(X_pre)
        inx = np.array(test_data)
        return self.predictR[inx[:, 0], inx[:, 1]]

    def predict(self, test_data):
        val = np.sum(self.U[test_data[:, 0], :] * self.U[test_data[:, 1], :], axis=1)
        val = np.exp(val)
        val = val / (1 + val)
        return val

    def evaluation(self, test_data, test_label):
        scores = self.predictR[test_data[:, 0], test_data[:, 1]]
        prec, rec, thr = precision_recall_curve(test_label, scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, scores)
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val

    def __str__(self):
        return f"Model: SRMF,  lambda_d:{self.lambda_d},  beta:{self.beta}, max_iter:{self.max_iter}"
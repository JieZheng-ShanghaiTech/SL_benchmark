import copy
import time
import numpy as np
import os
import scipy.sparse as sp
import wandb
from preprocess import cal_metrics, ChecktoSave
from models.grsmf import GRSMF

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.random.seed(456)

def train_grsmf(parameters, pos_samples, neg_samples, mode=None, save_mat=False, ex_compt=None,indep_test=None):
    if indep_test:
        graph_train_pos_kfold, _, _, train_pos_kfold, valid_pos_kfold, test_pos_kfold = pos_samples
        graph_train_neg_kfold, _, _, train_neg_kfold, valid_neg_kfold, test_neg_kfold = neg_samples
    else:
        graph_train_pos_kfold, _, train_pos_kfold, test_pos_kfold = pos_samples
        graph_train_neg_kfold, _, train_neg_kfold, test_neg_kfold = neg_samples

    kfold = parameters['kfold']
    num_node = parameters['num_node']
    p_n = parameters['pos_neg']
    d_s = parameters['division_strategy']
    n_s = parameters['negative_strategy']
    
    base_suffix = '_score_mats'
    if ex_compt:
        base_suffix = '_score_mats_wo_compt'
    
    if mode=='final_res':
        kfold=1
        p_n = d_s = n_s = 'final_res'

    ppi_sparse_mat = sp.load_npz('../data/preprocessed_data/ppi_sparse_upper_matrix_without_sl_relation_9845.npz')
    ppi_sparse_mat = (ppi_sparse_mat + ppi_sparse_mat.T).toarray()
    go_sim_mat = np.load('../data/preprocessed_data/final_gosim_bp_from_r_9845.npy')
    go_sim_cc_mat = np.load('../data/preprocessed_data/final_gosim_cc_from_r_9845.npy')

    run = wandb.init(
        # set the wandb project where this run will be logged
        project="Benchmarking",
        group="GRSMF",
        job_type=f"{p_n}_{d_s}_{n_s}",
        # track hyperparameters and run metadata
        config=parameters
    )
    checktosave = ChecktoSave(kfold)
    for fold_num in range(kfold):
        graph_train_pos = graph_train_pos_kfold[fold_num]
        # graph_train_neg = graph_train_neg_kfold[fold_num]
        train_pos = train_pos_kfold[fold_num]
        train_neg = train_neg_kfold[fold_num]

        model = GRSMF(lambda_d=2 ** (-7), beta=2 ** (-5), max_iter=10,wandb_run=run)

        x, y = train_pos[:, 0], train_pos[:, 1]
        IntMat = graph_train_pos.toarray()

        x_neg, y_neg = train_neg[:, 0], train_neg[:, 1]
        W = np.zeros((num_node, num_node))
        W[x, y] = 1
        W[y, x] = 1
        W[x_neg, y_neg] = 1
        W[y_neg, x_neg] = 1

        model.fix_model(W, IntMat, go_sim_mat, go_sim_cc_mat, ppi_sparse_mat)

        score_mat=model.predictR
        
        if indep_test:
            train_metrics = cal_metrics(score_mat, train_pos_kfold[fold_num], train_neg_kfold[fold_num])
            run.log({
                    'train_auc':train_metrics[0],'train_f1':train_metrics[1],'train_aupr':train_metrics[2],
                    'train_N10':train_metrics[3],'train_N20':train_metrics[4],'train_N50':train_metrics[5],
                    'train_R10':train_metrics[6],'train_R20':train_metrics[7],'train_R50':train_metrics[8],
                    'train_P10':train_metrics[9],'train_P20':train_metrics[10],'train_P50':train_metrics[11],
                    'train_M10':train_metrics[12],'train_M20':train_metrics[13],'train_M50':train_metrics[14],
                })
            checktosave.update_train_classify(fold_num, 0, np.asarray([train_metrics[0], train_metrics[2], train_metrics[1]]))
            checktosave.update_train_ranking(fold_num, 0, train_metrics[3:])
            
            valid_metrics = cal_metrics(score_mat, valid_pos_kfold[fold_num], valid_neg_kfold[fold_num], train_pos_kfold[fold_num])
            run.log({
                'valid_auc':valid_metrics[0],'valid_f1':valid_metrics[1],'valid_aupr':valid_metrics[2],
                'valid_N10':valid_metrics[3],'valid_N20':valid_metrics[4],'valid_N50':valid_metrics[5],
                'valid_R10':valid_metrics[6],'valid_R20':valid_metrics[7],'valid_R50':valid_metrics[8],
                'valid_P10':valid_metrics[9],'valid_P20':valid_metrics[10],'valid_P50':valid_metrics[11],
                'valid_M10':valid_metrics[12],'valid_M20':valid_metrics[13],'valid_M50':valid_metrics[14],
            })
            
            test_metrics = cal_metrics(score_mat, test_pos_kfold[fold_num], test_neg_kfold[fold_num], train_pos_kfold[fold_num])
            run.log({
                'test_auc':test_metrics[0],'test_f1':test_metrics[1],'test_aupr':test_metrics[2],
                'test_N10':test_metrics[3],'test_N20':test_metrics[4],'test_N50':test_metrics[5],
                'test_R10':test_metrics[6],'test_R20':test_metrics[7],'test_R50':test_metrics[8],
                'test_P10':test_metrics[9],'test_P20':test_metrics[10],'test_P50':test_metrics[11],
                'test_M10':test_metrics[12],'test_M20':test_metrics[13],'test_M50':test_metrics[14],
            })
            print(test_metrics)
            
            if save_mat:
                if checktosave.update_classify(fold_num, 0, np.asarray([valid_metrics[0], valid_metrics[2], valid_metrics[1]])):
                    if not os.path.exists(f'../results/{n_s}{base_suffix}/grsmf'):
                        os.makedirs(f'../results/{n_s}{base_suffix}/grsmf')
                    path = f'../results/{n_s}{base_suffix}/grsmf/grsmf_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_classify.npy'
                    checktosave.save_mat(path, score_mat)
                    checktosave.update_indep_test_classify(fold_num, 0, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]]))
                if checktosave.update_ranking(fold_num, 0, valid_metrics[3:]):
                    path = f'../results/{n_s}{base_suffix}/grsmf/grsmf_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_ranking.npy'
                    checktosave.save_mat(path, score_mat)
                    checktosave.update_indep_test_ranking(fold_num, 0, test_metrics[3:])
            else:
                if checktosave.update_classify(fold_num, 0, np.asarray([valid_metrics[0], valid_metrics[2], valid_metrics[1]])):
                    checktosave.update_indep_test_classify(fold_num, 0, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]]))
                if checktosave.update_ranking(fold_num, 0, valid_metrics[3:]):
                    checktosave.update_indep_test_ranking(fold_num, 0, test_metrics[3:])
                
        else:
            train_metrics = cal_metrics(score_mat, train_pos_kfold[fold_num], train_neg_kfold[fold_num])
            run.log({
                    'train_auc':train_metrics[0],'train_f1':train_metrics[1],'train_aupr':train_metrics[2],
                    'train_N10':train_metrics[3],'train_N20':train_metrics[4],'train_N50':train_metrics[5],
                    'train_R10':train_metrics[6],'train_R20':train_metrics[7],'train_R50':train_metrics[8],
                    'train_P10':train_metrics[9],'train_P20':train_metrics[10],'train_P50':train_metrics[11],
                    'train_M10':train_metrics[12],'train_M20':train_metrics[13],'train_M50':train_metrics[14],
                })
            checktosave.update_train_classify(fold_num, 0, np.asarray([train_metrics[0], train_metrics[2], train_metrics[1]]))
            checktosave.update_train_ranking(fold_num, 0, train_metrics[3:])
            test_metrics = cal_metrics(score_mat, test_pos_kfold[fold_num], test_neg_kfold[fold_num], train_pos_kfold[fold_num])
            run.log({
                'test_auc':test_metrics[0],'test_f1':test_metrics[1],'test_aupr':test_metrics[2],
                'test_N10':test_metrics[3],'test_N20':test_metrics[4],'test_N50':test_metrics[5],
                'test_R10':test_metrics[6],'test_R20':test_metrics[7],'test_R50':test_metrics[8],
                'test_P10':test_metrics[9],'test_P20':test_metrics[10],'test_P50':test_metrics[11],
                'test_M10':test_metrics[12],'test_M20':test_metrics[13],'test_M50':test_metrics[14],
            })
            print(test_metrics)
            
            if save_mat:
                if checktosave.update_classify(fold_num, 0, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]])):
                    if not os.path.exists(f'../results/{n_s}{base_suffix}/grsmf'):
                        os.makedirs(f'../results/{n_s}{base_suffix}/grsmf')
                    path = f'../results/{n_s}{base_suffix}/grsmf/grsmf_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_classify.npy'
                    checktosave.save_mat(path, score_mat)
                if checktosave.update_ranking(fold_num, 0, test_metrics[3:]):
                    path = f'../results/{n_s}{base_suffix}/grsmf/grsmf_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_ranking.npy'
                    checktosave.save_mat(path, score_mat)
            else:
                checktosave.update_classify(fold_num, 0, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]]))
                checktosave.update_ranking(fold_num, 0, test_metrics[3:])
                
            
        
    run.finish()
    # auc f1 aupr N10 N20 N50 R10 R20 R50 P10 P20 P50
    if indep_test:
        all_metrics = checktosave.get_all_indep_test_metrics()
    else:
        all_metrics = checktosave.get_all_metrics()
    
    return all_metrics

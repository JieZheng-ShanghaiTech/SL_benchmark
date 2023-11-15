import copy
import time
import os
import numpy as np
import wandb
from preprocess import cal_metrics, ChecktoSave
from models.sl2mf import LMF
from utils.sl2mf_utils import mean_confidence_interval, evalution_bal

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

np.random.seed(456)

def train_sl2mf(parameters, pos_samples, neg_samples):
    graph_train_pos_kfold, _, train_pos_kfold, test_pos_kfold = pos_samples
    graph_train_neg_kfold, _, train_neg_kfold, test_neg_kfold = neg_samples

    ppi_sparse_mat = np.load('../data/precessed_data/ppi_topo_sim_matrix.npy')
    go_sim_mat = np.load('../data/precessed_data/final_gosim_bp_from_r_9845.npy')
    go_sim_cc_mat = np.load('../data/precessed_data/final_gosim_cc_from_r_9845.npy')

    kfold=parameters['kfold']
    num_node=parameters['num_node']
    p_n = parameters['pos_neg']
    d_s = parameters['division_strategy']
    n_s = parameters['negative_strategy']

    run = wandb.init(
        # set the wandb project where this run will be logged
        project="Benchmarking",
        group="SL2MF",
        job_type=f"{p_n}_{d_s}_{n_s}",
        # track hyperparameters and run metadata
        config=parameters
    )
    checktosave = ChecktoSave(kfold)
    for fold_num in range(kfold):
        graph_train_pos=graph_train_pos_kfold[fold_num]
        graph_train_neg=graph_train_neg_kfold[fold_num]
        train_pos=train_pos_kfold[fold_num]
        train_neg=train_neg_kfold[fold_num]
        test_pos=test_pos_kfold[fold_num]
        test_neg=test_neg_kfold[fold_num]

        x, y = train_pos[:, 0], train_pos[:, 1]
        IntMat = graph_train_pos.toarray()
        W = np.ones((num_node, num_node))
        W[x, y] = 50
        W[y, x] = W[x, y]

        x_neg, y_neg = train_neg[:, 0], train_neg[:, 1]
        mask = np.zeros((num_node, num_node))
        mask[x, y] = 1
        mask[y, x] = 1
        mask[x_neg, y_neg] = 1
        mask[y_neg, x_neg] = 1

        for nn_size in [45]:
            print(f'nn_size : {nn_size}')
            auc_pair, aupr_pair = [], []
            
            model = LMF(num_factors=50, nn_size=nn_size, theta=2.0 ** (-5), reg=10.0 ** (-2), alpha=1 * 10.0 ** (0),
                        beta=1 * 10.0 ** (0), beta1=1 * 10.0 ** (0), beta2=1 * 10.0 ** (0), max_iter=100)
            # print(str(model))

            t = time.time()
            model.fix(IntMat, W, mask, go_sim_mat, go_sim_cc_mat, ppi_sparse_mat, run=run)
            # model.fix(IntMat, W, mask, go_sim_mat, go_sim_cc_mat, ppi_sparse_mat, co_pathway_mat)

            auc_val, aupr_val = evalution_bal(np.dot(model.U, model.U.T), test_pos, test_neg)

            score_mat = np.dot(model.U, model.U.T)
            
            train_metrics = cal_metrics(score_mat, train_pos_kfold[fold_num], train_neg_kfold[fold_num])
            run.log({
                    'train_auc':train_metrics[0],'train_aupr':train_metrics[2],'train_f1':train_metrics[1],
                    'train_N10':train_metrics[3],'train_N20':train_metrics[4],'train_N50':train_metrics[5],
                    'train_R10':train_metrics[6],'train_R20':train_metrics[7],'train_R50':train_metrics[8],
                    'train_P10':train_metrics[9],'train_P20':train_metrics[10],'train_P50':train_metrics[11],
                    'train_M10':train_metrics[12],'train_M20':train_metrics[13],'train_M50':train_metrics[14],
                })
            checktosave.update_train_classify(fold_num, 0, np.asarray([train_metrics[0], train_metrics[2], train_metrics[1]]))
            checktosave.update_train_ranking(fold_num, 0, train_metrics[3:])
            test_metrics = cal_metrics(score_mat, test_pos_kfold[fold_num], test_neg_kfold[fold_num], train_pos_kfold[fold_num])
            run.log({
                    'test_auc':test_metrics[0],'test_aupr':test_metrics[2],'test_f1':test_metrics[1],
                    'test_N10':test_metrics[3],'test_N20':test_metrics[4],'test_N50':test_metrics[5],
                    'test_R10':test_metrics[6],'test_R20':test_metrics[7],'test_R50':test_metrics[8],
                    'test_P10':test_metrics[9],'test_P20':test_metrics[10],'test_P50':test_metrics[11],
                    'test_M10':test_metrics[12],'test_M20':test_metrics[13],'test_M50':test_metrics[14],
                })
            if checktosave.update_classify(fold_num, 0, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]])):
                if not os.path.exists(f'../results/{n_s}_score_mats/sl2mf'):
                    os.mkdir(f'../results/{n_s}_score_mats/sl2mf')
                path = f'../results/{n_s}_score_mats/sl2mf/sl2mf_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_classify.npy'
                checktosave.save_mat(path, score_mat)
            if checktosave.update_ranking(fold_num, 0, test_metrics[3:]):
                path = f'../results/{n_s}_score_mats/sl2mf/sl2mf_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_ranking.npy'
                checktosave.save_mat(path, score_mat)
                
            print(test_metrics)
            print("metrics over protein pairs: auc %f, aupr %f, time: %f\n" % (auc_val, aupr_val, time.time() - t))

        m1, sdv1 = mean_confidence_interval(auc_pair)
        m2, sdv2 = mean_confidence_interval(aupr_pair)
        print("Average metrics over pairs: auc_mean:%s, auc_sdv:%s, aupr_mean:%s, aupr_sdv:%s\n" % ( m1, sdv1, m2, sdv2))

    run.finish()
    # auc f1 aupr N10 N20 N50 R10 R20 R50 P10 P20 P50
    all_metrics = checktosave.get_all_metrics()
    
    return all_metrics
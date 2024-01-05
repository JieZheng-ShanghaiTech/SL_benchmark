import random
import time
import wandb
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
from sklearn.model_selection import ShuffleSplit
from preprocess import load_kg, cal_metrics, ChecktoSave
from models.kg4sl import *
from utils.kg4sl_utils import eval_all_data, get_feed_dict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

random.seed(456)
np.random.seed(456)
tf.compat.v1.set_random_seed(456)

# torch.manual_seed(123)
# torch.cuda.manual_seed_all(123)


def train_kg4sl(parameters, pos_samples, neg_samples, mode=None, save_mat=False):
    _, _, train_pos_kfold, test_pos_kfold = pos_samples
    _, _, train_neg_kfold, test_neg_kfold = neg_samples

    kfold = parameters['kfold']
    n_epochs = parameters['n_epochs']
    batch_size = parameters['batch_size']
    earlystop_flag = parameters['earlystop_flag']
    dim = parameters['dim']
    n_hop = parameters['n_hop']
    neighbor_sample_size = parameters['neighbor_sample_size']
    l2_weight = parameters['l2_weight']
    lr = parameters['lr']
    num_node = parameters['num_node']
    p_n = parameters['pos_neg']
    d_s = parameters['division_strategy']
    n_s = parameters['negative_strategy']
    if mode == 'final_res':
        kfold=1
        p_n = 'final_res'
        d_s = 'final_res'
        n_s = 'final_res'

    _, _, n_entity, n_relation, adj_entity, adj_relation = load_kg(neighbor_sample_size)

    run=wandb.init(
        # set the wandb project where this run will be logged
        project="Benchmarking",
        group="KG4SL",
        job_type=f"{p_n}_{d_s}_{n_s}",
        # track hyperparameters and run metadata
        config=parameters
    )
    
    checktosave = ChecktoSave(kfold)
    for fold_num in range(kfold):

        print(f'{fold_num + 1}th Fold ...')

        tf.compat.v1.reset_default_graph()

        train_pos_data = train_pos_kfold[fold_num]
        train_neg_data = train_neg_kfold[fold_num]
        train_pos_data = np.hstack([train_pos_data, np.ones((len(train_pos_data), 1))])
        train_neg_data = np.hstack([train_neg_data, np.zeros((len(train_neg_data), 1))])
        print(train_pos_data.shape)
        print(train_neg_data.shape)
        train_data_con = np.vstack([train_pos_data, train_neg_data]).astype('int')
        ind = list(range(len(train_data_con)))
        random.shuffle(ind)
        train_data = train_data_con[ind]

        test_pos_data = test_pos_kfold[fold_num]
        test_neg_data = test_neg_kfold[fold_num]
        test_pos_data = np.hstack([test_pos_data, np.ones((len(test_pos_data), 1))])
        test_neg_data = np.hstack([test_neg_data, np.zeros((len(test_neg_data), 1))])
        test_data = np.vstack([test_pos_data, test_neg_data]).astype('int')

        ind = list(range(len(test_data)))
        random.shuffle(ind)
        test_data = test_data[ind]

        model = KG4SL(n_hop, batch_size, neighbor_sample_size, dim, l2_weight, lr, n_entity, n_relation,
                              adj_entity, adj_relation)
        saver = tf.compat.v1.train.Saver()
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            best_loss_flag = 1000000
            early_stopping_flag = 2

            for step in range(n_epochs):
                time_start = time.time()
                print(f'Epoch {step} ...')
                # training
                loss_list = []
                start = 0
                # skip the last incomplete minibatch if its size < batch size
                while start + batch_size <= train_data.shape[0]:
                    _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + batch_size))
                    start += batch_size
                    loss_list.append(loss)
                    
                loss_mean = np.mean(loss_list)
                run.log({
                    'train_loss':loss_mean
                })
                if (earlystop_flag):
                    if (loss_mean < best_loss_flag):
                        stopping_step = 0
                        best_loss_flag = loss_mean
                    else:
                        stopping_step += 1
                        if (stopping_step >= early_stopping_flag):
                            print(
                                'Early stopping is trigger at step:%.4f' % (step))
                            break

                print(f'Use {time.time() - time_start}s ... Loss {loss_mean} ...')

            print(f'Final Testing ...')
            ti = time.time()
            score_mat = eval_all_data(sess, model, batch_size, num_node)
            print(time.time()-ti)
            if sp.isspmatrix(score_mat):
                score_mat = score_mat.todense()
            score_mat = np.asarray(score_mat)
            score_mat[range(num_node), range(num_node)] = 0
            train_metrics = cal_metrics(score_mat, train_pos_kfold[fold_num], train_neg_kfold[fold_num])
            checktosave.update_train_classify(fold_num, 0, np.asarray([train_metrics[0], train_metrics[2], train_metrics[1]]))
            checktosave.update_train_ranking(fold_num, 0, train_metrics[3:])
            run.log({
                    'train_auc':train_metrics[0],'train_f1':train_metrics[1],'train_aupr':train_metrics[2],
                    'train_N10':train_metrics[3],'train_N20':train_metrics[4],'train_N50':train_metrics[5],
                    'train_R10':train_metrics[6],'train_R20':train_metrics[7],'train_R50':train_metrics[8],
                    'train_P10':train_metrics[9],'train_P20':train_metrics[10],'train_P50':train_metrics[11],
                    'train_M10':train_metrics[12],'train_M20':train_metrics[13],'train_M50':train_metrics[14],
                })
            
            test_metrics = cal_metrics(score_mat, test_pos_kfold[fold_num], test_neg_kfold[fold_num],train_pos_kfold[fold_num])
            run.log({
                    'test_auc':test_metrics[0],'test_f1':test_metrics[1],'test_aupr':test_metrics[2],
                    'test_N10':test_metrics[3],'test_N20':test_metrics[4],'test_N50':test_metrics[5],
                    'test_R10':test_metrics[6],'test_R20':test_metrics[7],'test_R50':test_metrics[8],
                    'test_P10':test_metrics[9],'test_P20':test_metrics[10],'test_P50':test_metrics[11],
                    'test_M10':test_metrics[12],'test_M20':test_metrics[13],'test_M50':test_metrics[14],
                })
            print(f'{test_metrics}')
            
            if save_mat:
                if checktosave.update_classify(fold_num, 0, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]])):
                    # print('Saving score matrix ...')
                    if not os.path.exists(f'../results/{n_s}_score_mats/kg4sl'):
                        os.makedirs(f'../results/{n_s}_score_mats/kg4sl')
                    path = f'../results/{n_s}_score_mats/kg4sl/kg4sl_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_classify.npy'
                    checktosave.save_mat(path, score_mat)
                if checktosave.update_ranking(fold_num, 0, test_metrics[3:]):
                    # print('Saving score matrix ...')
                    path = f'../results/{n_s}_score_mats/kg4sl/kg4sl_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_ranking.npy'
                    checktosave.save_mat(path, score_mat)
            else:
                checktosave.update_classify(fold_num, 0, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]]))
                checktosave.update_ranking(fold_num, 0, test_metrics[3:])

        tf.get_default_graph().finalize()
    run.finish()
        
    # auc f1 aupr N10 N20 N50 R10 R20 R50 P10 P20 P50
    all_metrics = checktosave.get_all_metrics()
    
    return all_metrics

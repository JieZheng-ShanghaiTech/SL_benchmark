import copy
import random
import time
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import os
import wandb
from preprocess import cal_metrics, load_gcatsl_features, ChecktoSave
from models.gcatsl import GAT
from utils.gcatsl_utils import ROC, masked_accuracy, sparse_to_tuple, random_walk_with_restart, extract_global_neighbors

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

random.seed(456)
np.random.seed(456)
tf.compat.v1.set_random_seed(456)

cuda_device='/gpu:0'

def train_gcatsl(parameters, pos_samples, neg_samples, mode=None, save_mat=False):
    graph_train_pos_kfold, graph_test_pos_kfold, train_pos_kfold, test_pos_kfold = pos_samples
    graph_train_neg_kfold, graph_test_neg_kfold, train_neg_kfold, test_neg_kfold = neg_samples

    kfold = parameters['kfold']
    batch_size = parameters['batch_size']
    l2_coef = parameters['l2_coef']
    hid_units = parameters['hid_units']

    nb_epochs = parameters['nb_epochs']
    lr = parameters['lr']
    weight_decay = parameters['weight_decay']
    n_heads = parameters['n_heads']
    dropout = parameters['dropout']
    n_node = parameters['n_node']
    p_n = parameters['pos_neg']
    d_s = parameters['division_strategy']
    n_s = parameters['negative_strategy']
    
    if mode=='final_res':
        kfold=1
        p_n = 'final_res'
        d_s = 'final_res'
        n_s = 'final_res'
        
    build_premat=0

    features_list_ori = load_gcatsl_features()

    nb_nodes = features_list_ori[0].shape[0]
    ft_size = features_list_ori[0].shape[1]

    features_list = [feature[np.newaxis] for feature in features_list_ori]

    wandb.init(
        # set the wandb project where this run will be logged
        project="Benchmarking",
        group="GCATSL",
        job_type=f"{p_n}_{d_s}_{n_s}",
        # track hyperparameters and run metadata
        config=parameters
    )
    checktosave = ChecktoSave(kfold)
    for fold_num in range(kfold):
        print("cross_validation:", '%01d' % (fold_num))

        graph_train_pos = graph_train_pos_kfold[fold_num]
        graph_test_pos = graph_test_pos_kfold[fold_num]
        graph_train_neg = graph_train_neg_kfold[fold_num].toarray().reshape([-1, 1])
        graph_test_neg = graph_test_neg_kfold[fold_num].toarray().reshape([-1, 1])

        residual = False
        nonlinearity = tf.nn.elu
        model = GAT

        y_train = graph_train_pos.toarray().reshape([-1, 1])
        y_test = graph_test_pos.toarray().reshape([-1, 1])

        # train_mask = graph_train_pos
        train_mask = np.array(y_train[:, 0], dtype=np.bool).reshape([-1, 1])
        train_neg_mask = graph_train_neg
        # test_mask = graph_test_pos
        test_mask = np.array(y_test[:, 0], dtype=np.bool).reshape([-1, 1])
        test_neg_mask = graph_test_neg
        # build_premat=1
        interaction_local = graph_train_pos.toarray() + np.eye(graph_train_pos.toarray().shape[0])
        interaction_local = sp.csr_matrix(interaction_local)
        interaction_local_list = [interaction_local, interaction_local, interaction_local]
        if 'All' in n_s:
            n_s = n_s.split('_')[1]
        path_global = os.path.normpath(f'../data/preprocessed_data/gcatsl_data/interaction_global_{fold_num}_{d_s}_{n_s}_1000.txt')
        if not os.path.exists(path_global):
            build_premat=1
            walk_matrix = random_walk_with_restart(graph_train_pos.toarray())
            interaction_global = extract_global_neighbors(graph_train_pos.toarray(), walk_matrix)
            np.savetxt(path_global, interaction_global)
            print(f'build global interaction matrix {fold_num}_{d_s}_{n_s}')
        if build_premat==1:
            continue
        
        interaction_global = np.loadtxt(path_global)

        interaction_global = interaction_global + np.eye(interaction_global.shape[0])
        interaction_global = sp.csr_matrix(interaction_global)
        interaction_global_list = [interaction_global, interaction_global, interaction_global]

        biases_local_list = [sparse_to_tuple(interaction) for interaction in interaction_local_list]
        biases_global_list = [sparse_to_tuple(interaction) for interaction in interaction_global_list]
        n = n_node
        entry_size = n * n
        with tf.Graph().as_default():
            with tf.name_scope('input'):
                feature_in_list = [tf.placeholder(dtype=tf.float32,
                                                  shape=(batch_size, nb_nodes, ft_size),
                                                  name='ftr_in_{}'.format(i))
                                   for i in range(len(features_list))]
                bias_in_local_list = [tf.compat.v1.sparse_placeholder(tf.float32, name='ftr_in_{}'.format(i)) for i in
                                      range(len(biases_local_list))]
                bias_in_global_list = [tf.compat.v1.sparse_placeholder(tf.float32, name='ftr_in_{}'.format(i)) for i in
                                       range(len(biases_global_list))]
                lbl_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(entry_size, batch_size))
                msk_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(entry_size, batch_size))
                neg_msk = tf.compat.v1.placeholder(dtype=tf.int32, shape=(entry_size, batch_size))
                attn_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=())
                ffd_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=())
                is_train = tf.compat.v1.placeholder(dtype=tf.bool, shape=())

            final_embedding = model.encoder(feature_in_list, nb_nodes, is_train, attn_drop, ffd_drop,
                                            bias_mat_local_list=bias_in_local_list,
                                            bias_mat_global_list=bias_in_global_list, hid_units=hid_units,
                                            n_heads=n_heads)

            pro_matrix = model.decoder_revised(final_embedding)

            loss = model.loss_overall(pro_matrix, lbl_in, msk_in, neg_msk, weight_decay, final_embedding)
            accuracy = masked_accuracy(pro_matrix, lbl_in, msk_in, neg_msk)

            train_op = model.training(loss, lr, l2_coef)

            init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
            print('Start train ...')
            # start to train
            with tf.compat.v1.Session() as sess:
                sess.run(init_op)

                for epoch in range(nb_epochs):
                    t = time.time()

                    ##########    train     ##############
                    tr_step = 0
                    train_loss_avg = 0
                    tr_size = features_list[0].shape[0]
                    while tr_step * batch_size < tr_size:
                        fd1 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                               for i, d in zip(feature_in_list, features_list)}
                        fd2 = {bias_in_local_list[i]: biases_local_list[i] for i in range(len(biases_local_list))}
                        fd3 = {bias_in_global_list[i]: biases_global_list[i] for i in range(len(biases_global_list))}
                        fd4 = {lbl_in: y_train,
                               msk_in: train_mask,
                               neg_msk: train_neg_mask,
                               is_train: True,
                               attn_drop: dropout,
                               ffd_drop: dropout}
                        fd = fd1
                        fd.update(fd2)
                        fd.update(fd3)
                        fd.update(fd4)
                        with tf.device(cuda_device):
                            score_matrix, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy], feed_dict=fd)
                        
                        train_loss_avg += loss_value_tr
                        # train_acc_avg += acc_tr
                        tr_step += 1
                    wandb.log({
                        'train_loss':loss_value_tr
                    })
                    
                    print('Epoch: %04d | Training: loss = %.5f, time = %.5f' % ((epoch + 1), train_loss_avg / tr_step, time.time() - t))

                    ###########     test      ############
                    ts_size = features_list[0].shape[0]
                    ts_step = 0
                    ts_loss = 0.0
                    ts_acc = 0.0
                    print("Start to test")
                    t = time.time()
                    while ts_step * batch_size < ts_size:
                        fd1 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                            for i, d in zip(feature_in_list, features_list)}
                        fd2 = {bias_in_local_list[i]: biases_local_list[i] for i in range(len(biases_local_list))}
                        fd3 = {bias_in_global_list[i]: biases_global_list[i] for i in range(len(biases_global_list))}
                        fd4 = {lbl_in: y_test,
                            msk_in: test_mask,
                            neg_msk: test_neg_mask,
                            is_train: False,
                            attn_drop: 0.0,
                            ffd_drop: 0.0}
                        fd = fd1
                        fd.update(fd2)
                        fd.update(fd3)
                        fd.update(fd4)
                        with tf.device(cuda_device):
                            score_matrix, loss_value_ts, acc_ts = sess.run([pro_matrix, loss, accuracy], feed_dict=fd)
                        wandb.log({
                            'test_loss':loss_value_ts
                        })
                        ts_loss += loss_value_ts
                        ts_acc += acc_ts
                        ts_step += 1
                    # print('Test loss:', ts_loss / ts_step)
                    print(f'Test loss: {ts_loss / ts_step}, Test time: {time.time()-t}')

                    score_matrix = score_matrix.reshape((n, n))

                    train_metrics = cal_metrics(score_matrix, train_pos_kfold[fold_num], train_neg_kfold[fold_num])
                    checktosave.update_train_classify(fold_num, epoch, np.asarray([train_metrics[0], train_metrics[2], train_metrics[1]]))
                    checktosave.update_train_ranking(fold_num, epoch, train_metrics[3:])
                    wandb.log({
                        'train_auc':train_metrics[0],'train_f1':train_metrics[1],'train_aupr':train_metrics[2],
                        'train_N10':train_metrics[3],'train_N20':train_metrics[4],'train_N50':train_metrics[5],
                        'train_R10':train_metrics[6],'train_R20':train_metrics[7],'train_R50':train_metrics[8],
                        'train_P10':train_metrics[9],'train_P20':train_metrics[10],'train_P50':train_metrics[11],
                    })

                    test_metrics = cal_metrics(score_matrix, test_pos_kfold[fold_num], test_neg_kfold[fold_num], train_pos_kfold[fold_num])
                    wandb.log({
                        'test_auc':test_metrics[0],'test_f1':test_metrics[1],'test_aupr':test_metrics[2],
                        'test_N10':test_metrics[3],'test_N20':test_metrics[4],'test_N50':test_metrics[5],
                        'test_R10':test_metrics[6],'test_R20':test_metrics[7],'test_R50':test_metrics[8],
                        'test_P10':test_metrics[9],'test_P20':test_metrics[10],'test_P50':test_metrics[11],
                    })
                    print(test_metrics)
                    if save_mat:
                        if checktosave.update_classify(fold_num, epoch, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]])):
                            # print('Saving score matrix ...')
                            if not os.path.exists(f'../results/{n_s}_score_mats/gcatsl'):
                                os.makedirs(f'../results/{n_s}_score_mats/gcatsl')
                            path = f'../results/{n_s}_score_mats/gcatsl/gcatsl_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_classify.npz'
                            checktosave.save_mat(path, score_matrix)
                        if checktosave.update_ranking(fold_num, epoch, test_metrics[3:]):
                            # print('Saving score matrix ...')
                            path = f'../results/{n_s}_score_mats/gcatsl/gcatsl_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_ranking.npz'
                            checktosave.save_mat(path, score_matrix)
                    else:
                        checktosave.update_classify(fold_num, epoch, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]]))
                        checktosave.update_ranking(fold_num, epoch, test_metrics[3:])

    wandb.finish()
    # auc f1 aupr N10 N20 N50 R10 R20 R50 P10 P20 P50
    all_metrics = checktosave.get_all_metrics()
    
    return all_metrics

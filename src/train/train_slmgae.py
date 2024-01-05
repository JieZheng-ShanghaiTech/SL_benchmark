from __future__ import division
from __future__ import print_function
import time
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn.model_selection import KFold
from models.slmgae import SLMGAE, Optimizer
from utils.slmgae_utils import load_data, sparse_to_tuple, preprocess_graph, construct_feed_dict
from preprocess import cal_metrics, ChecktoSave
import os
import wandb
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

np.random.seed(456)
tf.compat.v1.set_random_seed(456)

cuda_device='/gpu:0'

def train_slmgae(parameters, pos_samples, neg_samples, mode=None, save_mat=False):
    _, _, train_pos_kfold, test_pos_kfold = pos_samples
    _, _, train_neg_kfold, test_neg_kfold = neg_samples

    kfold = parameters['kfold']
    p_n = parameters['pos_neg']
    d_s = parameters['division_strategy']
    num_node = parameters['num_node']

    hidden_dim1 = parameters['hidden1']
    hidden_dim2 = parameters['hidden2']
    nn_size = parameters['nn_size']

    Alpha = parameters['Alpha']
    Coe = parameters['Coe']
    Beta = parameters['Beta']
    learning_rate = parameters['learning_rate']
    epochs = parameters['epochs']
    dropout = parameters['dropout']
    n_s = parameters['negative_strategy']
    
    if mode=='final_res':
        kfold=1
        n_s = 'final_res'
        d_s = 'final_res'
        p_n = 'final_res'
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="Benchmarking",
        group="SLMGAE",
        job_type=f"{p_n}_{d_s}_{n_s}",
        # track hyperparameters and run metadata
        config=parameters
    )

    # tensorflow config
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = False

    adjs_orig = load_data(knn=True, nnSize=nn_size)
    pos_edge = np.vstack([train_pos_kfold[0], test_pos_kfold[0]])
    neg_edge = np.vstack([train_neg_kfold[0], test_neg_kfold[0]])

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = sp.csr_matrix((np.ones(len(pos_edge)), (pos_edge[:, 0], pos_edge[:, 1])), shape=(9845, 9845))
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    
    checktosave = ChecktoSave(kfold)
    for fold_num in range(1, kfold + 1):
        tf.compat.v1.reset_default_graph()
        print("Training in the %02d fold..." % fold_num)
        train_pos, test_pos = train_pos_kfold[fold_num - 1], test_pos_kfold[fold_num - 1]
        train_neg, test_neg = train_neg_kfold[fold_num - 1], test_neg_kfold[fold_num - 1]

        row = train_pos[:, 0]
        col = train_pos[:, 1]
        val = np.ones(train_pos.shape[0])
        adj = sp.csr_matrix((val, (row, col)), shape=(9845, 9845))
        adj = adj + adj.T

        adjs = adjs_orig[0: 3]
        adjs.append(adj)

        # load features
        features = sparse_to_tuple(adj)
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]

        num_nodes = adj.shape[0]
        num_edges = adj.sum()

        train_index = np.vstack((train_pos, train_neg))

        # Some preprocessing
        supports = []
        for a in adjs:
            supports.append(preprocess_graph(a))
        num_supports = len(supports)
        placeholders = {
                'support': [tf.compat.v1.sparse_placeholder(tf.float32, name='adj_{}'.format(_)) for _ in range(num_supports)],
                'features': tf.compat.v1.sparse_placeholder(tf.float32, name='features'),
                'adj_orig': tf.compat.v1.sparse_placeholder(tf.float32, name='adj_orig'),
                'dropout': tf.compat.v1.placeholder_with_default(0., shape=(), name='dropout'),
            }
        # Create model
        model = SLMGAE(placeholders, num_features, features_nonzero, num_nodes, num_supports - 1,
                        name='SLMGAE_{}'.format(fold_num), hid1=hidden_dim1, hid2=hidden_dim2, Coe=Coe)
        # Create optimizer
        with tf.name_scope('optimizer'):
            opt = Optimizer(
                    supp=model.support_recs,
                    main=model.main_rec,
                    preds=model.reconstructions,
                    labels=tf.sparse.to_dense(placeholders['adj_orig'], validate_indices=False),
                    Alpha=Alpha,
                    Beta=Beta,
                    learning_rate=learning_rate,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                    index=train_index
                )
        # Initialize session
        sess = tf.compat.v1.Session(config=config)
        sess.run(tf.compat.v1.global_variables_initializer())
        adj_label = sparse_to_tuple(adj)

        # Construct feed dictionary
        feed_dict = construct_feed_dict(supports, features, adj_label, placeholders)

        # Train model
        for epoch in range(epochs):
            t = time.time()

            feed_dict = construct_feed_dict(supports, features, adj_label, placeholders)
            feed_dict.update({placeholders['dropout']: dropout})
            with tf.device(cuda_device):
                # One update of parameter matrices
                _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
            wandb.log({
                'train_loss':avg_cost
            })

            print("Epoch: " + '%04d' % (epoch + 1) +
                  " train_loss=" + "{:.5f}".format(avg_cost) +
                  " time= " + "{:.5f}".format(time.time() - t))

            print('Optimization Finished!')
            feed_dict.update({placeholders['dropout']: 0})
            with tf.device(cuda_device):
                adj_rec = sess.run(model.predict(), feed_dict=feed_dict)

            if sp.isspmatrix(adj_rec):
                adj_rec = adj_rec.todense()
            adj_rec = np.asarray(adj_rec)
            adj_rec[range(num_node), range(num_node)] = 0
            train_metrics = cal_metrics(adj_rec, train_pos_kfold[fold_num - 1], train_neg_kfold[fold_num - 1])
            checktosave.update_train_classify(fold_num-1, epoch, np.asarray([train_metrics[0], train_metrics[2], train_metrics[1]]))
            checktosave.update_train_ranking(fold_num-1, epoch, train_metrics[3:])
            wandb.log({
                'train_auc':train_metrics[0],'train_f1':train_metrics[1],'train_aupr':train_metrics[2],
                'train_N10':train_metrics[3],'train_N20':train_metrics[4],'train_N50':train_metrics[5],
                'train_R10':train_metrics[6],'train_R20':train_metrics[7],'train_R50':train_metrics[8],
                'train_P10':train_metrics[9],'train_P20':train_metrics[10],'train_P50':train_metrics[11],
                'train_M10':train_metrics[12],'train_M20':train_metrics[13],'train_M50':train_metrics[14],
            })

            test_metrics = cal_metrics(adj_rec, test_pos_kfold[fold_num - 1], test_neg_kfold[fold_num - 1],train_pos_kfold[fold_num - 1])
            wandb.log({
                'test_auc':test_metrics[0],'test_f1':test_metrics[1],'test_aupr':test_metrics[2],
                'test_N10':test_metrics[3],'test_N20':test_metrics[4],'test_N50':test_metrics[5],
                'test_R10':test_metrics[6],'test_R20':test_metrics[7],'test_R50':test_metrics[8],
                'test_P10':test_metrics[9],'test_P20':test_metrics[10],'test_P50':test_metrics[11],
                'test_M10':test_metrics[12],'test_M20':test_metrics[13],'test_M50':test_metrics[14],
            })
            # print(test_metrics)
            
            update_class, update_rank = False, False
            checktosave.update_classify(fold_num-1, epoch, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]]))
                print('Saving score matrix ...')
                # if not os.path.exists(f'../results/{n_s}_score_mats/slmgae'):
                    # os.makedirs(f'../results/{n_s}_score_mats/slmgae')
                # path = f'../results/{n_s}_score_mats/slmgae/slmgae_fold_{fold_num-1}_pos_neg_{p_n}_{d_s}_{n_s}_classify.npy'
                checktosave.save_mat(path, adj_rec)
                update_class = True
            checktosave.update_ranking(fold_num-1, epoch, test_metrics[3:])
                print('Saving score matrix ...')
                # path = f'../results/{n_s}_score_mats/slmgae/slmgae_fold_{fold_num-1}_pos_neg_{p_n}_{d_s}_{n_s}_ranking.npy'
                checktosave.save_mat(path, adj_rec)
                update_rank = True
                
            if update_class or update_rank:
                print(test_metrics)
                print('Refresh the stop count ...')
                stop_count = 0
            else:
                stop_count += 1
                print(f'Stop count: {stop_count}')
            if stop_count >= 20 and epoch>=100:
                print('Early stop!')
                break
                
            
            
        sess.close()
    
    wandb.finish()
    # auc f1 aupr N10 N20 N50 R10 R20 R50 P10 P20 P50
    all_metrics = checktosave.get_all_metrics()
    
    return all_metrics

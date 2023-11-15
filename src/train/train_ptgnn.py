import copy
import os
import time
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import wandb
from models.ptgnn import Model
from utils.ptgnn_utils import preprocess_graph, train_negative_sample, load_data
from preprocess import cal_metrics, ChecktoSave

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
pretrain_epochs=200

np.random.seed(456)
tf.compat.v1.set_random_seed(456)

cuda_device='/gpu:0'

def train_ptgnn(parameters,pos_samples, neg_samples):
    graph_train_pos_kfold, graph_test_pos_kfold, train_pos_kfold, test_pos_kfold = pos_samples
    graph_train_neg_kfold, graph_test_neg_kfold, train_neg_kfold, test_neg_kfold = neg_samples

    kfold=parameters['kfold']
    epochs=parameters['epochs']
    num_node=parameters['num_node']
    p_n=parameters['pos_neg']
    d_s=parameters['division_strategy']
    n_s = parameters['negative_strategy']
    
    # tensorflow config
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = False
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="Benchmarking",
        group="PTGNN",
        job_type=f"{p_n}_{d_s}_{n_s}",
        # track hyperparameters and run metadata
        config=parameters
    )
    checktosave = ChecktoSave(kfold)
    for fold_num in range(kfold):
        print(f'Fold {fold_num+1} ...')
        tf.compat.v1.reset_default_graph()

        interaction = graph_train_pos_kfold[fold_num] + sp.eye(num_node, num_node)
        logits_train = graph_train_pos_kfold[fold_num].toarray().reshape([-1, 1])
        logits_test = graph_test_pos_kfold[fold_num].toarray().reshape([-1, 1])
        train_mask = np.array(logits_train[:, 0], dtype=np.bool).reshape([-1, 1])
        test_mask = np.array(logits_test[:, 0], dtype=np.bool).reshape([-1, 1])
        word_matrix = np.load("../data/precessed_data/ptgnn_encod_by_word_sl_9845_800.npy")
        word_matrix = word_matrix[list(range(word_matrix.shape[0])), :600]
        word_matrix = word_matrix.astype(np.int32)

        biases = preprocess_graph(interaction)
        # save_path = "/home/yimiaofeng/PycharmProjects/SLBench/Bench/check_points/ptgnn/model.ckpt"
        model = Model(do_train=False)
        # saver = tf.train.Saver()
        init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
        
        sess = tf.compat.v1.Session(config=config)
        sess.run(init_op)
        
        # with tf.compat.v1.Session() as sess:
            # sess.run(init_op)

        neg_mask = graph_train_neg_kfold[fold_num].toarray().reshape([-1, 1])
        for epoch in range(epochs):
            t = time.time()
            fd={
                model.encoded_protein: word_matrix,
                model.bias_in: biases,
                model.lbl_in: logits_train,
                model.msk_in: train_mask,
                model.neg_msk: neg_mask,
            }
            with tf.device(cuda_device):
                _, loss_value_tr, acc_tr, emb = sess.run([model.train_op, model.loss, model.accuracy, model.embedding_tokens], feed_dict=fd)
            print('Epoch: %04d | Training: loss = %.5f, time = %.5f' % ((epoch + 1), loss_value_tr, time.time() - t))
            wandb.log({
                'train_loss':loss_value_tr
            })
            score_mat=acc_tr.reshape((num_node,num_node))
            train_metrics = cal_metrics(score_mat, train_pos_kfold[fold_num], train_neg_kfold[fold_num])
            wandb.log({
                    'train_auc':train_metrics[0],'train_aupr':train_metrics[2],'train_f1':train_metrics[1],
                    'train_N10':train_metrics[3],'train_N20':train_metrics[4],'train_N50':train_metrics[5],
                    'train_R10':train_metrics[6],'train_R20':train_metrics[7],'train_R50':train_metrics[8],
                    'train_P10':train_metrics[9],'train_P20':train_metrics[10],'train_P50':train_metrics[11],
                    'train_M10':train_metrics[12],'train_M20':train_metrics[13],'train_M50':train_metrics[14],
                })
            checktosave.update_train_classify(fold_num, epoch, np.asarray([train_metrics[0], train_metrics[2], train_metrics[1]]))
            checktosave.update_train_ranking(fold_num, epoch, train_metrics[3:])
            
            test_metrics = cal_metrics(score_mat, test_pos_kfold[fold_num], test_neg_kfold[fold_num],train_pos_kfold[fold_num])
            wandb.log({
                    'test_auc':test_metrics[0],'test_aupr':test_metrics[2],'test_f1':test_metrics[1],
                    'test_N10':test_metrics[3],'test_N20':test_metrics[4],'test_N50':test_metrics[5],
                    'test_R10':test_metrics[6],'test_R20':test_metrics[7],'test_R50':test_metrics[8],
                    'test_P10':test_metrics[9],'test_P20':test_metrics[10],'test_P50':test_metrics[11],
                    'test_M10':test_metrics[12],'test_M20':test_metrics[13],'test_M50':test_metrics[14],
                })
            if checktosave.update_classify(fold_num, epoch, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]])):
                print('Saving score matrix ...')
                if not os.path.exists(f'../results/{n_s}_score_mats/ptgnn'):
                    os.mkdir(f'../results/{n_s}_score_mats/ptgnn')
                path = f'../results/{n_s}_score_mats/ptgnn/ptgnn_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_classify.npy'
                checktosave.save_mat(path, score_mat)
            if checktosave.update_ranking(fold_num, epoch, test_metrics[3:]):
                print('Saving score matrix ...')
                path = f'../results/{n_s}_score_mats/ptgnn/ptgnn_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_ranking.npy'
                checktosave.save_mat(path, score_mat)
                
            print(test_metrics)
        sess.close()
    wandb.finish()
    # auc f1 aupr N10 N20 N50 R10 R20 R50 P10 P20 P50
    all_metrics = checktosave.get_all_metrics()
    
    return all_metrics



def train_ptgnn_pre():
    interaction1, interaction2, logits_train1, logits_train2, train_mask1, train_mask2, labels1, labels2, word_matrix = load_data()

    biases1 = preprocess_graph(interaction1)
    biases2 = preprocess_graph(interaction2)
    save_path = "../data/precessed_data/ptgnn_pretrain/model.ckpt"
    model = Model()
    saver = tf.train.Saver()
    init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        neg_mask1, neg_mask2 = train_negative_sample(logits_train1, logits_train2, len(labels1), len(labels2))
        for epoch in range(pretrain_epochs):
            t = time.time()
            fd={
                model.encoded_protein: word_matrix,
                model.bias_in1: biases1,
                model.bias_in2: biases2,
                model.lbl_in1: logits_train1,
                model.lbl_in2: logits_train2,
                model.msk_in1: train_mask1,
                model.msk_in2: train_mask2,
                model.neg_msk1: neg_mask1,
                model.neg_msk2: neg_mask2}
            _, loss_value_tr, acc_tr, emb = sess.run([model.train_op, model.loss, model.accuracy, model.embedding_tokens], feed_dict=fd)
            print('Epoch: %04d | Training: loss = %.5f, acc = %.5f, time = %.5f' % ((epoch + 1), loss_value_tr, acc_tr, time.time() - t))
        saver.save(sess, save_path)
        np.save('../data/precessed_data/ptgnn_pretrain/trained_word_embedding.npy', emb)
        sess.close()



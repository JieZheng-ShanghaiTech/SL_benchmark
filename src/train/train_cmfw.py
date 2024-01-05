import copy
import tensorflow as tf
import numpy as np
import os
import math
import time
import scipy.sparse as sp
import wandb
from preprocess import cal_metrics, ChecktoSave

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.random.seed(456)
tf.compat.v1.set_random_seed(456)

which_gpu=0

# define Frobenius norm square
def frob(z, k1, k2):
    vec_i = tf.reshape(z, [-1])
    return tf.reduce_sum(tf.multiply(vec_i, vec_i))

def frob_orig(z):  #by long
    vec_i = tf.reshape(z, [-1])
    return tf.reduce_sum(tf.multiply(vec_i, vec_i))

def train_cmfw(parameters, pos_samples, neg_samples, mode=None, save_mat=False):
    graph_train_pos_kfold, _, train_pos_kfold, test_pos_kfold = pos_samples
    graph_train_neg_kfold, _, train_neg_kfold, test_neg_kfold = neg_samples

    max_steps = 200
    tol = 1e-7
    kfold = parameters['kfold']
    p_n = parameters['pos_neg']
    d_s = parameters['division_strategy']
    n_s = parameters['negative_strategy']
    train_size = 9845
    du1 = 50
    du2 = 10
    lambda1 = 1e-4
    
    if mode=='final_res':
        kfold=1
        p_n = d_s = n_s = 'final_res'

    ppi_spm = sp.load_npz('../data/preprocessed_data/ppi_sparse_upper_matrix_without_sl_relation_9845.npz')
    ppi_spm = ppi_spm + ppi_spm.T
    xs_data = ppi_spm.toarray()
    xi_data = np.load('../data/preprocessed_data/final_gosim_bp_from_r_9845.npy')
    xg_data = np.load('../data/preprocessed_data/final_gosim_cc_from_r_9845.npy')

    run = wandb.init(
        # set the wandb project where this run will be logged
        project="Benchmarking",
        group="CMFW",
        job_type=f"{p_n}_{d_s}_{n_s}",
        # track hyperparameters and run metadata
        config=parameters
    )
    checktosave = ChecktoSave(kfold)
    for fold_num in range(kfold):
        print(f'{fold_num+1} fold ...')
        tf.compat.v1.reset_default_graph()
        xa_data = (graph_train_pos_kfold[fold_num]+graph_train_neg_kfold[fold_num]).toarray()
        y_train_pos = train_pos_kfold[fold_num]
        y_train_pos = np.hstack([y_train_pos, np.ones((y_train_pos.shape[0], 1))])
        y_train_neg = train_neg_kfold[fold_num]
        y_train_neg = np.hstack([y_train_neg, np.zeros((y_train_neg.shape[0], 1))])
        y_train = np.vstack([y_train_pos, y_train_neg])
        print(y_train.shape)

        y_test_pos = test_pos_kfold[fold_num]
        y_test_pos = np.hstack([y_test_pos, np.ones((y_test_pos.shape[0], 1))])
        y_test_neg = test_neg_kfold[fold_num]
        y_test_neg = np.hstack([y_test_neg, np.zeros((y_test_neg.shape[0], 1))])
        y_test = np.vstack([y_test_pos, y_test_neg])
        print(y_test.shape)


        kdti, da = xa_data.shape
        ndti, di = xi_data.shape
        n, dg = xg_data.shape
        _, ds = xs_data.shape
        tf.set_random_seed(456)
        best_result_ith=np.zeros((1,12))

        sess = tf.compat.v1.InteractiveSession()
        with tf.device(f'/gpu:{which_gpu}'):
            sess.run(tf.compat.v1.global_variables_initializer())

        # Input placeholders
        with tf.name_scope("input"):
            xa = tf.placeholder(tf.float32, shape=(None, da), name='xa-input')
            xi = tf.placeholder(tf.float32, shape=(None, di), name='xi-input')
            xg = tf.placeholder(tf.float32, shape=(None, dg), name='xg-input')
            xs = tf.placeholder(tf.float32, shape=(None, ds), name='xs-input')
            # xt = tf.placeholder(tf.float32, shape=(None, dg), name='xt-input')
            # xu = tf.placeholder(tf.float32, shape=(None, ds), name='xu-input')
            ytest = tf.placeholder(tf.float32, shape=(None, 3), name='ytest-input')
            keep_prob = tf.placeholder(tf.float32)

        print('svd ...')
        # initialize all factors by svd, you can choose other way to initialize all the variables
        with tf.device(f'/gpu:{which_gpu}'):
            with tf.name_scope('svd'):
                udti_svd1, _, _ = np.linalg.svd(xi_data, full_matrices=False)
                udti_svd2, _, _ = np.linalg.svd(udti_svd1, full_matrices=False)

                u1dti = udti_svd2[:, 0:du2]

        print('svd finish ...')

        [a0, b0] = udti_svd1.shape
        name = str(du1) + str(du2) + str(lambda1) + str(fold_num+1)

        w0 = tf.get_variable("w0_" + name, shape=(a0, b0), initializer=tf.contrib.layers.xavier_initializer())
        w1 = tf.get_variable("w1_" + name, shape=(a0, b0), initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable("w2_" + name, shape=(a0, b0), initializer=tf.contrib.layers.xavier_initializer())
        w3 = tf.get_variable("w3_" + name, shape=(a0, b0), initializer=tf.contrib.layers.xavier_initializer())
        # w4 = tf.get_variable("w4_" + name, shape=(a0, b0), initializer=tf.contrib.layers.xavier_initializer())
        # w5 = tf.get_variable("w5_" + name, shape=(a0, b0), initializer=tf.contrib.layers.xavier_initializer())

        u1i = tf.Variable(tf.cast(u1dti, tf.float32))

        # bias = tf.Variable(tf.constant(0.1, shape=[1, 1]))

        with tf.name_scope('output'):
            y_conf = tf.matmul(tf.matmul(u1i, tf.transpose(u1i)), w0)

        row_train = y_train[:, 0]
        col_train = y_train[:, 1]

        tf_row_train = tf.cast(row_train, tf.int32)
        tf_col_train = tf.cast(col_train, tf.int32)

        loss = frob(xa - tf.matmul((tf.matmul(u1i, tf.transpose(u1i))), w0), tf_row_train, tf_col_train) + \
               frob(xi - tf.matmul((tf.matmul(u1i, tf.transpose(u1i))), w1), tf_row_train, tf_col_train) + \
               frob(xg - tf.matmul((tf.matmul(u1i, tf.transpose(u1i))), w2), tf_row_train, tf_col_train) + \
               frob(xs - tf.matmul((tf.matmul(u1i, tf.transpose(u1i))), w3), tf_row_train, tf_col_train) + \
               lambda1 * frob_orig(u1i)
            #    frob(xt - tf.matmul((tf.matmul(u1i, tf.transpose(u1i))), w4), tf_row_train, tf_col_train) + \
            #    frob(xu - tf.matmul((tf.matmul(u1i, tf.transpose(u1i))), w5), tf_row_train, tf_col_train) + \
               
            # lambda1 * frob_orig(u1i)
        train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(loss)
        tf.compat.v1.global_variables_initializer().run()
        def feed_dict(training, prob_rate):
            xas = xa_data
            xis = xi_data
            xgs = xg_data
            xss = xs_data
            # xst = xt_data
            # xsu = xu_data
            ys = y_test
            return {xa: xas, xi: xis, xg: xgs, xs: xss, ytest: ys, keep_prob: prob_rate}
            # return {xa: xas, xi: xis, xg: xgs, xs: xss, xt: xst, xu: xsu, ytest: ys, keep_prob: prob_rate}

        print('Training ...')
        funval = []
        with tf.device(f'/gpu:{which_gpu}'):
            _, loss_iter = sess.run([train_step, loss], feed_dict=feed_dict(True, 0.8))
        funval.append(loss_iter)

        for i in range(max_steps):
            print(f'step {i}')
            with tf.device(f'/gpu:{which_gpu}'):
                _, loss_iter = sess.run([train_step, loss], feed_dict=feed_dict(True, 0.8))
            funval.append(loss_iter)
            run.log({
                'train_loss':loss_iter
            })

            if abs(funval[i + 1] - funval[i]) < tol:
                break
            if math.isnan(loss_iter):
                break

            print(f'loss {loss_iter}')

        print('Testing ...')
        with tf.device(f'/gpu:{which_gpu}'):
            pred_conf = sess.run([y_conf], feed_dict=feed_dict(False, 1.0))

        score_mat = np.reshape(np.array(pred_conf), (train_size, train_size))
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
        
        if save_mat:
            if checktosave.update_classify(fold_num, 0, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]])):
                if not os.path.exists(f'../results/{n_s}_score_mats/cmfw'):
                    os.makedirs(f'../results/{n_s}_score_mats/cmfw')
                path = f'../results/{n_s}_score_mats/cmfw/cmfw_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_classify.npy'
                checktosave.save_mat(path, score_mat)
            if checktosave.update_ranking(fold_num, 0, test_metrics[3:]):
                path = f'../results/{n_s}_score_mats/cmfw/cmfw_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_ranking.npy'
                checktosave.save_mat(path, score_mat)
        else:
            checktosave.update_classify(fold_num, 0, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]]))
            checktosave.update_ranking(fold_num, 0, test_metrics[3:])
        print(test_metrics)
        
        sess.close()

    run.finish()
    # auc f1 aupr N10 N20 N50 R10 R20 R50 P10 P20 P50
    all_metrics = checktosave.get_all_metrics()
    
    return all_metrics

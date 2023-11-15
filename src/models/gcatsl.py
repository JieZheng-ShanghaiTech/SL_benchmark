import tensorflow as tf
import numpy as np

# random.seed(123)
np.random.seed(456)
tf.compat.v1.set_random_seed(456)

def masked_accuracy(preds, labels, mask, negative_mask):
    """Accuracy with masking."""
    preds = tf.cast(preds, tf.float32)
    labels = tf.cast(labels, tf.float32)
    error = tf.square(preds-labels)
    mask += negative_mask
    mask = tf.cast(mask, dtype=tf.float32)
    error *= mask
    return tf.sqrt(tf.reduce_mean(error))

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


conv1d = tf.layers.conv1d


def sp_attn_head(seq, out_sz, adj_mat_local, adj_mat_global, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        seq_fts = seq

        latent_factor_size = 8
        nb_nodes = seq_fts.shape[1].value

        w_1 = glorot([seq_fts.shape[2].value, latent_factor_size])
        w_2 = glorot([3 * seq_fts.shape[2].value, latent_factor_size])

        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)

        # local neighbours
        logits = tf.add(f_1[0], tf.transpose(f_2[0]))
        logits_first = adj_mat_local * logits
        lrelu = tf.SparseTensor(indices=logits_first.indices,
                                values=tf.nn.leaky_relu(logits_first.values),
                                dense_shape=logits_first.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        neigh_embs = tf.sparse.sparse_dense_matmul(coefs, seq_fts)

        # non-local neighbours
        logits_global = adj_mat_global * logits
        lrelu_global = tf.SparseTensor(indices=logits_global.indices,
                                       values=tf.nn.leaky_relu(logits_global.values),
                                       dense_shape=logits_global.dense_shape)
        coefs_global = tf.sparse_softmax(lrelu_global)

        coefs_global = tf.sparse_reshape(coefs_global, [nb_nodes, nb_nodes])
        neigh_embs_global = tf.sparse.sparse_dense_matmul(coefs_global, seq_fts)

        neigh_embs_sum_1 = tf.matmul(tf.add(tf.add(seq_fts, neigh_embs), neigh_embs_global), w_1)
        neigh_embs_sum_2 = tf.matmul(tf.concat([tf.concat([seq_fts, neigh_embs], axis=-1), neigh_embs_global], axis=-1),
                                     w_2)

        final_embs = activation(neigh_embs_sum_1) + activation(neigh_embs_sum_2)

        return final_embs


def SimpleAttLayer(inputs, attention_size, time_major=False, return_alphas=False):
    hidden_size = inputs.shape[2].value

    # Trainable parameters
    w_omega = tf.Variable(tf.random.normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random.normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random.normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')
    alphas = tf.nn.softmax(vu, name='alphas')
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas

class BaseGAttN:
    def loss(logits, labels, nb_classes, class_weights):
        sample_wts = tf.reduce_sum(tf.multiply(tf.one_hot(labels, nb_classes), class_weights), axis=-1)
        xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits), sample_wts)
        return tf.reduce_mean(xentropy, name='xentropy_mean')

    def training(loss, lr, l2_coef):
        # weight decay
        vars = tf.compat.v1.trainable_variables()
        # print("LongYahui")
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef
        # optimizer
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

        # training op
        # train_op = opt.minimize(loss+lossL2)   #返回更新后的所有变量列表
        train_op = opt.minimize(loss)
        return train_op

    def preshape(logits, labels, nb_classes):
        new_sh_lab = [-1]
        new_sh_log = [-1, nb_classes]
        log_resh = tf.reshape(logits, new_sh_log)
        lab_resh = tf.reshape(labels, new_sh_lab)
        return log_resh, lab_resh

    def confmat(logits, labels):
        preds = tf.argmax(logits, axis=1)
        return tf.confusion_matrix(labels, preds)

    ##########################
    # Adapted from tkipf/gcn #
    ##########################

    def masked_softmax_cross_entropy(logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        print("logits:", logits)
        print("labels:", labels)
        print("mask:", mask)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_sigmoid_cross_entropy(logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        labels = tf.cast(labels, dtype=tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss, axis=1)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_accuracy(logits, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    def micro_f1(logits, labels, mask):
        """Accuracy with masking."""
        predicted = tf.round(tf.nn.sigmoid(logits))

        # Use integers to avoid any nasty FP behaviour
        predicted = tf.cast(predicted, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)
        mask = tf.cast(mask, dtype=tf.int32)

        # expand the mask so that broadcasting works ([nb_nodes, 1])
        mask = tf.expand_dims(mask, -1)

        # Count true positives, true negatives, false positives and false negatives.
        tp = tf.count_nonzero(predicted * labels * mask)
        tn = tf.count_nonzero((predicted - 1) * (labels - 1) * mask)
        fp = tf.count_nonzero(predicted * (labels - 1) * mask)
        fn = tf.count_nonzero((predicted - 1) * labels * mask)

        # Calculate accuracy, precision, recall and F1 score.
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fmeasure = (2 * precision * recall) / (precision + recall)
        fmeasure = tf.cast(fmeasure, tf.float32)
        return fmeasure


class GAT(BaseGAttN):
    def encoder(inputs_list, nb_nodes, training, attn_drop, ffd_drop, bias_mat_local_list, bias_mat_global_list, hid_units, n_heads, mp_att_size=16, activation=tf.nn.elu, residual=False):

        embed_list = []
        for inputs, bias_mat_local, bias_mat_global in zip(inputs_list, bias_mat_local_list, bias_mat_global_list):
            attns = []
            for _ in range(n_heads):
                attn_temp = sp_attn_head(inputs, adj_mat_local=bias_mat_local, adj_mat_global=bias_mat_global,
                                                out_sz=hid_units[0], activation=activation,
                                                in_drop=ffd_drop, coef_drop=attn_drop, residual=residual)
                attns.append(attn_temp)
            h_1 = tf.concat(attns, axis=-1)
            embed_list.append(tf.expand_dims(tf.squeeze(h_1), axis=1))
        multi_embed = tf.concat(embed_list, axis=1)
        final_embed, alpha = SimpleAttLayer(multi_embed, mp_att_size,
                                                   time_major=False,
                                                   return_alphas=True)
        return final_embed

    def decoder(embed):
        embed_size = embed.shape[1].value
        with tf.compat.v1.variable_scope("deco"):
            weight3 = glorot([embed_size, embed_size])
        U = embed
        V = embed
        logits = tf.matmul(tf.matmul(U, weight3), tf.transpose(V))
        logits = tf.reshape(logits, [-1, 1])
        return tf.nn.sigmoid(logits)

    def decoder_revised(embed):
        num_nodes = embed.shape[0].value
        embed_size = embed.shape[1].value
        with tf.compat.v1.variable_scope("deco_revised"):
            weight1 = glorot([embed_size, embed_size])
            weight2 = glorot([embed_size, embed_size])
            bias = glorot([num_nodes, embed_size])
        embedding = tf.add(tf.matmul(embed, weight1), bias)
        logits = tf.matmul(tf.matmul(embedding, weight2), tf.transpose(embedding))
        logits = tf.reshape(logits, [-1, 1])
        return tf.nn.sigmoid(logits)

    def loss_overall(scores, lbl_in, msk_in, neg_msk, weight_decay, emb):
        loss_basic = masked_accuracy(scores, lbl_in, msk_in, neg_msk)
        para_decode = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="deco_revised")
        loss_basic += weight_decay * tf.nn.l2_loss(para_decode[0])
        loss_basic += weight_decay * tf.nn.l2_loss(para_decode[1])
        loss_basic += weight_decay * tf.nn.l2_loss(para_decode[2])
        return loss_basic


class SpGAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat, hid_units, n_heads, activation=tf.nn.elu,
                  residual=False):
        attns = []
        for _ in range(n_heads[0]):
            attns.append(sp_attn_head(inputs,
                                      adj_mat=bias_mat,
                                      out_sz=hid_units[0], activation=activation, nb_nodes=nb_nodes,
                                      in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(sp_attn_head(h_1,
                                          adj_mat=bias_mat,
                                          out_sz=hid_units[i], activation=activation, nb_nodes=nb_nodes,
                                          in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)
        out = []
        for i in range(n_heads[-1]):
            out.append(sp_attn_head(h_1, adj_mat=bias_mat,
                                    out_sz=nb_classes, activation=lambda x: x, nb_nodes=nb_nodes,
                                    in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]

        return logits
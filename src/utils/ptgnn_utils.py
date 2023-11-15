import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import random
import tensorflow as tf
from collections import defaultdict

conv1d = tf.layers.conv1d

random.seed(456)
np.random.seed(456)
tf.compat.v1.set_random_seed(456)

# torch.manual_seed(123)
# torch.cuda.manual_seed_all(123)

def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0):
    with tf.name_scope('my_attn'):
        # generate linear features
        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
        # seq_fts = seq
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        return activation(ret), coefs[0]


def sp_attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0):
    with tf.name_scope('my_attn'):
        # seq_fts = seq
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        nb_nodes = seq_fts.shape[1].value

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)  # 1546x1
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)  # 1546x1

        logits = tf.add(f_1[0], tf.transpose(f_2[0]))
        logits_first = bias_mat * logits

        lrelu = tf.SparseTensor(indices=logits_first.indices,
                                values=tf.nn.leaky_relu(logits_first.values),
                                dense_shape=logits_first.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])  # 6375x6375
        seq_fts = tf.squeeze(seq_fts)  # 6375x64
        ret = tf.sparse.sparse_dense_matmul(coefs, seq_fts)

        return activation(ret)

def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


def normalize_features(feat):
    degree = np.asarray(feat.sum(1)).flatten()
    degree[degree == 0.] = np.inf
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    return feat_norm


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data():
    # labels1 = np.loadtxt("data/adj_ppi.txt")
    labels1 = np.load("../data/precessed_data/ppi_inter_arr.npy",allow_pickle=True)
    labels2 = np.load("../data/precessed_data/go_inter_arr.npy",allow_pickle=True)
    # labels2 = np.loadtxt("data/adj_go.txt")

    # interaction1 = sio.loadmat('data/graph_ppi.mat')
    interaction1 = np.load('../data/precessed_data/all_ppi_dense_sym_20398.npy')
    # interaction1 = interaction1['PPI']

    # interaction2 = sio.loadmat('data/graph_go.mat')
    interaction2 = np.load('../data/precessed_data/all_go_inter_dense_50_20398.npy')
    # interaction2 = interaction2['interaction']

    logits_train1 = interaction1
    logits_train1 = logits_train1.reshape([-1, 1])

    logits_train2 = interaction2
    logits_train2 = logits_train2.reshape([-1, 1])

    train_mask1 = np.array(logits_train1[:, 0], dtype=np.bool).reshape([-1, 1])
    train_mask2 = np.array(logits_train2[:, 0], dtype=np.bool).reshape([-1, 1])

    interaction1 = interaction1 + np.eye(interaction1.shape[0])
    interaction1 = sp.csr_matrix(interaction1)

    interaction2 = interaction2 + np.eye(interaction2.shape[0])
    interaction2 = sp.csr_matrix(interaction2)

    # word_matrix = np.loadtxt("data/sequence_to_word_matrix.txt")
    word_matrix = np.load("../data/precessed_data/ptgnn_encod_by_word_20398_800.npy")
    word_matrix = word_matrix[list(range(word_matrix.shape[0])), :600]
    word_matrix = word_matrix.astype(np.int32)

    return interaction1, interaction2, logits_train1, logits_train2, train_mask1, train_mask2, labels1, labels2, word_matrix


def load_data_for_fine_tuning(train_arr, test_arr):
    # labels = np.loadtxt("data/SynLethDB/adj.txt")
    labels = pd.read_csv('../data/precessed_data/human_sl_9845.csv')
    labels = np.asarray(labels[['unified_id_A','unified_id_B']])
    labels = np.hstack([labels, np.ones((labels.shape[0], 1))])
    num_node = 9845

    # logits_test = sp.csr_matrix((labels[test_arr, 2], (labels[test_arr, 0] - 1, labels[test_arr, 1] - 1)),
    logits_test = sp.csr_matrix((labels[test_arr, 2], (labels[test_arr, 0], labels[test_arr, 1])),
                                shape=(num_node, num_node)).toarray()
    logits_test = logits_test.reshape([-1, 1])

    # logits_train = sp.csr_matrix((labels[train_arr, 2], (labels[train_arr, 0] - 1, labels[train_arr, 1] - 1)),
    logits_train = sp.csr_matrix((labels[train_arr, 2], (labels[train_arr, 0], labels[train_arr, 1])),
                                 shape=(num_node, num_node)).toarray()
    logits_train = logits_train + logits_train.T
    interaction = logits_train
    logits_train = logits_train.reshape([-1, 1])

    train_mask = np.array(logits_train[:, 0], dtype=np.bool).reshape([-1, 1])
    test_mask = np.array(logits_test[:, 0], dtype=np.bool).reshape([-1, 1])

    interaction = interaction + np.eye(interaction.shape[0])
    interaction = sp.csr_matrix(interaction)

    word_matrix = np.loadtxt("../data/precessed_data/ptgnn_encod_by_word_sl_9845_800.npy")

    return interaction, logits_train, logits_test, train_mask, test_mask, labels, word_matrix


def train_negative_sample(logits_train1, logits_train2, num_pos_1, num_pos_2):
    num_node = 20398
    logits_train1 = logits_train1.reshape([num_node, num_node])
    logits_train2 = logits_train2.reshape([num_node, num_node])
    mask1 = np.zeros(logits_train1.shape)
    mask2 = np.zeros(logits_train2.shape)

    # sampling negatives for graph1
    num = 0
    while (num < 2 * num_pos_1):
        a = random.randint(0, num_node - 1)
        b = random.randint(0, num_node - 1)
        if logits_train1[a, b] != 1 and mask1[a, b] != 1:
            mask1[a, b] = 1
            num += 1
    mask1 = np.reshape(mask1, [-1, 1])

    # sampling negatives for graph2
    num = 0
    while (num < 2 * num_pos_2):
        a = random.randint(0, num_node - 1)
        b = random.randint(0, num_node - 1)
        if logits_train2[a, b] != 1 and mask2[a, b] != 1:
            mask2[a, b] = 1
            num += 1
    mask2 = np.reshape(mask2, [-1, 1])
    return mask1, mask2


def test_negative_sample(labels, N, negative_mask):
    num = 0
    (num_node, _) = negative_mask.shape
    A = sp.csr_matrix((labels[:, 2], (labels[:, 0] - 1, labels[:, 1] - 1)), shape=(num_node, num_node)).toarray()
    A = A + A.T
    mask = np.zeros(A.shape)
    test_neg = np.zeros((1 * N, 2))
    while (num < 1 * N):
        a = random.randint(0, num_node - 1)
        b = random.randint(0, num_node - 1)
        if a < b and A[a, b] != 1 and mask[a, b] != 1 and negative_mask[a, b] != 1:
            mask[a, b] = 1
            test_neg[num, 0] = a
            test_neg[num, 1] = b
            num += 1
    return test_neg


def div_list(ls, n):
    ls_len = len(ls)
    j = ls_len // n
    ls_return = []
    for i in range(0, (n - 1) * j, j):
        ls_return.append(ls[i:i + j])
    ls_return.append(ls[(n - 1) * j:])
    return ls_return


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def maxpooling(a):
    a = tf.cast(a, dtype=tf.float32)
    b = tf.reduce_max(a, axis=1, keepdims=True)
    c = tf.equal(a, b)
    mask = tf.cast(c, dtype=tf.float32)
    final = tf.multiply(a, mask)
    ones = tf.ones_like(a)
    zeros = tf.zeros_like(a)
    final = tf.where(final > 0.0, ones, zeros)
    return final


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj) + np.eye(adj.shape[0])
    return adj_normalized


def sparse_matrix(matrix):
    sigma = 0.001
    matrix = matrix.astype(np.int32)
    result = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        if matrix[i, 0] == 0:
            result[i, 0] = sigma
        else:
            result[i, 0] = 1
    return result


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = degree_mat_inv_sqrt.dot(adj_)
    adj_normalized = adj_normalized.dot(degree_mat_inv_sqrt).tocoo()

    return sparse_to_tuple(adj_normalized)




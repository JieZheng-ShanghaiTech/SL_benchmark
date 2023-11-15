import random
# import time
# from models.gcatsl import GAT
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
# import os

random.seed(456)
np.random.seed(456)
tf.compat.v1.set_random_seed(456)

# torch.manual_seed(123)
# torch.cuda.manual_seed_all(123)

def ROC(score_matrix, test_arr, label_neg):
    test_scores = []
    for i in range(len(test_arr)):
        test_scores.append(score_matrix[int(test_arr[i, 0]), int(test_arr[i, 1])])
    for i in range(len(label_neg)):
        test_scores.append(score_matrix[int(label_neg[i, 0]), int(label_neg[i, 1])])

    test_labels_pos = np.ones((len(test_arr), 1))
    test_labels_neg = np.zeros((len(label_neg), 1))

    test_labels = np.vstack((test_labels_pos, test_labels_neg))
    test_labels = np.array(test_labels, dtype=np.bool).reshape([-1, 1])
    return test_labels, test_scores


def masked_accuracy(preds, labels, mask, negative_mask):
    """Accuracy with masking."""
    preds = tf.cast(preds, tf.float32)
    labels = tf.cast(labels, tf.float32)
    error = tf.square(preds - labels)
    mask += negative_mask
    mask = tf.cast(mask, dtype=tf.float32)
    error *= mask
    return tf.sqrt(tf.reduce_mean(error))


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def random_walk_with_restart(interaction):
    p = 0.9  # 0.9
    iter_max = 1000  # 1000
    origi_matrix = np.identity(interaction.shape[0])
    sum_col = interaction.sum(axis=0)
    sum_col[sum_col == 0.] = 2
    interaction = np.divide(interaction, sum_col)
    pre_t = origi_matrix

    for i in range(iter_max):
        print("i:", i)
        t = (1 - p) * (np.dot(interaction, pre_t)) + p * origi_matrix
        pre_t = t
    return t


def extract_global_neighbors(interaction, walk_matrix):
    interaction = interaction.astype(int)
    interaction_mask = np.zeros_like(interaction)
    neigh_index = np.argsort(-walk_matrix, axis=0)

    for j in range(interaction.shape[1]):
        for i in range(np.sum(interaction[j, :])):
            interaction_mask[neigh_index[i, j], j] = 1
    return interaction_mask.T
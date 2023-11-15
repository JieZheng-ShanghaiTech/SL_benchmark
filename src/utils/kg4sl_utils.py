import numpy as np
import scipy.sparse as sp

np.random.seed(456)
# tf.compat.v1.set_random_seed(123)

def get_feed_dict(model, data, start, end):
    feed_dict = {model.nodea_indices: data[start:end, 0],
                 model.nodeb_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    return feed_dict

def eval_data(sess, model, batch_size, num_node):
    rol, col = np.triu_indices(num_node, k=1)
    mod = len(rol) % batch_size

    data = np.vstack([rol, col, np.zeros(len(rol))]).T
    patch = data[:(batch_size - mod)]
    data = np.vstack([data, patch])
    start = 0
    scores_list = []

    while start + batch_size <= data.shape[0]:
        scores = model.cal_scores(sess, get_feed_dict(model, data, start, start + batch_size))

        scores_list.append(scores)

        start += batch_size

    scores_flat = np.hstack(scores_list)
    score_mat = sp.csr_matrix((scores_flat[:len(rol)], (rol, col)), shape=(num_node, num_node))
    score_mat = score_mat + score_mat.T

    return score_mat

def eval_all_data(sess, model, batch_size, num_node):
    rol, col = np.triu_indices(num_node, k=1)
    mod = len(rol) % batch_size

    data = np.vstack([rol, col, np.zeros(len(rol))]).T
    patch = data[:(batch_size - mod)]
    data = np.vstack([data, patch])
    start = 0
    scores_list = []

    while start + batch_size <= data.shape[0]:
        scores = model.cal_scores(sess, get_feed_dict(model, data, start, start + batch_size))
        # nodea_emb, nodeb_emb = model.cal_scores(sess, get_feed_dict(model, data, start, start + batch_size))

        scores_list.append(scores)

        start += batch_size

    scores_flat = np.hstack(scores_list)
    score_mat = sp.csr_matrix((scores_flat[:len(rol)], (rol, col)), shape=(num_node, num_node))
    score_mat = score_mat + score_mat.T

    return score_mat
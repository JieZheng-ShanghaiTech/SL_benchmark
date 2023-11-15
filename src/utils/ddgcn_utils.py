import numpy as np
import scipy.sparse as sp
import torch

def normalize_mat(mat, normal_dim):
    # adj = sp.coo_matrix(adj)
    if normal_dim == 'Row&Column':
        # adj_ = mat + sp.eye(mat.shape[0])
        rowsum = np.array(mat.sum(1))
        inv = np.power(rowsum, -0.5).flatten()
        inv[np.isinf(inv)] = 0.
        degree_mat_inv_sqrt = sp.diags(inv)
        # D^{-0.5}AD^{-0.5}
        mat_normalized = mat.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return mat_normalized

    elif normal_dim == 'Row':
        rowsum = np.array(mat.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mat_normalized = r_mat_inv.dot(mat)
        return mat_normalized


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    :param sparse_mx:
    :return:
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def feature_loader(num_node):
    identity_matrix = torch.eye(num_node)
    is_sparse_feat = True
    # is_sparse_feat = False
    return identity_matrix, is_sparse_feat
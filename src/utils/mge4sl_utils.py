import os
import pandas as pd
import torch
import numpy as np
import math

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from models.mge4sl import SynlethDB_KG
import scipy.sparse as sp

from scipy import stats
from scipy.stats import t
import torch.nn.functional as F
from torch_geometric.utils import to_undirected

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
cuda_device=torch.device('cuda:0')
# random.seed(123)
np.random.seed(456)
# tf.compat.v1.set_random_seed(123)

torch.manual_seed(456)
torch.cuda.manual_seed_all(456)

def cal_confidence_interval(data, confidence=0.95):
    data = 1.0 * np.array(data)
    n = len(data)
    sample_mean = np.mean(data)
    se = stats.sem(data)
    t_ci = t.ppf((1 + confidence) / 2., n - 1)  # T value of Confidence Interval
    bound = se * t_ci
    return sample_mean, bound


def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def evaluate(y_true, y_score, pos_threshold=0.8):
    auc_test = roc_auc_score(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    aupr_test = auc(recall, precision)
    f1_test = f1_score(y_true, y_score > pos_threshold)
    return auc_test, aupr_test, f1_test


def train(model, optimizer, synlethdb_sl, synlethdb_ppi, synlethdb_rea, synlethdb_cor, synlethdb_go_F, \
          synlethdb_go_C, synlethdb_go_P, synlethdb_kegg):
    model.train()
    optimizer.zero_grad()

    pos_edge_index = synlethdb_sl.train_pos_edge_index
    neg_edge_index = synlethdb_sl.train_neg_edge_index

    x=synlethdb_sl.x
    ppi_train_pos_edge_index=synlethdb_ppi.train_pos_edge_index
    rea_train_pos_edge_index=synlethdb_rea.train_pos_edge_index
    cor_train_pos_edge_index=synlethdb_cor.train_pos_edge_index
    F_train_pos_edge_index=synlethdb_go_F.train_pos_edge_index
    C_train_pos_edge_index=synlethdb_go_C.train_pos_edge_index
    P_train_pos_edge_index=synlethdb_go_P.train_pos_edge_index
    kegg_train_pos_edge_index=synlethdb_kegg.train_pos_edge_index

    link_pred = model(x, pos_edge_index, neg_edge_index, ppi_train_pos_edge_index,
                      rea_train_pos_edge_index, cor_train_pos_edge_index,
                      F_train_pos_edge_index, C_train_pos_edge_index,
                      P_train_pos_edge_index, kegg_train_pos_edge_index)

    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy(link_pred, link_labels)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(model, num_node, synlethdb_sl, synlethdb_ppi, synlethdb_rea, synlethdb_cor, synlethdb_go_F, \
         synlethdb_go_C, synlethdb_go_P, synlethdb_kegg):
    model.eval()

    pos_edge_index = synlethdb_sl.val_pos_edge_index
    neg_edge_index = synlethdb_sl.val_neg_edge_index
    edge_index=np.hstack([pos_edge_index, neg_edge_index])
    edge_index=torch.tensor(edge_index)
    # edge_index=np.vstack(np.triu_indices(num_node,k=1))

    x=synlethdb_sl.x
    ppi_train_pos_edge_index=synlethdb_ppi.train_pos_edge_index
    rea_train_pos_edge_index=synlethdb_rea.train_pos_edge_index
    cor_train_pos_edge_index=synlethdb_cor.train_pos_edge_index
    F_train_pos_edge_index=synlethdb_go_F.train_pos_edge_index
    C_train_pos_edge_index=synlethdb_go_C.train_pos_edge_index
    P_train_pos_edge_index=synlethdb_go_P.train_pos_edge_index
    kegg_train_pos_edge_index=synlethdb_kegg.train_pos_edge_index


    perfs = []
    link_pred = model(x, edge_index, None, ppi_train_pos_edge_index,
                      rea_train_pos_edge_index, cor_train_pos_edge_index,
                      F_train_pos_edge_index, C_train_pos_edge_index,
                      P_train_pos_edge_index, kegg_train_pos_edge_index, 'test')
    score_mat=sp.csr_matrix((link_pred,(np.asarray(edge_index[0]),np.asarray(edge_index[1]))),shape=(num_node,num_node))
    score_mat=(score_mat+score_mat.T).toarray()

    return score_mat

@torch.no_grad()
def get_all_score_mat(model, num_node, synlethdb_sl, synlethdb_ppi, synlethdb_rea, synlethdb_cor, synlethdb_go_F, \
                    synlethdb_go_C, synlethdb_go_P, synlethdb_kegg):
    model.eval()

    edge_index=np.vstack(np.triu_indices(num_node,k=1))
    edge_index=torch.tensor(edge_index)

    x=synlethdb_sl.x
    ppi_train_pos_edge_index=synlethdb_ppi.train_pos_edge_index
    rea_train_pos_edge_index=synlethdb_rea.train_pos_edge_index
    cor_train_pos_edge_index=synlethdb_cor.train_pos_edge_index
    F_train_pos_edge_index=synlethdb_go_F.train_pos_edge_index
    C_train_pos_edge_index=synlethdb_go_C.train_pos_edge_index
    P_train_pos_edge_index=synlethdb_go_P.train_pos_edge_index
    kegg_train_pos_edge_index=synlethdb_kegg.train_pos_edge_index
 

    perfs = []
    link_pred = model(x, edge_index, None, ppi_train_pos_edge_index,
                      rea_train_pos_edge_index, cor_train_pos_edge_index,
                      F_train_pos_edge_index, C_train_pos_edge_index,
                      P_train_pos_edge_index, kegg_train_pos_edge_index, 'test')
    score_mat=sp.csr_matrix((link_pred,(np.asarray(edge_index[0]),np.asarray(edge_index[1]))),shape=(num_node,num_node))
    score_mat=(score_mat+score_mat.T).toarray()

    return score_mat


# random negative sample
def get_k_fold_data_random_neg(data, k=10):
    num_nodes = data.num_nodes

    row, col = data.edge_index
    num_edges = row.size(0)
    mask = row < col
    row, col = row[mask], col[mask]

    neg_row, neg_col = data.neg_edge_index
    neg_num_edges = neg_row.size(0)
    mask = neg_row < neg_col
    neg_row, neg_col = neg_row[mask], neg_col[mask]

    assert k > 1
    fold_size = num_edges // k

    perm = torch.randperm(num_edges)
    row, col = row[perm], col[perm]

    neg_perm = torch.randperm(neg_num_edges)
    neg_row, neg_col = neg_row[neg_perm], neg_col[neg_perm]

    res_neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)

    res_neg_adj_mask = res_neg_adj_mask.triu(diagonal=1).to(torch.bool)
    res_neg_adj_mask[row, col] = 0
    res_neg_row, res_neg_col = res_neg_adj_mask.nonzero(as_tuple=False).t()

    for j in range(k):
        val_start = j * fold_size
        val_end = (j + 1) * fold_size
        if j == k - 1:
            val_row, val_col = row[val_start:], col[val_start:]
            train_row, train_col = row[:val_start], col[:val_start]
        else:
            val_row, val_col = row[val_start:val_end], col[val_start:val_end]
            train_row, train_col = torch.cat([row[:val_start], row[val_end:]], 0), torch.cat(
                [col[:val_start], col[val_end:]], 0)

        # val
        data.val_pos_edge_index = torch.stack([val_row, val_col], dim=0)
        # train
        data.train_pos_edge_index = torch.stack([train_row, train_col], dim=0)

        add_val = data.val_pos_edge_index.shape[1]
        add_train = data.train_pos_edge_index.shape[1]
        perm = torch.randperm(res_neg_row.size(0))[:add_val + add_train]
        res_neg_row, res_neg_col = res_neg_row[perm], res_neg_col[perm]

        res_r, res_c = res_neg_row[:add_val], res_neg_col[:add_val]
        data.val_neg_edge_index = torch.stack([res_r, res_c], dim=0)

        res_r, res_c = res_neg_row[add_val:add_val + add_train], res_neg_col[add_val:add_val + add_train]
        data.train_neg_edge_index = torch.stack([res_r, res_c], dim=0)

        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)
        data.train_neg_edge_index = to_undirected(data.train_neg_edge_index)
        yield data


def train_test_split_edges_kg(data, test_ratio=0.1):
    num_nodes = data.num_nodes
    row, col = data.edge_index

    data.edge_index = None   # 是训练集的子集
    num_edges = row.size(0)

    # Return upper triangular portion.
    # mask = row < col
    # row, col = row[mask], col[mask]
    for i in range(len(row)):
        if row[i] > col[i]:
            row[i], col[i] = col[i], row[i]

    n_t = int(math.floor(test_ratio * num_edges))  # 0

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    # r, c = row[:n_t], col[:n_t]
    # data.test_pos_edge_index = torch.stack([r, c], dim=0)

    # r, c = row[n_t:], col[n_t:]
    data.train_pos_edge_index = torch.stack([row, col], dim=0)
    # data.train_pos_edge_index = torch.stack([r, c], dim=0)

    data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:num_edges]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    row, col = neg_row[:n_t], neg_col[:n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_t:], neg_col[n_t:]
    data.train_neg_edge_index = torch.stack([row, col], dim=0)

    data.train_neg_edge_index = to_undirected(data.train_neg_edge_index)
    return data


def construct_kg_sldb(num_nodes, data, sl_data, nosl_data):

    for i in range(len(data)):
        if data.iloc[i, 0] > data.iloc[i, 1]:
            data.iloc[i, 0], data.iloc[i, 1] = data.iloc[i, 1], data.iloc[i, 0]
    combined_score_data = pd.DataFrame(columns=data.columns, dtype='int')
    reactome_data = pd.DataFrame(columns=data.columns, dtype='int')
    corum_data = pd.DataFrame(columns=data.columns, dtype='int')
    go_F_data = pd.DataFrame(columns=data.columns, dtype='int')
    go_C_data = pd.DataFrame(columns=data.columns, dtype='int')
    go_P_data = pd.DataFrame(columns=data.columns, dtype='int')
    kegg_data = pd.DataFrame(columns=data.columns, dtype='int')
    for i in range(len(sl_data)):
        line = data[(data['unified_id_A'] == sl_data[i, 0]) & (data['unified_id_B'] == sl_data[i, 1])]
        if len(line) > 0:
            if line['combined_score'].values[0] > 0:
                combined_score_data=pd.concat([combined_score_data,line])
            if line['reactome'].values[0] > 0:
                reactome_data=pd.concat([reactome_data,line])
            if line['corum'].values[0] > 0:
                corum_data=pd.concat([corum_data,line])
            if line['go_F'].values[0] > 0:
                go_F_data=pd.concat([go_F_data,line])
            if line['go_C'].values[0] > 0:
                go_C_data=pd.concat([go_C_data,line])
            if line['go_P'].values[0] > 0:
                go_P_data=pd.concat([go_P_data,line])
            if line['kegg'].values[0] > 0:
                kegg_data=pd.concat([kegg_data,line])


    synlethdb_ppi = SynlethDB_KG(num_nodes, combined_score_data, 'combined_score')
    synlethdb_ppi.train_pos_edge_index=torch.tensor(combined_score_data[['unified_id_A','unified_id_B']].T.values, dtype=torch.long)
    synlethdb_ppi.train_pos_edge_index = to_undirected(synlethdb_ppi.train_pos_edge_index)
    synlethdb_ppi.train_neg_edge_index = torch.tensor(nosl_data.T, dtype=torch.long)
    synlethdb_ppi.train_neg_edge_index = to_undirected(synlethdb_ppi.train_neg_edge_index)
    # synlethdb_ppi = train_test_split_edges_kg(synlethdb_ppi, test_ratio=0)

    synlethdb_rea = SynlethDB_KG(num_nodes, reactome_data, 'reactome')
    synlethdb_rea.train_pos_edge_index = torch.tensor(reactome_data[['unified_id_A','unified_id_B']].T.values, dtype=torch.long)
    synlethdb_rea.train_pos_edge_index = to_undirected(synlethdb_rea.train_pos_edge_index)
    synlethdb_rea.train_neg_edge_index = torch.tensor(nosl_data.T, dtype=torch.long)
    synlethdb_rea.train_neg_edge_index = to_undirected(synlethdb_rea.train_neg_edge_index)
    # synlethdb_rea = train_test_split_edges_kg(synlethdb_rea, test_ratio=0)

    synlethdb_cor = SynlethDB_KG(num_nodes, corum_data, 'corum')
    synlethdb_cor.train_pos_edge_index = torch.tensor(corum_data[['unified_id_A','unified_id_B']].T.values, dtype=torch.long)
    synlethdb_cor.train_pos_edge_index = to_undirected(synlethdb_cor.train_pos_edge_index)
    synlethdb_cor.train_neg_edge_index = torch.tensor(nosl_data.T, dtype=torch.long)
    synlethdb_cor.train_neg_edge_index = to_undirected(synlethdb_cor.train_neg_edge_index)
    # synlethdb_cor = train_test_split_edges_kg(synlethdb_cor, test_ratio=0)

    synlethdb_go_F = SynlethDB_KG(num_nodes, go_F_data, 'go_F')
    synlethdb_go_F.train_pos_edge_index = torch.tensor(go_F_data[['unified_id_A','unified_id_B']].T.values, dtype=torch.long)
    synlethdb_go_F.train_pos_edge_index = to_undirected(synlethdb_go_F.train_pos_edge_index)
    synlethdb_go_F.train_neg_edge_index = torch.tensor(nosl_data.T, dtype=torch.long)
    synlethdb_go_F.train_neg_edge_index = to_undirected(synlethdb_go_F.train_neg_edge_index)
    # synlethdb_go_F = train_test_split_edges_kg(synlethdb_go_F, test_ratio=0)

    synlethdb_go_C = SynlethDB_KG(num_nodes, go_C_data, 'go_C')
    synlethdb_go_C.train_pos_edge_index = torch.tensor(go_C_data[['unified_id_A','unified_id_B']].T.values, dtype=torch.long)
    synlethdb_go_C.train_pos_edge_index = to_undirected(synlethdb_go_C.train_pos_edge_index)
    synlethdb_ppi.train_neg_edge_index = torch.tensor(nosl_data.T, dtype=torch.long)
    synlethdb_ppi.train_neg_edge_index = to_undirected(synlethdb_ppi.train_neg_edge_index)
    # synlethdb_go_C = train_test_split_edges_kg(synlethdb_go_C, test_ratio=0)

    synlethdb_go_P = SynlethDB_KG(num_nodes, go_P_data, 'go_P')
    synlethdb_go_P.train_pos_edge_index = torch.tensor(go_P_data[['unified_id_A','unified_id_B']].T.values, dtype=torch.long)
    synlethdb_go_P.train_pos_edge_index = to_undirected(synlethdb_go_P.train_pos_edge_index)
    synlethdb_go_P.train_neg_edge_index = torch.tensor(nosl_data.T, dtype=torch.long)
    synlethdb_go_P.train_neg_edge_index = to_undirected(synlethdb_go_P.train_neg_edge_index)
    # synlethdb_go_P = train_test_split_edges_kg(synlethdb_go_P, test_ratio=0)

    synlethdb_kegg = SynlethDB_KG(num_nodes, kegg_data, 'kegg')
    synlethdb_kegg.train_pos_edge_index = torch.tensor(kegg_data[['unified_id_A','unified_id_B']].T.values, dtype=torch.long)
    synlethdb_kegg.train_pos_edge_index = to_undirected(synlethdb_kegg.train_pos_edge_index)
    synlethdb_kegg.train_neg_edge_index = torch.tensor(nosl_data.T, dtype=torch.long)
    synlethdb_kegg.train_neg_edge_index = to_undirected(synlethdb_kegg.train_neg_edge_index)
    # synlethdb_kegg = train_test_split_edges_kg(synlethdb_kegg, test_ratio=0)

    return synlethdb_ppi, synlethdb_rea, synlethdb_cor, synlethdb_go_F, synlethdb_go_C, synlethdb_go_P, synlethdb_kegg
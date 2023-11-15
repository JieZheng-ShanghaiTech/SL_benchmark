import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.data import Data

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
cuda_device=torch.device('cuda:0')
# random.seed(123)
np.random.seed(456)
# tf.compat.v1.set_random_seed(123)

torch.manual_seed(456)
torch.cuda.manual_seed_all(456)

class GCNEncoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, 16)

    def forward(self, x, train_pos_edge_index):
        x = self.conv1(x, train_pos_edge_index)
        x = x.relu()
        x = self.conv2(x, train_pos_edge_index)
        return x


class GINEncoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(GINEncoder, self).__init__()
        self.conv1 = GINConv(input_dim, 128)
        self.conv2 = GINConv(128, 16)

    def forward(self, x, train_pos_edge_index):
        x = self.conv1(x, train_pos_edge_index)
        x = x.relu()
        x = self.conv2(x, train_pos_edge_index)
        return x


class MultiGraphEnsembleFC(torch.nn.Module):
    def __init__(self, n_graph, node_emb_dim, sl_input_dim, kg_input_dim):
        super(MultiGraphEnsembleFC, self).__init__()
        self.encode_sl = GCNEncoder(input_dim=sl_input_dim)
        self.encode_ppi = GCNEncoder(input_dim=kg_input_dim)
        self.encode_reactome = GCNEncoder(input_dim=kg_input_dim)
        self.encode_corum = GCNEncoder(input_dim=kg_input_dim)
        self.encode_go_f = GCNEncoder(input_dim=kg_input_dim)
        self.encode_go_c = GCNEncoder(input_dim=kg_input_dim)
        self.encode_go_p = GCNEncoder(input_dim=kg_input_dim)
        self.encode_kegg = GCNEncoder(input_dim=kg_input_dim)

        self.linear1 = torch.nn.Linear(n_graph * node_emb_dim, 32)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.linear2 = torch.nn.Linear(32, 16)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.linear3 = torch.nn.Linear(16, 1)

    def decode(self, z, pos_edge_index, neg_edge_index, mode='train'):
        if mode == 'test':
            # edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            emb_features = z[pos_edge_index[0]] + z[pos_edge_index[1]]
        else:
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            emb_features = z[edge_index[0]] + z[edge_index[1]]
        # return torch.Tensor(emb_features)
        return emb_features

    def forward(self, x, sl_pos, sl_neg, kg_ppi, kg_reactome, kg_corum, kg_go_f, kg_go_c, kg_go_p, kg_kegg, mode='train'):
        emb_sl = self.encode_sl(x, sl_pos)
        emb_ppi = self.encode_ppi(x, kg_ppi)
        emb_reactome = self.encode_reactome(x, kg_reactome)
        emb_corum = self.encode_corum(x, kg_corum)
        emb_go_f = self.encode_go_f(x, kg_go_f)
        emb_go_c = self.encode_go_c(x, kg_go_c)
        emb_go_p = self.encode_go_p(x, kg_go_p)
        emb_kegg = self.encode_kegg(x, kg_kegg)

        emb_all = [emb_sl, emb_ppi, emb_reactome, emb_corum, emb_go_f, emb_go_c, emb_go_p, emb_kegg]
        if mode == 'test':
            # concat all type of edge embedding
            features_all = torch.zeros((sl_pos.shape[1], 1))
            for emb in emb_all:
                emb_feature = self.decode(emb, sl_pos, sl_neg, mode)
                features_all = torch.cat((features_all, emb_feature), dim=1)

            features_all = features_all[:, 1:]
        else:
            # concat all type of edge embedding
            features_all = torch.zeros((sl_pos.shape[1] + sl_neg.shape[1], 1))
            for emb in emb_all:
                emb_feature = self.decode(emb, sl_pos, sl_neg)
                features_all = torch.cat((features_all, emb_feature), dim=1)

            features_all = features_all[:, 1:]
        hidden = self.linear1(features_all)
        hidden = F.relu(hidden)

        hidden = self.linear2(hidden)
        hidden = F.relu(hidden)
        hidden = self.linear3(hidden)
        y_pred = torch.sigmoid(hidden.squeeze(1))
        return y_pred


class MultiGraphEnsembleFC_SUM(torch.nn.Module):
    def __init__(self, n_graph, node_emb_dim, sl_input_dim, kg_input_dim):
        super(MultiGraphEnsembleFC_SUM, self).__init__()
        self.encode_sl = GCNEncoder(input_dim=sl_input_dim)
        self.encode_ppi = GCNEncoder(input_dim=kg_input_dim)
        self.encode_reactome = GCNEncoder(input_dim=kg_input_dim)
        self.encode_corum = GCNEncoder(input_dim=kg_input_dim)
        self.encode_go_f = GCNEncoder(input_dim=kg_input_dim)
        self.encode_go_c = GCNEncoder(input_dim=kg_input_dim)
        self.encode_go_p = GCNEncoder(input_dim=kg_input_dim)
        self.encode_kegg = GCNEncoder(input_dim=kg_input_dim)

        self.linear1 = torch.nn.Linear(16, 32)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.linear2 = torch.nn.Linear(32, 16)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.linear3 = torch.nn.Linear(16, 1)

    def decode(self, z, pos_edge_index, neg_edge_index, mode='train'):
        if mode == 'test':
            # edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            emb_features = z[pos_edge_index[0]] + z[pos_edge_index[1]]
        else:
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            emb_features = z[edge_index[0]] + z[edge_index[1]]
        return torch.Tensor(emb_features)

    def forward(self, x, sl_pos, sl_neg, kg_ppi, kg_reactome, kg_corum, kg_go_f, kg_go_c, kg_go_p, kg_kegg, mode='train'):
        emb_sl = self.encode_sl(x, sl_pos)
        emb_ppi = self.encode_ppi(x, kg_ppi)
        emb_reactome = self.encode_reactome(x, kg_reactome)
        emb_corum = self.encode_corum(x, kg_corum)
        emb_go_f = self.encode_go_f(x, kg_go_f)
        emb_go_c = self.encode_go_c(x, kg_go_c)
        emb_go_p = self.encode_go_p(x, kg_go_p)
        emb_kegg = self.encode_kegg(x, kg_kegg)

        emb_all = [emb_sl, emb_ppi, emb_reactome, emb_corum, emb_go_f, emb_go_c, emb_go_p, emb_kegg]
        if mode == 'test':
            # sum all type of edge embedding
            features_sum = torch.zeros((sl_pos.shape[1], 16))
            for emb in emb_all:
                emb_feature = self.decode(emb, sl_pos, sl_neg, mode)
                features_sum = features_sum + emb_feature
        else:
            # sum all type of edge embedding
            features_sum = torch.zeros((sl_pos.shape[1] + sl_neg.shape[1], 16))
            for emb in emb_all:
                emb_feature = self.decode(emb, sl_pos, sl_neg)
                features_sum = features_sum + emb_feature

        hidden = self.linear1(features_sum)
        hidden = F.relu(hidden)

        hidden = self.linear2(hidden)
        hidden = F.relu(hidden)
        hidden = self.linear3(hidden)
        y_pred = torch.sigmoid(hidden.squeeze(1))
        return y_pred


class CNNet(nn.Module):

    def __init__(self):
        super(CNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.dropout = nn.Dropout2d(p=0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.sigmoid(x)
        return y.squeeze(1)


class MultiGraphEnsembleCNN(torch.nn.Module):
    def __init__(self, n_graph, node_emb_dim, sl_input_dim, kg_input_dim):
        super(MultiGraphEnsembleCNN, self).__init__()
        self.encode_sl = GCNEncoder(input_dim=sl_input_dim)
        self.encode_ppi = GCNEncoder(input_dim=kg_input_dim)
        self.encode_reactome = GCNEncoder(input_dim=kg_input_dim)
        self.encode_corum = GCNEncoder(input_dim=kg_input_dim)
        self.encode_go_f = GCNEncoder(input_dim=kg_input_dim)
        self.encode_go_c = GCNEncoder(input_dim=kg_input_dim)
        self.encode_go_p = GCNEncoder(input_dim=kg_input_dim)
        self.encode_kegg = GCNEncoder(input_dim=kg_input_dim)

        self.cnn = CNNet()

    def get_emb_matrix(self, embeddings, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        edge_length = edge_index.shape[1]

        emb_matrix_trans = []
        for i in range(edge_length):
            emb_matrix = []
            for emb_index in range(len(embeddings)):
                cur_emb = embeddings[emb_index].detach().numpy()
                pair_emb = np.vstack((cur_emb[edge_index[0, i].item()], cur_emb[edge_index[1, i].item()])).T  # (16,2)
                emb_matrix.append(pair_emb)
            emb_matrix_trans.append(np.array(emb_matrix))
        emb_matrix_trans = np.array(emb_matrix_trans)
        return emb_matrix_trans

    def forward(self, x, sl_pos, sl_neg, kg_ppi, kg_reactome, kg_corum, kg_go_f, kg_go_c, kg_go_p, kg_kegg):
        self.emb_sl = self.encode_sl(x, sl_pos)
        self.emb_ppi = self.encode_ppi(x, kg_ppi)
        self.emb_reactome = self.encode_reactome(x, kg_reactome)
        self.emb_corum = self.encode_corum(x, kg_corum)
        self.emb_go_f = self.encode_go_f(x, kg_go_f)
        self.emb_go_c = self.encode_go_c(x, kg_go_c)
        self.emb_go_p = self.encode_go_p(x, kg_go_p)
        self.emb_kegg = self.encode_kegg(x, kg_kegg)

        # node embedding
        emb_all = [self.emb_sl, self.emb_reactome, self.emb_sl, self.emb_kegg, self.emb_sl, self.emb_go_f, self.emb_sl,
                   self.emb_corum, self.emb_sl, self.emb_go_c, self.emb_sl, self.emb_go_p, self.emb_sl, self.emb_ppi]

        input_image = self.get_emb_matrix(emb_all, sl_pos, sl_neg)  # shape 39494, 14, 16, 2
        input_image = np.transpose(input_image, (0, 3, 1, 2))  # Batch,Channel,Width,Height

        y_pred = self.cnn(torch.Tensor(input_image))
        return y_pred


# weight FC
class MultiGraphEnsembleWeightFC(torch.nn.Module):
    def __init__(self, n_graph, node_emb_dim, sl_input_dim, kg_input_dim):
        super(MultiGraphEnsembleWeightFC, self).__init__()
        self.encode_sl = GCNEncoder(input_dim=sl_input_dim)
        self.encode_ppi = GCNEncoder(input_dim=kg_input_dim)
        self.encode_reactome = GCNEncoder(input_dim=kg_input_dim)
        self.encode_corum = GCNEncoder(input_dim=kg_input_dim)
        self.encode_go_f = GCNEncoder(input_dim=kg_input_dim)
        self.encode_go_c = GCNEncoder(input_dim=kg_input_dim)
        self.encode_go_p = GCNEncoder(input_dim=kg_input_dim)
        self.encode_kegg = GCNEncoder(input_dim=kg_input_dim)

        self.w1 = torch.nn.Linear(16, 1)
        self.w2 = torch.nn.Linear(16, 1)
        self.w3 = torch.nn.Linear(16, 1)
        self.w4 = torch.nn.Linear(16, 1)
        self.w5 = torch.nn.Linear(16, 1)
        self.w6 = torch.nn.Linear(16, 1)
        self.w7 = torch.nn.Linear(16, 1)
        self.w8 = torch.nn.Linear(16, 1)

        self.linear1 = torch.nn.Linear(n_graph * node_emb_dim, 32)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.linear2 = torch.nn.Linear(32, 16)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.linear3 = torch.nn.Linear(16, 1)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        emb_features = z[edge_index[0]] + z[edge_index[1]]
        return torch.Tensor(emb_features)

    def forward(self, x, sl_pos, sl_neg, kg_ppi, kg_reactome, kg_corum, kg_go_f, kg_go_c, kg_go_p, kg_kegg):
        emb_sl = self.encode_sl(x, sl_pos)
        emb_ppi = self.encode_ppi(x, kg_ppi)
        emb_reactome = self.encode_reactome(x, kg_reactome)
        emb_corum = self.encode_corum(x, kg_corum)
        emb_go_f = self.encode_go_f(x, kg_go_f)
        emb_go_c = self.encode_go_c(x, kg_go_c)
        emb_go_p = self.encode_go_p(x, kg_go_p)
        emb_kegg = self.encode_kegg(x, kg_kegg)

        # emb_all = [emb_sl, emb_ppi, emb_reactome, emb_corum, emb_go_f, emb_go_c, emb_go_p, emb_kegg]
        emb_edge_sl = self.decode(emb_sl, sl_pos, sl_neg)

        emb_edge_ppi = self.decode(emb_ppi, sl_pos, sl_neg)
        emb_edge_reactome = self.decode(emb_reactome, sl_pos, sl_neg)
        emb_edge_corum = self.decode(emb_corum, sl_pos, sl_neg)
        emb_edge_go_f = self.decode(emb_go_f, sl_pos, sl_neg)
        emb_edge_go_c = self.decode(emb_go_c, sl_pos, sl_neg)
        emb_edge_go_p = self.decode(emb_go_p, sl_pos, sl_neg)
        emb_edge_kegg = self.decode(emb_kegg, sl_pos, sl_neg)

        self.w_value_sl = self.w1(emb_edge_sl)  # emb_edge_sl [114900, 16] w1 16*1 -> [114900, 1]

        self.w_value_ppi = self.w2(emb_edge_ppi)
        self.w_value_reactome = self.w3(emb_edge_reactome)
        # self.w_value_corum = self.w4(emb_edge_corum)
        self.w_value_go_f = self.w5(emb_edge_go_f)
        self.w_value_go_c = self.w6(emb_edge_go_c)
        self.w_value_go_p = self.w7(emb_edge_go_p)
        self.w_value_kegg = self.w8(emb_edge_kegg)

        features_all = torch.cat((emb_edge_sl * self.w_value_sl, emb_edge_ppi * self.w_value_ppi, \
                                  emb_edge_reactome * self.w_value_reactome, \
                                  emb_edge_go_f * self.w_value_go_f, emb_edge_go_c * self.w_value_go_c, \
                                  emb_edge_go_p * self.w_value_go_p, emb_edge_kegg * self.w_value_kegg), dim=1)

        hidden = self.linear1(features_all)
        hidden = F.relu(hidden)
        # hidden = self.bn1(hidden)
        hidden = self.linear2(hidden)
        hidden = F.relu(hidden)
        # hidden = F.dropout(hidden, p=0.5, training=self.training)
        # hidden = self.bn2(hidden)
        hidden = self.linear3(hidden)
        y_pred = torch.sigmoid(hidden.squeeze(1))
        return y_pred

class SynlethDB(Data):
    def __init__(self, num_nodes, sl_data,nosl_data):
        num_nodes = num_nodes
        num_edges = sl_data.shape[0]
        neg_num_edges = nosl_data.shape[0]
        feat_node_dim = 1
        feat_edge_dim = 1
        self.x = torch.ones(num_nodes, feat_node_dim)
        self.y = torch.randint(0, 2, (num_nodes,))
        self.edge_index = torch.tensor(sl_data.T, dtype=torch.long)
        # self.edge_index = torch.tensor(sl_data[['gene_a_encoder','gene_b_encoder']].T.values, dtype=torch.long)
        self.edge_attr = torch.ones(num_edges, feat_edge_dim)
        self.neg_edge_index = torch.tensor(nosl_data.T, dtype=torch.long)
        # self.neg_edge_index = torch.tensor(nosl_data[['gene_a_encoder', 'gene_b_encoder']].T.values, dtype=torch.long)
        self.neg_edge_attr = torch.ones(neg_num_edges, feat_edge_dim)

#related knowledge graph
class SynlethDB_KG(Data):
    def __init__(self, num_nodes, kg_data, types):
        self.type = types
        # num_nodes = num_nodes
        num_edges = kg_data.shape[0]
        feat_node_dim = 1
        feat_edge_dim = 1
        self.x = torch.ones(num_nodes, feat_node_dim)
        self.y = torch.randint(0, 2, (num_nodes,))
        self.edge_index = torch.tensor(kg_data[['unified_id_A','unified_id_B']].T.values, dtype=torch.long)
        # self.edge_index = torch.tensor(kg_data[['gene_a_encoder','gene_b_encoder']].T.values, dtype=torch.long)
        self.edge_attr = torch.tensor(kg_data[[self.type]].values, dtype = torch.long)

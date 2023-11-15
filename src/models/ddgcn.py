import copy
from scipy import stats
from scipy.stats import t
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits
import math
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, average_precision_score

# random.seed(123)
np.random.seed(456)
# tf.compat.v1.set_random_seed(123)

torch.manual_seed(456)
torch.cuda.manual_seed_all(456)

## DDGCN
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, init, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if use_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.use_bias = use_bias
        self.reset_parameters(init)

    def reset_parameters(self, init):
        if init == 'Xavier':
            fan_in, fan_out = self.weight.shape
            init_range = np.sqrt(6.0 / (fan_in + fan_out))
            self.weight.data.uniform_(-init_range, init_range)

            if self.use_bias:
                torch.nn.init.constant_(self.bias, 0.)

        elif init == 'Kaiming':
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

            if self.use_bias:
                fan_in, _ = self.weight.shape
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

        else:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.use_bias:
                self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        if inputs.is_sparse:
            support = torch.sparse.mm(inputs, self.weight)
        else:
            support = torch.mm(inputs, self.weight)
        outputs = torch.sparse.mm(adj, support)
        if self.use_bias:
            return outputs + self.bias
        else:
            return outputs

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNEncoder(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout, init, use_bias, is_sparse_feat1, is_sparse_feat2):
        """
        :param nfeat:
        :param nhid1: Node embedding dim in first GCN layer
        :param nhid2: Node embedding dim in second GCN layer
        :param dropout:
        :param init:
        :param use_bias:
        :param is_sparse_feat1:
        :param is_sparse_feat2:
        """
        super(GCNEncoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1, init, use_bias)
        self.gc2 = GraphConvolution(nhid1, nhid2, init, use_bias)
        self.dropout = dropout
        self.is_sparse_feat1 = is_sparse_feat1
        self.is_sparse_feat2 = is_sparse_feat2

    def forward(self, x1, x2, adj):
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.dropout(x2, self.dropout, training=self.training)
        if self.is_sparse_feat1:
            x1 = x1.to_sparse()
        if self.is_sparse_feat2:
            x2 = x2.to_sparse()
        x1 = F.relu(self.gc1(x1, adj))
        x2 = F.relu(self.gc1(x2, adj))
        if self.training:
            mask = torch.bernoulli(x1.data.new(x1.data.size()).fill_(1 - self.dropout)) / (1 - self.dropout)
            # mask = torch.FloatTensor(torch.bernoulli(x1.data.new(x1.data.size()).fill_(1 - self.dropout)) / (1 - self.dropout))
            x1 = x1 * mask
            x2 = x2 * mask
        x1 = self.gc2(x1, adj)
        x2 = self.gc2(x2, adj)
        return x1, x2


class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""

    def __init__(self, dropout):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout

    def forward(self, inputs1, inputs2):
        if self.training:
            mask = torch.bernoulli(inputs1.data.new(inputs1.data.size()).fill_(1 - self.dropout)) / (1 - self.dropout)
            inputs1 = inputs1 * mask
            inputs2 = inputs2 * mask
        outputs1 = torch.mm(inputs1, inputs1.t())
        outputs2 = torch.mm(inputs2, inputs2.t())

        return outputs1, outputs2


class GraphAutoEncoder(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout, init, use_bias, is_sparse_feat1, is_sparse_feat2):
        super(GraphAutoEncoder, self).__init__()
        self.encoder = GCNEncoder(nfeat, nhid1, nhid2, dropout, init, use_bias, is_sparse_feat1, is_sparse_feat2)
        self.decoder = InnerProductDecoder(dropout)


    def forward(self, x1, x2, adj):
        node_embed1, node_embed2 = self.encoder(x1, x2, adj)
        reconstruct_adj_logit1, reconstruct_adj_logit2 = self.decoder(node_embed1, node_embed2)

        return reconstruct_adj_logit1, reconstruct_adj_logit2


class ObjectiveFunction:
    def __init__(self, target_adj, weight_mask):
        num_edges = target_adj.sum()
        num_nodes = target_adj.shape[0]
        self.pos_weight = float(num_nodes ** 2 - num_edges) / num_edges
        self.norm = num_nodes ** 2 / float((num_nodes ** 2 - num_edges) * 2)
        self.target = target_adj
        self.weight_mask = weight_mask

    def cal_loss(self, logit1, logit2, rho):

        loss1 = self.norm * binary_cross_entropy_with_logits(logit1.cpu(), self.target, weight=self.weight_mask,
                                                             pos_weight=self.pos_weight, reduction='mean')

        loss2 = self.norm * binary_cross_entropy_with_logits(logit2.cpu(), self.target, weight=self.weight_mask,
                                                             pos_weight=self.pos_weight, reduction='mean')

        return loss1 + rho * loss2


class Evaluator:
    def __init__(self, train_adj, test_pos_adj=None, test_neg_adj=None, pos_threshold=None):
        node_num = train_adj.shape[0]

        r_pos, c_pos=test_pos_adj.nonzero()
        r_neg, c_neg=test_neg_adj.nonzero()

        self.sample_rows=np.hstack([r_pos,r_neg])
        self.sample_cols=np.hstack([c_pos,c_neg])

        if test_pos_adj is not None:

            self.y_true = test_pos_adj[self.sample_rows, self.sample_cols]

        self.pos_threshold = pos_threshold

    def remap_id(self, id_mapping):
        remapped_id = dict()
        for k in id_mapping.keys():
            remapped_id[id_mapping[k]] = k

        return remapped_id

    def get_test_adj(self, reconstruct_adj, all_eval_ind):
        pred_adj = np.zeros_like(reconstruct_adj)
        for i in all_eval_ind:
            pred_adj[:, i] = reconstruct_adj[:, i]
            pred_adj[i, :] = reconstruct_adj[i, :]

        return pred_adj

    @staticmethod
    def geometric_mean(reconstruct_adj1, reconstruct_adj2, rho):
        reconstruct_adj = np.power(reconstruct_adj1 * np.power(reconstruct_adj2, rho), 1 / (1 + rho))  # geometric mean

        return reconstruct_adj

    def eval(self, reconstruct_adj1, reconstruct_adj2, rho):
        reconstruct_adj = self.geometric_mean(reconstruct_adj1, reconstruct_adj2, rho)

        y_score = reconstruct_adj[self.sample_rows, self.sample_cols]

        auc_test = roc_auc_score(self.y_true, y_score)
        precision, recall, thresholds = precision_recall_curve(self.y_true, y_score)
        aupr_test = auc(recall, precision)

        f1_test = f1_score(self.y_true, y_score > self.pos_threshold)

        return reconstruct_adj, auc_test, aupr_test, f1_test


    def unknown_pairs_scores(self, reconstruct_adj1, reconstruct_adj2, rho):
        """
        :return y_score:1 D Tensor, e.g., tensor([ 1.0704,  0.6944, -0.5432])
        """
        reconstruct_adj = self.geometric_mean(reconstruct_adj1, reconstruct_adj2, rho)
        y_score = reconstruct_adj[self.sample_rows, self.sample_cols]  # unknown pairs scores

        return np.vstack([self.sample_rows, self.sample_cols]).T, y_score


def cal_confidence_interval(data, confidence=0.95):
    data = 1.0 * np.array(data)
    n = len(data)
    sample_mean = np.mean(data)
    se = stats.sem(data)
    t_ci = t.ppf((1 + confidence) / 2., n - 1)  # T value of Confidence Interval
    bound = se * t_ci
    return sample_mean, bound

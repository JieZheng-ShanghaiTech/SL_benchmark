import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv
from torch_scatter import scatter_mean
import dgl
from utils.slgnn_utils import DistanceCorrelation

device = 'cuda:0'


class GAT(nn.Module):
    def __init__(self, num_layers, in_dim, num_hidden, num_classes, heads,
                 activation, feat_drop, attn_drop, negative_slope, residual):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(
            GATConv(in_dim, num_hidden, heads[0], feat_drop, attn_drop,
                    negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                GATConv(num_hidden * heads[l - 1], num_hidden, heads[l],
                        feat_drop, attn_drop, negative_slope, residual,
                        self.activation))
        # output projection
        self.gat_layers.append(
            GATConv(num_hidden * heads[-2], num_classes, heads[-1], feat_drop,
                    attn_drop, negative_slope, residual, None))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits


class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self, n_genes, n_factors, reindex_dict):
        super(Aggregator, self).__init__()
        self.n_genes = n_genes
        self.n_factors = n_factors
        self.reindex_dict = reindex_dict

    def forward(self, GATConv, sl_graph, entity_emb, gene_sl_emb, latent_emb,
                edge_index, edge_type, interact_mat, weight, disen_weight_att):

        n_entities = entity_emb.shape[0]
        channel = entity_emb.shape[1]
        n_genes = self.n_genes
        n_factors = self.n_factors
        """KG aggregate"""

        # import pdb; pdb.set_trace()
        head, tail = edge_index
        edge_relation_emb = weight[edge_type]
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel] 
        entity_agg = scatter_mean(src=neigh_relation_emb,
                                  index=head,
                                  dim_size=n_entities,
                                  dim=0)

        latent_emb = torch.mm(nn.Softmax(dim=-1)(disen_weight_att),
                              weight)  #[n_factors, channel]
        score_ = torch.mm(gene_sl_emb, latent_emb.t())

        score = nn.Softmax(dim=1)(score_).unsqueeze(-1)

        gene_list = [i for i in range(self.n_genes)]
        reidx_dict = dict([(key, self.reindex_dict[key]) for key in gene_list])
        reidx = torch.tensor(list(reidx_dict.values()))
        gene_agg = torch.sparse.mm(interact_mat, entity_emb[reidx])
        output_sl = []
        input_emb = entity_agg[reidx]
        #input_emb = input_emb * latent_emb[i]
        for i in range(self.n_factors):
            output_emb = GATConv[i](sl_graph, input_emb)
            output_emb_i = output_emb * latent_emb[i]
            output_sl.append(output_emb_i)
        output_sl = torch.stack(output_sl)
        output_sl = output_sl.permute(1, 0, 2)
        disen_weight = torch.mm(nn.Softmax(dim=-1)(disen_weight_att),
                                weight).expand(n_genes, n_factors, channel)

        gene_return = gene_agg + (output_sl * score).sum(dim=1)
        #disen_weight_att [n_factors, n_relations]
        #weight [n_relations, channel]

        return entity_agg, gene_return, output_sl


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self,
                 channel,
                 n_hops,
                 n_genes,
                 n_factors,
                 n_relations,
                 interact_mat,
                 ind,
                 reindex_dict,
                 node_dropout_rate=0.5,
                 mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()
        self.GATConv = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.sl_graph = self.get_sl_graph()
        self.n_relations = n_relations
        self.n_genes = n_genes
        self.n_factors = n_factors
        self.reindex_dict = reindex_dict
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        for i in range(self.n_factors):
            self.GATConv.append(
                GAT(1, channel, channel, channel, [8, 8], F.leaky_relu, 0.2,
                    0.2, 0.2, True))
        self.temperature = 0.2
        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations, channel))
        self.weight = nn.Parameter(weight)

        disen_weight_att = initializer(torch.empty(n_factors, n_relations))
        self.disen_weight_att = nn.Parameter(disen_weight_att)
        # import pdb; pdb.set_trace()
        for i in range(n_hops):
            self.convs.append(
                Aggregator(n_genes=n_genes,
                           n_factors=n_factors,
                           reindex_dict=self.reindex_dict))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def get_sl_graph(self):
        interact_mat = self.interact_mat
        indices = interact_mat._indices()
        # import pdb; pdb.set_trace()
        g = dgl.graph((indices[0], indices[1]))
        return g.to(device)

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges,
                                          size=int(n_edges * rate),
                                          replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()  #number of none zeros
        #稀疏tensor的dropout
        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    # def _cul_cor_pro(self):
    #     # disen_T: [num_factor, dimension]
    #     disen_T = self.disen_weight_att.t()
    #
    #     # normalized_disen_T: [num_factor, dimension]
    #     normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)
    #
    #     pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
    #     ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)
    #
    #     pos_scores = torch.exp(pos_scores / self.temperature)
    #     ttl_scores = torch.exp(ttl_scores / self.temperature)
    #
    #     mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
    #     return mi_score

    def _cul_cor(self):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 *
                    normalized_tensor_2).sum(dim=0)**2  # no negative

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                   torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1**2, tensor_2**2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(
                dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(
                dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(
                torch.max((A * B).sum() / channel**2, zero) + 1e-8)
            dcov_AA = torch.sqrt(
                torch.max((A * A).sum() / channel**2, zero) + 1e-8)
            dcov_BB = torch.sqrt(
                torch.max((B * B).sum() / channel**2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.disen_weight_att.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T,
                                   dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att),
                                   dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = -torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.disen_weight_att[i],
                                                   self.disen_weight_att[j])
                    else:
                        cor += CosineSimilarity(self.disen_weight_att[i],
                                                self.disen_weight_att[j])
        return cor

    def forward(self,
                gene_emb,
                entity_emb,
                latent_emb,
                edge_index,
                edge_type,
                interact_mat,
                mess_dropout=True,
                node_dropout=False):
        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(
                edge_index, edge_type, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(interact_mat,
                                                self.node_dropout_rate)
        # import pdb; pdb.set_trace()
        entity_res_emb = entity_emb  # [n_entity, channel]
        gene_res_emb = gene_emb  #[n_gene, channel]
        cor = self._cul_cor()
        for_reg = None
        for i in range(len(self.convs)):
            entity_emb, gene_emb, output_gene = self.convs[i](
                self.GATConv, self.sl_graph, entity_emb, gene_emb, latent_emb,
                edge_index, edge_type, interact_mat, self.weight,
                self.disen_weight_att)
            """message dropout"""
            for_reg = output_gene.mean(dim=0)
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                gene_emb = self.dropout(gene_emb)
            entity_emb = F.normalize(entity_emb)
            gene_emb = F.normalize(gene_emb)
            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            gene_res_emb = torch.add(gene_res_emb, gene_emb)
            if i == 0:
                for_reg = output_gene.mean(dim=0)
            else:
                for_reg = torch.add(for_reg, output_gene.mean(dim=0))

        cor_2 = 0

        for i in range(self.n_factors):
            for j in range(i + 1, self.n_factors):
                cor_2 += DistanceCorrelation(for_reg[i], for_reg[j])

        return entity_res_emb, gene_res_emb, cor, cor_2


class SLModel(nn.Module):
    def __init__(self, n_genes, n_relations, n_entities, l2,sim_regularity,dim,context_hops,n_factors,
                 node_dropout,node_dropout_rate,mess_dropout,mess_dropout_rate,ind,cuda_device,edge_index,edge_type,sl_adj, reindex_dict):
        super(SLModel, self).__init__()
        self.n_genes = n_genes
        self.n_relations = n_relations
        self.reindex_dict = {y: x for x, y in reindex_dict.items()}
        self.n_entities = n_entities  # include items
        self.decay = l2
        self.sim_decay = sim_regularity
        self.emb_size = dim
        self.context_hops = context_hops
        self.n_factors = n_factors
        self.node_dropout = node_dropout
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout = mess_dropout
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.device = cuda_device
        self.sl_adj = sl_adj
        # self.kg = kg
        self.edge_index, self.edge_type = self._get_edges(edge_index, edge_type)
        # self.edge_index, self.edge_type = self._get_edges(kg)

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(
            torch.empty(self.n_genes + self.n_entities, self.emb_size))
        self.latent_emb = initializer(
            torch.empty(self.n_factors, self.emb_size))
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.sl_adj).to(
            self.device)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_genes=self.n_genes,
                         n_relations=self.n_relations,
                         n_factors=self.n_factors,
                         interact_mat=self.interact_mat,
                         ind=self.ind,
                         reindex_dict=self.reindex_dict,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor(np.array([coo.row, coo.col]))
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, edge_index, edge_type):
    # def _get_edges(self, graph):
        # graph_tensor = torch.tensor(list(graph))  # [-1, 3]
        # graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = torch.tensor(list(edge_index))  # [-1, 2]
        type = torch.tensor(list(edge_type))  # [-1, 1]
        # import pdb; pdb.set_trace()
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, gene_a, gene_b):
        gene_a = gene_a
        gene_b = gene_b
        gene_emb = self.all_embed[:self.n_genes, :]
        item_emb = self.all_embed[self.n_genes:, :]
        # import pdb; pdb.set_trace()
        entity_gcn_emb, gene_gcn_emb, cor, cor_2 = self.gcn(
            gene_emb,
            item_emb,
            self.latent_emb,
            self.edge_index,
            self.edge_type,
            self.interact_mat,
            mess_dropout=self.mess_dropout,
            node_dropout=self.node_dropout)
        e_u = gene_gcn_emb[gene_a]
        reidx_dict = dict([(key, self.reindex_dict[int(key)])
                           for key in gene_b])
        e_e = entity_gcn_emb[torch.tensor(list(reidx_dict.values()))]
        #e_e = entity_gcn_emb[gene_b]

        scores = (e_u * e_e).sum(dim=1)
        regularizer = (torch.norm(e_u)**2 + torch.norm(e_e)**2) / 2
        emb_loss = self.decay * regularizer / len(gene_a)
        cor_loss = self.sim_decay * (cor + cor_2)
        return torch.sigmoid(scores), emb_loss, cor_loss, cor, scores

    def generate(self):
        gene_emb = self.all_embed[:self.n_genes, :]
        item_emb = self.all_embed[self.n_genes:, :]
        return self.gcn(gene_emb,
                        item_emb,
                        self.latent_emb,
                        self.edge_index,
                        self.edge_type,
                        self.interact_mat,
                        mess_dropout=False,
                        node_dropout=False)[:-1]

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, genes, pos_items, neg_items, cor):
        batch_size = genes.shape[0]
        pos_scores = torch.sum(torch.mul(genes, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(genes, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(genes)**2 + torch.norm(pos_items)**2 +
                       torch.norm(neg_items)**2) / 2
        emb_loss = self.decay * regularizer / batch_size
        cor_loss = self.sim_decay * cor

        return mf_loss + emb_loss + cor_loss, mf_loss, emb_loss, cor

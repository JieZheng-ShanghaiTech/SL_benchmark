
import os
import abc
import logging
import numpy as np
import time
import dgl
import lmdb
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn import metrics
from utils.pilsl_utils import process_files,process_files,ssp_multigraph_to_dgl,deserialize,move_batch_to_device_dgl
from dgl.data.utils import save_graphs, load_graphs

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "2"

# random.seed(123)
np.random.seed(456)
# tf.compat.v1.set_random_seed(123)

torch.manual_seed(456)
torch.cuda.manual_seed_all(456)

class Trainer():
    def __init__(self, params, graph_classifier, train, train_evaluator=None, valid_evaluator=None,
                 test_evaluator=None, all_pairs_evaluator=None, wandb_runner=None):
        self.graph_classifier = graph_classifier
        self.train_evaluator = train_evaluator
        self.valid_evaluator = valid_evaluator
        self.all_pairs_evaluator = all_pairs_evaluator
        self.params = params
        self.train_data = train
        self.test_evaluator = test_evaluator
        self.updates_counter = 0
        self.wandb_runner=wandb_runner

        model_params = list(self.graph_classifier.parameters())

        print('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params['optimizer'] == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params['lr'], momentum=params['momentum'],
                                       weight_decay=self.params['l2'])
        if params['optimizer'] == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params['lr'], weight_decay=self.params['l2'])

        self.criterion = nn.BCELoss()
        self.reset_training_state()

        self.all_train_loss = []
        self.all_valid_loss = []
        self.all_test_loss = []

        self.all_train_auc = []
        self.all_valid_auc = []
        self.all_test_auc = []

        self.all_train_aupr = []
        self.all_valid_aupr = []
        self.all_test_aupr = []

        self.all_train_f1_score = []
        self.all_valid_f1_score = []
        self.all_test_f1_score = []

        self.best_test_result = {}

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def load_model(self):
        self.graph_classifier.load_state_dict(torch.load("my_resnet.pth"))

    def l2_regularization(self, weight):
        l2_loss = []
        for module in self.graph_classifier.modules():
            if type(module) is nn.Linear:
                l2_loss.append((module.weight ** 2).sum() / 2)
        return weight * sum(l2_loss)

    def train_epoch(self):
        total_loss = 0
        all_preds = []
        all_labels = []
        all_preds_scores = []

        train_all_auc = []
        train_all_aupr = []
        train_all_f1 = []

        dataloader = DataLoader(self.train_data, batch_size=self.params['batch_size'], shuffle=True,
                                num_workers=self.params['num_workers'], collate_fn=self.params['collate_fn'])
        self.graph_classifier.train()
        model_params = list(self.graph_classifier.parameters())
        bar = tqdm(enumerate(dataloader))

        self.params['eval_every_iter'] = int(self.train_data.num_graphs_pairs / self.params['batch_size']) + 1
        t_loss=[]
        print('training epoch')
        for b_idx, batch in bar:
            data_pos, r_labels_pos, targets_pos = move_batch_to_device_dgl(batch, self.params['device'])
            self.optimizer.zero_grad()
            score_pos, g_rep = self.graph_classifier(data_pos)

            # BCELoss
            m = nn.Sigmoid()
            score_pos = torch.squeeze(m(score_pos))

            loss_train = self.criterion(score_pos, r_labels_pos)
            model_params = list(self.graph_classifier.parameters())

            l2 = self.l2_regularization(self.params['l2'])
            loss = torch.sum(loss_train) + l2

            loss.backward()
            t_loss.append(loss.cpu().detach().numpy())
            clip_grad_norm_(self.graph_classifier.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()
            self.updates_counter += 1
            # calculate train metric
            bar.set_description('batchs: ' + str(b_idx + 1) + '/'+str(self.params['eval_every_iter'])+' | loss_train: ' + str(np.mean(t_loss)))
            with torch.no_grad():
                if self.wandb_runner is not None:
                    self.wandb_runner.log({
                        'train_loss':loss.item()
                    })
                total_loss += loss.item()
                target = r_labels_pos.to('cpu').numpy().flatten().tolist()
                all_labels += target

                y_preds = score_pos.cpu().flatten().tolist()
                all_preds += y_preds

                pred_scores = [1 if i else 0 for i in (np.asarray(y_preds) >= 0.5)]
                all_preds_scores += pred_scores

                if len(np.unique(target)) != 2:
                    train_auc = 0.5
                    train_aupr = 0.5
                    train_f1 = 0.5
                else:
                    train_auc = metrics.roc_auc_score(target, y_preds)
                    p, r, t = metrics.precision_recall_curve(target, y_preds)
                    train_aupr = metrics.auc(r, p)

                    train_f1 = metrics.f1_score(target, pred_scores)

                train_all_auc.append(train_auc)
                train_all_aupr.append(train_aupr)
                train_all_f1.append(train_f1)

            # calculate valida and test metric
            if self.updates_counter % self.params['eval_every_iter'] == 0:
                self.make_valid_test()


        weight_norm = sum(map(lambda x: torch.norm(x), model_params))

        train_loss = total_loss / b_idx
        train_auc = np.mean(train_all_auc)
        train_aupr = np.mean(train_all_aupr)
        train_f1 = np.mean(train_all_f1)

        return train_loss, train_auc, train_aupr, train_f1, weight_norm

    def train(self):
        self.reset_training_state()

        all_train_loss = []
        all_train_auc = []
        all_train_aupr = []
        all_train_f1_score = []

        for epoch in range(1, self.params['num_epochs'] + 1):
            time_start = time.time()
            print(f'epochs {epoch}')
            train_loss, train_auc, train_aupr, train_f1, weight_norm = self.train_epoch()
            all_train_loss.append(train_loss)
            all_train_auc.append(train_auc)
            all_train_aupr.append(train_aupr)
            all_train_f1_score.append(train_f1)

            self.all_train_loss.append(train_loss)
            self.all_train_auc.append(train_auc)
            self.all_train_aupr.append(train_aupr)
            self.all_train_f1_score.append(train_f1)

            time_elapsed = time.time() - time_start
            print(f'Epoch {epoch} with loss: {train_loss}, training auc: {train_auc}, training aupr: {train_aupr}, weight_norm: {weight_norm} in {time_elapsed}')

            np.save('../data/precessed_data/pilsl_database/ke_embed.npy', self.graph_classifier.gnn.embed.cpu().tolist())
            # early stop
            if self.not_improved_count > self.params['early_stop']:
                print('EARLY STOP HAPPEN!')
                break

        _, _, score_mat = self.test_evaluator.eval()
        self.score_mat = score_mat

        re = [self.all_train_loss, self.all_valid_loss, self.all_test_loss, self.all_train_auc, self.all_valid_auc,
              self.all_test_auc, self.all_train_aupr, self.all_valid_aupr, self.all_test_aupr, self.all_train_f1_score,
              self.all_valid_f1_score, self.all_test_f1_score]
        now = time.strftime("%Y-%m-%d-%H_%M", time.localtime(time.time()))
        np.save(os.path.join(self.params['exp_dir'], now + 'result.npy'), np.array(re))

    def make_valid_test(self):
        tic = time.time()
        test_result, test_reps, _ = self.test_evaluator.eval()
        
        print('\033[93m Test Performance:' + str(test_result) + ' in ' + str(time.time() - tic) + '\033[0m')
        if test_result['auc'] >= self.best_metric:
            # self.save_classifier()
            self.best_metric = test_result['auc']
            self.not_improved_count = 0
            self.best_test_result = test_result
            # self.save_representation(test_reps)

            print('\033[93m Test Performance Per Class:' + str(test_result) + 'in ' + str(time.time() - tic) + '\033[0m')
        else:
            self.not_improved_count += 1
            if self.not_improved_count > self.params['early_stop']:
                print(f"Validation performance didn\'t improve for {self.params['early_stop']} epochs. Training stops.")
                print('\033[93m Test Performance Per Class:' + str(self.best_test_result) + 'in ' + str(time.time() - tic) + '\033[0m')

        self.last_metric = test_result['auc']

        test_loss, test_auc, test_aupr, test_f1_score = test_result['loss'], test_result['auc'], test_result['aupr'], test_result['f1_score']


        self.all_test_loss.append(test_loss)
        self.all_test_auc.append(test_auc)
        self.all_test_aupr.append(test_aupr)
        self.all_test_f1_score.append(test_f1_score)
        # self.score_mat = score_mat

    def case_study(self):
        self.reset_training_state()
        self.test_evaluator.print_result(self.params['exp_dir'])

    def save_classifier(self):
        torch.save(self.graph_classifier, os.path.join(self.params['exp_dir'], 'best_graph_classifier.pth'))
        print('Better models found w.r.t accuracy. Saved it!')

    def save_representation(self, best_reps):
        np.savetxt(os.path.join(self.params['exp_dir'], 'pair_representation.csv'), best_reps[0], delimiter='\t')
        np.savetxt(os.path.join(self.params['exp_dir'], 'pair_pred_label.csv'), best_reps[1], delimiter='\t')
        np.savetxt(os.path.join(self.params['exp_dir'], 'pair_true_label.csv'), best_reps[2], delimiter='\t')


class Evaluator():
    def __init__(self, params, graph_classifier, data, used_pairs=None):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data
        self.used_pairs = used_pairs

    def print_result(self, exp_dir):
        y_preds = []
        targets = []
        pred_labels = []
        all_loss = 0

        all_idx = []
        all_edges = []
        all_edges_w = []
        g_reps = []

        dataloader = DataLoader(self.data, batch_size=self.params['batch_size'], shuffle=False,
                                num_workers=self.params['num_workers'], collate_fn=self.params['collate_fn'])
        self.graph_classifier.eval()
        new_bar = tqdm(enumerate(dataloader))
        tt_batch = len(new_bar)+1
        with torch.no_grad():
            for b_idx, batch in new_bar:
                
                data_pos, r_labels_pos, targets_pos = move_batch_to_device_dgl(batch, self.params['device'])
                output, g_rep = self.graph_classifier(data_pos)

                # Save pair-wise representation
                g_reps += g_rep.cpu().tolist()
                m = nn.Sigmoid()
                log = torch.squeeze(m(output))

                criterion = nn.BCELoss(reduce=False)
                loss_eval = criterion(log, r_labels_pos)
                loss = torch.sum(loss_eval)

                all_loss += loss.cpu().detach().numpy().item() / len(r_labels_pos)

                target = r_labels_pos.to('cpu').numpy().flatten().tolist()
                targets += target

                y_pred = output.cpu().flatten().tolist()
                y_preds += y_pred

                pred_label = [1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)]
                pred_labels += pred_label

                batch_graph = dgl.unbatch(data_pos)
                
                new_bar.set_description('test batchs: ' + str(b_idx + 1) + '/'+str(tt_batch)+' | loss_train: ' + str(np.mean(all_loss)))

                for g in batch_graph:
                    idx = g.ndata['idx'].to('cpu').numpy()
                    edges = g.edges()
                    edges_detach = (edges[0].cpu().numpy(), edges[1].cpu().numpy())
                    edges_w = g.edata['a'].to('cpu').numpy().reshape(1, -1)[0]

                    all_idx.append(idx)
                    all_edges.append(edges_detach)
                    all_edges_w.append(edges_w)

    def eval(self, all_pair=None):
        y_pred = []
        y_preds = []
        targets = []
        pred_labels = []
        all_auc = []
        all_aupr = []
        all_f1 = []
        all_loss = 0
        g_reps = []

        score_mat = None

        if all_pair == 'all_pair':
            dataloader = DataLoader(self.data, batch_size=4922, shuffle=False,
                                    num_workers=self.params['num_workers'], collate_fn=self.params['collate_fn'])
            self.graph_classifier.eval()
            new_bar = tqdm(enumerate(dataloader))
            tt_batch = len(new_bar)+1
            with torch.no_grad():
                for b_idx, batch in new_bar:
                    data_pos, r_labels_pos, targets_pos = move_batch_to_device_dgl(batch, self.params['device'])
                    output, g_rep = self.graph_classifier(data_pos)

                    y_pred = output.cpu().flatten().tolist()
                    y_preds += y_pred
                    
                    new_bar.set_description('test batchs: ' + str(b_idx + 1) + '/'+str(tt_batch))

                score_mat = sp.csr_matrix((y_preds, (self.used_pairs[:, 0], self.used_pairs[:, 1])), shape=(9845,9845))
        else:
            dataloader = DataLoader(self.data, batch_size=self.params['batch_size'], shuffle=False,
                                    num_workers=self.params['num_workers'], collate_fn=self.params['collate_fn'])
            self.graph_classifier.eval()
            new_bar = tqdm(enumerate(dataloader))
            tt_batch = int(self.data.num_graphs_pairs / self.params['batch_size']) + 1
            # tt_batch = len(dataloader)+1
            with torch.no_grad():
                for b_idx, batch in new_bar:
                    data_pos, r_labels_pos, targets_pos = move_batch_to_device_dgl(batch, self.params['device'])
                    output, g_rep = self.graph_classifier(data_pos)
                    # Save pair-wise representation
                    g_reps += g_rep.cpu().tolist()
                    m = nn.Sigmoid()
                    log = torch.squeeze(m(output))

                    criterion = nn.BCELoss(reduce = False)
                    loss_eval = criterion(log, r_labels_pos)
                    loss = torch.sum(loss_eval)

                    all_loss += loss.cpu().detach().numpy().item() / len(r_labels_pos)

                    target = r_labels_pos.to('cpu').numpy().flatten().tolist()
                    targets += target

                    y_pred = output.cpu().flatten().tolist()
                    y_preds += y_pred
                    
                    new_bar.set_description('test batchs: ' + str(b_idx + 1) + '/'+str(tt_batch)+' | loss_train: ' + str(all_loss/tt_batch))

                auc_ = metrics.roc_auc_score(targets, y_preds)
                p, r, t = metrics.precision_recall_curve(targets, y_preds)
                aupr = metrics.auc(r, p)

                pred_label = [1 if i else 0 for i in (np.asarray(y_preds) >= 0.5)]
                # pred_labels += pred_label

                f1 = metrics.f1_score(targets, pred_label)
                all_auc.append(auc_)
                all_aupr.append(aupr)
                all_f1.append(f1)

                score_mat = sp.csr_matrix((y_preds, (self.used_pairs[:, 0], self.used_pairs[:, 1])), shape=(9845,9845))


        return {'loss': all_loss / b_idx, 'auc': np.mean(all_auc), 'aupr': np.mean(all_aupr),
                'f1_score': np.mean(all_f1)}, (g_reps, pred_labels, targets), score_mat


class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    # triple file (KG)
    # entity_type file

    def __init__(self, db_path, db_name, raw_data_paths, used_pairs, included_relations=None, add_traspose_rels=False,
                 use_kge_embeddings=False, dataset='', kge_model='', file_name='', \
                 ssp_graph=None, relation2id=None, id2entity=None, id2relation=None, rel=None, graph=None,
                 morgan_feat=None):

        self.used_pairs = used_pairs
        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db_name = self.main_env.open_db(db_name.encode())
        self.node_features, self.kge_entity2id = (None, None)
        # self.file_name = file_name
        triple_file = '../data/precessed_data/fin_kg_wo_sl_9845.csv'
        self.entity_type = np.loadtxt('../data/precessed_data/pilsl_unified_etype_9845.csv')

        self.pair_to_dbindex = np.load('../data/precessed_data/pilsl_pair_to_dbindex.npy', allow_pickle=True).item()

        if not ssp_graph:
            if os.path.exists('../data/precessed_data/pilsl_database/processed_file.npy'):
                ssp_graph, id2entity, id2relation, rel = np.load('../data/precessed_data/pilsl_database/processed_file.npy',allow_pickle=True)
            else:
                ssp_graph, __, __, __, id2entity, id2relation, rel = process_files(raw_data_paths, triple_file,
                                                                               included_relations)
                np.save('../data/precessed_data/pilsl_database/processed_file.npy',[ssp_graph, id2entity, id2relation, rel])

            self.aug_num_rels = len(ssp_graph)
            # self.graph = ssp_multigraph_to_dgl(ssp_graph)
            if os.path.exists('../data/precessed_data/pilsl_database/pilsl_ssp_graph.bin'):
                loading_graph = load_graphs('../data/precessed_data/pilsl_database/pilsl_ssp_graph.bin')[0][0]
                spm_graph = loading_graph.adjacency_matrix_scipy()
                self.graph = dgl.DGLGraph()
                self.graph.from_scipy_sparse_matrix(spm_graph)
                self.graph.edata['type'] = loading_graph.edata['type']
                # self.graph._graph.is_readonly = False
            else:
                self.graph = ssp_multigraph_to_dgl(ssp_graph)
                save_graphs('../data/precessed_data/pilsl_database/pilsl_ssp_graph.bin', self.graph)
            self.ssp_graph = ssp_graph
        else:
            self.aug_num_rels = len(ssp_graph)
            self.graph = graph
            self.ssp_graph = ssp_graph

        self.num_rels = rel
        self.id2entity = id2entity
        self.id2relation = id2relation

        self.max_n_label = np.array([0, 0])
        with self.main_env.begin() as txn:
            self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
            self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')

        print(f"Max distance from sub : {self.max_n_label[0]}, Max distance from obj : {self.max_n_label[1]}")
        self.num_graphs_pairs = self.used_pairs.shape[0]
        # with self.main_env.begin(db=self.db_name) as txn:
            # self.num_graphs_pairs = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        self.__getitem__(0)

    def __getitem__(self, index):
        pair = self.used_pairs[index, :]
        id_indb = self.pair_to_dbindex[pair[0]][pair[1]]
        # id_indb = index
        with self.main_env.begin(db=self.db_name) as txn:
            str_id = '{:08}'.format(id_indb).encode('ascii')
            nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id)).values()
            subgraph_pos = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos)

        return subgraph_pos, g_label_pos, r_label_pos

    def __len__(self):
        return self.num_graphs_pairs

    def _prepare_subgraphs(self, nodes, r_label, n_labels):

        subgraph = dgl.DGLGraph(self.graph.subgraph(nodes))
        # print(subgraph, subgraph.nodes(), subgraph.ndata, subgraph.edges(), subgraph.edata)
        subgraph.edata['type'] = self.graph.edata['type'][self.graph.subgraph(nodes).parent_eid]

        subgraph.ndata['idx'] = torch.LongTensor(np.array(nodes))
        subgraph.ndata['ntype'] = torch.LongTensor(self.entity_type[nodes])
        subgraph.ndata['mask'] = torch.LongTensor(np.where(self.entity_type[nodes] == 1, 1, 0))
        try:
            edges_btw_roots = subgraph.edge_id(0, 1)
            rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)
        except AssertionError:
            pass

        kge_nodes = [self.kge_entity2id[self.id2entity[n]] for n in nodes] if self.kge_entity2id else None
        n_feats = self.node_features[kge_nodes] if self.node_features is not None else None
        subgraph = self._prepare_features_new(subgraph, n_labels, n_feats)
        # remove interaction
        try:
            edges_btw_roots = subgraph.edge_id(0, 1)
            subgraph.remove_edges(edges_btw_roots)
        except AssertionError:
            pass
        return subgraph  # , torch.LongTensor([head_idx, tail_idx])

    def _prepare_features(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1))
        label_feats[np.arange(n_nodes), n_labels] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph

    def _prepare_features_new(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        self.n_feat_dim = n_feats.shape[1]
        return subgraph



class Aggregator(nn.Module):
    def __init__(self, emb_dim):
        super(Aggregator, self).__init__()

    def forward(self, node):
        # print(node.mailbox['curr_emb'])
        curr_emb = node.mailbox['curr_emb'][:, 0, :]  # (B, F)
        nei_msg = torch.bmm(node.mailbox['alpha'].transpose(1, 2), node.mailbox['msg']).squeeze(1)  # (B, F)

        # curr_emb = node.mailbox['curr_emb']
        # nei_msg = node.mailbox['alpha'] *  node.mailbox['msg']  # (B, F)
        # nei_msg, _ = torch.max(node.mailbox['msg'], 1)  # (B, F)

        new_emb = self.update_embedding(curr_emb, nei_msg)

        return {'h': new_emb}

    @abc.abstractmethod
    def update_embedding(curr_emb, nei_msg):
        raise NotImplementedError


class SumAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(SumAggregator, self).__init__(emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = nei_msg + curr_emb

        return new_emb


class MLPAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(MLPAggregator, self).__init__(emb_dim)
        self.linear = nn.Linear(2 * emb_dim, emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        inp = torch.cat((nei_msg, curr_emb), 1)
        new_emb = F.relu(self.linear(inp))

        return new_emb


class GRUAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(GRUAggregator, self).__init__(emb_dim)
        self.gru = nn.GRUCell(emb_dim, emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = self.gru(nei_msg, curr_emb)

        return new_emb


"""
Class based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class GraphClassifier(nn.Module):
    def __init__(self, params):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.dropout = nn.Dropout(p=params['dropout'])
        self.relu = nn.ReLU()
        self.train_rels = params['train_rels']
        self.relations = params['num_rels']
        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)

        # MLP
        self.mp_layer1 = nn.Linear(self.params['feat_dim'], 256)
        self.mp_layer2 = nn.Linear(256, self.params['emb_dim'])
        self.bn1 = nn.BatchNorm1d(256)

        # Decoder
        if self.params['add_ht_emb'] and self.params['add_sb_emb']:
            if self.params['add_feat_emb'] and self.params['add_transe_emb']:
                self.fc_layer = nn.Linear(3 * (1 + self.params['num_gcn_layers']) * (
                            self.params['emb_dim'] + self.params['inp_dim']) + 2 * self.params['emb_dim'], 512)
            elif self.params['add_feat_emb']:
                self.fc_layer = nn.Linear(
                    3 * (self.params['num_gcn_layers']) * self.params['emb_dim'] + 2 * self.params['emb_dim'], 512)
            else:
                self.fc_layer = nn.Linear(
                    3 * (1 + self.params['num_gcn_layers']) * (self.params['emb_dim'] + self.params['inp_dim']), 512)
        elif self.params['add_ht_emb']:
            self.fc_layer = nn.Linear(2 * (1 + self.params['num_gcn_layers']) * self.params['emb_dim'], 512)
        else:
            self.fc_layer = nn.Linear(self.params['num_gcn_layers'] * self.params['emb_dim'], 512)
        self.fc_layer_1 = nn.Linear(512, 128)
        self.fc_layer_2 = nn.Linear(128, 1)

    def omics_feat(self, emb):
        self.genefeat = emb

    def get_omics_features(self, ids):
        a = []
        for i in ids:
            a.append(self.genefeat[i.cpu().numpy().item()])
        return np.array(a)

    def forward(self, data):
        g = data
        g.ndata['h'] = self.gnn(g)
        g_out = dgl.mean_nodes(g, 'repr')

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]

        # head_feat = torch.FloatTensor(self.get_omics_features(g.ndata['idx'][head_ids])).cuda()
        head_feat = torch.FloatTensor(self.get_omics_features(g.ndata['idx'][head_ids])).to(self.params['device'])
        # tail_feat = torch.FloatTensor(self.get_omics_features(g.ndata['idx'][tail_ids])).cuda()
        tail_feat = torch.FloatTensor(self.get_omics_features(g.ndata['idx'][tail_ids])).to(self.params['device'])

        if self.params['add_feat_emb']:
            fuse_feat1 = self.mp_layer2(self.bn1(self.relu(self.mp_layer1(head_feat))))
            fuse_feat2 = self.mp_layer2(self.bn1(self.relu(self.mp_layer1(tail_feat))))
            fuse_feat = torch.cat([fuse_feat1, fuse_feat2], dim=1)

        if self.params['add_ht_emb'] and self.params['add_sb_emb']:
            if self.params['add_feat_emb'] and self.params['add_transe_emb']:
                g_rep = torch.cat(
                    [g_out.view(-1, (1 + self.params['num_gcn_layers']) * (self.params['emb_dim'] + self.params['inp_dim'])),
                     head_embs.view(-1, (1 + self.params['num_gcn_layers']) * (self.params['emb_dim'] + self.params['inp_dim'])),
                     tail_embs.view(-1, (1 + self.params['num_gcn_layers']) * (self.params['emb_dim'] + self.params['inp_dim'])),
                     fuse_feat.view(-1, 2 * self.params['emb_dim'])
                     ], dim=1)
            elif self.params['add_feat_emb']:
                g_rep = torch.cat([g_out.view(-1, (self.params['num_gcn_layers']) * self.params['emb_dim']),
                                   head_embs.view(-1, (self.params['num_gcn_layers']) * self.params['emb_dim']),
                                   tail_embs.view(-1, (self.params['num_gcn_layers']) * self.params['emb_dim']),
                                   fuse_feat.view(-1, 2 * self.params['emb_dim'])
                                   ], dim=1)
            else:
                g_rep = torch.cat(
                    [g_out.view(-1, (1 + self.params['num_gcn_layers']) * (self.params['emb_dim'] + self.params['inp_dim'])),
                     head_embs.view(-1, (1 + self.params['num_gcn_layers']) * (self.params['emb_dim'] + self.params['inp_dim'])),
                     tail_embs.view(-1, (1 + self.params['num_gcn_layers']) * (self.params['emb_dim'] + self.params['inp_dim'])),
                     # fuse_feat.view(-1, 2*self.params.emb_dim)
                     ], dim=1)

        elif self.params['add_ht_emb']:
            g_rep = torch.cat([
                head_embs.view(-1, (1 + self.params['num_gcn_layers']) * self.params['emb_dim']),
                tail_embs.view(-1, (1 + self.params['num_gcn_layers']) * self.params['emb_dim'])
            ], dim=1)
        else:
            g_rep = g_out.view(-1, self.params['num_gcn_layers'] * self.params['emb_dim'])

        output = self.fc_layer_2(self.relu(self.fc_layer_1(self.relu(self.fc_layer(self.dropout(g_rep))))))

        return (output, g_rep)


"""
Class based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class RGCN(nn.Module):
    def __init__(self, params):
        super(RGCN, self).__init__()

        self.max_label_value = params['max_label_value']
        self.inp_dim = params['inp_dim']
        self.emb_dim = params['emb_dim']
        self.attn_rel_emb_dim = params['attn_rel_emb_dim']
        self.num_rels = params['num_rels']
        self.aug_num_rels = params['aug_num_rels']
        self.num_bases = params['num_bases']
        self.num_hidden_layers = params['num_gcn_layers']
        self.dropout = params['dropout']
        self.edge_dropout = params['edge_dropout']
        self.has_attn = params['has_attn']
        self.num_nodes = params['num_nodes']
        self.device = params['device']
        self.add_transe_emb = params['add_transe_emb']

        if self.has_attn:
            self.attn_rel_emb = nn.Embedding(self.aug_num_rels, self.attn_rel_emb_dim, sparse=False)
        else:
            self.attn_rel_emb = None

        # to incorporate the KG embeddings, you need to modify the code here and insert the KG embeddings
        if params['use_kge_embeddings']:
            kg_embed = np.load('data/SynLethKG/kg_embedding/kg_TransE_l2_entity.npy')
            # self.embed = torch.FloatTensor(kg_embed).cuda()
            self.embed = torch.FloatTensor(kg_embed).to(params['device'])
        else:
            self.embed = nn.Parameter(torch.Tensor(self.num_nodes, self.emb_dim), requires_grad=True)
            nn.init.xavier_uniform_(self.embed,
                                    gain=nn.init.calculate_gain('relu'))

        # initialize aggregators for input and hidden layers
        if params['gnn_agg_type'] == "sum":
            self.aggregator = SumAggregator(self.emb_dim)
        elif params['gnn_agg_type'] == "mlp":
            self.aggregator = MLPAggregator(self.emb_dim)
        elif params['gnn_agg_type'] == "gru":
            self.aggregator = GRUAggregator(self.emb_dim)

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers - 1):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)

    def build_input_layer(self):
        return RGCNBasisLayer(self.inp_dim + self.emb_dim,
                         self.inp_dim + self.emb_dim,
                         self.aggregator,
                         self.attn_rel_emb_dim,
                         self.aug_num_rels,
                         self.num_bases,
                         embed=self.embed,
                         num_nodes=self.num_nodes,
                         activation=F.relu,
                         dropout=self.dropout,
                         edge_dropout=self.edge_dropout,
                         is_input_layer=True,
                         has_attn=self.has_attn,
                         add_transe_emb=self.add_transe_emb,
                         one_attn=True)

    def build_hidden_layer(self, idx):
        return RGCNBasisLayer(
            self.inp_dim + self.emb_dim,
            self.inp_dim + self.emb_dim,
            self.aggregator,
            self.attn_rel_emb_dim,
            self.aug_num_rels,
            self.num_bases,
            embed=self.embed,
            activation=F.relu,
            dropout=self.dropout,
            edge_dropout=self.edge_dropout,
            has_attn=self.has_attn,
            add_transe_emb=self.add_transe_emb,
            one_attn=True)

    def forward(self, g):
        for layer in self.layers:
            layer(g, self.attn_rel_emb)
        return g.ndata.pop('h')


"""
Class baseed off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.
    (Identity has already been supported by PyTorch 1.2, we will directly
    import torch.nn.Identity in the future)
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x


class RGCNLayer(nn.Module):
    def __init__(self, inp_dim, out_dim, aggregator, bias=None, activation=None, num_nodes=122343142, dropout=0.0,
                 edge_dropout=0.0, is_input_layer=False, embed=False, add_transe_emb=True):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.num_nodes = num_nodes
        self.out_dim = out_dim
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))
        self.add_transe_emb = add_transe_emb
        self.aggregator = aggregator
        self.is_input_layer = is_input_layer
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if edge_dropout:
            self.edge_dropout = nn.Dropout(edge_dropout)
        else:
            self.edge_dropout = Identity()
        # if is_input_layer:
        if embed is not None:
            self.embed = embed

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g, attn_rel_emb=None):
        # message passing
        self.propagate(g, attn_rel_emb)

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout:
            node_repr = self.dropout(node_repr)

        g.ndata['h'] = node_repr

        if self.is_input_layer and self.add_transe_emb:
            # x = torch.cat([self.embed[g.ndata['idx']] , g.ndata['h']], dim = 1)
            init = torch.cat([g.ndata['feat'], self.embed[g.ndata['idx']]], dim=1)
            x = torch.cat([init, g.ndata['h']], dim=1)

            g.ndata['repr'] = x.unsqueeze(1).reshape(-1, 2, self.out_dim)
            # print(x.shape, g.ndata['repr'].shape)
        elif self.is_input_layer:
            g.ndata['repr'] = g.ndata['h'].unsqueeze(1)
        else:
            g.ndata['repr'] = torch.cat([g.ndata['repr'], g.ndata['h'].unsqueeze(1)], dim=1)


class RGCNBasisLayer(RGCNLayer):
    def __init__(self, inp_dim, out_dim, aggregator, attn_rel_emb_dim, num_rels, num_bases=-1, num_nodes=12345342,
                 bias=None,
                 activation=None, dropout=0.0, edge_dropout=0.0, is_input_layer=False, has_attn=False, embed=None,
                 add_transe_emb=True,
                 one_attn=False):
        super(
            RGCNBasisLayer,
            self).__init__(
            inp_dim,
            out_dim,
            aggregator,
            bias,
            activation,
            num_nodes=num_nodes,
            dropout=dropout,
            edge_dropout=edge_dropout,
            is_input_layer=is_input_layer,
            embed=embed,
            add_transe_emb=add_transe_emb)
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.attn_rel_emb_dim = attn_rel_emb_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        self.has_attn = has_attn
        self.num_nodes = num_nodes
        self.add_transe_emb = add_transe_emb
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # add basis weights
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.out_dim))
        self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
        self.one_attn = one_attn

        # print(self.weight.shape)
        if self.has_attn:
            self.A = nn.Linear(2 * self.inp_dim + self.attn_rel_emb_dim, self.inp_dim)
            self.B = nn.Linear(self.inp_dim, 1)

        self.self_loop_weight = nn.Parameter(torch.Tensor(self.inp_dim, self.out_dim))

        nn.init.xavier_uniform_(self.self_loop_weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

    def propagate(self, g, attn_rel_emb=None, nonKG=True):
        # generate all weights from bases
        weight = self.weight.view(self.num_bases, self.inp_dim * self.out_dim)
        weight = torch.matmul(self.w_comp, weight).view(self.num_rels, self.inp_dim, self.out_dim)

        g.edata['w'] = self.edge_dropout(torch.ones(g.number_of_edges(), 1).to(weight.device))

        input_ = 'feat' if self.is_input_layer else 'h'

        def msg_func(edges):
            w = weight.index_select(0, edges.data['type'])

            if input_ == 'feat' and self.add_transe_emb:
                x = torch.cat([edges.src[input_], self.embed[edges.src['idx']]], dim=1)
                y = torch.cat([edges.dst[input_], self.embed[edges.dst['idx']]], dim=1)
            else:
                x = edges.src[input_]
                y = edges.dst[input_]

            msg = edges.data['w'] * torch.bmm(x.unsqueeze(1), w).squeeze(1)
            curr_emb = torch.mm(y, self.self_loop_weight)  # (B, F)

            # attention
            if self.has_attn:
                e = torch.cat([x, y, attn_rel_emb(edges.data['type'])], dim=1)
                a = torch.sigmoid(self.B(F.relu(self.A(e))))
            else:
                # a = torch.ones((len(edges), 1)).cuda()
                a = torch.ones((len(edges), 1)).to(device=w.device)

            edges.data['a'] = a

            return {'curr_emb': curr_emb, 'msg': msg, 'alpha': a}

        g.update_all(msg_func, self.aggregator, None)

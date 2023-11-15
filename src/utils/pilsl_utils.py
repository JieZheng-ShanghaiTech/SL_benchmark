import time

import lmdb
import logging
import struct
import random
import torch
import dgl
import os
import json
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import pandas as pd
import multiprocessing as mp
import networkx as nx

from tqdm import tqdm,trange

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "2"

random.seed(456)
np.random.seed(456)
# tf.compat.v1.set_random_seed(123)

torch.manual_seed(456)
torch.cuda.manual_seed_all(456)

def generate_subgraph_datasets(params, splits=['train'], saved_relation2id=None, max_label_value=None):
    print('loading data')
    triple_file = '../data/precessed_data/fin_kg_wo_sl_9845.csv'  # raw kg triple
    print('load finish')
    # triple_file = 'data/SynLethKG/kg2id.txt'  # raw kg triple

    adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel = process_files(params['file_paths'], triple_file, saved_relation2id)
    graphs = {}
    # max_links = 250000
    for split_name in splits:
        graphs[split_name] = {'triplets': triplets[split_name], 'max_size': params['max_links']}

    '''
    graphs={train:{triplets:,max_size:,pairs:},test:{...},dev:{...}}
    '''

    for split_name, split in graphs.items():
        split['pairs'] = split['triplets']
        split['pos'], split['neg'] = split['triplets'], []
    print('start build sub graph ...')
    links2subgraphs(adj_list, graphs, params, max_label_value)
    print('done')


def process_files(files, saved_relation2id=None):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id

    triplets = {}

    ent = 0
    rel = 0

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if not saved_relation2id and triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(sp.csc_matrix((np.ones(len(idx), dtype=np.uint8),
                                    (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))),
                                   shape=(len(entity2id), len(entity2id))))

    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel


def process_files(files, triple_file, saved_relation2id=None, keeptrainone=False):
    # files gene gene interaction， gene,  gene, label
    # triple_file is the biomedical h, r, t (not contain this SL interaction)
    # relation2id is to count the relation in KG

    '''
    triple_file     kg biomedical h, r, t (not contain this SL interaction)
    '''
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id

    triplets = {}
    kg_triple = []
    ent = 0
    rel = 0

    # load train/valid/test data
    for file_type, file_path in files.items():
        data = []
        file_data = np.load(file_path)
        # file_data = np.loadtxt(file_path)
        for triplet in file_data:
            # print(triplet)
            triplet[0], triplet[1], triplet[2] = int(triplet[0]), int(triplet[1]), int(triplet[2])
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = triplet[0]
                # ent += 1
            if triplet[1] not in entity2id:
                entity2id[triplet[1]] = triplet[1]
                # ent += 1
            if not saved_relation2id and triplet[2] not in relation2id:
                if keeptrainone:
                    triplet[2] = 0
                    relation2id[triplet[2]] = 0
                    rel = 1
                else:
                    relation2id[triplet[2]] = triplet[2]
                    rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[2] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[1]], relation2id[triplet[2]]])

        triplets[file_type] = np.array(data)
    # load kg
    triplet_kg = pd.read_csv(triple_file).values
    # triplet_kg = np.loadtxt(triple_file)
    print(np.max(triplet_kg[:, -2]))
    for (h, t, r) in triplet_kg:
        h, t, r = int(h), int(r), int(t)
        if h not in entity2id:
            entity2id[h] = h
        if t not in entity2id:
            entity2id[t] = t
        if not saved_relation2id and rel + r not in relation2id:
            relation2id[rel + r] = rel + r
        kg_triple.append([h, t, r])
    kg_triple = np.array(kg_triple)
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(rel):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(sp.csc_matrix((np.ones(len(idx), dtype=np.uint8),
                                    (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))),
                                   shape=(len(entity2id), len(entity2id))))
    for i in range(rel, len(relation2id)):
        idx = np.argwhere(kg_triple[:, 2] == i - rel)
        adj_list.append(sp.csc_matrix(
            (np.ones(len(idx), dtype=np.uint8), (kg_triple[:, 0][idx].squeeze(1), kg_triple[:, 1][idx].squeeze(1))),
            shape=(len(entity2id), len(entity2id))))

    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel

def intialize_worker_feng(A, params, max_label_value,gene_neib):
    global A_, params_, max_label_value_,gene_neib_
    A_, params_, max_label_value_,gene_neib_ = A, params, max_label_value,gene_neib

def intialize_worker(A, params, max_label_value, three_hop_nb_mat):
    global A_, params_, max_label_value_, three_hop_nb_mat_
    A_, params_, max_label_value_, three_hop_nb_mat_ = A, params, max_label_value, three_hop_nb_mat

def extract_save_subgraph(args_):
    idx, (n1, n2, r_label), g_label = args_
    # id,h,t,r,1                                                                                                            #  3          True                   200
    nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A_, three_hop_nb_mat_ ,params_['hop'], params_['enclosing_sub_graph'], params_['max_nodes_per_hop'])

    # max_label_value_ is to set the maximum possible value of node label while doing double-radius labelling.
    if max_label_value_ is not None:
        n_labels = np.array([np.minimum(label, max_label_value_).tolist() for label in n_labels])

    datum = {'nodes': nodes, 'r_label': r_label, 'g_label': g_label, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
    str_id = '{:08}'.format(idx).encode('ascii')

    return (str_id, datum)

def extract_save_subgraph_feng(args_):

    # idx, (n1, n2, r_label), g_label = args_
    #
    # id_datum_dict=dict()
    tic=time.time()
    idx, (n1, n2, r_label), g_label = args_
    # id,h,t,r,1                                                                                                            #  3          True                   200
    nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling_feng((n1, n2), r_label, A_, params_['hop'], params_['enclosing_sub_graph'], params_['max_nodes_per_hop'])

    # max_label_value_ is to set the maximum possible value of node label while doing double-radius labelling.
    if max_label_value_ is not None:
        n_labels = np.array([np.minimum(label, max_label_value_).tolist() for label in n_labels])

    datum = {'nodes': nodes, 'r_label': r_label, 'g_label': g_label, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
    str_id = '{:08}'.format(idx).encode('ascii')
    toc=time.time()
    # print(f'extract_save_subgraph_feng {toc-tic}s')
    return (str_id, datum)



def links2subgraphs(A, graphs, params, max_label_value=None):
    '''
    A 所有关系对应的稀疏矩阵
    extract enclosing subgraphs, write map mode + named dbs
    '''
    max_n_label = {'value': np.array([0, 0])}
    subgraph_sizes = []
    enc_ratios = []
    num_pruned_nodes = []

    BYTES_PER_DATUM = 1000
    # BYTES_PER_DATUM = get_average_subgraph_size(100, list(graphs.values())[0]['pairs'], A, three_hop_nb_mat,params) * 1.5
    # print(f'BYTES_PER_DATUM done {BYTES_PER_DATUM}')
    links_length = 0
    for split_name, split in graphs.items():
        links_length += len(split['pairs']) * 2
    map_size = links_length * BYTES_PER_DATUM

    env = lmdb.open(params['db_path'], map_size=map_size, max_dbs=6)

    def extraction_helper(A, links, g_labels, split_env, split_name):

        # A, split['pairs'], labels, split_env, split_name
        flag = 0
        pairs = []
        # intialize_worker_feng(A, params, max_label_value, three_hop_nb_mat, three_hop_dict)
        print(f'{len(links)} subgraph to be build')
        gene_neib=np.load('../data/precessed_data/pilsl_3hop_neib.npy', allow_pickle=True).item()
        # args_ = zip(range(len(links)), links, g_labels)
        # datum_dict=dict()
        # extract the node subgraph
        with mp.Pool(processes=80, initializer=intialize_worker_feng, initargs=(A, params, max_label_value, gene_neib)) as p:
        # with mp.Pool(processes=None, initializer=intialize_worker, initargs=(A, params, max_label_value)) as p:
            args_ = zip(range(len(links)), links, g_labels)
            for (idx, datum) in tqdm(p.imap_unordered(extract_save_subgraph_feng, args_), total=len(links)):
                # check the subgraph nodes numbers
                nodes = datum['nodes']
                if len(nodes):
                    max_n_label['value'] = np.maximum(np.max(datum['n_labels'], axis=0), max_n_label['value'])
                    subgraph_sizes.append(datum['subgraph_size'])
                    enc_ratios.append(datum['enc_ratio'])
                    num_pruned_nodes.append(datum['num_pruned_nodes'])
                    str_id = '{:08}'.format(flag).encode('ascii')
                    # save pairs
                    pair = nodes[0:2] + [int(datum['r_label'])]
                    pairs.append(pair)
                    # write the data
                    with env.begin(write=True, db=split_env) as txn:
                        txn.put(str_id, serialize(datum))
                    flag += 1

        # write the graph numbers
        with env.begin(write=True, db=split_env) as txn:
            txn.put('num_graphs'.encode(), (flag).to_bytes(int.bit_length(len(links)), byteorder='little'))
        return pairs

    for split_name, split in graphs.items():
        print(f"Extracting enclosing subgraphs for gene pairs in {split_name} set")

        labels = np.ones(len(split['pairs']))
        db_name_pairs = split_name + '_pairs'
        split_env = env.open_db(db_name_pairs.encode())

        A_incidence = incidence_matrix(A)
        A_incidence += A_incidence.T

        # gene_set is the set of the unified kg_id of the genes (kg_id less than about 54000)
        gene_set = list(range(9845))
        if not os.path.exists(f'../data/precessed_data/pilsl_3hop_neib.npy'):
            n_hop_gene_nei_set = {}
            index = 1
            for root_gene in tqdm(gene_set):
                root_gene_nei = get_neighbor_nodes(set([root_gene]), A_incidence, params['hop'], params['max_nodes_per_hop'])
                n_hop_gene_nei_set[root_gene] = root_gene_nei
                # print(index)
                index = index + 1

            np.save(f'../data/precessed_data/pilsl_3hop_neib.npy', n_hop_gene_nei_set)

        print('extraction_helper running')

        pairs = extraction_helper(A, split['pairs'], labels, split_env, split_name)

    max_n_label['value'] = max_label_value if max_label_value is not None else max_n_label['value']
    print('start import to database')
    with env.begin(write=True) as txn:
        bit_len_label_sub = int.bit_length(int(max_n_label['value'][0]))
        bit_len_label_obj = int.bit_length(int(max_n_label['value'][1]))
        txn.put('max_n_label_sub'.encode(),
                (int(max_n_label['value'][0])).to_bytes(bit_len_label_sub, byteorder='little'))
        txn.put('max_n_label_obj'.encode(),
                (int(max_n_label['value'][1])).to_bytes(bit_len_label_obj, byteorder='little'))

        txn.put('avg_subgraph_size'.encode(), struct.pack('f', float(np.mean(subgraph_sizes))))
        txn.put('min_subgraph_size'.encode(), struct.pack('f', float(np.min(subgraph_sizes))))
        txn.put('max_subgraph_size'.encode(), struct.pack('f', float(np.max(subgraph_sizes))))
        txn.put('std_subgraph_size'.encode(), struct.pack('f', float(np.std(subgraph_sizes))))

        txn.put('avg_enc_ratio'.encode(), struct.pack('f', float(np.mean(enc_ratios))))
        txn.put('min_enc_ratio'.encode(), struct.pack('f', float(np.min(enc_ratios))))
        txn.put('max_enc_ratio'.encode(), struct.pack('f', float(np.max(enc_ratios))))
        txn.put('std_enc_ratio'.encode(), struct.pack('f', float(np.std(enc_ratios))))

        txn.put('avg_num_pruned_nodes'.encode(), struct.pack('f', float(np.mean(num_pruned_nodes))))
        txn.put('min_num_pruned_nodes'.encode(), struct.pack('f', float(np.min(num_pruned_nodes))))
        txn.put('max_num_pruned_nodes'.encode(), struct.pack('f', float(np.max(num_pruned_nodes))))
        txn.put('std_num_pruned_nodes'.encode(), struct.pack('f', float(np.std(num_pruned_nodes))))

def get_average_subgraph_size(sample_size, links, A,three_hop_nb_mat, params):
    # sample_size = 100
    total_size = 0
    # print(links, len(links))
    lst = np.random.choice(len(links), sample_size)
    for idx in lst:
        (n1, n2, r_label) = links[idx]
    # for (n1, n2, r_label) in links[np.random.choice(len(links), sample_size)]:
        nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A, three_hop_nb_mat, params['hop'], params['enclosing_sub_graph'], params['max_nodes_per_hop'])
        datum = {'nodes': nodes, 'r_label': r_label, 'g_label': 0, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
        total_size += len(serialize(datum))
    return total_size / sample_size

def serialize(data):
    data_tuple = tuple(data.values())
    return pkl.dumps(data_tuple)


def incidence_matrix(adj_list):
    '''
    adj_list: List of sparse adjacency matrices
    '''

    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    # remove SL relationship in KG
    flag = 0
    for adj in adj_list:
        if flag > 1:
            adjcoo = adj.tocoo()
            rows += adjcoo.row.tolist()
            cols += adjcoo.col.tolist()
            dats += adjcoo.data.tolist()
        flag += 1
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return sp.csc_matrix((data, (row, col)), shape=dim)

# """All functions in this file are from  dgl.contrib.data.knowledge_graph"""
def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    BFS for graphs.
    Modified from dgl.contrib.data.knowledge_graph to accomodate node sampling
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()
    # np.random.seed(0)
    random.seed(0)
    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)

def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors.
    Directly copied from dgl.contrib.data.knowledge_graph"""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(sp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors

def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return sp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def subgraph_extraction_labeling_feng(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None,
                                 max_node_label_value = None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    # A_incidence 包含所有信息在一个矩阵中
    # three_hop_nb_mat = sp.load_npz('/home/yimiaofeng/PycharmProjects/SLBench/scripts/clean_code/start_from_kg/pilsl_3_hop_spmat.npz')
    # ind, rel=[links[:,0],links[:,1]],links[:,2]
    # unique_ind=set(links[:,0])&set(links[:,1])
    # print(len(unique_ind))
    # tic=time.time()
    # A_incidence = incidence_matrix(A_list)
    # A_incidence += A_incidence.T
    # time.time()-tic

    ind = list(ind)
    ind[0], ind[1] = int(ind[0]), int(ind[1])
    ind = (ind[0], ind[1])
    random.seed(456)
    root1_nei = gene_neib_[ind[0]]
    root2_nei = gene_neib_[ind[1]]
    # root1_nei=set(three_hop_dict[ind[0]])
    # root2_nei=set(three_hop_dict[ind[1]])
    # root1_nei=set(random.sample(three_hop_dict[ind[0]],min(len(three_hop_dict[ind[0]]),max_nodes_per_hop)))
    # root2_nei=set(random.sample(three_hop_dict[ind[1]],min(len(three_hop_dict[ind[1]]),max_nodes_per_hop)))
    # root1_nei = get_neighbor_nodes(set([ind[0]]), three_hop_dict, h, max_nodes_per_hop)
    # root2_nei = get_neighbor_nodes(set([ind[1]]), three_hop_dict, h, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    if enclosing_sub_graph:
        if ind[0] in subgraph_nei_nodes_int:
            subgraph_nei_nodes_int.remove(ind[0])
        if ind[1] in subgraph_nei_nodes_int:
            subgraph_nei_nodes_int.remove(ind[1])
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
        # print('Subgraph neighbor len: ' + str(len(subgraph_nei_nodes_int)))
    else:
        if ind[0] in subgraph_nei_nodes_un:
            subgraph_nei_nodes_un.remove(ind[0])
        if ind[1] in subgraph_nei_nodes_un:
            subgraph_nei_nodes_un.remove(ind[1])
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)  # list(set(ind).union(subgraph_nei_nodes_un))

    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]

    labels, enclosing_subgraph_nodes = node_label(incidence_matrix(subgraph), max_distance=h)
    # print(ind, subgraph_nodes[:32],enclosing_subgraph_nodes[:32], labels)
    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]
    # pruned_subgraph_nodes = subgraph_nodes
    # pruned_labels = labels

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    subgraph_size = len(pruned_subgraph_nodes)
    # enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)
    enc_ratio = 0
    num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes)
    toc=time.time()
    # print(f'subgraph_extraction_labeling_feng {toc-tic}s')
    return pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes


def subgraph_extraction_labeling(ind, rel, A_list,three_hop_nb_mat, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None,
                                 max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    # A_incidence 包含所有信息在一个矩阵中
    # three_hop_nb_mat = sp.load_npz('/home/yimiaofeng/PycharmProjects/SLBench/scripts/clean_code/start_from_kg/pilsl_3_hop_spmat.npz')
    # A_incidence = incidence_matrix(A_list)
    # A_incidence += A_incidence.T
    ind = list(ind)
    ind[0], ind[1] = int(ind[0]), int(ind[1])
    ind = (ind[0], ind[1])
    root1_nei = get_neighbor_nodes(set([ind[0]]), three_hop_nb_mat, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), three_hop_nb_mat, h, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    if enclosing_sub_graph:
        if ind[0] in subgraph_nei_nodes_int:
            subgraph_nei_nodes_int.remove(ind[0])
        if ind[1] in subgraph_nei_nodes_int:
            subgraph_nei_nodes_int.remove(ind[1])
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
        # print('Subgraph neighbor len: ' + str(len(subgraph_nei_nodes_int)))
    else:
        if ind[0] in subgraph_nei_nodes_un:
            subgraph_nei_nodes_un.remove(ind[0])
        if ind[1] in subgraph_nei_nodes_un:
            subgraph_nei_nodes_un.remove(ind[1])
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)  # list(set(ind).union(subgraph_nei_nodes_un))

    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]

    labels, enclosing_subgraph_nodes = node_label(incidence_matrix(subgraph), max_distance=h)
    # print(ind, subgraph_nodes[:32],enclosing_subgraph_nodes[:32], labels)
    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]
    # pruned_subgraph_nodes = subgraph_nodes
    # pruned_labels = labels

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    subgraph_size = len(pruned_subgraph_nodes)
    enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)
    num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes)
    return pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes

def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]

def node_label(subgraph, max_distance=1):
    # implementation of the node labeling scheme described in the paper
    # add position information
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [
        np.clip(sp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for
        r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)
    # dist_to_roots = np.array([[2, 2]]*(len(subgraph)-2))

    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    return labels, enclosing_subgraph_nodes


##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################



def ssp_multigraph_to_dgl(graph, n_feats=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """

    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    for rel, adj in enumerate(graph):
        # Convert adjacency matrix to tuples for nx0
        nx_triplets = []
        adj = adj.tocoo()
        nonzeros = list(zip(adj.row, adj.col))
        # nx_triplets = list(zip(adj.row, adj.col, rel*np.ones(len(adj.col), dtype='int')))
        for src, dst in nonzeros:
            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)

    # make dgl graph
    g_dgl = dgl.DGLGraph(multigraph=True)
    g_dgl.from_networkx(g_nx, edge_attrs=['type'])
    # add node features
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl

def deserialize(data):
    data_tuple = pkl.loads(data)
    keys = ('nodes', 'r_label', 'g_label', 'n_label')
    return dict(zip(keys, data_tuple))


def initialize_experiment(params, file_name):
    '''
    Makes the experiment directory, sets standard paths and initializes the logger
    '''
    params['main_dir'] = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))), '..')
    exps_dir = os.path.join(params['main_dir'], 'experiments')
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)

    params['exp_dir'] = os.path.join(exps_dir, params['experiment_name'])

    if not os.path.exists(params['exp_dir']):
        os.makedirs(params['exp_dir'])

    if file_name == 'test_auc.py':
        params['test_exp_dir'] = os.path.join(params['exp_dir'], f"test_{params['dataset']}_{params['constrained_neg_prob']}")
        if not os.path.exists(params['test_exp_dir']):
            os.makedirs(params['test_exp_dir'])
        file_handler = logging.FileHandler(os.path.join(params['test_exp_dir'], f"log_test.txt"))
    else:
        file_handler = logging.FileHandler(os.path.join(params['exp_dir'], "log_train.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    # logger.info('============ Initialized logger ============')
    # logger.info('\t '.join('%s: %s' % (k, str(v)) for k, v
    #                        in sorted(dict(vars(params)).items())))
    # logger.info('============================================')

    # with open(os.path.join(params.exp_dir, "params.json"), 'w') as fout:
    #     json.dump(vars(params), fout)


def initialize_model(params, model, load_model=False):
    '''
    relation2id: the relation to id mapping, this is stored in the model and used when testing
    model: the type of model to initialize/load
    load_model: flag which decide to initialize the model or load a saved model
    '''
    print(params['exp_dir'])
    # assert 0
    if load_model and os.path.exists(os.path.join(params['exp_dir'], 'best_graph_classifier.pth')):
        print('Loading existing model from %s' % os.path.join(params['exp_dir'], 'best_graph_classifier.pth'))
        graph_classifier = torch.load(os.path.join(params['exp_dir'], 'best_graph_classifier.pth')).cuda()
        # graph_classifier = torch.load(os.path.join(params['exp_dir'], 'best_graph_classifier.pth')).to(device=params['device'])
    else:
        # relation2id_path = os.path.join(params.main_dir, f'data/{params.dataset}/relation2id.json')
        # with open(relation2id_path) as f:
        #     relation2id = json.load(f)

        print('No existing model found. Initializing new model..')
        # graph_classifier = model(params, relation2id).to(device=params.device)
        # graph_classifier = model(params).cuda()
        graph_classifier = model(params).to(device=params['device'])

    return graph_classifier


def collate_dgl(samples):
    # The input `samples` is a list of pairs
    graphs_pos, g_labels_pos, r_labels_pos = map(list, zip(*samples))
    # print(graphs_pos, g_labels_pos, r_labels_pos, samples)
    batched_graph_pos = dgl.batch(graphs_pos)
    # print(batched_graph_pos)

    # graphs_neg = [item for sublist in graphs_negs for item in sublist]
    # g_labels_neg = [item for sublist in g_labels_negs for item in sublist]
    # r_labels_neg = [item for sublist in r_labels_negs for item in sublist]
    # batched_graph_neg = dgl.batch(graphs_neg)

    return (batched_graph_pos, r_labels_pos), g_labels_pos


def move_batch_to_device_dgl(batch, device):
    (g_dgl_pos, r_labels_pos), targets_pos = batch

    targets_pos = torch.LongTensor(targets_pos).to(device=device)
    # targets_pos = torch.LongTensor(targets_pos).cuda()
    # r_labels_pos = torch.FloatTensor(r_labels_pos).cuda()
    r_labels_pos = torch.FloatTensor(r_labels_pos).to(device=device)

    g_dgl_pos = send_graph_to_device(g_dgl_pos, device)
    # g_dgl_neg = send_graph_to_device(g_dgl_neg, device)

    return g_dgl_pos, r_labels_pos, targets_pos

def move_batch_to_device_dgl_ddi2(batch, device):
    (g_dgl_pos, r_labels_pos), targets_pos = batch

    targets_pos = torch.LongTensor(targets_pos).to(device=device)
    r_labels_pos = torch.FloatTensor(r_labels_pos).to(device=device)
    # targets_pos = torch.LongTensor(targets_pos).cuda()
    # r_labels_pos = torch.FloatTensor(r_labels_pos).cuda()

    g_dgl_pos = send_graph_to_device(g_dgl_pos, device)
    # g_dgl_neg = send_graph_to_device(g_dgl_neg, device)

    return g_dgl_pos, r_labels_pos, targets_pos

def send_graph_to_device(g, device):
    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        # g.ndata[l] = g.ndata.pop(l).cuda()
        g.ndata[l] = g.ndata.pop(l).to(device)

    # edges
    labels = g.edge_attr_schemes()
    for l in labels.keys():
        # g.edata[l] = g.edata.pop(l).cuda()
        g.edata[l] = g.edata.pop(l).to(device)
    return g
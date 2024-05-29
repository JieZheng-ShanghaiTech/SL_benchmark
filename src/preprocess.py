import copy
import os

import numpy as np
import scipy.sparse as sp
import pandas as pd
import random
import pickle as pkl
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score, ndcg_score, auc

setup_all_seeds=123

random.seed(123)
np.random.seed(123)

############################### Unify ###############################

class ChecktoSave():
    def __init__(self, kfold):
        
        
        self.best_classify = np.zeros((kfold, 3))
        self.best_ranking = np.asarray([-np.inf]*12*kfold).reshape((kfold,12))
        
        self.best_train_classify = np.zeros((kfold, 3))
        self.best_train_ranking = np.zeros((kfold, 12))
        
        self.indep_test_classify = np.zeros((kfold, 3))
        self.indep_test_ranking = np.zeros((kfold, 12))
        
    def chechtostop(self, fold, stop_num, metrics):
        if metrics[-1]<self.best_ranking[fold][11]:
            stop_num += 1
        return stop_num

    def update_classify(self, fold, epoch, metrics):
        if metrics[2]>self.best_classify[fold][2]:
            self.best_classify[fold] = metrics
            self.best_class_epoch = epoch
            return True
    
    def update_train_classify(self, fold, epoch, metrics):
        if metrics[2]>self.best_train_classify[fold][2]:
            self.best_train_classify[fold] = metrics
            self.best_class_train_epoch = epoch
            return True
        
    def update_ranking(self, fold, epoch, metrics):
        if metrics[0]>self.best_ranking[fold][0]:
            self.best_ranking[fold] = metrics
            self.best_rank_epoch = epoch
            return True
        
    def update_train_ranking(self, fold, epoch, metrics):
        if metrics[0]>self.best_train_ranking[fold][0]:
            self.best_train_ranking[fold] = metrics
            self.best_rank_train_epoch = epoch
            return True

    def update_indep_test_classify(self, fold, epoch, metrics):
        self.indep_test_classify[fold] = metrics
        
    def update_indep_test_ranking(self, fold, epoch, metrics):
        self.indep_test_ranking[fold] = metrics
        
    def get_best_classify(self):
        return self.best_class_epoch, self.best_classify
    
    def get_best_ranking(self):
        return self.best_rank_epoch, self.best_ranking
    
    def save_mat(self, path, score_mat):
        np.save(path, np.asarray(score_mat).astype(np.float32))
    
    def get_all_indep_test_metrics(self):
        all_test_metrics = np.hstack([self.indep_test_classify, self.indep_test_ranking])
        all_valid_metrics = np.hstack([self.best_classify, self.best_ranking]) # valid
        all_metrics = np.vstack([all_valid_metrics, all_test_metrics])
        return all_metrics
    
    def get_all_metrics(self):
        all_test_metrics = np.hstack([self.best_classify, self.best_ranking])
        all_train_metrics = np.hstack([self.best_train_classify, self.best_train_ranking])
        all_metrics = np.vstack([all_train_metrics, all_test_metrics])
        return all_metrics
        
def sparse_matrix_from_adj(adj: np.array, num_node: int) -> sp.csr_matrix:
    r = np.asarray(adj)[:, 0]
    c = np.asarray(adj)[:, 1]
    # print(max(r))
    # print(num_node)
    spm = sp.csr_matrix((np.ones(len(adj)).reshape(-1, ), (r, c)), shape=(num_node, num_node))
    spm = spm + spm.T

    return spm

human_sl_pairs_df=pd.read_csv('../data/preprocessed_data/human_sl_6460.csv')
# human_sl_pairs_df=pd.read_csv('../data/preprocessed_data/human_sl_9845.csv')

meta_table = pd.read_csv('../data/preprocessed_data/meta_table_9845.csv')
all_uni_id_cut = np.load('../data/preprocessed_data/wo_compt/all_uni_id_cut.npy',allow_pickle=True).item()


def get_id_map():
    # meta_table = pd.read_csv(
    #     '/home/yimiaofeng/PycharmProjects/SLBench/scripts/clean_code/start_from_kg/fin_meta_table.csv')
    List_Proteins_in_SL = meta_table['symbol']

    id_mapping = dict(zip(list(meta_table['symbol']), list(meta_table['unified_id'])))
    num_node = len(set(human_sl_pairs_df['unified_id_A'])|set(human_sl_pairs_df['unified_id_B']))

    return num_node, id_mapping

def cv1_all(kfold,num_node,train_rat,valid_rat,test_rat,training_rat,xtimes, negative_strategy, exp_data_path=None):

    pos_position = copy.deepcopy(human_sl_pairs_df)
    pos_position = np.asarray(pos_position[['unified_id_A', 'unified_id_B']].values, dtype=int)
    for i in range(len(pos_position)):
        if pos_position[i, 0] > pos_position[i, 1]:
            pos_position[i, 0], pos_position[i, 1] = pos_position[i, 1], pos_position[i, 0]
    if negative_strategy == 'All_Random':
        neg_position = np.asarray(np.ones((num_node, num_node)), dtype='bool')

    elif negative_strategy == 'All_Exp' or negative_strategy == 'All_Dep':
        neg_position = np.load(exp_data_path)
        print(neg_position.shape)
        neg_position = sp.csr_matrix((np.ones(neg_position.shape[0]),(neg_position[:,0],neg_position[:,1])),shape=(num_node,num_node))
        neg_position = np.asarray((neg_position+neg_position.T).toarray(),dtype='bool')


    neg_position[pos_position[:, 1], pos_position[:, 0]] = False
    neg_position[pos_position[:, 0], pos_position[:, 1]] = False
    neg_position = np.triu(neg_position, k = 1)
    neg_position = np.where(neg_position == True)
    neg_position = np.vstack(neg_position).transpose()

    init_pos_index = list(range(len(pos_position)))
    init_neg_index = list(range(len(neg_position)))

    training_size = int(len(init_pos_index) * training_rat)
    print(training_size)
    # print(len(init_neg_index))
    # random.shuffle(init_pos_index)
    # random.shuffle(init_neg_index)
    random.seed(123)
    np.random.seed(123)
    pos_position = pos_position[random.sample(init_pos_index, training_size)]
    neg_position = neg_position[random.sample(init_neg_index, int(training_size*xtimes))]

    if train_rat + valid_rat + test_rat != 1:
        print('-' * 20)
        print('train_rat + valid_rat + test_rat != 1 !!!!!')
        print('-' * 20)
        return

    train_pos_kfold,train_neg_kfold,graph_train_pos_kfold,graph_train_neg_kfold = [],[],[],[]

    valid_pos_kfold,valid_neg_kfold,graph_valid_pos_kfold,graph_valid_neg_kfold = [],[],[],[]

    test_pos_kfold,test_neg_kfold,graph_test_pos_kfold,graph_test_neg_kfold = [],[],[],[]

    index_pos = np.array(range(len(pos_position)))
    kf = KFold(n_splits=kfold, shuffle=True, random_state=123)

    for train_index, test_index in kf.split(index_pos):
        train_pos_set = pos_position
        train_pos_spm = sparse_matrix_from_adj(train_pos_set, num_node)  # 对称矩阵
        train_pos_kfold.append(train_pos_set)
        graph_train_pos_kfold.append(train_pos_spm)

        test_pos_set = pos_position[test_index]
        test_pos_spm = sparse_matrix_from_adj(test_pos_set, num_node)
        test_pos_kfold.append(test_pos_set)
        graph_test_pos_kfold.append(test_pos_spm)

    index_neg = np.array(range(len(neg_position)))
    kf = KFold(n_splits=kfold, shuffle=True, random_state=123)

    for train_index, test_index in kf.split(index_neg):
        train_neg_set = neg_position
        train_neg_spm = sparse_matrix_from_adj(train_neg_set, num_node)
        train_neg_kfold.append(train_neg_set)
        graph_train_neg_kfold.append(train_neg_spm)

        test_neg_set = neg_position[test_index]
        test_neg_spm = sparse_matrix_from_adj(test_neg_set, num_node)
        test_neg_kfold.append(test_neg_set)
        graph_test_neg_kfold.append(test_neg_spm)

    pos_samples = [graph_train_pos_kfold, graph_test_pos_kfold, train_pos_kfold, test_pos_kfold]
    neg_samples = [graph_train_neg_kfold, graph_test_neg_kfold, train_neg_kfold, test_neg_kfold]

    return pos_samples, neg_samples

def cv1(kfold,num_node,train_rat,valid_rat,test_rat,training_rat,xtimes, negative_strategy, exp_data_path=None, score_data_path=None, ex_compt=None):

    if train_rat + valid_rat + test_rat != 1:
        print('-' * 20)
        print('train_rat + valid_rat + test_rat != 1 !!!!!')
        print('-' * 20)
        return

    pos_position = copy.deepcopy(human_sl_pairs_df)
    pos_position = np.asarray(pos_position[['unified_id_A', 'unified_id_B']].values,dtype=int)
    # print(set(pos_position[:,0])|set(pos_position[:,1]))
    for i in range(len(pos_position)):
        if pos_position[i, 0] > pos_position[i, 1]:
            pos_position[i, 0], pos_position[i, 1] = pos_position[i, 1], pos_position[i, 0]
    if negative_strategy == 'Random' or negative_strategy == 'All':
        sl_pos_set=set(zip(pos_position[:,0],pos_position[:,1]))
        # neg_position = np.asarray(np.ones((num_node, num_node)), dtype='bool')
        neg_position = []
        for ind_a in range(num_node):
            for ind_b in range(num_node):
                if (ind_a,ind_b) in sl_pos_set or (ind_b,ind_a) in sl_pos_set:
                    continue
                if ind_a < ind_b:
                    pair = (ind_a,ind_b)
                    neg_position.append(pair)
        neg_position = np.array(neg_position,dtype=float)
        # neg_position[pos_position[:, 1], pos_position[:, 0]] = False
        # neg_position[pos_position[:, 0], pos_position[:, 1]] = False
        # neg_position = np.triu(neg_position, k = 1)
        # neg_position = np.where(neg_position == True)
        # neg_position = np.vstack(neg_position).transpose()

        init_pos_index = list(range(len(pos_position)))
        init_neg_index = list(range(len(neg_position)))

        training_size = int(len(init_pos_index) * training_rat)
        print(training_size)
        random.seed(123)
        np.random.seed(123)
        pos_position = pos_position[random.sample(init_pos_index, training_size)]
        if xtimes==1000:
            neg_position = neg_position
        else:
            neg_position = neg_position[random.sample(init_neg_index, int(training_size*xtimes))]

    elif negative_strategy == 'Exp' or negative_strategy == 'Dep':
        neg_id_scores_data = np.load(score_data_path)
        neg_index_from_source = np.load(exp_data_path)
        neg_position = neg_id_scores_data[neg_index_from_source,:2]
        print(neg_position.shape)
        if ex_compt:
            new_neg_df = pd.DataFrame(neg_position)
            new_neg_df = new_neg_df[new_neg_df[0].isin(all_uni_id_cut.keys())&new_neg_df[1].isin(all_uni_id_cut.keys())]
            new_neg_df[0] = new_neg_df[0].map(all_uni_id_cut)
            new_neg_df[1] = new_neg_df[1].map(all_uni_id_cut)
            neg_position = new_neg_df.values
        neg_position = neg_position.astype('int')
        print(pos_position.shape)
        print(neg_position.shape)


    train_pos_kfold,train_neg_kfold,graph_train_pos_kfold,graph_train_neg_kfold = [],[],[],[]

    valid_pos_kfold,valid_neg_kfold,graph_valid_pos_kfold,graph_valid_neg_kfold = [],[],[],[]

    test_pos_kfold,test_neg_kfold,graph_test_pos_kfold,graph_test_neg_kfold = [],[],[],[]

    index_pos = np.array(range(len(pos_position)))
    kf = KFold(n_splits=kfold, shuffle=True, random_state=123)

    for train_index, test_index in kf.split(index_pos):
        train_pos_set = pos_position[train_index]
        train_pos_spm = sparse_matrix_from_adj(train_pos_set, num_node)  # 对称矩阵
        train_pos_kfold.append(train_pos_set)
        graph_train_pos_kfold.append(train_pos_spm)

        test_pos_set = pos_position[test_index]
        test_pos_spm = sparse_matrix_from_adj(test_pos_set, num_node)
        test_pos_kfold.append(test_pos_set)
        graph_test_pos_kfold.append(test_pos_spm)

    index_neg = np.array(range(len(neg_position)))
    kf = KFold(n_splits=kfold, shuffle=True, random_state=123)

    for train_index, test_index in kf.split(index_neg):
        train_neg_set = neg_position[train_index]
        train_neg_spm = sparse_matrix_from_adj(train_neg_set, num_node)
        train_neg_kfold.append(train_neg_set)
        graph_train_neg_kfold.append(train_neg_spm)

        test_neg_set = neg_position[test_index]
        test_neg_spm = sparse_matrix_from_adj(test_neg_set, num_node)
        test_neg_kfold.append(test_neg_set)
        graph_test_neg_kfold.append(test_neg_spm)

    pos_samples = [graph_train_pos_kfold, graph_test_pos_kfold, train_pos_kfold, test_pos_kfold]
    neg_samples = [graph_train_neg_kfold, graph_test_neg_kfold, train_neg_kfold, test_neg_kfold]

    return pos_samples, neg_samples


def cv2_division(row_set, col_set, SL_sparse, gene_a_sparse, gene_b_sparse, num_node, neg_num=0, neg=False):
    r_mat=np.asarray(np.zeros((num_node,num_node)),dtype='bool')
    r_mat[row_set,:]=True
    c_mat=np.asarray(np.zeros((num_node,num_node)),dtype='bool')
    c_mat[:,col_set]=True
    mask_mat=r_mat&c_mat
    if len(row_set) != len(col_set):
        mask_mat = mask_mat+mask_mat.T
    selected_sample = np.triu(mask_mat&(SL_sparse.toarray()),k=1)
    cv2_set = np.where(selected_sample == True)
    cv2_set = np.vstack(cv2_set).transpose()

    if neg:
        ind=list(range(len(cv2_set)))
        if neg_num < len(cv2_set):
            random.seed(123)
            np.random.seed(123)
            cv2_set = cv2_set[random.sample(ind, neg_num)]
        else:
            cv2_set = cv2_set

    return np.asarray(cv2_set, dtype='int')

def cv2(kfold,num_node,train_rat,valid_rat,test_rat,training_rat,xtimes, negative_strategy, exp_data_path=None, score_data_path=None, ex_compt=None):
    pos_position = copy.deepcopy(human_sl_pairs_df)
    pos_position = np.asarray(pos_position[['unified_id_A', 'unified_id_B']].values, dtype=int)
    for i in range(len(pos_position)):
        if pos_position[i, 0] > pos_position[i, 1]:
            pos_position[i, 0], pos_position[i, 1] = pos_position[i, 1], pos_position[i, 0]
    if negative_strategy == 'Random' or negative_strategy == 'All':
        # neg_position = np.asarray(np.ones((num_node, num_node)), dtype='bool')
        
        # neg_position[pos_position[:, 1], pos_position[:, 0]] = False
        # neg_position[pos_position[:, 0], pos_position[:, 1]] = False
        # neg_position = np.triu(neg_position, k = 1)
        # neg_position = np.where(neg_position == True)
        # neg_position = np.vstack(neg_position).transpose()
        sl_pos_set=set(zip(pos_position[:,0],pos_position[:,1]))
        # neg_position = np.asarray(np.ones((num_node, num_node)), dtype='bool')
        neg_position = []
        for ind_a in range(num_node):
            for ind_b in range(num_node):
                if (ind_a,ind_b) in sl_pos_set or (ind_b,ind_a) in sl_pos_set:
                    continue
                if ind_a < ind_b:
                    pair = (ind_a,ind_b)
                    neg_position.append(pair)
        neg_position = np.array(neg_position,dtype=float)

    elif negative_strategy == 'Exp' or negative_strategy == 'Dep':
        # neg_position = np.load(exp_data_path)
        neg_id_scores_data = np.load(score_data_path)
        neg_index_from_source = np.load(exp_data_path)
        neg_position = neg_id_scores_data[neg_index_from_source,:2]
        print(neg_position.shape)
        if ex_compt:
            new_neg_df = pd.DataFrame(neg_position)
            new_neg_df = new_neg_df[new_neg_df[0].isin(all_uni_id_cut.keys())&new_neg_df[1].isin(all_uni_id_cut.keys())]
            new_neg_df[0] = new_neg_df[0].map(all_uni_id_cut)
            new_neg_df[1] = new_neg_df[1].map(all_uni_id_cut)
            neg_position = new_neg_df.values
        neg_position = neg_position.astype('int')
        print(pos_position.shape)
        print(neg_position.shape)

    SL_pos_sparse = sp.csr_matrix((np.ones(len(pos_position)), (pos_position[:, 0], pos_position[:, 1])), shape=(num_node, num_node),dtype='bool')
    SL_pos_sparse = SL_pos_sparse+SL_pos_sparse.T
    gene_a_pos_sparse = sp.csr_matrix((pos_position[:, 0], (pos_position[:, 0], pos_position[:, 1])), shape=(num_node, num_node))
    gene_b_pos_sparse = sp.csr_matrix((pos_position[:, 1], (pos_position[:, 0], pos_position[:, 1])), shape=(num_node, num_node))

    SL_neg_sparse = sp.csr_matrix((np.ones(len(neg_position)), (neg_position[:, 0], neg_position[:, 1])), shape=(num_node, num_node),dtype='bool')
    SL_neg_sparse = SL_neg_sparse+SL_neg_sparse.T
    gene_a_neg_sparse = sp.csr_matrix((neg_position[:, 0], (neg_position[:, 0], neg_position[:, 1])), shape=(num_node, num_node))
    gene_b_neg_sparse = sp.csr_matrix((neg_position[:, 1], (neg_position[:, 0], neg_position[:, 1])), shape=(num_node, num_node))

    init_col_index = np.array(range(num_node))

    training_size = int(len(init_col_index) * training_rat)

    train_col = init_col_index[init_col_index[range(training_size)]]

    if train_rat + valid_rat + test_rat != 1:
        print('-' * 20)
        print('train_rat + valid_rat + test_rat != 1 !!!!!')
        print('-' * 20)
        return

    train_pos_kfold,train_neg_kfold,graph_train_pos_kfold,graph_train_neg_kfold = [],[],[],[]

    valid_pos_kfold,valid_neg_kfold,graph_valid_pos_kfold,graph_valid_neg_kfold = [],[],[],[]

    test_pos_kfold,test_neg_kfold,graph_test_pos_kfold,graph_test_neg_kfold = [],[],[],[]

    index_pos = np.array(range(len(train_col)))
    kf = KFold(n_splits=kfold, shuffle=True, random_state=123)


    for train_index, test_index in kf.split(index_pos):
        train_ind_set = train_col[train_index]
        test_ind_set = train_col[test_index]

        train_pos_sl_id = cv2_division(train_ind_set, train_ind_set, SL_pos_sparse, gene_a_pos_sparse, gene_b_pos_sparse,num_node)
        train_pos_spm = sparse_matrix_from_adj(train_pos_sl_id, num_node)
        train_pos_kfold.append(train_pos_sl_id)
        graph_train_pos_kfold.append(train_pos_spm)

        test_pos_sl_id = cv2_division(train_ind_set, test_ind_set, SL_pos_sparse, gene_a_pos_sparse, gene_b_pos_sparse,num_node)
        test_pos_spm = sparse_matrix_from_adj(test_pos_sl_id, num_node)
        test_pos_kfold.append(test_pos_sl_id)
        graph_test_pos_kfold.append(test_pos_spm)
        if xtimes==1000:
            train_neg_num=int(len(train_index)*len(train_index))
        else:
            train_neg_num=int(len(train_pos_sl_id)*xtimes)
        train_neg_sl_id = cv2_division(train_ind_set, train_ind_set, SL_neg_sparse, gene_a_neg_sparse, gene_b_neg_sparse,num_node,neg_num=train_neg_num, neg=True)
        train_neg_spm = sparse_matrix_from_adj(train_neg_sl_id, num_node)
        train_neg_kfold.append(train_neg_sl_id)
        graph_train_neg_kfold.append(train_neg_spm)

        if xtimes==1000:
            test_neg_num=int(len(train_index)*len(test_index))
        else:
            test_neg_num=int(len(test_pos_sl_id)*xtimes)
        test_neg_sl_id = cv2_division(train_ind_set, test_ind_set, SL_neg_sparse, gene_a_neg_sparse, gene_b_neg_sparse,num_node,neg_num=test_neg_num, neg=True)
        test_neg_spm = sparse_matrix_from_adj(test_neg_sl_id, num_node)
        test_neg_kfold.append(test_neg_sl_id)
        graph_test_neg_kfold.append(test_neg_spm)

    pos_samples = [graph_train_pos_kfold, graph_test_pos_kfold, train_pos_kfold, test_pos_kfold]
    neg_samples = [graph_train_neg_kfold, graph_test_neg_kfold, train_neg_kfold, test_neg_kfold]

    return pos_samples, neg_samples


def cv3_division(pos_set, SL_sparse, gene_a_sparse, gene_b_sparse, num_node, neg_num=0, neg=False):
    r_mat=np.asarray(np.zeros((num_node,num_node)),dtype='bool')
    r_mat[pos_set,:]=True
    c_mat=np.asarray(np.zeros((num_node,num_node)),dtype='bool')
    c_mat[:,pos_set]=True
    mask_mat=r_mat&c_mat
    selected_sample = np.triu(mask_mat&(SL_sparse.toarray()),k=1)
    cv3_set = np.where(selected_sample == True)
    cv3_set = np.vstack(cv3_set).transpose()

    if neg:
        ind = list(range(len(cv3_set)))
        if neg_num < len(cv3_set):
            random.seed(123)
            np.random.seed(123)
            cv3_set = cv3_set[random.sample(ind, neg_num)]
        else:
            cv3_set = cv3_set

    return np.asarray(cv3_set, dtype='int')

def cv3(kfold,num_node,train_rat,valid_rat,test_rat,training_rat,xtimes, negative_strategy, exp_data_path=None, score_data_path=None, ex_compt=None):
    pos_position = copy.deepcopy(human_sl_pairs_df)
    pos_position = np.asarray(pos_position[['unified_id_A', 'unified_id_B']].values, dtype=int)
    for i in range(len(pos_position)):
        if pos_position[i, 0] > pos_position[i, 1]:
            pos_position[i, 0], pos_position[i, 1] = pos_position[i, 1], pos_position[i, 0]

    if negative_strategy == 'Random' or negative_strategy == 'All':
        # neg_position = np.asarray(np.ones((num_node, num_node)), dtype='bool')
        
        # neg_position[pos_position[:, 1], pos_position[:, 0]] = False
        # neg_position[pos_position[:, 0], pos_position[:, 1]] = False
        # neg_position = np.triu(neg_position, k = 1)
        # neg_position = np.where(neg_position == True)
        # neg_position = np.vstack(neg_position).transpose()
        sl_pos_set=set(zip(pos_position[:,0],pos_position[:,1]))
        # neg_position = np.asarray(np.ones((num_node, num_node)), dtype='bool')
        neg_position = []
        for ind_a in range(num_node):
            for ind_b in range(num_node):
                if (ind_a,ind_b) in sl_pos_set or (ind_b,ind_a) in sl_pos_set:
                    continue
                if ind_a < ind_b:
                    pair = (ind_a,ind_b)
                    neg_position.append(pair)
        neg_position = np.array(neg_position,dtype=float)

    elif negative_strategy == 'Exp' or negative_strategy == 'Dep':
        # neg_position = np.load(exp_data_path)
        neg_id_scores_data = np.load(score_data_path)
        neg_index_from_source = np.load(exp_data_path)
        neg_position = neg_id_scores_data[neg_index_from_source,:2]
        print(neg_position.shape)
        if ex_compt:
            new_neg_df = pd.DataFrame(neg_position)
            new_neg_df = new_neg_df[new_neg_df[0].isin(all_uni_id_cut.keys())&new_neg_df[1].isin(all_uni_id_cut.keys())]
            new_neg_df[0] = new_neg_df[0].map(all_uni_id_cut)
            new_neg_df[1] = new_neg_df[1].map(all_uni_id_cut)
            neg_position = new_neg_df.values
        neg_position = neg_position.astype('int')
        print(pos_position.shape)
        print(neg_position.shape)

    SL_pos_sparse = sp.csr_matrix((np.ones(len(pos_position)), (pos_position[:, 0], pos_position[:, 1])), shape=(num_node, num_node),dtype='bool')
    SL_pos_sparse = SL_pos_sparse+SL_pos_sparse.T
    gene_a_pos_sparse = sp.csr_matrix((pos_position[:, 0], (pos_position[:, 0], pos_position[:, 1])), shape=(num_node, num_node))
    gene_b_pos_sparse = sp.csr_matrix((pos_position[:, 1], (pos_position[:, 0], pos_position[:, 1])), shape=(num_node, num_node))

    SL_neg_sparse = sp.csr_matrix((np.ones(len(neg_position)), (neg_position[:, 0], neg_position[:, 1])), shape=(num_node, num_node),dtype='bool')
    SL_neg_sparse = SL_neg_sparse+SL_neg_sparse.T
    gene_a_neg_sparse = sp.csr_matrix((neg_position[:, 0], (neg_position[:, 0], neg_position[:, 1])), shape=(num_node, num_node))
    gene_b_neg_sparse = sp.csr_matrix((neg_position[:, 1], (neg_position[:, 0], neg_position[:, 1])), shape=(num_node, num_node))

    init_col_index = np.array(range(num_node))

    training_size = int(len(init_col_index) * training_rat)

    train_col=init_col_index[init_col_index[range(training_size)]]

    if train_rat + valid_rat + test_rat != 1:
        print('-' * 20)
        print('train_rat + valid_rat + test_rat != 1 !!!!!')
        print('-' * 20)
        return

    train_pos_kfold,train_neg_kfold,graph_train_pos_kfold,graph_train_neg_kfold = [],[],[],[]

    valid_pos_kfold,valid_neg_kfold,graph_valid_pos_kfold,graph_valid_neg_kfold = [],[],[],[]

    test_pos_kfold,test_neg_kfold,graph_test_pos_kfold,graph_test_neg_kfold = [],[],[],[]

    index_pos = np.array(range(len(train_col)))
    kf = KFold(n_splits=kfold, shuffle=True, random_state=123)


    for train_index, test_index in kf.split(index_pos):
        train_ind_set = train_col[train_index]
        test_ind_set = train_col[test_index]

        train_pos_sl_id = cv3_division(train_ind_set, SL_pos_sparse, gene_a_pos_sparse, gene_b_pos_sparse,num_node)
        train_pos_spm = sparse_matrix_from_adj(train_pos_sl_id, num_node)
        train_pos_kfold.append(train_pos_sl_id)
        graph_train_pos_kfold.append(train_pos_spm)

        test_pos_sl_id = cv3_division(test_ind_set, SL_pos_sparse, gene_a_pos_sparse, gene_b_pos_sparse,num_node)
        test_pos_spm = sparse_matrix_from_adj(test_pos_sl_id, num_node)
        test_pos_kfold.append(test_pos_sl_id)
        graph_test_pos_kfold.append(test_pos_spm)

        if xtimes==1000:
            train_neg_num=int(len(train_index)*len(train_index))
        else:
            train_neg_num=int(len(train_pos_sl_id)*xtimes)
        train_neg_sl_id = cv3_division(train_ind_set, SL_neg_sparse, gene_a_neg_sparse, gene_b_neg_sparse,num_node,neg_num=train_neg_num, neg=True)
        train_neg_spm = sparse_matrix_from_adj(train_neg_sl_id, num_node)
        train_neg_kfold.append(train_neg_sl_id)
        graph_train_neg_kfold.append(train_neg_spm)

        if xtimes==1000:
            test_neg_num=int(len(test_index)*len(test_index))
        else:
            test_neg_num=int(len(test_pos_sl_id)*xtimes)
        test_neg_sl_id = cv3_division(test_ind_set, SL_neg_sparse, gene_a_neg_sparse, gene_b_neg_sparse,num_node,neg_num=test_neg_num, neg=True)
        test_neg_spm = sparse_matrix_from_adj(test_neg_sl_id, num_node)
        test_neg_kfold.append(test_neg_sl_id)
        graph_test_neg_kfold.append(test_neg_spm)

    pos_samples = [graph_train_pos_kfold, graph_test_pos_kfold, train_pos_kfold, test_pos_kfold]
    neg_samples = [graph_train_neg_kfold, graph_test_neg_kfold, train_neg_kfold, test_neg_kfold]

    return pos_samples, neg_samples


def get_kfold_data_pos_neg(kfold: int, num_node: int, train_rat: float, valid_rat: float, test_rat: float,
                           training_rat: float, pos_neg: float, division_strategy: str, negative_strategy: str, 
                           ex_compt=None, indep_test=None,cell_line=None):
    
    xtimes = int(1/pos_neg)
    
    if indep_test:
        suf=''
        if cell_line in ['k562','a549','293t','hela','merged']:
            suf=f'_cell_{cell_line}'
        elif cell_line=='indep_test':
            suf = '_indep_test'
        elif cell_line=='kr4sl':
            suf = '_kr4sl'
        elif 'nsf4sl' in cell_line:
            suf = '_'+cell_line
        if ex_compt:
            if division_strategy == 'CV1' and negative_strategy == 'Random':
                with open(f'../data/data_split_wo_comp/CV1_{xtimes}{suf}.pkl','rb') as f:
                    pos_samples, neg_samples = pkl.load(f)
                    print(f'CV1_{xtimes}{suf}.pkl loaded')
            if division_strategy == 'CV1' and negative_strategy == 'Exp':
                with open(f'../data/data_split_wo_comp/CV1_{xtimes}_Exp{suf}.pkl','rb') as f:
                    pos_samples, neg_samples = pkl.load(f)
                    print(f'CV1_{xtimes}_Exp{suf}.pkl loaded')
        else:
            if division_strategy == 'CV1' and negative_strategy == 'Random':
                with open(f'../data/data_split/CV1_{xtimes}{suf}.pkl','rb') as f:
                    pos_samples, neg_samples = pkl.load(f)
                    print(f'CV1_{xtimes}{suf}.pkl loaded')
            if division_strategy in ['CV3'] and negative_strategy == 'Random':
                with open(f'../data/data_split/{division_strategy}_{xtimes}{suf}.pkl','rb') as f:
                    pos_samples, neg_samples = pkl.load(f)
                    print(f'{division_strategy}_{xtimes}{suf}.pkl loaded')
            if division_strategy == 'CV1' and negative_strategy == 'Exp':
                with open(f'../data/data_split/CV1_{xtimes}_Exp{suf}.pkl','rb') as f:
                    pos_samples, neg_samples = pkl.load(f)
                    print(f'CV1_{xtimes}_Exp{suf}.pkl loaded')
        
        return pos_samples, neg_samples
    
    


    exp_data_paths = ['../data/preprocessed_data/one_time_neg_index_exp.npy',
                    '../data/preprocessed_data/five_time_neg_index_exp.npy',
                    '../data/preprocessed_data/twenty_time_neg_index_exp.npy',
                    '../data/preprocessed_data/fifty_time_neg_index_exp.npy',]
    dep_data_paths = ['../data/preprocessed_data/one_time_neg_index_dep.npy',
                    '../data/preprocessed_data/five_time_neg_index_dep.npy',
                    '../data/preprocessed_data/twenty_time_neg_index_dep.npy',
                    '../data/preprocessed_data/fifty_time_neg_index_dep.npy',]
    
    exp_score_data_path = '../data/preprocessed_data/sorted_neg_ids_scores_exp.npy'
    dep_score_data_path = '../data/preprocessed_data/sorted_neg_ids_scores_dep.npy'
    
    train_data_path = '../data/data_split/'
    
    if ex_compt:
        exp_data_paths = ['../data/preprocessed_data/wo_compt/one_time_neg_index_exp_wo_compt.npy',
                        '../data/preprocessed_data/wo_compt/five_time_neg_index_exp_wo_compt.npy',
                        '../data/preprocessed_data/wo_compt/twenty_time_neg_index_exp_wo_compt.npy',
                        '../data/preprocessed_data/wo_compt/fifty_time_neg_index_exp_wo_compt.npy']
        dep_data_paths = ['../data/preprocessed_data/wo_compt/one_time_neg_index_dep_wo_compt.npy',
                        '../data/preprocessed_data/wo_compt/five_time_neg_index_dep_wo_compt.npy',
                        '../data/preprocessed_data/wo_compt/twenty_time_neg_index_dep_wo_compt.npy',
                        '../data/preprocessed_data/wo_compt/fifty_time_neg_index_dep_wo_compt.npy']
        
        exp_score_data_path = '../data/preprocessed_data/wo_compt/sorted_neg_ids_scores_exp_wo_compt.npy'
        dep_score_data_path = '../data/preprocessed_data/wo_compt/sorted_neg_ids_scores_dep_wo_compt.npy'
        
        train_data_path = '../data/data_split_wo_comp/'

    if xtimes == 1:
        exp_data_path = exp_data_paths[0]
        dep_data_path = dep_data_paths[0]
    elif xtimes == 5:
        exp_data_path = exp_data_paths[1]
        dep_data_path = dep_data_paths[1]
    elif xtimes == 20:
        exp_data_path = exp_data_paths[2]
        dep_data_path = dep_data_paths[2]
    elif xtimes == 50:
        exp_data_path = exp_data_paths[3]
        dep_data_path = dep_data_paths[3]

    if negative_strategy == 'Exp':
        if division_strategy == 'CV1':
            if os.path.exists(f'{train_data_path}CV1_{xtimes}_Exp.npy'):
                pos_samples, neg_samples = np.load(f'{train_data_path}CV1_{xtimes}_Exp.npy', allow_pickle=True)
            else:
                pos_samples, neg_samples = cv1(kfold, num_node, train_rat, valid_rat, test_rat, training_rat, xtimes,
                                               negative_strategy, exp_data_path, exp_score_data_path, ex_compt)
                np.save(f'{train_data_path}CV1_{xtimes}_Exp.npy', [pos_samples, neg_samples])
        elif division_strategy == 'CV2':
            if os.path.exists(f'{train_data_path}CV2_{xtimes}_Exp.npy'):
                pos_samples, neg_samples = np.load(f'{train_data_path}CV2_{xtimes}_Exp.npy', allow_pickle=True)
            else:
                pos_samples, neg_samples = cv2(kfold, num_node, train_rat, valid_rat, test_rat, training_rat, xtimes,
                                               negative_strategy, exp_data_path, exp_score_data_path, ex_compt)
                np.save(f'{train_data_path}CV2_{xtimes}_Exp.npy', [pos_samples, neg_samples])
        elif division_strategy == 'CV3':
            if os.path.exists(f'{train_data_path}CV3_{xtimes}_Exp.npy'):
                pos_samples, neg_samples = np.load(f'{train_data_path}CV3_{xtimes}_Exp.npy', allow_pickle=True)
            else:
                pos_samples, neg_samples = cv3(kfold, num_node, train_rat, valid_rat, test_rat, training_rat, xtimes,
                                               negative_strategy, exp_data_path, exp_score_data_path, ex_compt)
                np.save(f'{train_data_path}CV3_{xtimes}_Exp.npy', [pos_samples, neg_samples])
        else:
            print('Please select data division strategy. (one of ["CV1","CV2","CV3"])')
    elif negative_strategy == 'Dep':
        if division_strategy == 'CV1':
            if os.path.exists(f'{train_data_path}CV1_{xtimes}_Dep.npy'):
                pos_samples, neg_samples = np.load(f'{train_data_path}CV1_{xtimes}_Dep.npy', allow_pickle=True)
            else:
                pos_samples, neg_samples = cv1(kfold, num_node, train_rat, valid_rat, test_rat, training_rat, xtimes,
                                               negative_strategy, dep_data_path, dep_score_data_path, ex_compt)
                np.save(f'{train_data_path}CV1_{xtimes}_Dep.npy', [pos_samples, neg_samples])
        elif division_strategy == 'CV2':
            if os.path.exists(f'{train_data_path}CV2_{xtimes}_Dep.npy'):
                pos_samples, neg_samples = np.load(f'{train_data_path}CV2_{xtimes}_Dep.npy', allow_pickle=True)
            else:
                pos_samples, neg_samples = cv2(kfold, num_node, train_rat, valid_rat, test_rat, training_rat, xtimes,
                                               negative_strategy, dep_data_path, dep_score_data_path, ex_compt)
                np.save(f'{train_data_path}CV2_{xtimes}_Dep.npy', [pos_samples, neg_samples])
        elif division_strategy == 'CV3':
            if os.path.exists(f'{train_data_path}CV3_{xtimes}_Dep.npy'):
                pos_samples, neg_samples = np.load(f'{train_data_path}CV3_{xtimes}_Dep.npy', allow_pickle=True)
            else:
                pos_samples, neg_samples = cv3(kfold, num_node, train_rat, valid_rat, test_rat, training_rat, xtimes,
                                               negative_strategy, dep_data_path, dep_score_data_path, ex_compt)
                np.save(f'{train_data_path}CV3_{xtimes}_Dep.npy', [pos_samples, neg_samples])
        else:
            print('Please select data division strategy. (one of ["CV1","CV2","CV3"])')
    elif negative_strategy == 'Random':
        if division_strategy == 'CV1':
            if os.path.exists(f'{train_data_path}CV1_{xtimes}.npy'):
                pos_samples, neg_samples = np.load(f'{train_data_path}CV1_{xtimes}.npy',allow_pickle=True)
            else:
                pos_samples, neg_samples = cv1(kfold, num_node, train_rat, valid_rat, test_rat, training_rat, xtimes, negative_strategy)
                np.save(f'{train_data_path}CV1_{xtimes}.npy',[pos_samples, neg_samples])
        elif division_strategy == 'CV2':
            if os.path.exists(f'{train_data_path}CV2_{xtimes}.npy'):
                pos_samples, neg_samples = np.load(f'{train_data_path}CV2_{xtimes}.npy', allow_pickle=True)
            else:
                pos_samples, neg_samples = cv2(kfold, num_node, train_rat, valid_rat, test_rat, training_rat, xtimes, negative_strategy)
                np.save(f'{train_data_path}CV2_{xtimes}.npy', [pos_samples, neg_samples])
        elif division_strategy == 'CV3':
            if os.path.exists(f'{train_data_path}CV3_{xtimes}.npy'):
                pos_samples, neg_samples = np.load(f'{train_data_path}CV3_{xtimes}.npy', allow_pickle=True)
            else:
                pos_samples, neg_samples = cv3(kfold, num_node, train_rat, valid_rat, test_rat, training_rat, xtimes, negative_strategy)
                np.save(f'{train_data_path}CV3_{xtimes}.npy', [pos_samples, neg_samples])
        else:
            print('Please select data division strategy. (one of ["CV1","CV2","CV3"])')
    elif negative_strategy == 'All':
        if division_strategy == 'CV1':
            if os.path.exists(f'{train_data_path}CV1_1000.npy'):
                pos_samples, neg_samples = np.load(f'{train_data_path}CV1_1000.npy',allow_pickle=True)
            else:
                pos_samples, neg_samples = cv1(kfold, num_node, train_rat, valid_rat, test_rat, training_rat, xtimes, negative_strategy)
                np.save(f'{train_data_path}CV1_1000.npy',[pos_samples, neg_samples])
        elif division_strategy == 'CV2':
            if os.path.exists(f'{train_data_path}CV2_1000.npy'):
                pos_samples, neg_samples = np.load(f'{train_data_path}CV2_1000.npy', allow_pickle=True)
            else:
                pos_samples, neg_samples = cv2(kfold, num_node, train_rat, valid_rat, test_rat, training_rat, xtimes, negative_strategy)
                np.save(f'{train_data_path}CV2_1000.npy', [pos_samples, neg_samples])
        elif division_strategy == 'CV3':
            if os.path.exists(f'{train_data_path}CV3_1000.npy'):
                pos_samples, neg_samples = np.load(f'{train_data_path}CV3_1000.npy', allow_pickle=True)
            else:
                pos_samples, neg_samples = cv3(kfold, num_node, train_rat, valid_rat, test_rat, training_rat, xtimes, negative_strategy)
                np.save(f'{train_data_path}CV3_1000.npy', [pos_samples, neg_samples])
        else:
            print('Please select data division strategy. (one of ["CV1","CV2","CV3"])')
    elif negative_strategy == 'All_Random':
        if division_strategy == 'CV1':
            if os.path.exists(f'{train_data_path}CV1_{xtimes}_All_Random.npy'):
                pos_samples, neg_samples = np.load(f'{train_data_path}CV1_{xtimes}_All_Random.npy',allow_pickle=True)
            else:
                pos_samples, neg_samples = cv1_all(kfold, num_node, train_rat, valid_rat, test_rat, training_rat, xtimes, negative_strategy)
                np.save(f'{train_data_path}CV1_{xtimes}_All_Random.npy',[pos_samples, neg_samples])
        else:
            print('There are no more other strategys')
            return None
    elif negative_strategy == 'All_Exp':
        if division_strategy == 'CV1':
            if os.path.exists(f'{train_data_path}CV1_{xtimes}_All_Exp.npy'):
                pos_samples, neg_samples = np.load(f'{train_data_path}CV1_{xtimes}_All_Exp.npy',allow_pickle=True)
            else:
                pos_samples, neg_samples = cv1_all(kfold, num_node, train_rat, valid_rat, test_rat, training_rat, xtimes, negative_strategy,exp_data_path)
                np.save(f'{train_data_path}CV1_{xtimes}_All_Exp.npy',[pos_samples, neg_samples])
        else:
            print('There are no more other strategys')
            return None
    elif negative_strategy == 'All_Dep':
        if division_strategy == 'CV1':
            if os.path.exists(f'{train_data_path}CV1_{xtimes}_All_Dep.npy'):
                pos_samples, neg_samples = np.load(f'{train_data_path}CV1_{xtimes}_All_Dep.npy',allow_pickle=True)
            else:
                pos_samples, neg_samples = cv1_all(kfold, num_node, train_rat, valid_rat, test_rat, training_rat, xtimes, negative_strategy,dep_data_path)
                np.save(f'{train_data_path}CV1_{xtimes}_All_Dep.npy',[pos_samples, neg_samples])
        else:
            print('There are no more other strategys')
            return None

    return pos_samples, neg_samples


def NDCG(y_pos_index, y_not_seen, topk):
    # hit topk
    hit_topk = len(set(list(y_pos_index)) & set(list(y_not_seen[:topk])))
    if hit_topk == 0:
        return 0, 0, 0

    GT = set(list(y_pos_index))
    if len(GT) > topk :
        sent_list = [1.0] * topk
    else:
        sent_list = [1.0]*len(GT) + [0.0]*(topk-len(GT))


    hit_list = []
    for i in range(len(set(list(y_not_seen)))):
        if y_not_seen[i] in set(list(y_pos_index)):
            hit_list.append(1)
        else:
            hit_list.append(0)

    hit_list = np.asfarray(hit_list)[:topk]
    idcg_topk = np.sum(sent_list / np.log2(np.arange(2, len(sent_list) + 2)))
    dcg_topk = np.sum(hit_list / np.log2(np.arange(2, len(hit_list) + 2)))
    return 0 if dcg_topk == 0 or idcg_topk == 0 else dcg_topk / idcg_topk, \
           hit_topk / len(y_pos_index), hit_topk / min(topk, len(y_pos_index))

def calculate_AP_at_k(result, k):
    relevant_docs = 0
    precision_sum = 0
    for i in range(min(k, len(result))):
        if result[i] == 1:
            relevant_docs += 1
            precision_sum += relevant_docs / (i + 1)
    return precision_sum / relevant_docs if relevant_docs > 0 else 0

def calculate_MAP_at_k(results, k):
    AP_sum = 0
    for result in results:
        AP_sum += calculate_AP_at_k(result, k)
    return AP_sum / len(results)

def cal_metrics(score_mat, pos_index, neg_index, seen_index=None):
    score_matrix = copy.deepcopy(score_mat)
    if sp.isspmatrix(score_matrix):
        score_matrix = score_matrix.todense()
    score_matrix = np.asarray(score_matrix)
    n_gene = score_matrix.shape[0]
    score_matrix[range(n_gene), range(n_gene)] = 0

    pos_matrix=sp.csr_matrix((np.ones(pos_index.shape[0]), (pos_index[:, 0], pos_index[:, 1])),shape=score_matrix.shape)
    pos_matrix=pos_matrix+pos_matrix.T

    auroc_p, f1_p, aupr_p = 0, 0, 0
    
    pos_s = score_matrix[pos_index[:, 0], pos_index[:, 1]]
    neg_s = score_matrix[neg_index[:, 0], neg_index[:, 1]]
    
    y_pred_score = np.hstack([pos_s, neg_s]).reshape(-1, 1)
    y_true = np.hstack([np.ones([1, len(pos_s)]), np.zeros([1, len(neg_s)])]).reshape(-1, 1)

    precision, recall, _ = precision_recall_curve(y_true, y_pred_score)
    aupr_p = auc(recall, precision)
    auroc_p = roc_auc_score(y_true, y_pred_score)
    f1_p = max(2 * precision * recall / (precision + recall))
    
    score_matrix_ndcg = copy.deepcopy(score_mat)
    score_matrix_ndcg[range(n_gene), range(n_gene)] = 0
    seen_score = -999999
    if seen_index is not None:
        score_matrix_ndcg[seen_index[:,0], seen_index[:,1]] = seen_score  # del seen samples
        score_matrix_ndcg[seen_index[:,1], seen_index[:,0]] = seen_score
        
    y_top100_list = []
    y_bool_list = []
    y_sorted_score_list = []
    y_pos_num_list = []
    ### onli for kr4sl
    test_gene_set = list(set(pos_index[:, 0])|set(pos_index[:, 1]))
    for i in test_gene_set:
    ### onli for kr4sl
    # for i in range(n_gene):
        y_pos_index = pos_matrix[i, :].nonzero()[1]
        if len(y_pos_index) == 0:
            continue
        y_pos_num_list.append(len(y_pos_index))
        sort_score = np.asarray(score_matrix_ndcg[i, :])
        y_top100 = np.argsort(sort_score)[::-1][:100]
        y_bool = pos_matrix[i, :].toarray()[0][y_top100]
        y_sorted_score = score_matrix_ndcg[i, :][y_top100]
        y_top100_list.append(y_top100)
        y_bool_list.append(y_bool)
        y_sorted_score_list.append(y_sorted_score)
    
    y_bool_list = np.asarray(y_bool_list)
    y_sorted_score_list = np.asarray(y_sorted_score_list)
    y_top100_list = np.asarray(y_top100_list)
    
    ndcg_k, precision_k, recall_k, map_k = np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3)
    ndcg_k[0] = ndcg_score(y_bool_list, y_sorted_score_list, k=10)
    ndcg_k[1] = ndcg_score(y_bool_list, y_sorted_score_list, k=20)
    ndcg_k[2] = ndcg_score(y_bool_list, y_sorted_score_list, k=50)
    
    precision_k[0] = (y_bool_list[:, :10].sum(axis=1) / np.minimum(y_pos_num_list,10)).mean()
    precision_k[1] = (y_bool_list[:, :20].sum(axis=1) / np.minimum(y_pos_num_list,20)).mean()
    precision_k[2] = (y_bool_list[:, :50].sum(axis=1) / np.minimum(y_pos_num_list,50)).mean()
    recall_k[0] = (y_bool_list[:, :10].sum(axis=1) / y_pos_num_list).mean()
    recall_k[1] = (y_bool_list[:, :20].sum(axis=1) / y_pos_num_list).mean()
    recall_k[2] = (y_bool_list[:, :50].sum(axis=1) / y_pos_num_list).mean()
    
    map_k[0] = calculate_MAP_at_k(y_bool_list[:, :10],10)
    map_k[1] = calculate_MAP_at_k(y_bool_list[:, :20],20)
    map_k[2] = calculate_MAP_at_k(y_bool_list[:, :50],50)

    by_pair = [auroc_p, f1_p, aupr_p, ndcg_k, recall_k, precision_k, map_k]

    return list(np.hstack(by_pair))


def load_kg(neighbor_sample_size):
    kg_wo_sl = pd.read_csv('../data/preprocessed_data/fin_kg_wo_sl_9845.csv')
    sl_pairs = pd.read_csv('../data/preprocessed_data/human_sl_9845.csv')

    n_node_a = len(set(sl_pairs['unified_id_A']))
    n_node_b = len(set(sl_pairs['unified_id_B']))

    n_entity = len(set(kg_wo_sl['unified_id_A']) | set(kg_wo_sl['unified_id_B']))
    n_relations = len(set(kg_wo_sl['type(r)']))

    kg2id_np = np.asarray(kg_wo_sl.values)

    kg2dict = dict()
    for triple in kg2id_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]

        if head not in kg2dict:
            kg2dict[head] = []
        kg2dict[head].append((tail, relation))

        # treat the KG as an undirected graph
        if tail not in kg2dict:
            kg2dict[tail] = []
        kg2dict[tail].append((head, relation))
    if neighbor_sample_size == 'slgnn':
        adj_entity = kg_wo_sl[['unified_id_A','unified_id_B']].values
        adj_relation = kg_wo_sl['type(r)'].values
    else:
        isolated_point = []
        adj_entity = np.zeros([n_entity, neighbor_sample_size], dtype=np.int64)
        adj_relation = np.zeros([n_entity, neighbor_sample_size], dtype=np.int64)
        for entity, entity_name in enumerate(kg2dict.keys()):
            if (entity in kg2dict.keys()):
                neighbors = kg2dict[entity]
            else:
                neighbors = [(entity, 24)]
                isolated_point.append(entity)

            n_neighbors = len(neighbors)
            if n_neighbors >= neighbor_sample_size:
                sampled_indices = np.random.choice(list(range(n_neighbors)), size=neighbor_sample_size, replace=False)
            else:
                sampled_indices = np.random.choice(list(range(n_neighbors)), size=neighbor_sample_size, replace=True)
            adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
            adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    return n_node_a, n_node_b, n_entity, n_relations, adj_entity, adj_relation


# def save_epoch_result(raw_res, model_name, p_n):
#     col_names = [
#         'Model',
#         'pos/neg',
#         'AUROC',
#         'F1',
#         'AUPR',
#         'NDCG@10',
#         'NDCG@20',
#         'NDCG@50',
#         'Recall@10',
#         'Recall@20',
#         'Recall@50',
#         'Precision@10',
#         'Precision@20',
#         'Precision@50'
#     ]
#     epoch_res = pd.DataFrame(columns=col_names)
#     for res in raw_res:
#         res = [model_name, str(p_n)] + list(np.hstack(res))
#         row = pd.DataFrame(np.asarray(res).reshape(1, -1), columns=col_names)
#         epoch_res = pd.concat([epoch_res, row])

# GCATSL
def normalize_features(feat):
    degree = np.asarray(feat.sum(1)).flatten()
    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)
    return feat_norm

def load_gcatsl_features():
    pca = PCA(n_components=128)
    scaler = MinMaxScaler(feature_range=(0, 1))

    ppi_spm = sp.load_npz('../data/preprocessed_data/ppi_sparse_upper_matrix_without_sl_relation_9845.npz')
    ppi_spm = ppi_spm + ppi_spm.T
    ppi_spm = ppi_spm.toarray()
    ppi_spm = pca.fit_transform(ppi_spm)
    ppi_spm = scaler.fit_transform(ppi_spm)
    ppi_spm = normalize_features(ppi_spm)
    go_sim = np.load('../data/preprocessed_data/final_gosim_bp_from_r_9845.npy')
    # go_sim = go_sim + go_sim.T
    # go_sim = go_sim.toarray()
    go_sim = pca.fit_transform(go_sim)
    go_sim = scaler.fit_transform(go_sim)
    go_sim = normalize_features(go_sim)
    go_sim_cc = np.load('../data/preprocessed_data/final_gosim_cc_from_r_9845.npy')
    # go_sim_cc = go_sim_cc + go_sim_cc.T
    # go_sim_cc = go_sim_cc.toarray()
    go_sim_cc = pca.fit_transform(go_sim_cc)
    go_sim_cc = scaler.fit_transform(go_sim_cc)
    go_sim_cc = normalize_features(go_sim_cc)

    features_list_ori = [ppi_spm, go_sim, go_sim_cc]

    return features_list_ori

# SLMGAE

def build_KNN_mateix(S, nn_size):
    m, n = S.shape
    X = np.zeros((m, n))
    for i in range(m):
        ii = np.argsort(S[i, :])[::-1][:min(nn_size, n)]
        X[i, ii] = S[i, ii]
    return X


def array2coo(S, t=0):
    m, n = np.shape(S)
    row, col = [], []
    for i in range(len(S)):
        for j in range(i, len(S[i])):
            if S[i][j] > t:
                row.append(i)
                col.append(j)

    coo_matrix = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(m, n))
    coo_matrix = coo_matrix + coo_matrix.T

    return coo_matrix

# def load_slmgae_feature(knn, nn_size):
#     feature1 = np.load('./data/preprocessed_data/final_gosim_bp_from_r_9845.npy')
#     # feature1 = sp.load_npz('./data/preprocessed_data/gosim_bp_upper_tri_spm_9845.npz')
#     # feature1 = feature1 + feature1.T
#     if knn == True:
#         feature1 = build_KNN_mateix(feature1, nn_size=nn_size)
#     feature1 = feature1 + feature1.T
#     feature1 = array2coo(feature1)

#     feature2 = np.load('./data/preprocessed_data/final_gosim_cc_from_r_9845.npy')
#     # feature1 = sp.load_npz('./data/preprocessed_data/gosim_cc_upper_tri_spm_9845.npz')
#     # feature2 = feature2 + feature2.T
#     if knn == True:
#         feature2 = build_KNN_mateix(feature2, nn_size=nn_size)
#     feature2 = feature2 + feature2.T
#     feature2 = array2coo(feature2)

#     feature3 = np.load('./data/preprocessed_data/ppi_sparse_upper_matrix_without_sl_relation_9845.npy')
#     # feature3 = sp.load_npz('./data/preprocessed_data/ppi_sparse_upper_matrix_without_sl_relation_9845.npz')
#     # feature3 = feature3 + feature3.T
#     # feature3 = feature3.tocoo()
#     feature3 = feature3

#     adjs_orig = [feature1, feature2, feature3]

#     return adjs_orig
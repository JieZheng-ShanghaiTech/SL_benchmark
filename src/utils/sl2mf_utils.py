# import pdb
import numpy as np
from collections import defaultdict
# from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score
from sklearn.metrics import auc
from scipy import sparse
import scipy.io as sio

# random.seed(123)
np.random.seed(456)
# tf.compat.v1.set_random_seed(123)

# torch.manual_seed(123)
# torch.cuda.manual_seed_all(123)

def load_data_from_file(dataset, folder):
    int_file_name = folder + dataset + "_admat_dgc.txt"  # the target-drug interaction file
    with open(int_file_name, "r") as inf:
        inf.next()
        int_array = [line.strip("\n").split()[1:] for line in inf]

    with open(folder + dataset + "_simmat_dc.txt", "r") as inf:  # the drug similarity file
        inf.next()
        drug_sim = [line.strip("\n").split()[1:] for line in inf]

    with open(folder + dataset + "_simmat_dg.txt", "r") as inf:  # the target similarity file
        inf.next()
        target_sim = [line.strip("\n").split()[1:] for line in inf]

    intMat = np.array(int_array, dtype=np.float64).T
    drugMat = np.array(drug_sim, dtype=np.float64)
    targetMat = np.array(target_sim, dtype=np.float64)
    return intMat, drugMat, targetMat


def get_drugs_targets_names(dataset, folder):
    int_file_name = folder + dataset + "_admat_dgc.txt"
    with open(int_file_name, "r") as inf:
        drugs = inf.next().strip("\n").split()
        targets = [line.strip("\n").split()[0] for line in inf]
    return drugs, targets


def cross_validation(intMat, seeds, cv=0, num=10):
    cv_data = defaultdict(list)
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cv == 0:
            index = prng.permutation(num_drugs)
        if cv == 1:
            index = prng.permutation(intMat.size)
        step = index.size / num
        for i in range(num):
            if i < num - 1:
                ii = index[i * step:(i + 1) * step]
            else:
                ii = index[i * step:]
            if cv == 0:
                test_data = np.array([[k, j] for k in ii for j in range(num_targets)], dtype=np.int32)
            elif cv == 1:
                test_data = np.array([[k / num_targets, k % num_targets] for k in ii], dtype=np.int32)
            x, y = test_data[:, 0], test_data[:, 1]
            test_label = intMat[x, y]
            W = np.ones(intMat.shape)
            W[x, y] = 0
            cv_data[seed].append((W, test_data, test_label))
    return cv_data


def kfold_cv(intMat, seed, cvs=3, num_folds=10, nest_cv=True):
    m, n = intMat.shape
    prng = np.random.RandomState(seed)
    cv_data = []
    if cvs == 1:
        kf = KFold(intMat.size, n_folds=num_folds, shuffle=True, random_state=prng)
    elif cvs == 2:
        kf = KFold(m, n_folds=num_folds, shuffle=True, random_state=prng)
    elif cvs == 3:
        kf = KFold(n, n_folds=num_folds, shuffle=True, random_state=prng)
    for train, test in kf:
        W = np.ones(intMat.shape)
        if cvs == 1:
            x, y = test / n, test % n
        elif cvs == 2:
            x = np.repeat(test, n)
            y = np.tile(np.arange(n), test.size)
        elif cvs == 3:
            x = np.tile(np.arange(m), test.size)
            y = np.repeat(test, m)
        W[x, y] = 0
        if nest_cv:
            inner_kf = KFold(train.size, n_folds=3, shuffle=True, random_state=prng)
            inner_cv_data = []
            for inner_train, inner_test in inner_kf:
                if cvs == 1:
                    x1, y1 = train[inner_test] / n, train[inner_test] % n
                elif cvs == 2:
                    x1 = np.repeat(train[inner_test], n)
                    y1 = np.tile(np.arange(n), inner_test.size)
                elif cvs == 3:
                    x1 = np.tile(np.arange(m), inner_test.size)
                    y1 = np.repeat(train[inner_test], m)
                W1 = W.copy()
                W1[x1, y1] = 0
                inner_cv_data.append((W1, x1, y1, intMat[x1, y1]))
            cv_data.append((inner_cv_data, W, x, y, intMat[x, y]))
        else:
            cv_data.append((W, x, y, intMat[x, y]))
    return cv_data


def train(model, cv_data, intMat, drugMat, targetMat, N=5):
    aupr, auc = [], []
    for seed in cv_data.keys():
        for W, test_data, test_label in cv_data[seed]:
            model.fix_model(W, intMat, drugMat, targetMat, seed)
            # model.fix_model(W*intMat, drugMat, targetMat, seed)
            scores = model.predict_scores(test_data)
            aupr_val, auc_val = evaluation(scores, test_label.astype(int))
            aupr.append(aupr_val)
            auc.append(auc_val)
    return np.array(aupr, dtype=np.float64), np.array(auc, dtype=np.float64)


def evaluation(scores, test_label):
    prec, rec, thr = precision_recall_curve(test_label, scores)
    aupr_val = auc(rec, prec)
    fpr, tpr, thr = roc_curve(test_label, scores)
    auc_val = auc(fpr, tpr)
    return aupr_val, auc_val


def svd_init(M, num_factors):
    from scipy.linalg import svd
    U, s, V = svd(M, full_matrices=False)
    ii = np.argsort(s)[::-1][:num_factors]
    s1 = np.sqrt(np.diag(s[ii]))
    U0, V0 = U[:, ii].dot(s1), s1.dot(V[ii, :])
    return U0, V0.T


def mean_confidence_interval(data, confidence=0.95):
    import scipy as sp
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h


def write_metric_vector_to_file(auc_vec, file_name):
    np.savetxt(file_name, auc_vec, fmt='%.6f')


def load_metric_vector(file_name):
    return np.loadtxt(file_name, dtype=np.float64)


def statistical_significance_analysis():
    # import pdb
    import scipy.stats as st
    for cv in ["1", ]:
        print("cross_validation setting:" + cv)
        for dataset in ["nr", "gpcr", "ic", "e"]:
            nrlmf_auc = load_metric_vector("../output/nrlmf_auc_" + cv + "_" + dataset + ".txt")
            nrlmf_aupr = load_metric_vector("../output/nrlmf_aupr_" + cv + "_" + dataset + ".txt")
            for cp in ["netlaprls", "blm", "wnnrls", "kbmf", "cmf"]:
                cp_auc = load_metric_vector("../output/" + cp + "_auc_" + cv + "_" + dataset + ".txt")
                cp_aupr = load_metric_vector("../output/" + cp + "_aupr_" + cv + "_" + dataset + ".txt")
                x1, y1 = st.ttest_ind(nrlmf_auc, cp_auc)
                x2, y2 = st.ttest_ind(nrlmf_aupr, cp_aupr)
                print(dataset, cp, x1, y1, x2, y2)
            print("")


def load_ppi_data():
    with open("datasets/List_Proteins_in_SL.txt", "r") as inf:
        ppis = [line.rstrip() for line in inf]
        id_mapping = dict(zip(ppis, range(len(set(ppis)))))
    num = len(ppis)
    tp_pairs, tp_sim = [], []
    with open("datasets/PPI_TopologySim.txt", "r") as inf:
        for line in inf:
            id1, id2, s = line.rstrip().split()
            # pdb.set_trace()
            tp_pairs.append((id_mapping[id1], id_mapping[id2]))
            tp_sim.append(float(s))
    tp_pairs = np.array(tp_pairs, dtype=np.int32)
    tp_sim_mat = sparse.coo_matrix((tp_sim, (tp_pairs[:, 0], tp_pairs[:, 1])), shape=(num, num))
    inter_pairs, inter_scores = [], []
    with open("datasets/SL_Human_FinalCheck.txt", "r") as inf:
        for line in inf:
            id1, id2, s = line.rstrip().split()
            inter_pairs.append((id_mapping[id1], id_mapping[id2]))
            inter_scores.append(float(s))
    inter_pairs = np.array(inter_pairs, dtype=np.int32)
    inter_scores = np.array(inter_scores)
    go_pairs, go_sim = [], []
    with open("datasets/Human_GOsim.txt", "r") as inf:
        for i, line in enumerate(inf):
            data = line.rstrip().split()
            for j, s in enumerate(data):
                go_pairs.append((i, num - len(data) + j))
                go_sim.append(float(s))
    go_pairs = np.array(go_pairs, dtype=np.int32)
    go_sim_mat = sparse.coo_matrix((go_sim, (go_pairs[:, 0], go_pairs[:, 1])), shape=(num, num))
    return inter_pairs, inter_scores, tp_sim_mat, go_sim_mat, id_mapping


def load_ppi_data_long(flag):
    if flag == 0:
        with open("datasets/List_Proteins_in_SL.txt", "r") as inf:
            ppis = [line.rstrip() for line in inf]
            id_mapping = dict(zip(ppis, range(len(set(ppis)))))
        num = len(ppis)
        inter_pairs, inter_scores = [], []
        with open("datasets/SL_Human_FinalCheck.txt", "r") as inf:
            for line in inf:
                id1, id2, s = line.rstrip().split()
                inter_pairs.append((id_mapping[id1], id_mapping[id2]))
                inter_scores.append(float(s))
        inter_pairs = np.array(inter_pairs, dtype=np.int32)
        inter_scores = np.array(inter_scores)
        go_pairs, go_sim = [], []
        with open("datasets/Human_GOsim.txt", "r") as inf:
            for i, line in enumerate(inf):
                data = line.rstrip().split()
                for j, s in enumerate(data):
                    go_pairs.append((i, num - len(data) + j))
                    go_sim.append(float(s))
        go_pairs = np.array(go_pairs, dtype=np.int32)
        go_sim_mat = sparse.coo_matrix((go_sim, (go_pairs[:, 0], go_pairs[:, 1])), shape=(num, num))

        GOsim_CC = sio.loadmat('datasets/Human_GOsim_CC.mat')
        go_sim_cc_mat = GOsim_CC['Human_GOsim_CC']

        ppi_sparse = sio.loadmat('datasets/gene_ppi_sparse.mat')
        ppi_sparse_mat = ppi_sparse['gene_ppi_sparse']

        co_pathway = sio.loadmat('datasets/gene_co_pathway.mat')
        co_pathway_mat = co_pathway['gene_co_pathway']

    elif flag == 1:
        with open("SynlethDB_extension/List_Proteins_in_SL.txt", "r") as inf:
            ppis = [line.rstrip() for line in inf]
            id_mapping = dict(zip(ppis, range(len(set(ppis)))))
        num = len(ppis)
        inter_pairs, inter_scores = [], []
        with open("SynlethDB_extension/SL_Human_FinalCheck.txt", "r") as inf:
            for line in inf:
                id1, id2, s = line.rstrip().split()
                if id1 in id_mapping and id2 in id_mapping:
                    if id_mapping[id1] > id_mapping[id2]:
                        inter_pairs.append((id_mapping[id2], id_mapping[id1]))
                    elif id_mapping[id1] < id_mapping[id2]:
                        inter_pairs.append((id_mapping[id1], id_mapping[id2]))
                # inter_scores.append(float(s))
        inter_pairs = np.array(inter_pairs, dtype=np.int32)
        inter_scores = np.array(inter_scores)

        go_sim_mat = sio.loadmat('SynlethDB_extension/gene_similarity_BP.mat')
        go_sim_mat = go_sim_mat['gene_similarity_BP']

        go_sim_cc_mat = sio.loadmat('SynlethDB_extension/gene_similarity_CC.mat')
        go_sim_cc_mat = go_sim_cc_mat['gene_similarity_CC']

        ppi_sparse_mat = sio.loadmat('SynlethDB_extension/gene_ppi_sparse.mat')
        ppi_sparse_mat = ppi_sparse_mat['gene_ppi_sparse']

        co_pathway_mat = sio.loadmat('SynlethDB_extension/gene_ppi_pathway.mat')
        co_pathway_mat = co_pathway_mat['gene_ppi_pathway']

        return inter_pairs, go_sim_mat, go_sim_cc_mat, ppi_sparse_mat, co_pathway_mat, id_mapping


def evalution_bal(adj_rec, edges_pos, edges_neg):
    # Predict on test set of edges
    preds = []
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]])

    preds_neg = []

    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]])
        if len(preds_neg) == len(preds):
            break

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    fpr, tpr, th = roc_curve(labels_all, preds_all)
    roc_score = auc(fpr, tpr)

    prec, rec, th = precision_recall_curve(labels_all, preds_all)
    aupr_score = auc(rec, prec)
    labels_all = labels_all.astype(np.float32)
    # accuracy_score_ = accuracy_score(labels_all, preds_all)

    return roc_score, aupr_score

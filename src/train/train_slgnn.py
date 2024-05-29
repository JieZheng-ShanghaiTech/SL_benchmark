import argparse
import time
import numpy as np
import torch
from tqdm import tqdm
import wandb
import os
import random
import scipy.sparse as sp
import torch.optim as optim
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.metrics import f1_score
from preprocess import load_kg, cal_metrics, ChecktoSave

from models.slgnn import SLModel
from utils.slgnn_utils import load_data, KGCNDataset, SLDataset


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

random.seed(456)
np.random.seed(456)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
cuda_device = torch.device('cuda:0')

def train_slgnn(parameters, pos_samples, neg_samples, mode=None, save_mat=False, ex_compt=None,indep_test=None):
    if indep_test:
        graph_train_pos_kfold, _, _, train_pos_kfold, valid_pos_kfold, test_pos_kfold = pos_samples
        _, _, _, train_neg_kfold, valid_neg_kfold, test_neg_kfold = neg_samples
    else:
        graph_train_pos_kfold, _, train_pos_kfold, test_pos_kfold = pos_samples
        _, _, train_neg_kfold, test_neg_kfold = neg_samples

    kfold = parameters['kfold']
    inverse_r = parameters['inverse_r']
    n_factors = parameters['n_factors']
    node_dropout = parameters['node_dropout']
    node_dropout_rate = parameters['node_dropout_rate']
    mess_dropout = parameters['mess_dropout']
    mess_dropout_rate = parameters['mess_dropout_rate']
    ind = parameters['ind']
    n_epochs = parameters['n_epochs']
    batch_size = parameters['batch_size']
    earlystop_flag = parameters['earlystop_flag']
    dim = parameters['dim']
    context_hops = parameters['n_hop']
    sim_regularity = parameters['sim_regularity']
    l2_weight = parameters['l2_weight']
    lr = parameters['lr']
    num_node = parameters['num_node']
    p_n = parameters['pos_neg']
    d_s = parameters['division_strategy']
    n_s = parameters['negative_strategy']
    
    base_suffix = '_score_mats'
    if ex_compt:
        base_suffix = '_score_mats_wo_compt'
    
    if mode == 'final_res':
        kfold=1
        p_n = 'final_res'
        d_s = 'final_res'
        n_s = 'final_res'

    run=wandb.init(
        # set the wandb project where this run will be logged
        project="Benchmarking",
        group="SLMGNN",
        job_type=f"{p_n}_{d_s}_{n_s}",
        # track hyperparameters and run metadata
        config=parameters
    )
    
    checktosave = ChecktoSave(kfold)

    _, _, n_entities, n_relations, edge_index, edge_type = load_kg('slgnn')

    for fold_num in range(kfold):
        print('5-fold-validation:{}'.format(fold_num + 1))
        graph_train_pos = graph_train_pos_kfold[fold_num]
        graph_sl = graph_train_pos
        graph_sl += sp.eye(graph_sl.shape[0], format='csr')
        
        train_pos_data = train_pos_kfold[fold_num]
        train_neg_data = train_neg_kfold[fold_num]
        train_pos_data = np.hstack([train_pos_data, np.ones((len(train_pos_data), 1))])
        train_neg_data = np.hstack([train_neg_data, np.zeros((len(train_neg_data), 1))])

        print(train_pos_data.shape)
        print(train_neg_data.shape)
        train_data_con = np.vstack([train_pos_data, train_neg_data]).astype('int')
        ind = list(range(len(train_data_con)))
        random.shuffle(ind)
        train_data = train_data_con[ind]
        
        test_pos_data = test_pos_kfold[fold_num]
        test_neg_data = test_neg_kfold[fold_num]
        test_pos_data = np.hstack([test_pos_data, np.ones((len(test_pos_data), 1))])
        test_neg_data = np.hstack([test_neg_data, np.zeros((len(test_neg_data), 1))])
        test_data_con = np.vstack([test_pos_data, test_neg_data]).astype('int')

        ind = list(range(len(test_data_con)))
        random.shuffle(ind)
        test_data = test_data_con[ind]
        print('Prepare data set')
        if indep_test:
            valid_pos_data = valid_pos_kfold[fold_num]
            valid_neg_data = valid_neg_kfold[fold_num]
            valid_pos_data = np.hstack([valid_pos_data, np.ones((len(valid_pos_data), 1))])
            valid_neg_data = np.hstack([valid_neg_data, np.zeros((len(valid_neg_data), 1))])

            valid_data_con = np.vstack([valid_pos_data, valid_neg_data]).astype('int')
            ind = list(range(len(valid_data_con)))
            random.shuffle(ind)
            valid_data = valid_data_con[ind]
            
            train_dataset = KGCNDataset(train_data)
            valid_dataset = KGCNDataset(valid_data)
        else:
            train_dataset = KGCNDataset(train_data)
            valid_dataset = KGCNDataset(test_data)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=batch_size)
        
        reindex_dict = {i:i for i in range(num_node)}

        model = SLModel(num_node, n_relations, n_entities, l2_weight,sim_regularity,dim,context_hops,n_factors,
                        node_dropout,node_dropout_rate,mess_dropout,mess_dropout_rate,ind,cuda_device,edge_index,edge_type,
                        graph_sl,reindex_dict).to(cuda_device)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        cnt_wait = 0
        best_valid_auc = 0
        best_test_auc = 0
        best_test_aupr = 0
        best_test_f1 = 0
        print('train')
        for epoch in range(n_epochs):
            labels_train = torch.tensor([]).to(cuda_device)
            logits_train = torch.tensor([]).to(cuda_device)
            running_loss = 0
            train_cor = 0
            for i, (gene_a, gene_b, labels) in enumerate(train_loader):
                # import pdb; pdb.set_trace()
                gene_a = gene_a.to(cuda_device)
                gene_b = gene_b.to(cuda_device)
                labels = labels.to(cuda_device)
                logits, emb_loss, cor_loss, cor, _ = model(gene_a, gene_b)
                labels_train = torch.cat((labels_train, labels))
                logits_train = torch.cat((logits_train, logits))
                loss = criterion(logits, labels)
                loss = loss + emb_loss + cor_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                train_cor += cor.item()
            running_loss = running_loss / len(train_loader)
            train_cor = train_cor / len(train_loader)

            with torch.no_grad():
                valid_loss = 0
                total_roc = 0
                valid_cor = 0
                labels_valid = torch.tensor([]).to(cuda_device)
                logits_valid = torch.tensor([]).to(cuda_device)
                for i, (gene_a, gene_b, labels) in enumerate(valid_loader):
                    gene_a, gene_b, labels = gene_a.to(cuda_device), gene_b.to(cuda_device), labels.to(cuda_device)
                    logits, _, _, cor, _ = model(gene_a, gene_b)
                    labels_valid = torch.cat((labels_valid, labels))
                    logits_valid = torch.cat((logits_valid, logits))
                    valid_cor += cor.item()
                valid_loss = criterion(logits_valid, labels_valid).item()
                valid_cor = valid_cor / len(valid_loader)
                valid_auc = roc_auc_score(labels_valid.cpu().detach().numpy(),
                                        logits_valid.cpu().detach().numpy())
                prec, reca, _ = precision_recall_curve(labels_valid.cpu().detach().numpy(),logits_valid.cpu().detach().numpy())

                valid_f1 = f1_score(
                    labels_valid.cpu().detach().numpy(),
                    (torch.round(logits_valid)).cpu().detach().numpy())
                valid_aupr = auc(reca, prec)


            print(
                '[Epoch {}] train_loss:{:.4f}, valid_loss:{:.4f}, valid_auc:{:.4f}, valid_aupr:{:.4f}, valid_f1:{:.4f}, cor:{:.4f}'
                .format(epoch + 1, (running_loss), (valid_loss), (valid_auc),
                        (valid_aupr), (valid_f1), (train_cor)))
            
            if valid_auc > best_valid_auc:
                best_valid_auc = valid_auc
                best_model = model
                cnt_wait = 0
            else:
                cnt_wait += 1
            if (cnt_wait == 5):
                print('Early stopped.')
                break
            
        print('Overall best test_auc:{:.4f}, test_aupr:{:.4f}, test_f1:{:.4f}'.
            format(best_test_auc, best_test_aupr, best_test_f1))
        
        print(f'Final Testing ...')
        ti = time.time()
        score_mat = final_test(best_model, batch_size, num_node)
        print(time.time()-ti)
        if sp.isspmatrix(score_mat):
            score_mat = score_mat.todense()
        score_mat = np.asarray(score_mat)
        score_mat[range(num_node), range(num_node)] = 0
        train_metrics = cal_metrics(score_mat, train_pos_kfold[fold_num], train_neg_kfold[fold_num])
        checktosave.update_train_classify(fold_num, 0, np.asarray([train_metrics[0], train_metrics[2], train_metrics[1]]))
        checktosave.update_train_ranking(fold_num, 0, train_metrics[3:])
        run.log({
                'train_auc':train_metrics[0],'train_f1':train_metrics[1],'train_aupr':train_metrics[2],
                'train_N10':train_metrics[3],'train_N20':train_metrics[4],'train_N50':train_metrics[5],
                'train_R10':train_metrics[6],'train_R20':train_metrics[7],'train_R50':train_metrics[8],
                'train_P10':train_metrics[9],'train_P20':train_metrics[10],'train_P50':train_metrics[11],
                'train_M10':train_metrics[12],'train_M20':train_metrics[13],'train_M50':train_metrics[14],
            })
        
        if indep_test:
            valid_metrics = cal_metrics(score_mat, valid_pos_kfold[fold_num], valid_neg_kfold[fold_num],train_pos_kfold[fold_num])
            run.log({
                    'valid_auc':valid_metrics[0],'valid_f1':valid_metrics[1],'valid_aupr':valid_metrics[2],
                    'valid_N10':valid_metrics[3],'valid_N20':valid_metrics[4],'valid_N50':valid_metrics[5],
                    'valid_R10':valid_metrics[6],'valid_R20':valid_metrics[7],'valid_R50':valid_metrics[8],
                    'valid_P10':valid_metrics[9],'valid_P20':valid_metrics[10],'valid_P50':valid_metrics[11],
                    'valid_M10':valid_metrics[12],'valid_M20':valid_metrics[13],'valid_M50':valid_metrics[14],
                })
            
            test_metrics = cal_metrics(score_mat, test_pos_kfold[fold_num], test_neg_kfold[fold_num],train_pos_kfold[fold_num])
            run.log({
                    'test_auc':test_metrics[0],'test_f1':test_metrics[1],'test_aupr':test_metrics[2],
                    'test_N10':test_metrics[3],'test_N20':test_metrics[4],'test_N50':test_metrics[5],
                    'test_R10':test_metrics[6],'test_R20':test_metrics[7],'test_R50':test_metrics[8],
                    'test_P10':test_metrics[9],'test_P20':test_metrics[10],'test_P50':test_metrics[11],
                    'test_M10':test_metrics[12],'test_M20':test_metrics[13],'test_M50':test_metrics[14],
                })
            print(f'{test_metrics}')
            
            if save_mat:
                if checktosave.update_classify(fold_num, 0, np.asarray([valid_metrics[0], valid_metrics[2], valid_metrics[1]])):
                    # print('Saving score matrix ...')
                    if not os.path.exists(f'../results/{n_s}{base_suffix}/slgnn'):
                        os.makedirs(f'../results/{n_s}{base_suffix}/slgnn')
                    path = f'../results/{n_s}{base_suffix}/slgnn/slgnn_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_classify.npy'
                    checktosave.save_mat(path, score_mat)
                    checktosave.update_indep_test_classify(fold_num, 0, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]]))
                if checktosave.update_ranking(fold_num, 0, valid_metrics[3:]):
                    # print('Saving score matrix ...')
                    path = f'../results/{n_s}{base_suffix}/slgnn/slgnn_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_ranking.npy'
                    checktosave.save_mat(path, score_mat)
                    checktosave.update_indep_test_ranking(fold_num, 0, test_metrics[3:])
            else:
                if checktosave.update_classify(fold_num, 0, np.asarray([valid_metrics[0], valid_metrics[2], valid_metrics[1]])):
                    checktosave.update_indep_test_classify(fold_num, 0, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]]))
                if checktosave.update_ranking(fold_num, 0, valid_metrics[3:]):
                    checktosave.update_indep_test_ranking(fold_num, 0, test_metrics[3:])
        else:
            test_metrics = cal_metrics(score_mat, test_pos_kfold[fold_num], test_neg_kfold[fold_num],train_pos_kfold[fold_num])
            run.log({
                    'test_auc':test_metrics[0],'test_f1':test_metrics[1],'test_aupr':test_metrics[2],
                    'test_N10':test_metrics[3],'test_N20':test_metrics[4],'test_N50':test_metrics[5],
                    'test_R10':test_metrics[6],'test_R20':test_metrics[7],'test_R50':test_metrics[8],
                    'test_P10':test_metrics[9],'test_P20':test_metrics[10],'test_P50':test_metrics[11],
                    'test_M10':test_metrics[12],'test_M20':test_metrics[13],'test_M50':test_metrics[14],
                })
            print(f'{test_metrics}')
            
            if save_mat:
                if checktosave.update_classify(fold_num, 0, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]])):
                    # print('Saving score matrix ...')
                    if not os.path.exists(f'../results/{n_s}{base_suffix}/slgnn'):
                        os.makedirs(f'../results/{n_s}{base_suffix}/slgnn')
                    path = f'../results/{n_s}{base_suffix}/slgnn/slgnn_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_classify.npy'
                    checktosave.save_mat(path, score_mat)
                if checktosave.update_ranking(fold_num, 0, test_metrics[3:]):
                    # print('Saving score matrix ...')
                    path = f'../results/{n_s}{base_suffix}/slgnn/slgnn_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_ranking.npy'
                    checktosave.save_mat(path, score_mat)
            else:
                checktosave.update_classify(fold_num, 0, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]]))
                checktosave.update_ranking(fold_num, 0, test_metrics[3:])
        
        import gc
        del model
        del best_model
        gc.collect()  # Python的垃圾回收器
        torch.cuda.empty_cache()  # 清空未使用的缓存
    
    run.finish()
    if indep_test:
        all_metrics = checktosave.get_all_indep_test_metrics()
    else:
        all_metrics = checktosave.get_all_metrics()
    
    return all_metrics

def final_test(model, batch_size, num_node):
    
    with torch.no_grad():
        rol, col = np.triu_indices(num_node, k=1)

        data = np.vstack([rol, col, np.zeros(len(rol))]).T
        test_dataset = KGCNDataset(data)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=batch_size*10)
        
        iterable = tqdm(total=len(test_loader), desc='Testing', leave=True)
        scores_list = []
        for i, (gene_a, gene_b, labels) in enumerate(test_loader):
            gene_a, gene_b = gene_a.long().to(cuda_device), gene_b.long().to(cuda_device)
            _, _, _, _, scores = model(gene_a, gene_b)
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            scores_list.append(scores)
            iterable.update(1)

        scores_flat = np.concatenate(scores_list)
        score_mat = sp.csr_matrix((scores_flat[:len(rol)], (rol, col)), shape=(num_node, num_node))
        score_mat = score_mat + score_mat.T
    return score_mat
import copy
import time
import torch
import numpy as np
import pandas as pd
import wandb
from preprocess import cal_metrics, ChecktoSave
import os

from models.mge4sl import MultiGraphEnsembleFC, MultiGraphEnsembleFC_SUM, MultiGraphEnsembleCNN, \
    MultiGraphEnsembleWeightFC, SynlethDB, SynlethDB_KG
from utils.mge4sl_utils import get_k_fold_data_random_neg, construct_kg_sldb, train, test, cal_confidence_interval, get_all_score_mat
from torch_geometric.utils import to_undirected

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

np.random.seed(456)
torch.manual_seed(456)
torch.cuda.manual_seed_all(456)

cuda_device=torch.device('cuda:0')

def train_mge4sl(parameters, pos_samples, neg_samples, mode=None, save_mat=False):
    data_path = '../data/preprocessed_data/mge4sl_processed_data.csv'

    _, _, train_pos_kfold, test_pos_kfold = pos_samples
    _, _, train_neg_kfold, test_neg_kfold = neg_samples

    kfold = parameters['kfold']
    epochs = parameters['epochs']
    lr = parameters['lr']
    p_n = parameters['pos_neg']
    d_s = parameters['division_strategy']
    num_nodes = parameters['num_node']
    n_s = parameters['negative_strategy']
    
    if mode=='final_res':
        kfold=1
        p_n = d_s = n_s = 'final_res'

    wandb.init(
        # set the wandb project where this run will be logged
        project="Benchmarking",
        group="MGE4SL",
        job_type=f"{p_n}_{d_s}_{n_s}",
        # track hyperparameters and run metadata
        config=parameters
    )
    
    data = pd.read_csv(data_path)
    checktosave = ChecktoSave(kfold)
    for fold_num in range(kfold):
        sl_data = train_pos_kfold[fold_num]
        nosl_data = train_neg_kfold[fold_num]
        sl_test_data = test_pos_kfold[fold_num]
        nosl_test_data = test_neg_kfold[fold_num]

        synlethdb = SynlethDB(num_nodes, sl_data, nosl_data)

        synlethdb.train_pos_edge_index = torch.tensor(sl_data.T)
        synlethdb.train_pos_edge_index = to_undirected(synlethdb.train_pos_edge_index)
        synlethdb.train_neg_edge_index = torch.tensor(nosl_data.T)
        synlethdb.train_neg_edge_index = to_undirected(synlethdb.train_neg_edge_index)

        synlethdb.val_pos_edge_index = torch.tensor(sl_test_data.T)
        # synlethdb.val_pos_edge_index = to_undirected(synlethdb.val_pos_edge_index)
        synlethdb.val_neg_edge_index = torch.tensor(nosl_test_data.T)
        # synlethdb.val_neg_edge_index = to_undirected(synlethdb.val_neg_edge_index)

        k_data = synlethdb

        synlethdb_ppi, synlethdb_rea, synlethdb_cor, synlethdb_go_F, \
        synlethdb_go_C, synlethdb_go_P, synlethdb_kegg = construct_kg_sldb(num_nodes, data, sl_data, nosl_data)

        model = MultiGraphEnsembleFC(n_graph=8, node_emb_dim=16, sl_input_dim=1, kg_input_dim=1)
        # model = MultiGraphEnsembleFC_SUM(n_graph=8, node_emb_dim=16, sl_input_dim=1, kg_input_dim=1)
        # model = MultiGraphEnsembleCNN(n_graph=8, node_emb_dim=16, sl_input_dim=1, kg_input_dim=1)
        # model = MultiGraphEnsembleWeightFC(n_graph=7, node_emb_dim=16, sl_input_dim=1, kg_input_dim=1)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        explr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

        print(f'start {fold_num} fold:')

        for epoch in range(0, epochs):
            train_loss = train(model, optimizer, k_data, synlethdb_ppi, synlethdb_rea, synlethdb_cor, synlethdb_go_F, \
                               synlethdb_go_C, synlethdb_go_P, synlethdb_kegg)
            wandb.log({
                'train_loss':train_loss
            })
            if (epoch+1) % 1 == 0:
                all_score_matrix = get_all_score_mat(model, num_nodes, k_data, synlethdb_ppi, synlethdb_rea, synlethdb_cor, synlethdb_go_F, \
                                    synlethdb_go_C, synlethdb_go_P, synlethdb_kegg)
                train_metrics = cal_metrics(all_score_matrix, train_pos_kfold[fold_num],train_neg_kfold[fold_num])
                checktosave.update_train_classify(fold_num, epoch, np.asarray([train_metrics[0], train_metrics[2], train_metrics[1]]))
                checktosave.update_train_ranking(fold_num, epoch, train_metrics[3:])
                wandb.log({
                    'train_auc':train_metrics[0],'train_f1':train_metrics[1],'train_aupr':train_metrics[2],
                    'train_N10':train_metrics[3],'train_N20':train_metrics[4],'train_N50':train_metrics[5],
                    'train_R10':train_metrics[6],'train_R20':train_metrics[7],'train_R50':train_metrics[8],
                    'train_P10':train_metrics[9],'train_P20':train_metrics[10],'train_P50':train_metrics[11],
                })
                test_metrics = cal_metrics(all_score_matrix, test_pos_kfold[fold_num],test_neg_kfold[fold_num], train_pos_kfold[fold_num])
                wandb.log({
                    'test_auc':test_metrics[0],'test_f1':test_metrics[1],'test_aupr':test_metrics[2],
                    'test_N10':test_metrics[3],'test_N20':test_metrics[4],'test_N50':test_metrics[5],
                    'test_R10':test_metrics[6],'test_R20':test_metrics[7],'test_R50':test_metrics[8],
                    'test_P10':test_metrics[9],'test_P20':test_metrics[10],'test_P50':test_metrics[11],
                })
                
                if save_mat:
                    if checktosave.update_classify(fold_num, epoch, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]])):
                        # print('Saving score matrix ...')
                        if not os.path.exists(f'../results/{n_s}_score_mats/mge4sl'):
                            os.makedirs(f'../results/{n_s}_score_mats/mge4sl')
                        path = f'../results/{n_s}_score_mats/mge4sl/mge4sl_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_classify.npy'
                        checktosave.save_mat(path, all_score_matrix)
                    if checktosave.update_ranking(fold_num, epoch, test_metrics[3:]):
                        # print('Saving score matrix ...')
                        path = f'../results/{n_s}_score_mats/mge4sl/mge4sl_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_ranking.npy'
                        checktosave.save_mat(path, all_score_matrix)
                else:
                    checktosave.update_classify(fold_num, epoch, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]]))
                    checktosave.update_ranking(fold_num, epoch, test_metrics[3:])
                    
                print(test_metrics)
                
                explr_scheduler.step()

                log = 'Epoch: {:03d}, Loss: {:.4f}, Val_AUC: {:.4f}, Val_AUPR:{:.4f}, Val_F1:{:.4f},'
                print(log.format(epoch, train_loss, test_metrics[0], test_metrics[2], test_metrics[1]))
    wandb.finish()
    # auc f1 aupr N10 N20 N50 R10 R20 R50 P10 P20 P50
    all_metrics = checktosave.get_all_metrics()
    
    return all_metrics

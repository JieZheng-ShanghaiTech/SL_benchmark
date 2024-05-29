'''
Ref: https://github.com/hwwang55/KGNN-LS/blob/master/src/train.py
for i, (label, inp) in enumerate(loader):
    = model(adj_entity, adj_relation, inp)
'''
import torch
import copy
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
import random
import time
from models.nsf4sl import Net, SLDataset
from utils.nsf4sl_utils import loadKGData, evaluate, print_eval_results, cal_score_mat, map_genes, cal_final_result
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from preprocess import cal_metrics, ChecktoSave
import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

random.seed(456)
np.random.seed(456)
# tf.compat.v1.set_random_seed(123)

torch.manual_seed(456)
torch.cuda.manual_seed_all(456)

cuda_device=torch.device('cuda:0')

def train_nsf4sl(parameters, pos_samples, neg_samples, mode=None, save_mat=False, ex_compt=None,indep_test=None):
    
    p_name = parameters['p_name']
    train_ratio = parameters['train_ratio']
    aug_ratio = parameters['aug_ratio']
    weight_decay = parameters['weight_decay']
    lr = parameters['lr']
    batch_size = parameters['batch_size']
    epochs = parameters['epochs']
    early_stop = parameters['early_stop']
    latent_size = parameters['latent_size']
    momentum = parameters['momentum']
    gpu = parameters['gpu']
    num_node = parameters['num_node']
    p_n = parameters['pos_neg']
    d_s = parameters['division_strategy']
    kfold = parameters['kfold']
    n_s = parameters['negative_strategy']
    
    base_suffix = '_score_mats'
    if ex_compt:
        base_suffix = '_score_mats_wo_compt'
    
    if mode == 'final_res':
        kfold = 1
        n_s = 'final_res'
        d_s = 'final_res'
        p_n = 'final_res'

    wandb.init(
        # set the wandb project where this run will be logged
        project="Benchmarking",
        group="NSF4SL",
        job_type=f"{p_n}_{d_s}_{n_s}",
        # track hyperparameters and run metadata
        config=parameters
    )

    if indep_test:
        _, _, _, train_pos_kfold, valid_pos_kfold, test_pos_kfold = pos_samples
        _, _, _, train_neg_kfold, valid_neg_kfold, test_neg_kfold = neg_samples
    else:
        _, _, train_pos_kfold, test_pos_kfold = pos_samples
        _, _, train_neg_kfold, test_neg_kfold = neg_samples

    gene_kgemb, gene_id = loadKGData()

    id2orig, _ = map_genes()
    all_dataset = SLDataset(None, gene_kgemb, gene_id, aug_ratio)

    all_gene_feature = []
    for i in id2orig.keys():
        all_gene_feature.append(all_dataset.getFeat(id2orig[i]))
    all_gene_feature = np.asarray(all_gene_feature)

    loss_time = []
    checktosave = ChecktoSave(kfold)
    for fold_num in range(kfold):

        print(f"<<<<<<<<<<<<<[ FOLD {fold_num + 1} ]>>>>>>>>>>>>>>>")
        print('============= Start Training ... ==============')
        input_size = all_gene_feature.shape[1]
        model = Net(input_size, latent_size, momentum)
        model = model.to(cuda_device)

        optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # every 10 steps, lr=gamma*lr

        train_dataset = SLDataset(train_pos_kfold[fold_num], gene_kgemb, gene_id, aug_ratio)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        early_stop_cnt = 0
        total_step = 0
        for epoch in range(epochs):
            print(f" epoch {epoch + 1} ", end="")
            tic1 = time.time()
            train_loss = []
            for i, (_, gene1_feat, gene1_feat_aug, _, gene2_feat, gene2_feat_aug) in enumerate(train_loader):
                # Move tensors to GPU
                gene1_feat = gene1_feat.to(cuda_device)
                gene2_feat = gene2_feat.to(cuda_device)
                gene1_feat_aug = gene1_feat_aug.to(cuda_device)
                gene2_feat_aug = gene2_feat_aug.to(cuda_device)

                # Forward
                model.train()
                output = model(
                    [gene1_feat.float(), gene2_feat.float(), gene1_feat_aug.float(), gene2_feat_aug.float()])
                batch_loss = model.get_loss(output)
                train_loss.append(batch_loss)
                total_step += 1

                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                model._update_target()

            train_loss = torch.mean(torch.stack(train_loss)).data.cpu().numpy()
            toc1 = time.time()
            loss_time.append([fold_num, train_loss, toc1 - tic1])
            print(f"train_loss: {train_loss}")
            wandb.log({
                'trian_loss':train_loss
            })

            scheduler.step()
            if (epoch + 1) % 3 == 0:
                model.eval()
                with torch.no_grad():
                    score_mat = cal_score_mat(model, all_gene_feature, gpu)
                    if sp.isspmatrix(score_mat):
                        score_mat = score_mat.todense()
                    score_mat = np.asarray(score_mat)
                    score_mat[range(num_node), range(num_node)] = 0
                    train_metrics = cal_metrics(score_mat, train_pos_kfold[fold_num], train_neg_kfold[fold_num])
                    checktosave.update_train_classify(fold_num, epoch, np.asarray([train_metrics[0], train_metrics[2], train_metrics[1]]))
                    checktosave.update_train_ranking(fold_num, epoch, train_metrics[3:])
                    wandb.log({
                        'train_auc':train_metrics[0],'train_f1':train_metrics[1],'train_aupr':train_metrics[2],
                        'train_N10':train_metrics[3],'train_N20':train_metrics[4],'train_N50':train_metrics[5],
                        'train_R10':train_metrics[6],'train_R20':train_metrics[7],'train_R50':train_metrics[8],
                        'train_P10':train_metrics[9],'train_P20':train_metrics[10],'train_P50':train_metrics[11],
                        'train_M10':train_metrics[12],'train_M20':train_metrics[13],'train_M50':train_metrics[14],
                    })
                    if indep_test:
                        valid_metrics = cal_metrics(score_mat, valid_pos_kfold[fold_num], valid_neg_kfold[fold_num],train_pos_kfold[fold_num])
                        wandb.log({
                            'valid_auc':valid_metrics[0],'valid_f1':valid_metrics[1],'valid_aupr':valid_metrics[2],
                            'valid_N10':valid_metrics[3],'valid_N20':valid_metrics[4],'valid_N50':valid_metrics[5],
                            'valid_R10':valid_metrics[6],'valid_R20':valid_metrics[7],'valid_R50':valid_metrics[8],
                            'valid_P10':valid_metrics[9],'valid_P20':valid_metrics[10],'valid_P50':valid_metrics[11],
                            'valid_M10':valid_metrics[12],'valid_M20':valid_metrics[13],'valid_M50':valid_metrics[14],
                        })
                        
                        test_metrics = cal_metrics(score_mat, test_pos_kfold[fold_num], test_neg_kfold[fold_num],train_pos_kfold[fold_num])
                        wandb.log({
                            'test_auc':test_metrics[0],'test_f1':test_metrics[1],'test_aupr':test_metrics[2],
                            'test_N10':test_metrics[3],'test_N20':test_metrics[4],'test_N50':test_metrics[5],
                            'test_R10':test_metrics[6],'test_R20':test_metrics[7],'test_R50':test_metrics[8],
                            'test_P10':test_metrics[9],'test_P20':test_metrics[10],'test_P50':test_metrics[11],
                            'test_M10':test_metrics[12],'test_M20':test_metrics[13],'test_M50':test_metrics[14],
                        })
                        print(test_metrics)
                        if save_mat:
                            if checktosave.update_classify(fold_num, epoch, np.asarray([valid_metrics[0], valid_metrics[2], valid_metrics[1]])):
                                # print('Saving score matrix ...')
                                if not os.path.exists(f'../results/{n_s}{base_suffix}/nsf4sl'):
                                    os.makedirs(f'../results/{n_s}{base_suffix}/nsf4sl')
                                path = f'../results/{n_s}{base_suffix}/nsf4sl/nsf4sl_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_classify.npy'
                                checktosave.save_mat(path, score_mat)
                                checktosave.update_indep_test_classify(fold_num, 0, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]]))
                            if checktosave.update_ranking(fold_num, epoch, valid_metrics[3:]):
                                # print('Saving score matrix ...')
                                path = f'../results/{n_s}{base_suffix}/nsf4sl/nsf4sl_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_ranking.npy'
                                checktosave.save_mat(path, score_mat)
                                checktosave.update_indep_test_ranking(fold_num, 0, test_metrics[3:])
                        else:
                            if checktosave.update_classify(fold_num, epoch, np.asarray([valid_metrics[0], valid_metrics[2], valid_metrics[1]])):
                                checktosave.update_indep_test_classify(fold_num, epoch, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]]))
                            if checktosave.update_ranking(fold_num, epoch, valid_metrics[3:]):
                                checktosave.update_indep_test_ranking(fold_num, epoch, test_metrics[3:])
                    else:
                        test_metrics = cal_metrics(score_mat, test_pos_kfold[fold_num], test_neg_kfold[fold_num],train_pos_kfold[fold_num])
                        wandb.log({
                            'test_auc':test_metrics[0],'test_f1':test_metrics[1],'test_aupr':test_metrics[2],
                            'test_N10':test_metrics[3],'test_N20':test_metrics[4],'test_N50':test_metrics[5],
                            'test_R10':test_metrics[6],'test_R20':test_metrics[7],'test_R50':test_metrics[8],
                            'test_P10':test_metrics[9],'test_P20':test_metrics[10],'test_P50':test_metrics[11],
                            'test_M10':test_metrics[12],'test_M20':test_metrics[13],'test_M50':test_metrics[14],
                        })
                        print(test_metrics)
                        if save_mat:
                            if checktosave.update_classify(fold_num, epoch, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]])):
                                # print('Saving score matrix ...')
                                if not os.path.exists(f'../results/{n_s}{base_suffix}/nsf4sl'):
                                    os.makedirs(f'../results/{n_s}{base_suffix}/nsf4sl')
                                path = f'../results/{n_s}{base_suffix}/nsf4sl/nsf4sl_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_classify.npy'
                                checktosave.save_mat(path, score_mat)
                            if checktosave.update_ranking(fold_num, epoch, test_metrics[3:]):
                                # print('Saving score matrix ...')
                                path = f'../results/{n_s}{base_suffix}/nsf4sl/nsf4sl_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_ranking.npy'
                                checktosave.save_mat(path, score_mat)
                        else:
                            checktosave.update_classify(fold_num, epoch, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]]))
                            checktosave.update_ranking(fold_num, epoch, test_metrics[3:])
                        
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    score_mat = cal_score_mat(model, all_gene_feature, gpu)
                    if sp.isspmatrix(score_mat):
                        score_mat = score_mat.todense()
                    score_mat = np.asarray(score_mat)
                    score_mat[range(num_node), range(num_node)] = 0
                    test_metrics = cal_metrics(score_mat, test_pos_kfold[fold_num], test_neg_kfold[fold_num],train_pos_kfold[fold_num])
                    print(test_metrics)
                    
                early_stop_cnt = checktosave.chechtostop(fold_num, early_stop_cnt, test_metrics)
                if early_stop_cnt == early_stop:
                    print('[ EARLY STOP HAPPEN! ]')
                    break
                
    wandb.finish()
    # auc f1 aupr N10 N20 N50 R10 R20 R50 P10 P20 P50
    if indep_test:
        all_metrics = checktosave.get_all_indep_test_metrics()
    else:
        all_metrics = checktosave.get_all_metrics()
    
    return all_metrics

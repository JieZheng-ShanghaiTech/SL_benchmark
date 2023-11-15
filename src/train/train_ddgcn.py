from models.ddgcn import *
import numpy as np
import scipy.sparse as sp
import torch
import time
import os
import wandb
from preprocess import cal_metrics, ChecktoSave
from utils.ddgcn_utils import normalize_mat, scipy_sparse_mat_to_torch_sparse_tensor, feature_loader

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.manual_seed(456)
torch.cuda.manual_seed_all(456)

cuda_device = torch.device('cuda:0')


def train_ddgcn(parameters, pos_samples, neg_samples):
    graph_train_pos_kfold, _, train_pos_kfold, test_pos_kfold = pos_samples
    graph_train_neg_kfold, _, train_neg_kfold, test_neg_kfold = neg_samples

    KFold = parameters['KFold']
    EPOCH = parameters['EPOCH']
    KERNAL_SIZE1 = parameters['KERNAL_SIZE1']
    KERNAL_SIZE2 = parameters['KERNAL_SIZE2']
    DROPOUT = parameters['DROPOUT']
    INIT_TYPE = parameters['INIT_TYPE']
    USE_BIAS = parameters['USE_BIAS']
    num_node = parameters['num_node']
    POS_THRESHOLD = parameters['POS_THRESHOLD']
    RHO = parameters['RHO']
    NORMAL_DIM = parameters['NORMAL_DIM']
    TOLERANCE_EPOCH = parameters['TOLERANCE_EPOCH']
    LR = parameters['LR']
    STOP_THRESHOLD = parameters['STOP_THRESHOLD']
    EVAL_INTER = parameters['EVAL_INTER']
    p_n = parameters['pos_neg']
    d_s = parameters['division_strategy']
    n_s = parameters['negative_strategy']

    feature1, is_sparse_feat1 = feature_loader(num_node)
    nfeat = feature1.shape[1]

    LOOP_FEAT2 = False

    loss_time = []
    wandb.init(
        # set the wandb project where this run will be logged
        project="Benchmarking",
        group="DDGCN",
        job_type=f"{p_n}_{d_s}_{n_s}",
        # track hyperparameters and run metadata
        config=parameters
    )
    checktosave = ChecktoSave(KFold)
    for fold_num in range(KFold):
        
        print("Using {} th fold dataset.".format(fold_num + 1))
        graph_train = graph_train_pos_kfold[fold_num].toarray()
        graph_train_neg = graph_train_neg_kfold[fold_num].toarray()

        adj_norm = normalize_mat(sp.coo_matrix(graph_train) + sp.eye(num_node), NORMAL_DIM)
        adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(adj_norm)

        adj_traget = torch.FloatTensor(graph_train + np.eye(num_node))
        pair_mask = torch.FloatTensor(graph_train + graph_train_neg)
        if LOOP_FEAT2:
            feature2 = torch.FloatTensor(graph_train + np.eye(num_node))
            is_sparse_feat2 = True
        else:
            feature2 = torch.FloatTensor(graph_train)
            is_sparse_feat2 = True

        model = GraphAutoEncoder(nfeat, KERNAL_SIZE1, KERNAL_SIZE2, DROPOUT, INIT_TYPE, USE_BIAS, is_sparse_feat1, is_sparse_feat2)
        model = model.to(cuda_device)

        obj = ObjectiveFunction(adj_traget, pair_mask)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=True)
        
        last_loss=1e-5

        for j in range(EPOCH):
            tic = time.time()
            model.train()
            
            feature1 = feature1.to(cuda_device)
            feature2 = feature2.to(cuda_device)
            adj_norm = adj_norm.to(cuda_device)

            reconstruct_adj_logit1, reconstruct_adj_logit2 = model(feature1, feature2, adj_norm)
            loss = obj.cal_loss(reconstruct_adj_logit1, reconstruct_adj_logit2, RHO)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            toc = time.time()
            loss_time.append([fold_num, loss.item(), toc - tic])
            wandb.log({
                'trian_loss':loss.item()
            })
            need_early_stop_check = j > TOLERANCE_EPOCH and abs((loss.item() - last_loss) / last_loss) < STOP_THRESHOLD

            if ((j + 1) % EVAL_INTER == 0) or need_early_stop_check or j + 1 >= EPOCH:
                print(f'Training {j}/{EPOCH} ...')
                
                model.eval()
                with torch.no_grad():

                    reconstruct_adj_logit1, reconstruct_adj_logit2 = model(feature1, feature2, adj_norm)
                    
                    reconstruct_adj1 = torch.sigmoid(reconstruct_adj_logit1)
                    reconstruct_adj2 = torch.sigmoid(reconstruct_adj_logit2)

                    reconstruct_adj = np.power(reconstruct_adj1.cpu() * np.power(reconstruct_adj2.cpu(), RHO), 1 / (1 + RHO))

                    reconstruct_adj = np.asarray(reconstruct_adj)
                    reconstruct_adj[range(num_node), range(num_node)] = 0
                    train_metrics = cal_metrics(reconstruct_adj, train_pos_kfold[fold_num], train_neg_kfold[fold_num])
                    
                    checktosave.update_train_classify(fold_num, j, np.asarray([train_metrics[0], train_metrics[2], train_metrics[1]]))
                    checktosave.update_train_ranking(fold_num, j, train_metrics[3:])
                    wandb.log({
                        'train_auc':train_metrics[0],'train_aupr':train_metrics[2],'train_f1':train_metrics[1],
                        'train_N10':train_metrics[3],'train_N20':train_metrics[4],'train_N50':train_metrics[5],
                        'train_R10':train_metrics[6],'train_R20':train_metrics[7],'train_R50':train_metrics[8],
                        'train_P10':train_metrics[9],'train_P20':train_metrics[10],'train_P50':train_metrics[11],
                        'train_M10':train_metrics[12],'train_M20':train_metrics[13],'train_M50':train_metrics[14],
                    })
                    
                    print('Calculating all metrics ...')
                    test_metrics = cal_metrics(reconstruct_adj, test_pos_kfold[fold_num], test_neg_kfold[fold_num], train_pos_kfold[fold_num])
                    
                    print(test_metrics)
                    wandb.log({
                        'test_auc':test_metrics[0],'test_aupr':test_metrics[2],'test_f1':test_metrics[1],
                        'test_N10':test_metrics[3],'test_N20':test_metrics[4],'test_N50':test_metrics[5],
                        'test_R10':test_metrics[6],'test_R20':test_metrics[7],'test_R50':test_metrics[8],
                        'test_P10':test_metrics[9],'test_P20':test_metrics[10],'test_P50':test_metrics[11],
                        'test_M10':test_metrics[12],'test_M20':test_metrics[13],'test_M50':test_metrics[14],
                    })
                    
                    if checktosave.update_classify(fold_num, j, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]])):
                        print('Saving score matrix ...')
                        if not os.path.exists(f'../results/{n_s}_score_mats/ddgcn'):
                            os.mkdir(f'../results/{n_s}_score_mats/ddgcn')
                        path = f'../results/{n_s}_score_mats/ddgcn/ddgcn_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_classify.npy'
                        checktosave.save_mat(path, reconstruct_adj)
                    if checktosave.update_ranking(fold_num, j, test_metrics[3:]):
                        print('Saving score matrix ...')
                        path = f'../results/{n_s}_score_mats/ddgcn/ddgcn_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_ranking.npy'
                        checktosave.save_mat(path, reconstruct_adj)

                if need_early_stop_check or j + 1 >= EPOCH:

                    if need_early_stop_check:
                        print("Early stopping...")
                    else:
                        print("Arrived at the last Epoch...")
                    break

            last_loss = loss.item()
    wandb.finish()
    # auc f1 aupr N10 N20 N50 R10 R20 R50 P10 P20 P50
    all_metrics = checktosave.get_all_metrics()
    
    return all_metrics

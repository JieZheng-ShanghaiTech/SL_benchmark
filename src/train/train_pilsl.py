import copy
import os
import argparse
import logging
import torch
from scipy.sparse import SparseEfficiencyWarning
import wandb

from utils.pilsl_utils import generate_subgraph_datasets, initialize_experiment, initialize_model, collate_dgl, move_batch_to_device_dgl
from models.pilsl import SubgraphDataset, Evaluator, Trainer
from models.pilsl import GraphClassifier as dgl_model

import numpy as np
from warnings import simplefilter
from preprocess import cal_metrics, ChecktoSave

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

np.random.seed(456)
torch.manual_seed(456)
torch.cuda.manual_seed_all(456)

def train_pilsl(parameters, pos_samples, neg_samples, mode=None, save_mat=False):
    _, _, train_pos_kfold, test_pos_kfold = pos_samples
    _, _, train_neg_kfold, test_neg_kfold = neg_samples
    kfold = parameters['kfold']
    p_n = parameters['pos_neg']
    d_s = parameters['division_strategy']
    n_s = parameters['negative_strategy']
    
    if mode=='final_res':
        kfold=1
        p_n = d_s = n_s = 'final_res'

    pilsl_paras_dict={}
    pilsl_paras_dict['num_epochs']=parameters['num_epochs']
    pilsl_paras_dict['eval_every']=parameters['eval_every']
    pilsl_paras_dict['eval_every_iter']=parameters['eval_every_iter']
    pilsl_paras_dict['save_every']=parameters['save_every']
    pilsl_paras_dict['early_stop']=parameters['early_stop']
    pilsl_paras_dict['optimizer']=parameters['optimizer']
    pilsl_paras_dict['lr']=parameters['lr']
    pilsl_paras_dict['clip']=parameters['clip']
    pilsl_paras_dict['l2']=parameters['l2']
    pilsl_paras_dict['max_links']=parameters['max_links']
    pilsl_paras_dict['hop']=parameters['hop']
    pilsl_paras_dict['max_nodes_per_hop']=parameters['max_nodes_per_hop']
    pilsl_paras_dict['batch_size']=parameters['batch_size']

    # parser = argparse.ArgumentParser(description='TransE model')
    pilsl_paras_dict['gpu'] = 0,1,2,3
    pilsl_paras_dict['experiment_name']="default1"
    pilsl_paras_dict['dataset']=None
    pilsl_paras_dict['disable_cuda']=False
    pilsl_paras_dict['load_model']=False
    pilsl_paras_dict['train_file']='train'
    pilsl_paras_dict['valid_file']='dev'
    pilsl_paras_dict['test_file']='test'
    pilsl_paras_dict['use_kge_embeddings']=False
    pilsl_paras_dict['kge_model']='TransE'
    pilsl_paras_dict['model_type']='dgl'
    pilsl_paras_dict['constrained_neg_prob']=0.0
    pilsl_paras_dict['num_workers']=10
    pilsl_paras_dict['add_traspose_rels']=False
    pilsl_paras_dict['enclosing_sub_graph']=True
    pilsl_paras_dict['rel_emb_dim']=64
    pilsl_paras_dict['attn_rel_emb_dim']=64
    pilsl_paras_dict['emb_dim']=64
    pilsl_paras_dict['num_gcn_layers']=3
    pilsl_paras_dict['num_bases']=10
    pilsl_paras_dict['dropout']=0.2
    pilsl_paras_dict['edge_dropout']=0.2
    pilsl_paras_dict['gnn_agg_type']='sum'
    pilsl_paras_dict['add_ht_emb']=True
    pilsl_paras_dict['add_sb_emb']=True
    pilsl_paras_dict['has_attn']=True
    pilsl_paras_dict['has_kg']=True
    pilsl_paras_dict['feat_dim']=600
    pilsl_paras_dict['add_feat_emb']=False
    pilsl_paras_dict['add_transe_emb']=True

    if not pilsl_paras_dict['disable_cuda'] and torch.cuda.is_available():
        pilsl_paras_dict['device'] = torch.device('cuda')
        # pilsl_paras_dict['device'] = torch.device('cuda:%d' % pilsl_paras_dict['gpu'])
    else:
        pilsl_paras_dict['device'] = torch.device('cpu')

    pilsl_paras_dict['collate_fn'] = collate_dgl
    pilsl_paras_dict['move_batch_to_device'] = move_batch_to_device_dgl

    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    pilsl_paras_dict['main_dir']='../data/preprocessed_data/pilsl_data/pilsl_database'
    pilsl_paras_dict['dataset']='CV1'
    # pilsl_paras_dict['dataset']=d_s
    pilsl_paras_dict['db_path']=os.path.join(pilsl_paras_dict['main_dir'], f"data/{pilsl_paras_dict['dataset']}/graph_hop_{pilsl_paras_dict['hop']}")

    pilsl_paras_dict['file_paths'] = {
        'train': '../data/preprocessed_data/pilsl_data/all_pairs_used_9845.npy',
    }

    if not os.path.isdir(pilsl_paras_dict['db_path']):
        print('preprocessing...')
        generate_subgraph_datasets(pilsl_paras_dict)

    initialize_experiment(pilsl_paras_dict, __file__)

    ith_fold_result = []
    pilsl_result_kfold = []
    run=wandb.init(
        # set the wandb project where this run will be logged
        project="Benchmarking",
        group="PiLSL",
        job_type=f"{p_n}_{d_s}_{n_s}",
        # track hyperparameters and run metadata
        config=parameters
    )
    checktosave = ChecktoSave(kfold)
    for fold_num in range(kfold):
        print(f'fold:{fold_num}')
        train_pos=np.hstack([train_pos_kfold[fold_num], np.zeros((train_pos_kfold[fold_num].shape[0], 1))])
        train_neg=np.hstack([train_neg_kfold[fold_num], np.zeros((train_neg_kfold[fold_num].shape[0], 1))])
        test_pos=np.hstack([test_pos_kfold[fold_num], np.zeros((test_pos_kfold[fold_num].shape[0], 1))])
        test_neg=np.hstack([test_neg_kfold[fold_num], np.zeros((test_neg_kfold[fold_num].shape[0], 1))])

        train_pairs=np.vstack([train_neg,train_pos])
        test_pairs=np.vstack([train_neg,train_pos,test_neg,test_pos])

        print(pilsl_paras_dict['db_path'])

        train = SubgraphDataset(pilsl_paras_dict['db_path'], 'train_pairs', pilsl_paras_dict['file_paths'], train_pairs)
        print('train set done.')
        test = SubgraphDataset(pilsl_paras_dict['db_path'], 'train_pairs', pilsl_paras_dict['file_paths'], test_pairs,
                               ssp_graph=train.ssp_graph,
                               id2entity=train.id2entity, id2relation=train.id2relation, rel=train.num_rels,
                               graph=train.graph)
        print('test set done.')

        pilsl_paras_dict['num_rels']=train.num_rels
        pilsl_paras_dict['aug_num_rels']=train.aug_num_rels
        pilsl_paras_dict['inp_dim']=train.n_feat_dim
        pilsl_paras_dict['train_rels']=pilsl_paras_dict['num_rels'] - 1
        pilsl_paras_dict['num_nodes']=54012

        # Log the max label value to save it in the model. This will be used to cap the labels generated on test set.
        pilsl_paras_dict['max_label_value'] = train.max_n_label
        print(f"Device: {pilsl_paras_dict['device']}")
        # logging.info(f"Device: {pilsl_paras_dict['device']}")
        print(
            f"Input dim : {pilsl_paras_dict['inp_dim']}, # Relations : {pilsl_paras_dict['num_rels']}, # Augmented relations : {pilsl_paras_dict['aug_num_rels']}")

        # get in
        graph_classifier = initialize_model(pilsl_paras_dict, dgl_model, pilsl_paras_dict['load_model'])

        mfeat = np.load('../data/preprocessed_data/pilsl_data/pilsl_random_feature.npy', allow_pickle=True).item()
        graph_classifier.omics_feat(mfeat)

        test_evaluator = Evaluator(pilsl_paras_dict, graph_classifier, test, test_pairs)
        train_evaluator = Evaluator(pilsl_paras_dict, graph_classifier, train, train_pairs)

        trainer = Trainer(pilsl_paras_dict, graph_classifier, train, train_evaluator, None, test_evaluator, None, wandb_runner=run)

        print('Starting training with full batch...')
        trainer.train()
        
        score_mat = copy.deepcopy(trainer.score_mat)
        score_mat = score_mat.todense()
        score_mat = np.asarray(score_mat)
        train_metrics = cal_metrics(score_mat, train_pos_kfold[fold_num], train_neg_kfold[fold_num])
                    
        checktosave.update_train_classify(fold_num, 0, np.asarray([train_metrics[0], train_metrics[2], train_metrics[1]]))
        checktosave.update_train_ranking(fold_num, 0, train_metrics[3:])
        wandb.log({
            'train_auc':train_metrics[0],'train_f1':train_metrics[1],'train_aupr':train_metrics[2],
            'train_N10':train_metrics[3],'train_N20':train_metrics[4],'train_N50':train_metrics[5],
            'train_R10':train_metrics[6],'train_R20':train_metrics[7],'train_R50':train_metrics[8],
            'train_P10':train_metrics[9],'train_P20':train_metrics[10],'train_P50':train_metrics[11],
            'train_M10':train_metrics[12],'train_M20':train_metrics[13],'train_M50':train_metrics[14],
        })

        test_metrics = cal_metrics(score_mat, test_pos_kfold[fold_num], test_neg_kfold[fold_num], train_pos_kfold[fold_num])
        run.log({
            'test_auc':test_metrics[0],'test_f1':test_metrics[1],'test_aupr':test_metrics[2],
            'test_N10':test_metrics[3],'test_N20':test_metrics[4],'test_N50':test_metrics[5],
            'test_R10':test_metrics[6],'test_R20':test_metrics[7],'test_R50':test_metrics[8],
            'test_P10':test_metrics[9],'test_P20':test_metrics[10],'test_P50':test_metrics[11],
            'test_M10':test_metrics[12],'test_M20':test_metrics[13],'test_M50':test_metrics[14],
        })
        
        if save_mat:
            if checktosave.update_classify(fold_num, 0, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]])):
                # print('Saving score matrix ...')
                if not os.path.exists(f'../results/{n_s}_score_mats/pilsl'):
                    os.makedirs(f'../results/{n_s}_score_mats/pilsl')
                path = f'../results/{n_s}_score_mats/pilsl/pilsl_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_classify.npy'
                checktosave.save_mat(path, score_mat)
            if checktosave.update_ranking(fold_num, 0, test_metrics[3:]):
                # print('Saving score matrix ...')
                path = f'../results/{n_s}_score_mats/pilsl/pilsl_fold_{fold_num}_pos_neg_{p_n}_{d_s}_{n_s}_ranking.npy'
                checktosave.save_mat(path, score_mat)
        else:
            checktosave.update_classify(fold_num, 0, np.asarray([test_metrics[0], test_metrics[2], test_metrics[1]]))
            checktosave.update_ranking(fold_num, 0, test_metrics[3:])
            
        print(f'{test_metrics}')
        pilsl_result_kfold.append(test_metrics)

    run.finish()
    # auc f1 aupr N10 N20 N50 R10 R20 R50 P10 P20 P50
    all_metrics = checktosave.get_all_metrics()
    
    return all_metrics



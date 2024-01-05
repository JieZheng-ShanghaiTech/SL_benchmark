from config import get_config_dict
import copy
import random
import preprocess
import argparse
import wandb

import pandas as pd
import time
import torch
import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Set random seed
random.seed(123)
np.random.seed(123)
tf.compat.v1.set_random_seed(123)

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

PTGNN_init = 1  # Pretrain PTGNN is ready, set to 0 if you want to repretrain PTGNN

def save_resultes(d_s, p_n, old_res, model_res, model_name, res_name, col_names):

    train_res = model_res[:5,:]
    test_res = model_res[5:,:]
    
    def get_mean_std(res):
        mean_res = np.mean(np.asarray(res), 0)
        std_res = np.std(np.asarray(res), 0)
        mean_res = [d_s, model_name, str(int(1/p_n))] + list(mean_res)
        std_res = [d_s, model_name, str(int(1/p_n))] + list(std_res)
        mean_row = pd.DataFrame(np.asarray(mean_res).reshape(1, -1), columns=col_names)
        std_row = pd.DataFrame(np.asarray(std_res).reshape(1, -1), columns=col_names)
        
        return mean_row, std_row
    mean_row, std_row = get_mean_std(train_res)
    old_res = pd.concat([old_res, mean_row])
    old_res = pd.concat([old_res, std_row])
    mean_row, std_row = get_mean_std(test_res)
    old_res = pd.concat([old_res, mean_row])
    old_res = pd.concat([old_res, std_row])
    

    new_res = copy.deepcopy(old_res)
    # save results
    old_res.to_csv(f'{res_name}', index=False)

    return new_res

def set_gpu_memory_growth():
    for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def main(params_main):
    # Define a list of all available models
    all_models = ['DDGCN', 'NSF4SL', 'GCATSL', 'SLMGAE', 'KG4SL', 'PTGNN', 'PiLSL', 'MGE4SL', 'SL2MF', 'GRSMF', 'CMFW']

    # Define a list of all negative strategies
    negative_strategy = ['Random', 'Exp', 'Dep']

    # Check if the negative_strategy parameter is set to 'all'
    if params_main.negative_strategy == 'all':
        negative_strategy = ['Random', 'Exp', 'Dep']
    else:
        # Split the negative_strategy parameter by comma and assign it to the negative_strategy list
        negative_strategy = params_main.negative_strategy.split(',')

    # Check if the train_models parameter is set to 'all'
    if params_main.train_models == 'all':
        which_model = all_models
    else:
        # Split the train_models parameter by comma and assign it to the which_model list
        which_model = params_main.train_models.split(',')

    # Check if the pos_neg parameter is set to 'all'
    if params_main.pos_neg == 'all':
        pos_neg = [1, 1 / 5, 1 / 20, 1 / 50]
    else:
        # Split the pos_neg parameter by comma, convert each element to float, and assign it to the pos_neg list
        pos_neg = list(1.0/np.asarray(params_main.pos_neg.split(','), dtype='float'))

    # Check if the division_strategy parameter is set to 'all'
    if params_main.division_strategy == 'all':
        division_strategy = ['CV1', 'CV2', 'CV3']
    else:
        # Split the division_strategy parameter by comma and assign it to the division_strategy list
        division_strategy = params_main.division_strategy.split(',')

    tail_star = None
    # Check if the output_name parameter is set
    if params_main.output_name:
        # Assign the value of output_name parameter to tail_star variable
        tail_star = params_main.output_name
    
    # Build result dataframe and the path to save the evaluation metrics
    def build_res_df(model_name):
        res_name_time = time.strftime(
            '%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
        if tail_star:
            res_name = f'../results/metrics/{n_s}/{model_name.lower()}/results_{res_name_time}_{p_n}_{d_s}_{n_s}_{tail_star}.csv'
        else:
            res_name = f'../results/metrics/{n_s}/{model_name.lower()}/results_{res_name_time}_{p_n}_{d_s}_{n_s}.csv'
        print(res_name)
        return res_name
        
    for p_n in pos_neg:
        for d_s in division_strategy:
            for n_s in negative_strategy:
                
                # Loading data
                print(f'Loading data ... (pos/neg={p_n}, division_strategy: {d_s}, negative_strategy: {n_s})')
                num_node, _ = preprocess.get_id_map()
                pos_samples, neg_samples = preprocess.get_kfold_data_pos_neg(5, num_node, 0.8, 0, 0.2, 1, p_n, d_s, n_s)
                # initialize result dataframe
                col_names = ['Strategy','Model','pos/neg','AUROC','AUPR','F1',
                'NDCG@10','NDCG@20','NDCG@50','Recall@10','Recall@20','Recall@50',
                'Precision@10','Precision@20','Precision@50','MAP@10','MAP@20','MAP@50',
                ]
                if params_main.mode == 'final_res': # final_res means the training data is ALL known SL data
                    pos_samples, neg_samples = np.load(f'../data/data_split/{which_model[0]}_fin_res.npy', allow_pickle=True)
                summary_result_df = pd.DataFrame(columns=col_names)
                
                # DDGCN
                if 'DDGCN' in which_model:
                    from train.train_ddgcn import train_ddgcn
                    parameter_set = get_config_dict('DDGCN', 9845, p_n, d_s, n_s)
                    print(f'DDGCN starting ...')
                    res_name = build_res_df('DDGCN')
                    ddgcn_metrics = train_ddgcn(parameter_set, pos_samples, neg_samples, params_main.mode, params_main.save_mat)
                    summary_result_df = save_resultes(d_s, p_n, summary_result_df, ddgcn_metrics, 'DDGCN', res_name, col_names)
                    print(f'DDGCN done ...')

                    set_gpu_memory_growth()

                # KG4SL
                if 'KG4SL' in which_model:
                    from train.train_kg4sl import train_kg4sl
                    parameter_set = get_config_dict('KG4SL', 9845, p_n, d_s, n_s)
                    print(f'KG4SL starting ...')
                    res_name = build_res_df('KG4SL')
                    kg4sl_metrics = train_kg4sl(parameter_set, pos_samples, neg_samples, params_main.mode, params_main.save_mat)
                    summary_result_df = save_resultes(d_s, p_n, summary_result_df, kg4sl_metrics, 'KG4SL', res_name, col_names)
                    print(f'KG4SL done ...')

                    set_gpu_memory_growth()
                
                # NSF4SL
                if 'NSF4SL' in which_model:
                    from train.train_nsf4sl import train_nsf4sl
                    parameter_set = get_config_dict('NSF4SL', 9845, p_n, d_s, n_s)
                    print(f'NSF4SL starting ...')
                    res_name = build_res_df('NSF4SL')
                    nsf4sl_metrics = train_nsf4sl(parameter_set, pos_samples, neg_samples, params_main.mode, params_main.save_mat)
                    summary_result_df = save_resultes(d_s, p_n, summary_result_df, nsf4sl_metrics, 'NSF4SL', res_name, col_names)
                    print(f'NSF4SL done ...')

                    set_gpu_memory_growth()

                # GCATSL
                if 'GCATSL' in which_model:
                    from train.train_gcatsl import train_gcatsl
                    parameter_set = get_config_dict('GCATSL', 9845, p_n, d_s, n_s)
                    print(f'GCATSL starting ...')
                    res_name = build_res_df('GCATSL')
                    gcatsl_metrics = train_gcatsl(parameter_set, pos_samples, neg_samples, params_main.mode, params_main.save_mat)
                    summary_result_df = save_resultes(d_s, p_n, summary_result_df, gcatsl_metrics, 'GCATSL', res_name, col_names)
                    print(f'GCATSL done ...')

                    set_gpu_memory_growth()
                    
                # SLMGAE
                if 'SLMGAE' in which_model:
                    from train.train_slmgae import train_slmgae
                    parameter_set = get_config_dict('SLMGAE', 9845, p_n, d_s, n_s)
                    print(f'SLMGAE starting ...')
                    res_name = build_res_df('SLMGAE')
                    slmgae_metrics = train_slmgae(parameter_set, pos_samples, neg_samples, params_main.mode, params_main.save_mat)
                    summary_result_df = save_resultes(d_s, p_n, summary_result_df, slmgae_metrics, 'SLMGAE', res_name, col_names)
                    print(f'SLMGAE done ...')

                    set_gpu_memory_growth()

                # PiLSL
                if 'PiLSL' in which_model:
                    from train.train_pilsl import train_pilsl
                    parameter_set = get_config_dict('PiLSL', 9845, p_n, d_s, n_s)
                    print(f'PiLSL starting ...')
                    res_name = build_res_df('PiLSL')
                    pilsl_metrics = train_pilsl(parameter_set, pos_samples, neg_samples, params_main.mode, params_main.save_mat)
                    summary_result_df = save_resultes(d_s, p_n, summary_result_df, pilsl_metrics, 'PiLSL', res_name, col_names)
                    print(f'PiLSL done ...')

                    set_gpu_memory_growth()

                # PTGNN
                if 'PTGNN' in which_model:
                    from train.train_ptgnn import train_ptgnn, train_ptgnn_pre
                    parameter_set = get_config_dict('PTGNN', 9845, p_n, d_s, n_s)
                    if PTGNN_init == 0:
                        print(f'PTGNN pretraining ...')
                        train_ptgnn_pre()
                        print(f'PTGNN done ...')

                    if PTGNN_init == 1:
                        print(f'PTGNN training ...')
                        res_name = build_res_df('PTGNN')
                        ptgnn_metrics = train_ptgnn(parameter_set, pos_samples, neg_samples, params_main.mode, params_main.save_mat)
                        summary_result_df = save_resultes(d_s, p_n, summary_result_df, ptgnn_metrics, 'PTGNN', res_name, col_names)
                        print(f'PTGNN done ...')

                        set_gpu_memory_growth()

                # MGE4SL
                if 'MGE4SL' in which_model:
                    from train.train_mge4sl import train_mge4sl
                    parameter_set = get_config_dict('MGE4SL', 9845, p_n, d_s, n_s)
                    print(f'MGE4SL starting ...')
                    res_name = build_res_df('MGE4SL')
                    mge4sl_metrics = train_mge4sl(parameter_set, pos_samples, neg_samples, params_main.mode, params_main.save_mat)
                    summary_result_df = save_resultes(d_s, p_n, summary_result_df, mge4sl_metrics, 'MGE4SL', res_name, col_names)
                    print(f'MGE4SL done ...')

                    set_gpu_memory_growth()
                    
                # SL2MF
                if 'SL2MF' in which_model:
                    from train.train_sl2mf import train_sl2mf
                    parameter_set = get_config_dict('SL2MF', 9845, p_n, d_s, n_s)
                    print(f'SL2MF starting ...')
                    res_name = build_res_df('SL2MF')
                    sl2mf_metrics = train_sl2mf(parameter_set, pos_samples, neg_samples, params_main.mode, params_main.save_mat)
                    summary_result_df = save_resultes(d_s, p_n, summary_result_df, sl2mf_metrics, 'SL2MF', res_name, col_names)
                    print(f'SL2MF done ...')
                    
                    set_gpu_memory_growth()

                # GRSMF
                if 'GRSMF' in which_model:
                    from train.train_grsmf import train_grsmf
                    parameter_set = get_config_dict('GRSMF', 9845, p_n, d_s, n_s)
                    print(f'GRSMF starting ...')
                    res_name = build_res_df('GRSMF')
                    grsmf_metrics = train_grsmf(parameter_set, pos_samples, neg_samples, params_main.mode, params_main.save_mat)
                    summary_result_df = save_resultes(d_s, p_n, summary_result_df, grsmf_metrics, 'GRSMF', res_name, col_names)
                    print(f'GRSMF done ...')

                    set_gpu_memory_growth()

                # CMFW
                if 'CMFW' in which_model:
                    from train.train_cmfw import train_cmfw
                    parameter_set = get_config_dict('CMFW', 9845, p_n, d_s, n_s)
                    print(f'CMFW starting ...')
                    res_name = build_res_df('CMFW')
                    cmfw_metrics = train_cmfw(parameter_set, pos_samples, neg_samples, params_main.mode, params_main.save_mat)
                    summary_result_df = save_resultes(d_s, p_n, summary_result_df, cmfw_metrics, 'CMFW', res_name, col_names)
                    print(f'CMFW done ...')

                    set_gpu_memory_growth()


if __name__ == "__main__":
    params = argparse.ArgumentParser(description='SL model benchmark')
    params.add_argument("--train_models", '-m', type=str, default='KG4SL',  # 'all_models',
                        help="Select model from ['DDGCN', 'NSF4SL', 'GCATSL', 'SLMGAE', 'KG4SL', 'PTGNN', 'PiLSL', 'MGE4SL', 'SL2MF', 'GRSMF', 'CMFW'].")
    params.add_argument("--negative_strategy", "-ns", type=str, default='Random',
                        help="Select negative sample strategy from 'Random' 'Dep' or 'Exp'. ")
    params.add_argument("--pos_neg", "-pn", type=str, default='1',
                        help="Select ratio of neg/pos = [1, 5, 20, 50, 100]")
    params.add_argument("--division_strategy", "-ds", type=str, default='CV1',
                        help="Select data division strategy from ['CV1','CV2','CV3']")
    params.add_argument("--mode", type=str, default=None,
                        help="Get final prediction result with all data.")
    params.add_argument("--save_mat", action="store_true",
                        help="Save the matrix by calculating predictions.")
    params.add_argument("--output_name", "-out", type=str, default='wandb')
    params = params.parse_args()
    
    main(params)

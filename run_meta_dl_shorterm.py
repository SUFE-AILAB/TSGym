import os
import sys
import torch
import re
import yaml
import numpy as np
import pandas as pd
from itertools import chain
from torch import nn
from utils.myutils import Utils
from sklearn import preprocessing
from torch.utils.data import Subset, DataLoader, TensorDataset, random_split, ConcatDataset
from meta.networks import meta_predictor
from tqdm import tqdm
import logging
import random
import joblib

class Meta():
    def __init__(self,
                 seed: int=42,
                 task_name: str='LTF',
                 early_stopping=True,
                 batch_size=128,
                 d_model=64,
                 weight_decay=0.01,
                 lr=0.001,
                 epochs=100):
        self.seed = seed
        self.task_name = task_name
        self.utils = Utils()
        self.early_stopping = early_stopping
        self.batch_size = batch_size

        self.d_model = d_model
        self.weight_decay = weight_decay
        self.lr = lr
        self.epochs = epochs

    def components_processing(self, task_name, datasets, test_dataset,
                              frequency='Yearly',
                              metric='mse',
                              components_path='./meta/components.yaml',
                              components_add_Transformer_path='./meta/components_add_Transformer.yaml',
                              components_add_LLM_TSFM_path='./meta/components_add_LLM_TSFM.yaml',
                              result_path_non_transformer='./results_short_term_forecasting/resultsGym_non_transformer',
                              result_path_transformer='./results_short_term_forecasting/resultsGym_transformer',
                              result_path_LLM='./results_short_term_forecasting/resultsGym_LLM',
                              result_path_TSFM='./results_short_term_forecasting/resultsGym_TSFM',
                              meta_feature_path='./get_meta_feature/meta_features',
                              arg_component_balance=False,
                              arg_add_new_dataset=False,
                              arg_add_transformer=False,
                              arg_add_LLM_TSFM=False,
                              arg_all_periods=False):
        self.pred_len_1, self.pred_len_2 = pred_len_1, pred_len_2
        
        self.arg_component_balance = arg_component_balance
        self.arg_add_new_dataset = arg_add_new_dataset
        self.arg_add_transformer = arg_add_transformer
        self.arg_add_LLM_TSFM = arg_add_LLM_TSFM
        self.arg_all_periods = arg_all_periods
        self.test_dataset = test_dataset

        # 统计跑完组合个数
        for dataset in datasets:
            file_list_non_transformer = os.listdir(os.path.join(result_path_non_transformer, dataset))
            file_list_transformer = os.listdir(os.path.join(result_path_transformer, dataset))
            if arg_add_LLM_TSFM:
                file_list_LLM = os.listdir(os.path.join(result_path_LLM, dataset))
                file_list_TSFM = os.listdir(os.path.join(result_path_TSFM, dataset))
            pred_lens = [96, 192, 336, 720] if dataset != 'ili' else [24, 36, 48, 60]
            for pl in pred_lens:
                file_list_sub_non_transformer = [_ for _ in file_list_non_transformer if f'pl{pl}' in _]
                file_list_sub_transformer = [_ for _ in file_list_transformer if f'pl{pl}' in _]
                if arg_add_LLM_TSFM:
                    file_list_sub_LLM = [_ for _ in file_list_LLM if f'pl{pl}' in _]
                    file_list_sub_TSFM = [_ for _ in file_list_TSFM if f'pl{pl}' in _]
                    logger.info(f'Dataset: {dataset}, Pred Length: {pl}, number of combinations (non Transformer/Transformer/LLM/TSFM): {len(file_list_sub_non_transformer)}/{len(file_list_sub_transformer)}/{len(file_list_sub_LLM)}/{len(file_list_sub_TSFM)}')
                else:
                    logger.info(f'Dataset: {dataset}, Pred Length: {pl}, number of combinations (non Transformer/Transformer): {len(file_list_sub_non_transformer)}/{len(file_list_sub_transformer)}')
        
        if arg_add_transformer:
            file_dict = {dataset: os.listdir(os.path.join(result_path_non_transformer, dataset)) +\
                                  os.listdir(os.path.join(result_path_transformer, dataset)) for dataset in datasets}
        elif arg_add_LLM_TSFM:
            file_dict = {dataset: os.listdir(os.path.join(result_path_non_transformer, dataset)) +\
                                  os.listdir(os.path.join(result_path_LLM, dataset)) +\
                                  os.listdir(os.path.join(result_path_TSFM, dataset)) for dataset in datasets}
        else:
            file_dict = {dataset: os.listdir(os.path.join(result_path_non_transformer, dataset)) for dataset in datasets}
        
        if arg_all_periods:
            file_dict_test = file_dict.copy()
            file_dict_test = {k: [_ for _ in v if f'pl{pred_len_2}' in _] if k == 'ili' 
                            else [_ for _ in v if f'pl{pred_len_1}' in _]for k, v in file_dict_test.items()}
        else:
            file_dict = {k: [_ for _ in v if f'pl{pred_len_2}' in _] if k == 'ili' 
                       else [_ for _ in v if f'pl{pred_len_1}' in _]for k, v in file_dict.items()}
            file_dict_test = file_dict.copy()

        logger.info(f'number of combinations: {sum([len(_) for _ in file_dict.values()])}')

        if arg_component_balance:
            self.utils.set_seed(self.seed)
            print(f'Before balance: {sum([len(_) for _ in file_dict.values()])}')
            file_dict_resample = file_dict.copy()
            num_intersection = min([len(_) for _ in file_dict_resample.values()])
            file_dict_resample = {k: random.sample(v, num_intersection) for k,v in file_dict_resample.items()}
            file_dict = file_dict_resample
            print(f'After balance: {sum([len(_) for _ in file_dict.values()])}')

        # load meta features
        name_dict = {dataset: dataset for dataset in datasets}
        name_dict['ECL'] = 'electricity'
        name_dict['Exchange'] = 'exchange_rate'
        name_dict['ili'] = 'national_illness'
        pred_dict = {dataset: 24 if dataset == 'ili' else 96 for dataset in datasets}
        self.meta_features = {dataset: np.load(f'{meta_feature_path}/meta_feature_{name_dict[dataset]}_train_{pred_dict[dataset]}.npz',
                                               allow_pickle=True)['meta_feature'] for dataset in datasets}
        
        # todo: 利用全部多元序列的统计量信息(而不是简单平均)
        # self.meta_features = {k:np.mean(v, axis=0).squeeze() for k,v in self.meta_features.items()}
        # self.meta_features = {k: v[0, :] for k,v in self.meta_features.items()} # mean
        self.meta_features = {k: v.flatten() for k,v in self.meta_features.items()}
        
        assert len(set([v.shape for v in self.meta_features.values()])) == 1
        self.meta_feature_dim = list(self.meta_features.values())[0].shape[0]

        # z-score on different datasets
        mu = np.nanmean(np.stack(list(self.meta_features.values())), axis=0)
        std = np.nanstd(np.stack(list(self.meta_features.values())), axis=0)
        self.meta_features = {k: (v - mu) / (std + 1e-6) for k,v in self.meta_features.items()}
        # clip values
        self.meta_features = {k: np.clip(v, -1e4, 1e4) for k, v in self.meta_features.items()}
        # fillna (e.g., covid-19)
        self.meta_features = {k: np.where(np.isnan(v), 0, v) for k, v in self.meta_features.items()}
        assert (~np.isnan(np.stack(list(self.meta_features.values())))).all()

        # training datasets and testing dataset
        datasets_train = [_ for _ in datasets if _ != test_dataset]
        if not arg_add_new_dataset:
            datasets_train = [_ for _ in datasets_train if _ not in ['covid-19', 'fred-md']]
        dataset_test = [_ for _ in datasets if _ == test_dataset][0]
        print(f'training dataset: {datasets_train}, testing dataset: {dataset_test}')

        # load components
        if arg_add_transformer:
            with open(components_add_Transformer_path, 'r') as f:
                self.components = yaml.safe_load(f)
        elif arg_add_LLM_TSFM:
            with open(components_add_LLM_TSFM_path, 'r') as f:
                self.components = yaml.safe_load(f)
        else:
            with open(components_path, 'r') as f:
                self.components = yaml.safe_load(f)
        self.components = {k: {kk:vv for kk, vv in zip(v, preprocessing.LabelEncoder().fit_transform(v))} for k,v in self.components.items()}

        # load result, metrics: mae(√), mse, rmse, mape, mspe
        metric_matrix_train, metric_matrix_test, metric_matrix_test_mae = {}, {}, {}
        for dataset in file_dict.keys():
            for _ in file_dict[dataset]:
                if 'Transformer' in _:
                    result_path = result_path_transformer
                elif 'LLM' in _:
                    result_path = result_path_LLM
                elif 'TSFM' in _:
                    result_path = result_path_TSFM
                else:
                    result_path = result_path_non_transformer
                
                idx = 0 if metric == 'mae' else 1
                try:
                    if dataset in datasets_train:
                        metric_matrix_train['_'.join([dataset, _])] = np.load(f'{result_path}/{dataset}/{_}/metrics.npy')[idx]
                    else:
                        pass
                except FileNotFoundError:
                    pass
                    continue
        for dataset in file_dict_test.keys():
            for _ in file_dict_test[dataset]:
                if 'Transformer' in _:
                    result_path = result_path_transformer
                elif 'LLM' in _:
                    result_path = result_path_LLM
                elif 'TSFM' in _:
                    result_path = result_path_TSFM
                else:
                    result_path = result_path_non_transformer
                
                idx = 0 if metric == 'mae' else 1
                try:
                    if dataset == dataset_test:
                        metric_matrix_test['_'.join([dataset, _])] = np.load(f'{result_path}/{dataset}/{_}/metrics.npy')[idx]
                        metric_matrix_test_mae['_'.join([dataset, _])] = np.load(f'{result_path}/{dataset}/{_}/metrics.npy')[0]
                    else:
                        pass
                except FileNotFoundError:
                    pass
                    continue

        # training set
        trainset_components, trainset_meta_features, trainset_targets = [], [], []
        for k, v in metric_matrix_train.items():
            k = k.replace(f'{task_name}_', '')
            d = k.split('_')[0]
            if np.isnan(v):
                continue
            
            k_HP = '_'.join(k[re.search(re.escape('TSGym'), k).end()+1: ].split('_')[11:])
            current_components = k[re.search(re.escape('TSGym'), k).end()+1: ].split('_')[:11] +\
                                  [re.search(r'_sl(\d+)_', k_HP).group(1),
                                   re.search(r'_dm(\d+)_', k_HP).group(1),
                                   re.search(r'_df(\d+)_', k_HP).group(1),
                                   re.search(r'_el(\d+)_', k_HP).group(1),
                                   re.search(r'_epochs(\d+)_', k_HP).group(1),
                                   re.search(r'lf([^_]+)', k_HP).group(1),
                                   re.search(r'_lr([\d.]+)_', k_HP).group(1),
                                   re.search(r'lrs([^_]+)', k_HP).group(1)]
            assert len(current_components) == len(self.components)
            current_components = {list(self.components.keys())[i]: v for i, v in enumerate(current_components)}
            try:
                current_components = [self.components[k][v] for k, v in current_components.items()]
            except KeyError:
                print(f'{k} is not in components')
                continue
            trainset_components.append(current_components)
            trainset_meta_features.append(self.meta_features[d])
            trainset_targets.append([d, v])

        # testing set
        testset_components, testset_meta_features, testset_targets, testset_targets_mae, self.name_components = [], [], [], [], []
        for k, v in metric_matrix_test.items():
            self.name_components.append(k)
            v_mae = metric_matrix_test_mae[k]
            k = k.replace(f'{task_name}_', '')
            d = k.split('_')[0]

            k_HP = '_'.join(k[re.search(re.escape('TSGym'), k).end()+1: ].split('_')[11:])
            current_components = k[re.search(re.escape('TSGym'), k).end()+1: ].split('_')[:11] +\
                                  [re.search(r'_sl(\d+)_', k_HP).group(1),
                                   re.search(r'_dm(\d+)_', k_HP).group(1),
                                   re.search(r'_df(\d+)_', k_HP).group(1),
                                   re.search(r'_el(\d+)_', k_HP).group(1),
                                   re.search(r'_epochs(\d+)_', k_HP).group(1),
                                   re.search(r'lf([^_]+)', k_HP).group(1),
                                   re.search(r'_lr([\d.]+)_', k_HP).group(1),
                                   re.search(r'lrs([^_]+)', k_HP).group(1)]
            assert len(current_components) == len(self.components)
            current_components = {list(self.components.keys())[i]: v for i, v in enumerate(current_components)}
            current_components = [self.components[k][v] for k, v in current_components.items()]
            testset_components.append(current_components)
            testset_meta_features.append(self.meta_features[d])
            testset_targets.append(v)
            testset_targets_mae.append(v_mae)

        # metric to rank
        trainset_targets = pd.DataFrame(trainset_targets, columns=['dataset', 'metric'])
        for dataset in datasets_train:
            idx = (trainset_targets['dataset'] == dataset)
            trainset_targets_sub = trainset_targets.loc[idx, 'metric']
            # NA影响排序
            assert not trainset_targets_sub.isna().any()
            # to rank
            trainset_targets.loc[idx, 'metric'] = (np.argsort(np.argsort(trainset_targets_sub)).astype(np.float32) + 1) / trainset_targets_sub.shape[0]


        # to tensor
        trainset_components, trainset_meta_features, trainset_targets = (torch.from_numpy(np.stack(trainset_components)).long(),
                                                                         torch.from_numpy(np.stack(trainset_meta_features)).float(),
                                                                         torch.from_numpy(np.stack(trainset_targets['metric'].values)).float())
        testset_components, testset_meta_features, testset_targets, testset_targets_mae = (torch.from_numpy(np.stack(testset_components)).long(),
                                                                      torch.from_numpy(np.stack(testset_meta_features)).float(),
                                                                      torch.from_numpy(np.stack(testset_targets)).float(),
                                                                      torch.from_numpy(np.stack(testset_targets_mae)).float())
        print(trainset_components.shape, trainset_meta_features.shape, trainset_targets.shape)
        print(testset_components.shape, testset_meta_features.shape, testset_targets.shape, testset_targets_mae.shape)
        
        # to device
        self.device = self.utils.get_device()
        trainset_components, trainset_meta_features, trainset_targets = trainset_components.to(self.device), trainset_meta_features.to(self.device), trainset_targets.to(self.device)
        testset_components, testset_meta_features, testset_targets, testset_targets_mae = testset_components.to(self.device), testset_meta_features.to(self.device), testset_targets.to(self.device), testset_targets_mae.to(self.device)

        if self.early_stopping:
            # splitting training and validation set (70% vs 30%)
            train_size = int(0.7 * trainset_components.shape[0])
            val_size = trainset_components.shape[0] - train_size
            self.utils.set_seed(self.seed)
            trainset, valset = random_split(TensorDataset(trainset_components, trainset_meta_features, trainset_targets),
                                                      [train_size, val_size])
            # to dataloader
            self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            self.valloader = DataLoader(valset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        else:
            self.trainloader = DataLoader(TensorDataset(trainset_components, trainset_meta_features, trainset_targets),
                                          batch_size=self.batch_size, shuffle=True, drop_last=True)
            self.valloader = None

        self.testloader = DataLoader(TensorDataset(testset_components, testset_meta_features, testset_targets, testset_targets_mae),
                                     batch_size=self.batch_size, shuffle=False, drop_last=False)
        
    def loss_pearson(self, y_pred, y_true, cal_loss=True):
        vx = y_pred - torch.mean(y_pred)
        vy = y_true - torch.mean(y_true)
        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        if cal_loss:
            return 1 - corr
        else:
            return corr
        
    def meta_init(self):
        # set seed for reproductive results
        self.utils.set_seed(self.seed)
        self.model = meta_predictor(n_col=[len(_) for _ in self.components.values()], d_model=self.d_model,
                                    embed_dim_meta_feature=self.meta_feature_dim)
        self.model.to(self.device)
        self.optimizer = self.model.configure_optimizers(weight_decay=self.weight_decay, learning_rate=self.lr, device_type='cuda')
        # self.criterion = nn.MSELoss()
        self.criterion = self.loss_pearson

    def meta_fit(self, best_metric=999, best_epoch=0, es_count=0, es_tol=5, es_stopped=False):
        pred_ranks_for_true_topk_epoch, true_ranks_for_pred_topk_epoch, top1_perf_epoch, top1_perf_epoch_mae, top_name_epoch = [], [], [], [], []
        for epoch in tqdm(range(self.epochs)):
            loss_batch = []; self.model.train()
            for batch in self.trainloader:
                component, meta_feature, y_true = batch

                # clear gradient
                self.model.zero_grad()

                # loss forward
                _, y_pred = self.model(component, meta_feature)
                loss = self.criterion(y_pred.squeeze(), y_true.squeeze())

                # loss backward
                loss.backward()
                loss_batch.append(loss.item())

                # gradient update
                self.optimizer.step()

            print(f'Epoch {epoch} loss: {np.mean(loss_batch)}')
            if self.valloader is not None and not es_stopped:
                val_loss = self.meta_evaluate()
                if val_loss < best_metric:
                    print(f'best val metric: {best_metric}, current val metric: {val_loss}, continue training..')
                    best_metric = val_loss
                    es_count = 0
                else:
                    es_count += 1

                if es_count > es_tol:
                    print(f'Early stopping at epoch: {epoch}')
                    best_epoch = epoch
                    es_stopped = True
                    # break

            if self.testloader is not None:
                pred_ranks_for_true_topk, true_ranks_for_pred_topk, total_num, top1_perf, top1_perf_mase, top_name = self.meta_predict()
                pred_ranks_for_true_topk_epoch.append(pred_ranks_for_true_topk)
                true_ranks_for_pred_topk_epoch.append(true_ranks_for_pred_topk)
                top1_perf_epoch.append(top1_perf)
                top1_perf_epoch_mae.append(top1_perf_mase)
                top_name_epoch.append(top_name)

        np.savez_compressed(f'./meta/results/{self.test_dataset}-component_balance_{self.arg_component_balance}-add_transformer_{self.arg_add_transformer}-add_LLM_TSFM_{self.arg_add_LLM_TSFM}-all_periods_{self.arg_all_periods}_{self.pred_len_1}_{self.pred_len_2}.npz',
                            pred_ranks_for_true_topk_epoch=pred_ranks_for_true_topk_epoch,
                            true_ranks_for_pred_topk_epoch=true_ranks_for_pred_topk_epoch,
                            total_num=total_num,
                            top1_perf_epoch=top1_perf_epoch,
                            top1_perf_epoch_mae=top1_perf_epoch_mae,
                            top_name_epoch=top_name_epoch,
                            best_epoch=best_epoch)

    @torch.no_grad()
    def meta_evaluate(self):
        self.model.eval(); print(f'validating model...') # eval mode
        y_preds, y_trues = [], []
        for batch in tqdm(self.valloader):
            component, meta_feature, y_true = batch
            _, y_pred = self.model(component, meta_feature)
            y_preds.append(y_pred.squeeze())
            y_trues.append(y_true.squeeze())

        # validation loss
        y_preds = torch.hstack(y_preds)
        y_trues = torch.hstack(y_trues)

        loss = self.criterion(y_preds, y_trues)
        return loss.item()

    @torch.no_grad()
    def meta_predict(self, topk=10):
        self.model.eval(); print(f'testing model...') # eval mode
        y_preds, y_trues, y_trues_mae = [], [], []
        for batch in tqdm(self.testloader):
            component, meta_feature, y_true, y_true_mae = batch
            _, y_pred = self.model(component, meta_feature)
            y_preds.append(y_pred.squeeze())
            y_trues.append(y_true.squeeze())
            y_trues_mae.append(y_true_mae.squeeze())

        # validation loss
        y_preds = torch.hstack(y_preds).cpu().numpy()
        y_trues = torch.hstack(y_trues).cpu().numpy()
        y_trues_mae = torch.hstack(y_trues_mae).cpu().numpy()
        top1_perf = y_trues[np.argmin(y_preds)]
        top1_perf_mae = y_trues_mae[np.argmin(y_preds)]

        # 输出最好组合名称
        assert len(y_preds) == len(self.name_components)
        top_name = self.name_components[np.argmin(y_preds)]

        # 计算全局排名（升序排列，数值越小排名越高）
        pred_ranks = np.argsort(np.argsort(y_preds)) + 1  # 预测误差的升序排名
        true_ranks = np.argsort(np.argsort(y_trues)) + 1  # 真实误差的升序排名
        assert len(pred_ranks) == len(true_ranks)

        # 指标1：真实误差最小的top5在预测中的排名
        true_topk_indices = np.argsort(y_trues)[:topk]  # 真实误差最小的topk个索引
        pred_ranks_for_true_topk = pred_ranks[true_topk_indices]
        # print(f"{self.test_dataset}: 真实最小Topk在预测中的排名: {pred_ranks_for_true_topk}, 总数: {len(pred_ranks)}")
        logger.info(f"{self.test_dataset}: 真实最小Topk在预测中的排名: {pred_ranks_for_true_topk}, 总数: {len(pred_ranks)}")

        # 指标2：预测误差最小的top5在真实中的排名
        pred_topk_indices = np.argsort(y_preds)[:topk]  # 预测误差最小的topk个索引
        true_ranks_for_pred_topk = true_ranks[pred_topk_indices]
        # print(f"{self.test_dataset}: 预测最小Topk在真实中的排名: {true_ranks_for_pred_topk}, 总数: {len(true_ranks)}\n")
        logger.info(f"{self.test_dataset}: 预测最小Topk在真实中的排名: {true_ranks_for_pred_topk}, 总数: {len(true_ranks)}\n")
        assert len(pred_ranks) == len(true_ranks)

        return np.mean(pred_ranks_for_true_topk), np.mean(true_ranks_for_pred_topk), len(pred_ranks), top1_perf, top1_perf_mae, top_name
    
os.makedirs('meta/logfiles', exist_ok=True)
os.makedirs('meta/results', exist_ok=True)

logging.basicConfig(filename=f'meta/logfiles/meta.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()

meta = Meta(); task_name = 'LTF'
arg_component_balance, arg_add_transformer, arg_add_LLM_TSFM, arg_all_periods = True, True, False, True
if arg_add_LLM_TSFM:
    datasets = ['ETTh1', 'ETTh2', 'Exchange', 'ili']
else:
    datasets = sorted([_ for _ in os.listdir('./results_long_term_forecasting/resultsGym_non_transformer')])

for test_dataset in datasets:
    for pred_len_1, pred_len_2 in zip([96, 192, 336, 720], [24, 36, 48, 60]):
        # processing data for meta learning
        meta.components_processing(task_name=task_name,
                                   datasets=datasets,
                                   test_dataset=test_dataset,
                                   pred_len_1=pred_len_1,
                                   pred_len_2=pred_len_2,
                                   arg_component_balance=arg_component_balance,
                                   arg_add_transformer=arg_add_transformer,
                                   arg_add_LLM_TSFM=arg_add_LLM_TSFM,
                                   arg_all_periods=arg_all_periods)
        # init model
        meta.meta_init()
        # fitting meta-learner
        meta.meta_fit()
        # predicting
        meta.meta_predict()
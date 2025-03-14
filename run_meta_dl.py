import os
import sys
import torch
import re
import yaml
import numpy as np
import pandas as pd
from torch import nn
from utils.myutils import Utils
from sklearn import preprocessing
from torch.utils.data import Subset, DataLoader, TensorDataset, random_split, ConcatDataset
from meta.networks import meta_predictor
from tqdm import tqdm
import logging

class Meta():
    def __init__(self,
                 seed: int=42,
                 task_name: str='long_term_forecast',
                 early_stopping=True,
                 batch_size=128):
        self.seed = seed
        self.task_name = task_name
        self.utils = Utils()
        self.early_stopping = early_stopping
        self.batch_size = batch_size

    def components_processing(self, task_name, test_dataset,
                              components_path='./meta/components.yaml',
                              result_path='./resultsGym',
                              meta_feature_path='./meta_learner_cc/meta_feature_copy/meta_feature'):

        self.test_dataset = test_dataset
        
        file_list = [_ for _ in os.listdir(result_path) if task_name in _]
        datasets = list(set([_[re.search(re.escape(task_name), _).end()+1:].split('_')[0] for _ in file_list]))

        # load meta features
        self.meta_features = {dataset: np.load(f'{meta_feature_path}/meta_feature_{dataset}.npz', allow_pickle=True)['meta_feature'] for dataset in datasets}
        # z-score on different datasets
        mu = np.nanmean(np.stack(self.meta_features.values()), axis=0)
        std = np.nanstd(np.stack(self.meta_features.values()), axis=0)
        self.meta_features = {k: (v - mu) / (std + 1e-6) for k,v in self.meta_features.items()}
        assert (~np.isnan(np.stack(self.meta_features.values()))).all()


        datasets_train = [_ for _ in datasets if _ != test_dataset]
        dataset_test = [_ for _ in datasets if _ == test_dataset][0]

        # load components
        with open(components_path, 'r') as f:
            self.components = yaml.safe_load(f)
        self.components = {k: {kk:vv for kk, vv in zip(v, preprocessing.LabelEncoder().fit_transform(v))} for k,v in self.components.items()}

        # load result, metrics: mae, mse (√), rmse, mape, mspe
        metric_matrix_train, metric_matrix_test = {}, {}
        metric_matrix_train = {_: np.load(f'{result_path}/{_}/metrics.npy')[1] for _ in file_list if dataset_test not in _}
        metric_matrix_test = {_: np.load(f'{result_path}/{_}/metrics.npy')[1] for _ in file_list if dataset_test in _}

        # training set
        trainset_components, trainset_meta_features, trainset_targets = [], [], []
        for k, v in metric_matrix_train.items():
            k = k[re.search(re.escape(task_name), k).end()+1: ]
            d = k.split('_')[0]
            if np.isnan(v):
                continue

            current_components = k[re.search(re.escape('TSGym'), k).end()+1: ].split('_')[:8] +\
                                  [re.search(r'_sl(\d+)_', k).group(1),
                                   re.search(r'_dm(\d+)_', k).group(1),
                                   re.search(r'_df(\d+)_', k).group(1),
                                   re.search(r'_el(\d+)_', k).group(1),
                                   re.search(r'_epochs(\d+)_', k).group(1),
                                   re.search(r'_lr([\d.]+)_', k).group(1)]
            assert len(current_components) == len(self.components)
            current_components = {list(self.components.keys())[i]: v for i, v in enumerate(current_components)}
            current_components = [self.components[k][v] for k, v in current_components.items()]
            trainset_components.append(current_components)
            trainset_meta_features.append(self.meta_features[d])
            trainset_targets.append([d, v])

        # testing set
        testset_components, testset_meta_features, testset_targets = [], [], []
        for k, v in metric_matrix_test.items():
            k = k[re.search(re.escape(task_name), k).end()+1: ]
            d = k.split('_')[0]

            current_components = k[re.search(re.escape('TSGym'), k).end()+1: ].split('_')[:8] +\
                                  [re.search(r'_sl(\d+)_', k).group(1),
                                   re.search(r'_dm(\d+)_', k).group(1),
                                   re.search(r'_df(\d+)_', k).group(1),
                                   re.search(r'_el(\d+)_', k).group(1),
                                   re.search(r'_epochs(\d+)_', k).group(1),
                                   re.search(r'_lr([\d.]+)_', k).group(1)]
            assert len(current_components) == len(self.components)
            current_components = {list(self.components.keys())[i]: v for i, v in enumerate(current_components)}
            current_components = [self.components[k][v] for k, v in current_components.items()]
            testset_components.append(current_components)
            testset_meta_features.append(self.meta_features[d])
            testset_targets.append(v)

        # metric to rank
        trainset_targets = pd.DataFrame(trainset_targets, columns=['dataset', 'metric'])
        for dataset in datasets_train:
            idx = (trainset_targets['dataset'] == dataset)
            trainset_targets_sub = trainset_targets.loc[idx, 'metric']
            # NA影响排序
            assert not trainset_targets_sub.isna().any()
            # to rank
            trainset_targets.loc[idx, 'metric'] = np.argsort(np.argsort(trainset_targets_sub)) / trainset_targets_sub.shape[0]


        # to tensor
        trainset_components, trainset_meta_features, trainset_targets = (torch.from_numpy(np.stack(trainset_components)).long(),
                                                                         torch.from_numpy(np.stack(trainset_meta_features)).float(),
                                                                         torch.from_numpy(np.stack(trainset_targets['metric'].values)).float())
        testset_components, testset_meta_features, testset_targets = (torch.from_numpy(np.stack(testset_components)).long(),
                                                                      torch.from_numpy(np.stack(testset_meta_features)).float(),
                                                                      torch.from_numpy(np.stack(testset_targets)).float())
        
        # to device
        self.device = self.utils.get_device()
        trainset_components, trainset_meta_features, trainset_targets = trainset_components.to(self.device), trainset_meta_features.to(self.device), trainset_targets.to(self.device)
        testset_components, testset_meta_features, testset_targets = testset_components.to(self.device), testset_meta_features.to(self.device), testset_targets.to(self.device)

        if self.early_stopping:
            # splitting training and validation set
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

        self.testloader = DataLoader(TensorDataset(testset_components, testset_meta_features, testset_targets),
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
        self.model = meta_predictor(n_col=[len(_) for _ in self.components.values()])
        self.model.to(self.device)
        self.optimizer = self.model.configure_optimizers(weight_decay=0.01, learning_rate=0.001, device_type='cuda')
        # self.criterion = nn.MSELoss()
        self.criterion = self.loss_pearson

    def meta_fit(self, best_metric=999, es_count=0, es_tol=5):
        for epoch in range(50):
            loss_batch = []
            for i, batch in tqdm(enumerate(self.trainloader)):
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
            if self.valloader is not None:
                val_loss = self.meta_evaluate()
                if val_loss < best_metric:
                    print(f'best val metric: {best_metric}, current val metric: {val_loss}, continue training..')
                    best_metric = val_loss
                    es_count = 0
                else:
                    es_count += 1

                if es_count > es_tol:
                    print(f'Early stopping at epoch: {epoch}')
                    break
    
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
        self.model.train() # back to train mode
        return loss.item()

    @torch.no_grad()
    def meta_predict(self, topk=10):
        self.model.eval(); print(f'testing model...') # eval mode
        y_preds, y_trues = [], []
        for batch in tqdm(self.testloader):
            component, meta_feature, y_true = batch
            _, y_pred = self.model(component, meta_feature)
            y_preds.append(y_pred.squeeze())
            y_trues.append(y_true.squeeze())

        # validation loss
        y_preds = torch.hstack(y_preds).cpu().numpy()
        y_trues = torch.hstack(y_trues).cpu().numpy()

        # 计算全局排名（升序排列，数值越小排名越高）
        pred_ranks = np.argsort(np.argsort(y_preds)) + 1  # 预测误差的升序排名
        true_ranks = np.argsort(np.argsort(y_trues)) + 1  # 真实误差的升序排名
        assert len(pred_ranks) == len(true_ranks)

        # 指标1：真实误差最小的top5在预测中的排名
        true_topk_indices = np.argsort(y_trues)[:topk]  # 真实误差最小的topk个索引
        pred_ranks_for_true_topk = pred_ranks[true_topk_indices]
        logger.info(f"{self.test_dataset}: 真实最小Topk在预测中的排名: {pred_ranks_for_true_topk}, 总数: {len(pred_ranks)}")

        # 指标2：预测误差最小的top5在真实中的排名
        pred_topk_indices = np.argsort(y_preds)[:topk]  # 预测误差最小的topk个索引
        true_ranks_for_pred_topk = true_ranks[pred_topk_indices]
        logger.info(f"{self.test_dataset}: 预测最小Topk在真实中的排名: {true_ranks_for_pred_topk}, 总数: {len(true_ranks)}")

        return None

meta = Meta(); task_name = 'long_term_forecast'
file_list = [_ for _ in os.listdir('./resultsGym') if task_name in _]
datasets = list(set([_[re.search(re.escape(task_name), _).end()+1:].split('_')[0] for _ in file_list]))

os.makedirs('logfiles', exist_ok=True)
logging.basicConfig(filename=f'./logfiles/meta.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()

for test_dataset in datasets:
    # processing data for meta learning
    meta.components_processing(task_name=task_name,
                               test_dataset=test_dataset)
    # init model
    meta.meta_init()
    # fitting meta-learner
    meta.meta_fit()
    # predicting
    meta.meta_predict()
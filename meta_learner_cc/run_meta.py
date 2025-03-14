import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import set_seed
import xgboost as xgb
from tqdm import tqdm
from meta_predictor import meta_predictor_dl
from torch.utils.data import DataLoader, TensorDataset
import torch
import warnings
warnings.filterwarnings("ignore")



def get_model(model_name):
    if model_name == 'xgboost':
        return xgb.XGBRegressor(random_state=seed)
    else:
        raise NotImplementedError

def get_meta_feature(dataset_name):
    meta_feature_dir_path = '/data/coding/chaochuan/cc/meta_learner/meta_feature'
    meta_feature_path = os.path.join(meta_feature_dir_path,f'meta_feature_{dataset_name}.npz')
    meta_feature = np.load(meta_feature_path)['meta_feature']
    return meta_feature

def run_meta_dl(result_df,test_dataset,seed,scaler_meta_feature,metrics,model_name,model_save_path,dl_model_configs):
    y_pred_dict = {}
    y_test_dict = {}
    train_data = result_df[result_df['Dataset'] != test_dataset]
    test_data = result_df[result_df['Dataset'] == test_dataset]
    train_pred_len, test_pred_len = train_data['pred_len'].values, test_data.pred_len.values
    train_method, test_method = train_data['method'].values, test_data.method.values
    train_meta_feature = [scaler_meta_feature.transform(get_meta_feature(dataset).reshape(1,-1)).reshape(-1) for dataset in train_data['Dataset'].values]
    test_meta_feature = [scaler_meta_feature.transform(get_meta_feature(dataset).reshape(1,-1)).reshape(-1) for dataset in test_data['Dataset'].values]
    train_meta_feature,test_meta_feature = np.stack(train_meta_feature,axis=0), np.stack(test_meta_feature,axis=0)
    y_train, y_test = train_data[metrics].values, test_data[metrics].values

    # to tensor
    train_meta_feature = torch.from_numpy(train_meta_feature).float()
    train_pred_len = torch.from_numpy(train_pred_len).float()
    train_method = torch.from_numpy(train_method).float()
    y_train = torch.tensor(y_train).float()

    test_meta_feature = torch.from_numpy(test_meta_feature).float()
    test_pred_len = torch.from_numpy(test_pred_len).float()
    test_method = torch.from_numpy(test_method).float()
    y_test = torch.tensor(y_test).float()


    train_loader = DataLoader(TensorDataset(train_meta_feature, train_pred_len, train_method, y_train),
                            batch_size=dl_model_configs['batch_size'], shuffle=True, drop_last=False)

    model = meta_predictor_dl(
        model_name=model_name,
        meta_feature_dim=train_meta_feature.shape[1],
        n_models=dl_model_configs['n_models'],
        learning_rate=dl_model_configs['lr'],
        n_epochs=dl_model_configs['epochs'],
        optimizer=dl_model_configs['optimizer'],
        lradj=dl_model_configs['lradj'],
        train_iter=len(train_loader),
        loss_fn=dl_model_configs['loss_fn'],
        embedding_dim=dl_model_configs['model_embedding_dim'])
        
    model = model.fit(train_loader,model_save_path)

    for pl in [96,192,336,720]:
        indices = np.where(test_pred_len == pl)
        y_test_dict[pl] = y_test[indices]
        y_pred_dict[pl] = model.predict(test_meta_feature[indices],test_pred_len[indices],test_method[indices])
        y_pred_dict[pl] = y_test_dict[pl][np.argmin(y_pred_dict[pl])]      

    return y_pred_dict

def run_meta_ml(result_df,test_dataset,seed,scaler_meta_feature,metrics,model_name,model_save_path):
    X_train = []
    y_train = []
    X_test_dict = {96:[],192:[],336:[],720:[]}
    y_test_dict = {96:[],192:[],336:[],720:[]}
    dict_predlen2onehot = {96:[1,0,0,0],192:[0,1,0,0],336:[0,0,1,0],720:[0,0,0,1]}
    y_pred_dict = {}
    for index, row in result_df.iterrows():
        dataset_meta_feature = scaler_meta_feature.transform(get_meta_feature(row['Dataset']).reshape(1,-1)).reshape(-1).tolist()
        dataset_meta_feature.extend(dict_predlen2onehot[row['pred_len']])
        dataset_meta_feature.append(row['method'])
        if row['Dataset'] != test_dataset:
            X_train.append(dataset_meta_feature)
            y_train.append(row[metrics])
        else:
            X_test_dict[row['pred_len']].append(dataset_meta_feature)
            y_test_dict[row['pred_len']].append(row[metrics])
    if os.path.exists(os.path.join(model_save_path,'xgboost.json')):
        print('model file exist!')
        model = get_model(model_name)
        model.load_model(os.path.join(model_save_path,'xgboost.json'))
    else:
        model = get_model(model_name).fit(X_train,y_train)
        model.save_model(os.path.join(model_save_path,'xgboost.json'))
    for pl in [96,192,336,720]:
        y_pred_dict[pl] = y_test_dict[pl][np.argmin(model.predict(X_test_dict[pl]))]

    return y_pred_dict

def get_benchmark_result(metrics):
    result_df = pd.read_csv(f'./benchmark_baseline/result_multivarite_{metrics}.csv')
    result_df = result_df.fillna(999)
    result_df['Dataset'] = [x.split('/')[1].split('-')[0] for x in result_df['Dataset-Quantity-metrics'].values]
    result_df['pred_len'] = [int(x.split('/')[1].split('-')[1]) for x in result_df['Dataset-Quantity-metrics'].values]
    result_df = result_df[result_df.pred_len.isin([96,192,336,720])]
    # result_df = result_df[result_df.Dataset.isin(['ETTh1','ETTh2','ETTm1','ETTm2','Electricity','Exchange','Traffic','Weather'])]
    pred_len = result_df.pred_len.values
    result_df = result_df.drop(columns=['Dataset-Quantity-metrics','pred_len'],axis=1)
    
    # 转换为相对排名
    result_df = result_df.set_index('Dataset').rank(axis=1, ascending=True).reset_index()

    # 将排名转换为0-1之间的小数
    result_df = result_df.set_index('Dataset').apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1).reset_index()
    result_df['pred_len'] = pred_len
    return result_df

def get_meta_feature_scaler(result_df):
    meta_feature_list = []
    for dataset_name in list(set(result_df['Dataset'].values)):
        meta_feature = get_meta_feature(dataset_name)
        meta_feature_list.append(meta_feature)
    scaler_meta_feature = MinMaxScaler(clip=True).fit(np.array(meta_feature_list))
    return scaler_meta_feature

        
def run(metrics,seed,model_name,model_save_path,dl_model_configs,suffix):
    result_df = get_benchmark_result(metrics)
    scaler_meta_feature = get_meta_feature_scaler(result_df.copy())
    model_list = [x for x in result_df.columns.tolist() if 'Dataset' != x and 'pred_len' != x]
    
    model_labelencoder = preprocessing.LabelEncoder().fit(model_list)
    result_df = pd.melt(result_df, id_vars=['Dataset','pred_len'], var_name='method', value_name=metrics)
    result_df['method'] = model_labelencoder.transform(result_df['method'])
    result = pd.DataFrame(columns=['Dataset','pred_len','method',metrics])
    for test_dataset in tqdm(list(set(result_df['Dataset'].values))):
        model_save_path = os.path.join(model_save_path,test_dataset)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if model_name in {'xgboost'}:
            y_pred_dict = run_meta_ml(result_df.copy(),test_dataset,seed,scaler_meta_feature,metrics,model_name,model_save_path)
        elif model_name in {'mlp'}:
            dl_model_configs['n_models'] = len(model_list)
            y_pred_dict = run_meta_dl(result_df.copy(),test_dataset,seed,scaler_meta_feature,metrics,model_name,model_save_path,dl_model_configs)
        else:
            raise NotImplementedError
        for pl in [96,192,336,720]:
            result = result._append({'Dataset':test_dataset,'pred_len':pl,'method':'Meta',metrics:y_pred_dict[pl].item()},ignore_index=True)
    result_df['method'] =  model_labelencoder.inverse_transform(result_df['method'])
    result_meta = result_df._append(result).sort_values(['Dataset','pred_len',metrics])
    result_meta['Rank'] = result_meta.groupby(['Dataset','pred_len'])[metrics].rank(method='min', ascending=True).astype(int)
    model_avg_rank = result_meta.groupby(['method','pred_len'])['Rank'].mean().reset_index()
    model_avg_rank.rename(columns={'Rank': 'Average_Rank'}, inplace=True)

    if not os.path.exists(f'result/{suffix}/'):
        os.makedirs(f'result/{suffix}/')
    result_meta = result_meta.merge(model_avg_rank, on=['method','pred_len'], how='left')
    result_meta.to_csv(f'result/{suffix}/result_meta_{metrics}_{model_name}_seed{seed}.csv',index=False)
    result_meta = result_meta[['pred_len','method','Average_Rank']].sort_values(['pred_len','Average_Rank'])
    result_meta = result_meta.drop_duplicates()
    result_meta.to_csv(f'result/{suffix}/result_meta_{metrics}_{model_name}_seed{seed}_rankmodel.csv',index=False)


seed_list = [0]
metrics_list = ['mse','mae']
meta_model_list = ['mlp']
suffix = 'large'

for seed in seed_list:
    set_seed(seed)
    for metrics in metrics_list:
        for model_name in meta_model_list:
            model_save_path = f'./checkpoint/checkpoint_{model_name}_{metrics}_{suffix}_{seed}'
            dl_model_configs={
                'lr':1e-2,
                'epochs':20,
                'optimizer':'Adam',
                'lradj':'type1',
                'loss_fn':'mse',
                'batch_size':16,
                'model_embedding_dim':5}
            run(metrics,seed,model_name,model_save_path,dl_model_configs,suffix)
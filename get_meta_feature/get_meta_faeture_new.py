import numpy as np
import warnings
import tsfel
import os
import pandas as pd
import logging
from scipy.stats import ks_2samp, wasserstein_distance
from joblib import Parallel, delayed
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

def calculate_delta(X, m=100):
    '''
    from tfb
    '''
    X = np.array(X).flatten()
    # 步骤1：计算z-score归一化
    mean = np.mean(X)
    std = np.std(X)
    if std == 0:
        return 0.5  # 处理常数值情况
    Z = (X - mean) / std
    # 步骤2：获取Z的最小值和最大值
    Z_min = np.min(Z)
    Z_max = np.max(Z)
    # 步骤3：生成阈值集合S
    S = np.linspace(Z_min, Z_max, num=m, endpoint=False)
    # 步骤4：计算每个s_i对应的中位数M_i
    M = []
    for s_i in S:
        # 找出所有满足Z[j] > s_i的索引j
        K = np.where(Z > s_i)[0]
        if len(K) == 0:
            # 理论上不会发生，若有则跳过或处理
            M_i = np.nan
        else:
            M_i = np.median(K)
        M.append(M_i)
    M = np.array(M)
    # 处理可能的NaN值（例如用中位数填充）
    if np.isnan(M).any():
        M[np.isnan(M)] = np.nanmedian(M)
    # 步骤5：Min-Max归一化M
    M_min, M_max = M.min(), M.max()
    if M_max == M_min:
        M_prime = np.full_like(M, 0.5)
    else:
        M_prime = (M - M_min) / (M_max - M_min)
    # 步骤6：计算δ作为M_prime的中位数
    delta = np.median(M_prime)
    # 确保结果在(0,1)范围内
    delta = np.clip(delta, 1e-10, 1 - 1e-10)
    return delta

def calculate_drift_metrics(reference_data, current_data, n_jobs=-1):
    '''
    统计meta feature的前后偏移量来描述时序数据的偏移量
    '''
    metrics = []
    n_features = reference_data.shape[1]
    
    # 预计算参考数据的统计量
    ref_mean = np.mean(reference_data, axis=0)
    ref_var = np.var(reference_data, axis=0)
    current_mean = np.mean(current_data, axis=0)
    current_var = np.var(current_data, axis=0)
    
    # 并行化单变量指标计算
    def _parallel_ks(f):
        return ks_2samp(reference_data[:, f], current_data[:, f])[0]
    ks_values = Parallel(n_jobs=n_jobs)(delayed(_parallel_ks)(f) for f in range(n_features))
    metrics.append(np.mean(ks_values))
    metrics.append(np.max(ks_values))
    metrics.append(np.median(ks_values))
    
    def _parallel_wd(f):
        return wasserstein_distance(reference_data[:, f], current_data[:, f])
    wd_values = Parallel(n_jobs=n_jobs)(delayed(_parallel_wd)(f) for f in range(n_features))
    metrics.append(np.mean(wd_values))
    metrics.append(np.max(wd_values))
    metrics.append(np.median(wd_values))
    
    # 均值方差差异（使用预计算值）
    metrics.append(np.mean(np.abs(ref_mean - current_mean)))
    metrics.append(np.mean(np.abs(ref_var - current_var)))
    
    # 欧氏距离
    euc_distance = np.sqrt(np.sum((reference_data - current_data)**2))
    metrics.append(euc_distance)
    # 余弦距离
    dot_product = np.dot(reference_data.flatten(), current_data.flatten())
    norm1 = np.linalg.norm(reference_data)
    norm2 = np.linalg.norm(current_data)
    metrics.append((norm1 * norm2) / dot_product)
    # 曼哈顿距离
    manhattan_distance = np.sum(np.abs(reference_data - current_data))
    metrics.append(manhattan_distance)
    return metrics

def get_meta_feature_series(numpy_array):
    '''
    input: 部分时间序列: num_samples,num_features
    '''
    # get meat feature
    cfg = tsfel.get_features_by_domain() # Extracts the temporal, statistical and spectral feature sets.
    meta_feature = []
    for i in tqdm(range(numpy_array.shape[1])):# num_features
        X = tsfel.time_series_features_extractor(cfg, numpy_array[:,i], fs=100,verbose=0,n_jobs=-1).values # (num_meta-feartures, )
        meta_feature.append(X)
    meta_feature = np.concatenate(meta_feature,axis=0) # num_features,num_meta_features
    mean = np.mean(meta_feature,axis=0)
    std = np.std(meta_feature,axis=0)
    min_val = np.min(meta_feature,axis=0)
    q25 = np.percentile(meta_feature, 25,axis=0)
    median = np.median(meta_feature,axis=0)
    q75 = np.percentile(meta_feature, 75,axis=0)
    max_val = np.max(meta_feature,axis=0)
    range_val = max_val - min_val
    iqr = q75 - q25
    combined_features = np.stack([mean, std, min_val, q25, median, q75, max_val, range_val, iqr])#(9,num_meta_features)
    return combined_features

def get_meata_feature(df_raw, data_name,flag,seq_len,num_subseries=20):
    '''
    提取10段子序列的元特征和对应的特征漂移问题
    '''
    assert flag in ['train','test','val','all']
    type_map = {'train': 0, 'val': 1, 'test': 2, 'all':3}
    set_type = type_map[flag]
    df_raw = df_raw.drop(columns='date')
    if 'ETT' in data_name:
        seq_len = 24 * 4 * 4
        if 'ETTh' in data_name:
            border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len, 0]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif 'ETTm' in data_name:
            border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len, 0]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            raise NotImplementedError
    else:
        seq_len = seq_len
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - seq_len, len(df_raw) - num_test - seq_len, 0]
        border2s = [num_train, num_train + num_vali, len(df_raw), len(df_raw)]
    border1 = border1s[set_type]
    border2 = border2s[set_type]
    data = df_raw.values
    data = data[border1:border2]# num_sample,num_features
    whole_meta_features = get_meta_feature_series(data)
    # 计算总体data的分布漂移情况（from TFB）
    print('**************start calculate delta***************')
    delta_array_list = []
    for i in tqdm(range(data.shape[1])):
        array = data[:,i]
        delta_array = calculate_delta(array) # 标量
        delta_array_list.append(delta_array)
    delta_array_list = np.array(delta_array_list) # (num_features, )
    
    print('start calculate 10 meta features')
    # 计算起始子序列和后续9个子序列的元特征的差异
    diff_sub_series_meta_feature = []
    time_lenth = 10*(data.shape[0] // 10)
    data_time_split_list = np.split(data[-time_lenth:],10)
    meta_feature_list = [get_meta_feature_series(array) for array in tqdm(data_time_split_list)] # list: (9,num_meta_features)
    first_sub_series = meta_feature_list[0]
    for sub_series in meta_feature_list[1:]:
        diff_metrics = calculate_drift_metrics(first_sub_series, sub_series) #(num_metrics,)
        diff_sub_series_meta_feature.extend(diff_metrics)
    diff_sub_series_meta_feature = np.array(diff_sub_series_meta_feature)#(num_metrics*9, )
 
    return whole_meta_features,delta_array_list,np.array(meta_feature_list),diff_sub_series_meta_feature
    
    
def main(root_path):
    data_dir_path_list = os.listdir(root_path)
    data_path_list = []
    data_name_list = []
    for data_dir_path in data_dir_path_list:
        data_path = [x for x in os.listdir(os.path.join(root_path,data_dir_path)) if 'csv' in x]
        data_path_list.extend([os.path.join(root_path,data_dir_path,x) for x in data_path])
        data_name_list.extend([x.split('.csv')[0] for x in data_path])
    print(data_path_list)
    print(data_name_list)
    for flag in ['train']: # train and all 不受seqlen影响
        for data_path,data_name in zip(data_path_list,data_name_list):
            if 'm4' in data_path:
                continue
            if data_name in ['solar', 'czelan', 'metr-la', 'aqshunyi', 'pems08', 'nyse', 'wind', 'pems04', 'us_births_dataset_1', 'wike2000',
                             'aqwan', 'pems-bay', 'zafnoo','nasdaq', 'covid-19', 'fred-md', 'Yearly-test', 'Weekly-test', 'submission-Naive2', 'nn5']:
                continue
            print(data_path)
            print(data_name)
            data = pd.read_csv(data_path)
            for seq_len in [96]:
                if 'ill' in data_name:
                    seq_len = 24
                try:
                    data_meta_feature,data_delta,subseries_meta_feature,subseries_diff = get_meata_feature(data,data_name,flag,seq_len)
                    np.savez_compressed(f'./meta_features_new/meta_feature_{data_name}_{flag}_{seq_len}.npz',
                                        meta_feature=data_meta_feature,
                                        data_delta=data_delta,
                                        subseries_meta_feature=subseries_meta_feature,
                                        subseries_diff=subseries_diff)
                    logger.info(f"meta_feature_{data_name}_{flag}_{seq_len} succees!\n")
                except Exception as e:
                    logger.error(f"meta_feature_{data_name}_{flag}_{seq_len} unfinished:{e}!\n")


os.makedirs('logfiles', exist_ok=True)
os.makedirs('meta_features_new', exist_ok=True)
logging.basicConfig(filename=f'logfiles/meta0428.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()

main('/root/minqi/TSGym/dataset')
    
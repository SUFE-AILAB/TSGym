import numpy as np
import warnings
import tsfel
import os
import pandas as pd
import logging
import warnings
warnings.filterwarnings("ignore")

def get_meata_feature(df_raw, data_name,flag,seq_len):
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
    data = data[border1:border2]
    
    # get meat feature
    cfg = tsfel.get_features_by_domain() # Extracts the temporal, statistical and spectral feature sets.
    meta_feature = []
    for i in range(data.shape[1]):
        X = tsfel.time_series_features_extractor(cfg, data[:,i], fs=100,verbose=0).values
        meta_feature.append(X)
    # print(len(meta_feature))
    meta_feature = np.concatenate(meta_feature,axis=0)
    # print(meta_feature.shape)
    
    mean = np.mean(meta_feature,axis=0)
    # print(mean.shape)
    std = np.std(meta_feature,axis=0)
    min_val = np.min(meta_feature,axis=0)
    q25 = np.percentile(meta_feature, 25,axis=0)
    median = np.median(meta_feature,axis=0)
    q75 = np.percentile(meta_feature, 75,axis=0)
    max_val = np.max(meta_feature,axis=0)
    range_val = max_val - min_val
    iqr = q75 - q25
    
    combined_features = np.stack([mean, std, min_val, q25, median, q75, max_val, range_val, iqr])
    # print(combined_features.shape)
    return combined_features
    
    
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
    for flag in ['train','all','test','val']: # train and all 不受seqlen影响
        for data_path,data_name in zip(data_path_list,data_name_list):
            print(data_path)
            print(data_name)
            data = pd.read_csv(data_path)
            for seq_len in [96]:
                if 'ill' in data_name:
                    seq_len = 24
                try:
                    meta_feature = get_meata_feature(data,data_name,flag,seq_len)
                    np.savez_compressed(f'./meta_features/meta_feature_{data_name}_{flag}_{seq_len}.npz',meta_feature=meta_feature)
                    logger.info(f"meta_feature_{data_name}_{flag}_{seq_len} succees!\n")
                except Exception as e:
                    logger.error(f"meta_feature_{data_name}_{flag}_{seq_len} unfinished:{e}!\n")


os.makedirs('logfiles', exist_ok=True)
os.makedirs('meta_features', exist_ok=True)
logging.basicConfig(filename=f'logfiles/meta.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()

main('/root/minqi/TSGym/dataset')
    
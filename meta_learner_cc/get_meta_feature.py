import tsfel
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

root_path = '/data/coding/chaochuan/TSGym/dataset'
dataset_dir_list = [x for x in os.listdir(root_path) if 'm4' not in x and '.' not in x]

print(dataset_dir_list)

for dataset_name in dataset_dir_list:
    # print(dataset_name)
    whole_data = []
    for dataset_path in [x for x in os.listdir(os.path.join(root_path,dataset_name)) if '.' not in x]:
        print(dataset_path)
        data = pd.read_csv(os.path.join(root_path,dataset_name,dataset_path))
        data = data.drop(columns=['date'],axis=1)
        # whole_data.append(data)
    
    # whole_data = pd.concat(whole_data, ignore_index=True).values
        cfg = tsfel.get_features_by_domain() # Extracts the temporal, statistical and spectral feature sets.
        meta_feature = []
        for i in range(whole_data.shape[1]):
            X = tsfel.time_series_features_extractor(cfg, whole_data[:,i], fs=100,verbose=0).values
            meta_feature.append(X)
        meta_feature = np.concatenate(meta_feature,axis=0).mean(axis=0)

        np.savez_compressed(f'./meta_feature/meta_feature_{dataset_name}.npz',meta_feature=meta_feature)
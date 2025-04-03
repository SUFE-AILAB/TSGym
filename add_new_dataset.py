import pandas as pd
import os
import subprocess
from multiprocessing import Pool

def run_task(task_command, task_id):
    try:
        # 使用 subprocess 来执行 shell 命令
        subprocess.run(task_command, shell=True, check=True)
        print(f"Task {task_id} completed successfully.", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError executing task {task_id}: \n{e}", flush=True)

def create_task_list():
    # https://decisionintelligence.github.io/OpenTS/datasets/#Multivariate-time-series
    # Prepare the cleaned data from the OCR result
    data = [
        # ["metr-la", "Traffic", "5 mins", 34272, 207, 712],
        # ["pems-bay", "Traffic", "5 mins", 52116, 325, 712],
        # ["pems04", "Traffic", "5 mins", 16992, 307, 622],
        # ["pems08", "Traffic", "5 mins", 17856, 170, 622],
        # # ["traffic", "Traffic", "1 hour", 17544, 862, 712,'Traffic_script'],
        # # ["electricity", "Electricity", "1 hour", 26304, 321, 712,'ECL_script'],
        # ["solar", "Energy", "10 mins", 52560, 137, 622],
        # ["wind", "Energy", "15 mins", 48673, 7, 712, 622],
        # # ["weather", "Environment", "10 mins", 52696, 21, 712,'Weather_script'],
        # ["aqshunyi", "Environment", "1 hour", 35064, 11, 622],
        # ["aqwan", "Environment", "1 hour", 35064, 11, 622],
        # ["zafnoo", "Nature", "30 mins", 19225, 1, 712],
        # ["czelan", "Nature", "30 mins", 19934, 11, 712],
        ["fred-md", "Economic", "1 month", 728, 107, 712],
        # # ["exchange_rate", "Economic", "1 day", 7588, 8, 712,'Exchange_script'],
        # ["nasdaq", "Stock", "1 day", 1244, 5, 712],
        # ["nyse", "Stock", "1 day", 1243, 5, 712],
        # ["nn5", "Banking", "1 day", 791, 111, 712],
        # # ["iliness", "Health", "1 week", 966, 7, 712,'ILI_script'],
        ["covid-19", "Health", "1 day", 1392, 948, 712],
        # ["wike2000", "Web", "1 day", 792, 2000, 712],
    ]
    df = pd.DataFrame(data, columns=["Dataset", "Domain", "Frequency", "Lengths", "Dim", "Split"])

    task_list = []

    model_list =  ['Autoformer', 'PatchTST', 'TimesNet', 'DLinear', 'LightTS', 'Pyraformer', 'MICN', 'Mamba', 'Koopa', 'FEDformer', 'Reformer', 'SegRNN', 'Crossformer', 'debug', 'TimeMixer', 'TSGym', 'Nonstationary_Transformer', 'FiLM', 'ETSformer', 'TSMixer', 'TimeXer', 'iTransformer', 'Informer', 'Transformer']
    for model_name in model_list:
        for i in range(len(df)):
            data_name = df.loc[i, 'Dataset']
            data_fea_num = df.loc[i, 'Dim']
            task_command = f"""python3 -u run.py \
                    --task_name long_term_forecast \
                    --is_training 1 \
                    --root_path ./dataset/{data_name}/ \
                    --data_path {data_name}.csv \
                    --model_id {data_name}_36_24 \
                    --model {model_name} \
                    --data custom \
                    --features M \
                    --seq_len 36 \
                    --label_len 18 \
                    --pred_len 24 \
                    --e_layers 2 \
                    --d_layers 1 \
                    --factor 3 \
                    --enc_in {data_fea_num} \
                    --dec_in {data_fea_num} \
                    --c_out {data_fea_num} \
                    --des 'Exp' \
                    --itr 1 """
            task_list.append(task_command)
    
    return task_list

def main():
    # 创建任务列表
    task_list = create_task_list()

    # 使用 multiprocessing.Pool 来并行运行任务
    with Pool(processes=1) as pool:  # 设置进程池的大小（例如5个并行进程）
        pool.starmap(run_task, [(command, idx + 1) for idx, command in enumerate(task_list)])

        # 等待所有任务完成
        pool.close()  # 关闭进程池，停止接受新任务
        pool.join()  # 等待池中的所有任务完成

    print("All tasks started.")


if __name__ == "__main__":
    main()
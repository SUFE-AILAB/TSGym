#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 指定目录路径
dir_path="./scripts/long_term_forecast/ECL_script/gym"
echo "Current directory: $(pwd)"

# 找到所有 .sh 脚本并使用 xargs 并行执行，限制为12个并行进程
# find "$dir_path" -name '*.sh' -print0 | xargs -0 -n 1 -P 10 bash
find "$dir_path" -name '*Transformer*.sh' -print0| shuf -z | xargs -0 -n 1 -P 3 bash

echo "All scripts have been started."
#!/bin/bash

# # 指定目录路径
# dir_path="./scripts/short_term_forecast"
# echo "Current directory: $(pwd)"

# # 假设所有的脚本都在 short_term_forecast 目录下，并且都以 .sh 结尾
# for script in "$dir_path"/*.sh; do
#     echo "Running $script..."
#     bash "$script"
# done

# 指定目录路径
dir_path="./scripts/short_term_forecast"
echo "Current directory: $(pwd)"

# 找到所有 .sh 脚本并使用 xargs 并行执行，限制为10个并行进程
find "$dir_path" -name '*.sh' -print0 | xargs -0 -n 1 -P 10 bash

echo "All scripts have been started."
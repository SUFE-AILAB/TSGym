#!/bin/bash

# 指定目录路径
dir_path="./scripts/short_term_forecast/gym"
echo "Current directory: $(pwd)"

# 找到所有 .sh 脚本并使用 xargs 并行执行，限制为12个并行进程
find "$dir_path" -name '*.sh' -print0 | xargs -0 -n 1 -P 10 bash

echo "All scripts have been started."
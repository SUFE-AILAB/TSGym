# 指定目录路径
dir_path="./scripts/long_term_forecast/Traffic_script"
echo "Current directory: $(pwd)"

# 找到所有 .sh 脚本并使用 xargs 并行执行，限制为10个并行进程
find "$dir_path" -name '*.sh' -print0 | xargs -0 -n 1 -P 3 bash
# find "$dir_path" -name '*.sh' -print0 | xargs -0 -n 1 -P 3 nohup bash &

echo "All scripts have been started."
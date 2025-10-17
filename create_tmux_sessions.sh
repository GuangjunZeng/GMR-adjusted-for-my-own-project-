#!/bin/bash

# 批量创建tmux会话的脚本
# 基于multiple_retarget.txt中的配置

# 切换到正确的工作目录
cd /home/retarget_proj/workbench/data/locomotion/GMR

# 定义会话配置数组
# 格式: "会话名:起始行:线程数"
sessions=(
    "smplx_robot001:8000:2"
    "smplx_robot002:8000:2"
    "smplx_robot003:8000:2"
    "smplx_robot004:8000:2"
    "smplx_robot005:8000:2"
    "smplx_robot006:8000:2"
)

echo "开始批量创建tmux会话..."

# 遍历配置数组创建会话
for session_config in "${sessions[@]}"; do
    # 解析配置
    IFS=':' read -r session_name start_row num_threads <<< "$session_config"
    
    echo "创建会话: $session_name (起始行: $start_row, 线程数: $num_threads)"
    
    # 检查会话是否已存在
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "会话 $session_name 已存在，跳过创建"
        continue
    fi
    
    # 创建新的tmux会话并运行命令
    tmux new-session -d -s "$session_name" \
        "python scripts/smplx_to_robot_multiplenpz.py --csv_file ../raw/manifest.csv --save_path ../robot/ik_based/pkl/ --start_row $start_row --num_threads $num_threads"
    
    if [ $? -eq 0 ]; then
        echo "✓ 会话 $session_name 创建成功"
    else
        echo "✗ 会话 $session_name 创建失败"
    fi
    
    # 添加短暂延迟避免冲突
    sleep 1
done

echo ""
echo "所有会话创建完成！"
echo ""
echo "查看所有tmux会话:"
tmux list-sessions

echo ""
echo "查看GPU使用情况:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{print "GPU Memory: " $1 "MB / " $2 "MB (" int($1/$2*100) "%)"}'

echo ""
echo "=== 显存使用汇总 ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{
    used = $1
    total = $2
    free = total - used
    percentage = int(used/total*100)
    print "总显存: " total "MB"
    print "已使用: " used "MB (" percentage "%)"
    print "剩余: " free "MB (" int(free/total*100) "%)"
}'

#!/bin/bash

# 定义重试函数
retry_command() {
    local command=("$@")
    local max_retries=100
    local delay_seconds=10
    local retries=0

    while [ $retries -le $max_retries ]; do
        echo "执行命令: ${command[*]}"
        "${command[@]}"
        
        # 检查命令执行结果
        if [ $? -eq 0 ]; then
            echo "命令成功完成: ${command[*]}"
            return 0
        else
            retries=$((retries + 1))
            if [ $retries -gt $max_retries ]; then
                echo "达到最大重试次数，命令执行失败: ${command[*]}"
                return 1
            fi
            echo "命令执行失败，将在$delay_seconds秒后进行第$retries次重试 (共$max_retries次)"
            sleep $delay_seconds
        fi
    done
}

# 定义要执行的下载命令
commands=(
    "huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./pretrained_models/Wan2.1-T2V-14B"
    "huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./pretrained_models/wav2vec2-base-960h"
    "huggingface-cli download OmniAvatar/OmniAvatar-14B --local-dir ./pretrained_models/OmniAvatar-14B"
)

# 逐个执行命令
for cmd in "${commands[@]}"; do
    # 将命令字符串拆分为数组
    IFS=' ' read -ra cmd_array <<< "$cmd"
    retry_command "${cmd_array[@]}"
    
    # 如果当前命令最终失败，是否继续执行下一个命令
    if [ $? -ne 0 ]; then
        echo "警告：当前命令执行失败，但将继续执行下一个命令"
    fi
done

echo "所有命令执行完毕"
    
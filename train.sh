#!/bin/bash

# 设置环境变量
export WORLD_SIZE=1
export LOCAL_WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=350
export CUDA_VISIBLE_DEVICES=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 设置配置路径和参数
TEST_MODE=false
CONFIG_PATH="/home/trx/workplace/Grounding_LLMs_with_online_RL/lamorel/examples/PPO_finetuning/"
CONFIG_NAME="local_gpu_config"
RL_SCRIPT_PATH="/home/trx/workplace/Grounding_LLMs_with_online_RL/lamorel/examples/PPO_finetuning/mac_main.py"
OUTPUT_DIR="/home/trx/workplace/trx/output"

UE_NUM=5
SEED=1113
WANDB_PROJECT="LLM_PPO_MacEnv"
WANDB_NAME="UE_num_${UE_NUM}"
# 执行 Python 脚本
python -m lamorel_launcher.launch \
    --config-path "$CONFIG_PATH" \
    --config-name "$CONFIG_NAME" \
    "rl_script_args.path=$RL_SCRIPT_PATH"\
    "rl_script_args.test=$TEST_MODE" \
    "rl_script_args.output_dir=$OUTPUT_DIR" \
    "rl_script_args.wandb_project=$WANDB_PROJECT" \
    "rl_script_args.wandb_name=$WANDB_NAME" \
    "macEnv_args.UE_num=$UE_NUM" \
    "macEnv_test_args.UE_num=$UE_NUM" \
    "macEnv_test_args.seed=$SEED" \

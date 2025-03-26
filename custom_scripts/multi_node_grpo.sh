set -x

echo "!--- Master_Addr and Master Port: ${MASTER_ADDR}:${MASTER_PORT} ---!"
echo "!--- Node Rank: ${RANK} ---!"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NNODES=2
export NPROC_PER_NODE=7
export NODE_RNAK=${RANK}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

swift rlhf \
    --rlhf_type grpo \
    --model ${MODEL_DIR} \
    --reward_funcs accuracy format \
    --use_vllm true \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len ${MAX_LENGTH:-16384} \
    --num_infer_workers 1 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'AI-MO/NuminaMath-TIR#10000' \
    --max_completion_length ${MCL:-8192} \
    --num_train_epochs 50 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 7 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length ${MAX_LENGTH:-16384} \
    --output_dir ${OUTPUT_DIR} \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 7 \
    --temperature 0.9 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero2 \
    --log_completions true \
    --report_to swanlab \
    --swanlab_token ${SWANLAB_TOKEN} \
    --swanlab_project ${SWANLAB_PROJECT:-easy_r1} \
    --swanlab_exp_name ${SWANLAB_EXP_NAME} \
    --swanlab_mode ${SWANLAB_MODE:-cloud}
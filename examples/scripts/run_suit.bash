# cd /home/ubisec/swh/codes/swh-trl/.
# python3 grpo_train.py > logs/gspo_250730.log1

# cd /home/ubisec/swh/codes/LLaMA-Factory
# conda activate swh-llama
# llamafactory-cli export examples/merge_lora/qwen3_lora_gspo.yaml >> log/merge_250724.log

# cd /home/ubisec/swh/codes/AssessModel
# conda activate swh-vllm
# python3 llm_pre_assess.py >> data/R1_0528_Qwen3_8B-lora-gspo_20250724/cpp_gspo_train1_20250730.cpp


# DISABLE_VERSION_CHECK=1  CUDA_VISIBLE_DEVICES=0 python3 ./orpo.py \
#     --dataset_name /home/ubisec/swh/codes/AssessModel/data/train_data/random800/cpp_completion_orpo_800_train_dataset_20250609.parquet \
#     --model_name_or_path=/home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-2 \
#     --per_device_train_batch_size 1 \
#     --max_steps 9600 \
#     --learning_rate 8e-5 \
#     --gradient_accumulation_steps 1 \
#     --eval_steps 500 \
#     --output_dir=/home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-orpo_adapter6 \
#     --optim rmsprop \
#     --warmup_steps 150 \
#     --report_to none \
#     --bf16 true \
#     --logging_first_step \
#     --no_remove_unused_columns \
#     --use_peft \
#     --lora_r=32 \
#     --lora_alpha=16 \
#     --lora_dropout=0.0 \
#     --max_length=12288 \
#     --max_prompt_length=4096 \
#     --max_completion_length=8192 \
#     --beta=0.1 \
#     --truncation_mode=keep_end \
#     --disable_dropout=false > ./logs/orpo6_250813.log 


# DISABLE_VERSION_CHECK=1  CUDA_VISIBLE_DEVICES=0 python3 ./orpo.py \
#     --dataset_name /home/ubisec/swh/codes/AssessModel/data/train_data/random800/cpp_completion_orpo_800_train_dataset_20250609.parquet \
#     --model_name_or_path=/home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-2 \
#     --per_device_train_batch_size 1 \
#     --max_steps 9600 \
#     --learning_rate 8e-5 \
#     --gradient_accumulation_steps 1 \
#     --eval_steps 500 \
#     --output_dir=/home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-orpo_adapter7 \
#     --optim rmsprop \
#     --warmup_steps 150 \
#     --report_to none \
#     --bf16 true \
#     --logging_first_step \
#     --no_remove_unused_columns \
#     --use_peft \
#     --lora_r=64 \
#     --lora_alpha=16 \
#     --lora_dropout=0.0 \
#     --max_length=12288 \
#     --max_prompt_length=4096 \
#     --max_completion_length=8192 \
#     --beta=0.1 \
#     --truncation_mode=keep_end \
#     --disable_dropout=false > ./logs/orpo7_250813.log

# DISABLE_VERSION_CHECK=1  CUDA_VISIBLE_DEVICES=0 python3 ./orpo.py \
#     --dataset_name /home/ubisec/swh/codes/AssessModel/data/train_data/random800/cpp_completion_orpo_800_train_dataset_20250609.parquet \
#     --model_name_or_path=/home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-2 \
#     --per_device_train_batch_size 1 \
#     --max_steps 9600 \
#     --learning_rate 8e-5 \
#     --gradient_accumulation_steps 1 \
#     --eval_steps 500 \
#     --output_dir=/home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-orpo_adapter8 \
#     --optim rmsprop \
#     --warmup_steps 150 \
#     --report_to none \
#     --bf16 true \
#     --logging_first_step \
#     --no_remove_unused_columns \
#     --use_peft \
#     --lora_r=128 \
#     --lora_alpha=16 \
#     --lora_dropout=0.0 \
#     --max_length=12288 \
#     --max_prompt_length=4096 \
#     --max_completion_length=8192 \
#     --beta=0.1 \
#     --truncation_mode=keep_end \
#     --disable_dropout=false > ./logs/orpo8_250813.log


# DISABLE_VERSION_CHECK=1  CUDA_VISIBLE_DEVICES=0 python3 ./orpo.py \
#     --dataset_name /home/ubisec/swh/codes/AssessModel/data/train_data/random800/cpp_completion_orpo_800_train_dataset_20250609.parquet \
#     --model_name_or_path=/home/ubisec/swh/train_models/selekt_stage1_instruction_train20/checkpoint-184-2 \
#     --per_device_train_batch_size 1 \
#     --max_steps 9600 \
#     --learning_rate 8e-05 \
#     --gradient_accumulation_steps 1 \
#     --eval_steps 500 \
#     --output_dir=/home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-orpo_adapter22 \
#     --optim rmsprop \
#     --warmup_steps 150 \
#     --report_to none \
#     --bf16 true \
#     --logging_first_step \
#     --no_remove_unused_columns \
#     --use_peft \
#     --lora_r=128 \
#     --lora_alpha=32 \
#     --lora_dropout=0.05 \
#     --max_length=12288 \
#     --max_prompt_length=4096 \
#     --max_completion_length=8192 \
#     --beta=0.05 \
#     --truncation_mode=keep_end \
#     --disable_dropout=false > ./logs/orpo22_250820.log 

# DISABLE_VERSION_CHECK=1  CUDA_VISIBLE_DEVICES=0 python3 ./orpo.py \
#     --dataset_name /home/ubisec/swh/codes/AssessModel/data/train_data/random800/cpp_completion_orpo_800_train_dataset_20250609.parquet \
#     --model_name_or_path=/home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-2 \
#     --per_device_train_batch_size 1 \
#     --max_steps 9600 \
#     --learning_rate 8e-05 \
#     --gradient_accumulation_steps 1 \
#     --eval_steps 500 \
#     --output_dir=/home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-orpo_adapter23 \
#     --optim rmsprop \
#     --warmup_steps 150 \
#     --report_to none \
#     --bf16 true \
#     --logging_first_step \
#     --no_remove_unused_columns \
#     --use_peft \
#     --lora_r=128 \
#     --lora_alpha=32 \
#     --lora_dropout=0.05 \
#     --max_length=12288 \
#     --max_prompt_length=8192 \
#     --max_completion_length=4096 \
#     --beta=0.05 \
#     --truncation_mode=keep_end \
#     --disable_dropout=false > ./logs/orpo23_250820.log 


DISABLE_VERSION_CHECK=1  CUDA_VISIBLE_DEVICES=0,2 python3 ./orpo.py \
    --dataset_name /home/ubisec/swh/codes/AssessModel/data/train_data/random800/cpp_completion_orpo_800_train_dataset_20250609.parquet \
    --model_name_or_path=/home/ubisec/swh/models/Qwen3-Coder-30B-A3B-Instruct \
    --per_device_train_batch_size 1 \
    --max_steps 9600 \
    --learning_rate 8e-05 \
    --gradient_accumulation_steps 1 \
    --eval_steps 500 \
    --output_dir=/home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-orpo_adapter24 \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to none \
    --bf16 true \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16 \
    --lora_dropout=0.05 \
    --max_length=12288 \
    --max_prompt_length=4096 \
    --max_completion_length=8192 \
    --beta=0.05 \
    --truncation_mode=keep_end \
    --disable_dropout=false > ./logs/orpo24_250820.log 

DISABLE_VERSION_CHECK=1  CUDA_VISIBLE_DEVICES=0,2 accelerate launch  --gpu_ids 0,2 --main_process_port 0 --config_file fsdp_config.yaml ./examples/scripts/orpo.py     --dataset_name /home/ubisec/swh/codes/AssessMod
el/data/train_data/random800/cpp_completion_orpo_800_train_dataset_20250609.parquet     --model_name_or_path=/home/ubisec/swh/models/Qwen3-Coder-30B-A3B-Instruct     --per_device_train_batch_size 1     --max_steps 9600     --learning_rate 8e-05     --gradient
_accumulation_steps 1     --eval_steps 500     --output_dir=/home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-orpo_adapter24     --optim rmsprop     --warmup_steps 150     --report_to none     --bf16 true     --logging_first_step 
    --no_remove_unused_columns     --use_peft     --lora_r=16     --lora_alpha=16     --lora_dropout=0.05     --max_length=12288     --max_prompt_length=4096     --max_completion_length=8192     --beta=0.05     --truncation_mode=keep_end     --disable_dropout
=false > ./examples/scripts/logs/orpo24_250820.log
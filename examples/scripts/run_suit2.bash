# cd /home/ubisec/swh/codes/swh-trl/.
# python3 grpo_train.py > logs/gspo_250730.log1

# cd /home/ubisec/swh/codes/LLaMA-Factory
# conda activate swh-llama
# llamafactory-cli export examples/merge_lora/qwen3_lora_gspo.yaml >> log/merge_250724.log

# cd /home/ubisec/swh/codes/AssessModel
# conda activate swh-vllm
# python3 llm_pre_assess.py >> data/R1_0528_Qwen3_8B-lora-gspo_20250724/cpp_gspo_train1_20250730.cpp


# DISABLE_VERSION_CHECK=1  CUDA_VISIBLE_DEVICES=1 python3 ./orpo.py \
#     --dataset_name /home/ubisec/swh/codes/AssessModel/data/train_data/random800/cpp_completion_orpo_800_train_dataset_20250609.parquet \
#     --model_name_or_path=/home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-2 \
#     --per_device_train_batch_size 1 \
#     --max_steps 9600 \
#     --learning_rate 8e-5 \
#     --gradient_accumulation_steps 1 \
#     --eval_steps 500 \
#     --output_dir=/home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-orpo_adapter9 \
#     --optim rmsprop \
#     --warmup_steps 150 \
#     --report_to none \
#     --bf16 true \
#     --logging_first_step \
#     --no_remove_unused_columns \
#     --use_peft \
#     --lora_r=128 \
#     --lora_alpha=32 \
#     --lora_dropout=0.0 \
#     --max_length=12288 \
#     --max_prompt_length=4096 \
#     --max_completion_length=8192 \
#     --beta=0.1 \
#     --truncation_mode=keep_end \
#     --disable_dropout=false > ./logs/orpo9_250813.log 


# DISABLE_VERSION_CHECK=1  CUDA_VISIBLE_DEVICES=1 python3 ./orpo.py \
#     --dataset_name /home/ubisec/swh/codes/AssessModel/data/train_data/random800/cpp_completion_orpo_800_train_dataset_20250609.parquet \
#     --model_name_or_path=/home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-2 \
#     --per_device_train_batch_size 1 \
#     --max_steps 9600 \
#     --learning_rate 1e-5 \
#     --gradient_accumulation_steps 1 \
#     --eval_steps 500 \
#     --output_dir=/home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-orpo_adapter10 \
#     --optim rmsprop \
#     --warmup_steps 150 \
#     --report_to none \
#     --bf16 true \
#     --logging_first_step \
#     --no_remove_unused_columns \
#     --use_peft \
#     --lora_r=16 \
#     --lora_alpha=16 \
#     --lora_dropout=0.0 \
#     --max_length=12288 \
#     --max_prompt_length=4096 \
#     --max_completion_length=8192 \
#     --beta=0.1 \
#     --truncation_mode=keep_end \
#     --disable_dropout=false > ./logs/orpo10_250813.log

# DISABLE_VERSION_CHECK=1  CUDA_VISIBLE_DEVICES=1 python3 ./orpo.py \
#     --dataset_name /home/ubisec/swh/codes/AssessModel/data/train_data/random800/cpp_completion_orpo_800_train_dataset_20250609.parquet \
#     --model_name_or_path=/home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-2 \
#     --per_device_train_batch_size 1 \
#     --max_steps 9600 \
#     --learning_rate 5e-6 \
#     --gradient_accumulation_steps 1 \
#     --eval_steps 500 \
#     --output_dir=/home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-orpo_adapter11 \
#     --optim rmsprop \
#     --warmup_steps 150 \
#     --report_to none \
#     --bf16 true \
#     --logging_first_step \
#     --no_remove_unused_columns \
#     --use_peft \
#     --lora_r=16 \
#     --lora_alpha=16 \
#     --lora_dropout=0.0 \
#     --max_length=12288 \
#     --max_prompt_length=4096 \
#     --max_completion_length=8192 \
#     --beta=0.1 \
#     --truncation_mode=keep_end \
#     --disable_dropout=false > ./logs/orpo11_250811.log

# DISABLE_VERSION_CHECK=1  CUDA_VISIBLE_DEVICES=1 python3 ./orpo.py \
#     --dataset_name /home/ubisec/swh/codes/AssessModel/data/train_data/random800/cpp_completion_orpo_800_train_dataset_20250609.parquet \
#     --model_name_or_path=/home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-2 \
#     --per_device_train_batch_size 1 \
#     --max_steps 9600 \
#     --learning_rate 8e-05 \
#     --gradient_accumulation_steps 1 \
#     --eval_steps 1600 \
#     --output_dir=/home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-orpo_adapter19 \
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
#     --disable_dropout=false > ./logs/orpo19_250818.log 

# DISABLE_VERSION_CHECK=1  CUDA_VISIBLE_DEVICES=2 python3 ./orpo.py \
#     --dataset_name /home/ubisec/swh/codes/AssessModel/data/train_data/random800/cpp_completion_orpo_800_train_dataset_20250609.parquet \
#     --model_name_or_path=/home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-2 \
#     --per_device_train_batch_size 1 \
#     --max_steps 9600 \
#     --learning_rate 8e-05 \
#     --gradient_accumulation_steps 1 \
#     --eval_steps 1600 \
#     --output_dir=/home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-orpo_adapter20 \
#     --optim rmsprop \
#     --warmup_steps 150 \
#     --report_to none \
#     --bf16 true \
#     --logging_first_step \
#     --no_remove_unused_columns \
#     --use_peft \
#     --lora_r=128 \
#     --lora_alpha=32 \
#     --lora_dropout=0.075 \
#     --max_length=12288 \
#     --max_prompt_length=4096 \
#     --max_completion_length=8192 \
#     --beta=0.05 \
#     --truncation_mode=keep_end \
#     --disable_dropout=false > ./logs/orpo20_250819.log 

DISABLE_VERSION_CHECK=1  CUDA_VISIBLE_DEVICES=2 python3 ./orpo.py \
    --dataset_name /home/ubisec/swh/codes/AssessModel/data/train_data/random800/cpp_completion_orpo_800_train_dataset_20250609.parquet \
    --model_name_or_path=/home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-2 \
    --per_device_train_batch_size 1 \
    --max_steps 9600 \
    --learning_rate 8e-05 \
    --gradient_accumulation_steps 1 \
    --eval_steps 500 \
    --output_dir=/home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-orpo_adapter21 \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to none \
    --bf16 true \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=128 \
    --lora_alpha=32 \
    --lora_dropout=0.1 \
    --max_length=12288 \
    --max_prompt_length=4096 \
    --max_completion_length=8192 \
    --beta=0.05 \
    --truncation_mode=keep_end \
    --disable_dropout=false > ./logs/orpo21_250819.log 
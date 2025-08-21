from unsloth import FastLanguageModel
import torch

# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "peft",
# ]
# ///

"""
Run the ORPO training script with the following command with some example arguments.
In general, the optimal configuration for ORPO will be similar to that of DPO without the need for a reference model:

# regular:
python examples/scripts/orpo.py \
    --dataset_name trl-internal-testing/hh-rlhf-helpful-base-trl-style \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --learning_rate 8e-6 \
    --gradient_accumulation_steps 1 \
    --eval_steps 500 \
    --output_dir="gpt2-aligned-orpo" \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns

# peft:
DISABLE_VERSION_CHECK=1  CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file ./fsdp_config.yaml examples/scripts/orpo.py \
    --dataset_name /home/ubisec/swh/codes/AssessModel/data/train_data/random800/cpp_completion_orpo_800_train_dataset_20250609.parquet \
    --model_name_or_path=/home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-2 \
    --per_device_train_batch_size 1 \
    --max_steps 9600 \
    --learning_rate 8e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 500 \
    --output_dir=/home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-orpo_adapter2 \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to none \
    --bf16 true \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=128 \
    --lora_alpha=32 \
    --lora_dropout=0.0 \
    --max_length=12288 \
    --max_prompt_length=4096 \
    --max_completion_length=8192 \
    --beta=0.05 \
    --truncation_mode=keep_end \
    --disable_dropout=false > examples/scripts/logs/orpo2_250811.log 

"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import ModelConfig, ORPOConfig, ORPOTrainer, ScriptArguments, get_peft_config
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, ORPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    print(f'script_args={script_args}, training_args={training_args}, model_args={model_args}')
    ################
    # Model & Tokenizer
    ################
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    # )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    # )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        max_seq_length=training_args.max_prompt_length + training_args.max_completion_length,
        dtype=torch.bfloat16,            # float16 / bfloat16
        load_in_4bit=True,     # 显存立省 70 %[^31^]
    )
    print(f'FastLanguageModel loaded with model_id: {model_args.model_name_or_path}, {training_args.max_prompt_length} {training_args.max_completion_length}')
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    dataset = load_dataset("parquet", data_files=script_args.dataset_name) # ,split="train"
    # load_dataset('json', data_files=r'/home/ubisec/swh/codes/AssessModel/data/train_data/random800/cpp_completion_curlora_800_train_dataset_20250609.json')
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    ################
    # Training
    ################
    trainer = ORPOTrainer(
        model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # train and save the model
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

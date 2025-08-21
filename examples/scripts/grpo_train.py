import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["UNSLOTH_DISABLE_CHAT_TEMPLATE_CHECK"] = "1"
# os.environ["UNSLOTH_DISABLE_SMART_GRADIENT_CHECKPOINTING"] = "0"
from unsloth import FastLanguageModel
import numpy as np

"""
Usage:

CUDA_VISIBLE_DEVICES=0,1 python online_dpo_train.py \
    --model_name_or_path /home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B  \
    --reward_model_path /home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B \
    --dataset_name /home/ubisec/swh/codes/AssessModel/data/train_data/online_dpo/c_cpp_completion_trl_online_dpo_train_data_20250609.parquet \
    --learning_rate 5.0e-7 \
    --output_dir  \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \
    --use_peft --max_new_tokens 8192 --temperature 0.6 --max_length 16384
    # --loss_type single_sample # sigmoid ipo

"""

import torch
import hashlib, json, re
from datetime import datetime
from typing import Dict
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
# torch.cuda.set_device(0)

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, GenerationConfig
# from trl import (
#     ModelConfig,
#     get_peft_config,
#     get_quantization_config,
# )
# from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


def get_prompt_hash(prompt: str) -> str:
    """
    将prompt转换为SHA-1哈希值
    
    Args:
        prompt: 输入的prompt字符串
        
    Returns:
        SHA-1哈希值的十六进制字符串
    """
    return hashlib.sha1(prompt.encode('utf-8')).hexdigest()


def sanitize_json_string(json_str):
    """
    处理包含未转义双引号和多行代码的 JSON 字符串
    
    Args:
        json_str: 原始 JSON 字符串
    Returns:
        dict: 解析后的字典
    """
    # 1. 首先尝试提取三个主要部分
    try:
        # 使用非贪婪匹配和多行模式
        pattern = (
            r'"gt_retention":\s*(\d+),\s*'
            r'"gt_retention_code":\s*"(.*?)",\s*'
            r'"reason":\s*"(.*?)"}'
        )
        match = re.search(pattern, json_str, re.DOTALL)
        
        if not match:
            raise ValueError("Invalid JSON string format")
            
        gt_retention = int(match.group(1))
        gt_retention_code = match.group(2)
        reason = match.group(3)
        
        # 2. 处理代码中的双引号和换行
        # 替换所有未转义的双引号
        gt_retention_code = re.sub(r'(?<!\\)"', r'\"', gt_retention_code)
        # 处理换行符
        gt_retention_code = gt_retention_code.replace('\n', '\\n').replace('\r', '')
        
        # 3. 处理 reason 中的双引号
        reason = reason.replace('"', '\\"')
        
        # 4. 构建新的 JSON 字符串
        cleaned_json = (
            '{{'
            '"gt_retention": {}, '
            '"gt_retention_code": "{}", '
            '"reason": "{}"'
            '}}'
        ).format(gt_retention, gt_retention_code, reason)
        
        # 5. 解析成字典
        return json.loads(cleaned_json)
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")
    except Exception as e:
        raise ValueError(f"Error processing JSON string: {e}")

def sanitize_json_string2(json_str):
    """
    处理包含未转义双引号、转义字符和多行代码的 JSON 字符串
    
    Args:
        json_str: 原始 JSON 字符串
    Returns:
        dict: 解析后的字典
    """
    try:
        # 1. 首先尝试直接解析
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # 2. 如果直接解析失败，尝试提取各个部分
        pattern = (
            r'"gt_retention":\s*(\d+),\s*'
            r'"gt_retention_code":\s*"(.*?)",\s*'
            r'"reason":\s*"(.*?)"}'
        )
        match = re.search(pattern, json_str, re.DOTALL)
        
        if not match:
            raise ValueError("Invalid JSON string format")
            
        gt_retention = int(match.group(1))
        gt_retention_code = match.group(2)
        reason = match.group(3)
        
        # 3. 预处理字符串
        # 处理已经转义的序列
        gt_retention_code = gt_retention_code.replace(r'\\', '___BACKSLASH___')
        gt_retention_code = gt_retention_code.replace(r'\n', '___NEWLINE___')
        gt_retention_code = gt_retention_code.replace(r'\t', '___TAB___')
        gt_retention_code = gt_retention_code.replace(r'\r', '___RETURN___')
        gt_retention_code = gt_retention_code.replace(r'\"', '___QUOTE___')
        
        # 处理未转义的双引号
        gt_retention_code = gt_retention_code.replace('"', '\\"')
        reason = reason.replace('"', '\\"')
        
        # 还原转义序列
        gt_retention_code = gt_retention_code.replace('___BACKSLASH___', '\\\\')
        gt_retention_code = gt_retention_code.replace('___NEWLINE___', '\\n')
        gt_retention_code = gt_retention_code.replace('___TAB___', '\\t')
        gt_retention_code = gt_retention_code.replace('___RETURN___', '\\r')
        gt_retention_code = gt_retention_code.replace('___QUOTE___', '\\"')
        
        # 4. 构建结果字典
        result = {
            "gt_retention": gt_retention,
            "gt_retention_code": gt_retention_code,
            "reason": reason
        }
        
        # 5. 验证结果
        json.dumps(result, ensure_ascii=False)  # 如果有问题会抛出异常
        return result
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")
    except Exception as e:
        raise ValueError(f"Error processing JSON string: {e}")
    
def extract_retention(completion: str):    
    json_data = None
    gt_retention, gt_retention_code, reason = -1, '', ''
    parse_num = 0
    
    try:
        # 尝试从文本中提取JSON部分
        # 查找可能的JSON格式内容
        json_pattern = r'\{[^{}]*"gt_retention"[^{}]*\}'
        matches = re.findall(json_pattern, completion, re.DOTALL)
        for match in matches:
            try:
                json_data = json.loads(match)
                if json_data is not None and 'gt_retention' in json_data:
                    break
            except json.JSONDecodeError:
                continue
    except Exception:
        print(f'直接提取失败, 尝试其他方法')
        pass
    
    # 第一种解析方法
    try:
        if json_data is None or 'gt_retention' not in json_data or 'gt_retention_code' not in json_data or 'reason' not in json_data:
            parse_num=1
            fixed_json = completion.split('```json\n', 1)[-1]
            fixed_json = fixed_json.replace('```', '')
            # 仅转义字符串值内的换行符（非结构换行）
            fixed_json = re.sub(
                r'("[^"]*")', 
                lambda m: m.group(0).replace('\n', '\\\\n'), 
                fixed_json
            )
            json_data = json.loads(fixed_json, strict=False) #
    except Exception as e:
        print(f'[Error-1] {datetime.now()} fail to convert completion to json_data "{json_data}": {e}')

    # 第二种解析方法
    try:
        if json_data is None or 'gt_retention' not in json_data or 'gt_retention_code' not in json_data or 'reason' not in json_data:
            parse_num=2
            fixed_json = re.sub(r'^```json\s*|\s*```$', '', completion, flags=re.DOTALL).strip()
            # 修复字符串值内的特殊字符
            def fix_json_string(match):
                s = match.group(1)
                # 转义双引号（但保留已转义的双引号）
                s = re.sub(r'(?<!\\)"', r'\\"', s)
                # 转义换行符
                s = s.replace('\n', '\\n')
                # 转义反斜杠（但保留已转义的）
                s = re.sub(r'(?<!\\)\\', r'\\\\', s)
                return f'"{s}"'
            # 处理所有字符串值
            fixed_json = re.sub(
                r'"((?:[^"\\]|\\.)*)"', 
                fix_json_string, 
                fixed_json,
                flags=re.DOTALL
            )
            json_data = json.loads(fixed_json) # # strict=False
        
    except Exception as e:
        print(f'[Error-2] {datetime.now()} fail to convert completion to json_data "{json_data}": {e}')
    
    # 第三种解析方法
    try:
        if json_data is None or 'gt_retention' not in json_data or 'gt_retention_code' not in json_data or 'reason' not in json_data:
            parse_num=3
            json_data = sanitize_json_string(completion)
    except Exception as e:
        print(f'[Error-3] {datetime.now()} fail to convert completion to json_data "{json_data}": {e}')
    
    # 第四种 json 解析方法
    try:
        if json_data is None or 'gt_retention' not in json_data or 'gt_retention_code' not in json_data or 'reason' not in json_data:
            parse_num=4
            json_data = sanitize_json_string2(completion)
    except Exception as e:
        print(f'[Error-4] {datetime.now()} fail to convert completion to json_data "{json_data}": {e}')
    
    if json_data is not None and 'gt_retention' in json_data and 'gt_retention_code' in json_data and 'reason' in json_data:
        gt_retention = json_data['gt_retention']
        gt_retention_code = json_data['gt_retention_code']
        reason = json_data['reason']
        
    # 第五种保底评价结果提取方法, split 硬截取
    try:
        if gt_retention==-1:
            parse_num=5
            if '"gt_retention"' in completion:
                text = completion.split('"gt_retention"')[1]
                if '"gt_retention_code"' in text:
                    text = text.split('"gt_retention_code"')
                    if '0' in text[0]:
                        gt_retention=0
                    elif '1' in text[0]:
                        gt_retention=1
                    elif '2' in text[0]:
                        gt_retention=2
                    if '"reason"' in text[1]:
                        text = text[1].split('"reason"')
                        first_quote = text[0].find('"')  # 找到第一个双引号的位置
                        last_quote = text[0].rfind('"')  # 找到最后一个双引号的位置
                        # 如果找到双引号，去掉第一个和最后一个双引号
                        if first_quote != -1 and last_quote != -1:
                            gt_retention_code = text[0][first_quote + 1:last_quote]
                        first_quote = text[1].find('"')  
                        last_quote = text[1].rfind('"')
                        if first_quote != -1 and last_quote != -1:
                            reason = text[1][first_quote + 1:last_quote]
    except Exception as e:
        print(f"extract_retention() exception: {e} completion={[completion]}")

    if gt_retention not in [0, 1, 2]:
        parse_num = 6
        print(f'[Error-5] {datetime.now()} finally fail to convert completion "{[completion]}" to json_data="{json_data}"')
    else:
        print(f'extract_retention(): extract successfully, parse_num={parse_num}, retention={gt_retention}, retention_code={gt_retention_code}, reason={reason}')
    
    return gt_retention, gt_retention_code, reason, parse_num


# 第3、4轮
# def old_self_reward(prompts, completions, validator, reason_quality_model=None, **kwargs):
#     artificial_mark_list = []
#     if prompts[0]==prompts[-1]:
#         print(f'self_reward(): len(prompts)={len(prompts)}, prompts={prompts[0]}\ncompletions={completions}')
#     else:
#         print(f'self_reward(): len(prompts)={len(prompts)}, prompts={prompts}\n\ncompletions={completions}')
#     for prompt in prompts:
#         if prompt.startswith('User: '):
#             prompt=prompt[6:]
#         if prompt.endswith('\n\n\n'):
#             prompt=prompt[:-2]
#         prompt_hash = hashlib.sha1(prompt.encode('utf-8')).hexdigest()
#         if prompt_hash in validator:
#             artificial_mark_list.append(validator[prompt_hash])
#         else:
#             print(f'[Error]: prompt_hash={prompt_hash} not found in validator, using None as artificial mark.')
#             artificial_mark_list.append(None)
#     json_penaltys, label_rewards, code_rewards = [], [], []
#     for am, completion in zip(artificial_mark_list, completions):
#         reward = 0
#         #  json 质量检测
#         retention, retention_code, reason, parse_num = extract_retention(completion)
#         if parse_num<2:
#             json_penaltys.append(0.0)
#         elif parse_num==2:
#             json_penaltys.append(-0.1)
#         elif parse_num==3:
#             json_penaltys.append(-0.2)
#         elif parse_num==4:
#             json_penaltys.append(-0.5)
#         elif parse_num==5:
#             json_penaltys.append(-0.7)
#         elif parse_num==6:
#             json_penaltys.append(-1.0)
        
#         # 标签检测
#         if am is None:
#             print(f'[Error]: artificial mark is None, set reward = 0')
#             rewards.append(reward)
#             continue

#         gt_retention, original_retention_code, gt_retention_code, gt_reason = am['gt_retention'], am['retention_code'], am['gt_retention_code'], am['reason']
#         if len(json_penaltys)==1:
#             print(f'gt_retention={gt_retention}, original_retention_code={original_retention_code}, gt_retention_code={gt_retention_code}, gt_reason={gt_reason}')
#         if gt_retention==1 and gt_retention_code=='':
#             gt_retention_code = original_retention_code
#         if gt_retention==2:
#             gt_retention_code = ''
        
#         if retention==gt_retention:
#             label_rewards.append(1.0)
#         elif retention==1 and gt_retention==2:
#             label_rewards.append(0.4)
#         elif retention==2 and gt_retention==1:
#             label_rewards.append(0.2)
#         elif retention==2 and gt_retention==0:
#             label_rewards.append(-0.2)
#         elif retention==1 and gt_retention==0:
#             label_rewards.append(-0.4)
#         elif retention==0 and gt_retention==2:
#             label_rewards.append(-0.6)
#         elif retention==0 and gt_retention==1:
#             label_rewards.append(-1.0)
#         else:
#             label_rewards.append(-2.0)
            
#         # 推荐代码质量检测
#         if retention_code==gt_retention_code:
#             if retention==gt_retention:
#                 code_rewards.append(0.5)
#             else:
#                 code_rewards.append(0.2)
#         else:
#             code_rewards.append(0)
        
#     rewards = []
#     for penalty, label_reward, code_reward in zip(json_penaltys, label_rewards, code_rewards):
#         reward = penalty + label_reward + code_reward
#         rewards.append(reward)
#     print(f'self_reward(): json_penaltys={json_penaltys}, label_rewards={label_rewards}, code_rewards={code_rewards}, rewards={rewards}')
    
#     return rewards


# 第 5 轮
def self_reward(prompts, completions, validator, reason_quality_model=None, **kwargs):
    artificial_mark_list = []
    if prompts[0]==prompts[-1]:
        print(f'self_reward(): len(prompts)={len(prompts)}, prompts={prompts[0]}\ncompletions={completions}')
    else:
        print(f'self_reward(): len(prompts)={len(prompts)}, prompts={prompts}\n\ncompletions={completions}')
    for prompt in prompts:
        if prompt.startswith('User: '):
            prompt=prompt[6:]
        if prompt.endswith('\n\n\n'):
            prompt=prompt[:-2]
        prompt_hash = hashlib.sha1(prompt.encode('utf-8')).hexdigest()
        if prompt_hash in validator:
            artificial_mark_list.append(validator[prompt_hash])
        else:
            print(f'[Error]: prompt_hash={prompt_hash} not found in validator, using None as artificial mark.')
            artificial_mark_list.append(None)
    json_penaltys, label_rewards, code_rewards, reason_rewards = [], [], [], []
    for am, completion in zip(artificial_mark_list, completions):
        reward = 0
        #  json 质量检测
        retention, retention_code, reason, parse_num = extract_retention(completion)
        if parse_num<2:
            json_penaltys.append(0.0)
        elif parse_num==2:
            json_penaltys.append(-0.1)
        elif parse_num==3:
            json_penaltys.append(-0.2)
        elif parse_num==4:
            json_penaltys.append(-0.5)
        elif parse_num==5:
            json_penaltys.append(-0.7)
        else:
            json_penaltys.append(-1.0)
        
        # 标签检测
        if am is None:
            print(f'[Error]: artificial mark is None, set reward = 0')
            label_rewards.append(0), code_rewards.append(0), reason_rewards.append(0)
            continue

        gt_retention, original_retention_code, gt_retention_code, gt_reason = am['gt_retention'], am['retention_code'], am['gt_retention_code'], am['reason']
        if len(json_penaltys)==1:
            print(f'gt_retention={gt_retention}, original_retention_code={original_retention_code}, gt_retention_code={gt_retention_code}, gt_reason={gt_reason}')
        if gt_retention==1 and gt_retention_code=='':
            gt_retention_code = original_retention_code
        if gt_retention==2:
            gt_retention_code = ''
        
        if retention==gt_retention:
            label_rewards.append(1.0)
        elif retention==1 and gt_retention==2:
            label_rewards.append(0.4)
        elif retention==2 and gt_retention==1:
            label_rewards.append(0.2)
        elif retention==2 and gt_retention==0:
            label_rewards.append(-0.2)
        elif retention==1 and gt_retention==0:
            label_rewards.append(-0.4)
        elif retention==0 and gt_retention==2:
            label_rewards.append(-0.6)
        elif retention==0 and gt_retention==1:
            label_rewards.append(-1.0)
        else:
            label_rewards.append(-2.0)
            
        # 推荐代码质量检测
        if retention_code==gt_retention_code:
            if retention==gt_retention:
                code_rewards.append(0.5)
            else:
                code_rewards.append(0.2)
        else:
            code_rewards.append(0)
        
        # 理由质量检测
        if reason_quality_model: # 非空表明开启评价原因相似性检测
            if retention==gt_retention:
                reason_sim_score = reason_quality_model.predict([(gt_reason, reason)])[0]
                if reason_sim_score<0.6:
                    reason_rewards.append(-0.2)
                elif reason_sim_score<=0.7:
                    reason_rewards.append(0)
                else:
                    reason_rewards.append(0.2)
            else:
                reason_rewards.append(0)
        else:
            reason_rewards.append(0)
    
    if np.average(label_rewards)>0.8:
        for i in range(len(reason_rewards)):
            code_rewards[i], reason_rewards[i] = 2 * code_rewards[i], 2 * reason_rewards[i]
    
    rewards = []
    for penalty, label_reward, code_reward, reason_reward in zip(json_penaltys, label_rewards, code_rewards, reason_rewards):
        reward = penalty + label_reward + code_reward + reason_reward
        rewards.append(reward)
    print(f'self_reward(): json_penaltys={json_penaltys}, label_rewards={label_rewards}, code_rewards={code_rewards}, reason_rewards={reason_rewards}, rewards={rewards}')
    
    return rewards


if __name__ == "__main__":

    dataset = load_dataset("parquet", data_files=r'/home/ubisec/swh/codes/AssessModel/data/train_data/grpo/cpp_completion_grpo_train_data_20250609_150.parquet',split="train")
    print(f'dataset={dataset}\nlen(dataset)={len(dataset)}')
    training_args = GRPOConfig(
        output_dir="/home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-gspo_adapter5",
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8,
        num_generations=8,
        num_iterations=2,
        num_train_epochs=2,
        max_completion_length=8192,
        max_prompt_length=4096,
        max_grad_norm=1.0,
        top_p=1.0,
        importance_sampling_level='sequence',
        beta=0.1,
        # loss_type='dr_grpo',
        # top_entropy_quantile=0.2,
        use_vllm=True,
        vllm_mode="openai",
        # vllm_mode="server",
        vllm_server_host='0.0.0.0',
        vllm_server_port=12300,
        fp16=False,
        bf16=True,
        save_steps=50
    )
    print(f'training_args={training_args}')
    trainer = GRPOTrainer(
        model="/home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-2",
        reward_funcs=self_reward,
        args=training_args,
        train_dataset=dataset,
        peft_config=True
    )
    
    trainer.train()

# CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model /home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-2 --host 0.0.0.0 --port 12300 --max-model-len 16384 --dtype bfloat16
# CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model /home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-2 --host 0.0.0.0 --port 12300 --max-model-len 16384 --dtype bfloat16 --enable-prefix-caching true --tensor-parallel-size 1
# CUDA_VISIBLE_DEVICES=1  vllm serve /home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-2 --host 0.0.0.0 --port 12300 --max-model-len 16384 --dtype bfloat16 \
    # --enable-prefix-caching  --tensor-parallel-size 1 --enable-lora  --lora-modules lora1=/home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-gspo_adapter3/checkpoint-100

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

import concurrent.futures
import logging
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from accelerate import Accelerator
from huggingface_hub import InferenceClient
from transformers.utils import is_openai_available

from ..import_utils import is_llm_blender_available


if is_llm_blender_available():
    import llm_blender

if is_openai_available():
    from openai import OpenAI


DEFAULT_PAIRWISE_SYSTEM_PROMPT = '''I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": """{prompt}""",
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "0",
        "output": """{response0}"""
    }},
    {{
        "model_identifier": "1",
        "output": """{response1}"""
    }}
}}

## Task

Evaluate the models on the basis of the quality and relevance of their results, and select the model that generated the best result. Reply with the identifier of the best model. Our evaluation will only take into account the first character of your answer, so make sure it contains only one of the identifiers and nothing else (no quotation marks, no spaces, no new lines, ...).
'''


class BaseJudge(ABC):
    """
    Base class for judges. The subclasses of this class should implement the `judge` method.
    """

    @abstractmethod
    def judge(self, prompts: list[str], completions: list[str], shuffle_order: bool = True) -> list:
        raise NotImplementedError("Judge subclasses must implement the `judge` method.")


class BaseRankJudge(ABC):
    """
    Base class for LLM ranking judges.

    **Example**:
    ```python
    class MyRankJudge(BaseRankJudge):
        def judge(self, prompts, completions, shuffle_order=True):
            return ...  # Your ranking logic here


    judge = MyRankJudge()
    judge.judge(
        prompts=["The capital of France is", "The capital of Germany is"],
        completions=[[" Paris", " Marseille", "Lyon"], [" Munich", " Berlin"]],
    )  # [[0, 1, 2], [1, 0]]
    ```
    """

    @abstractmethod
    def judge(self, prompts: list[str], completions: list[list[str]], shuffle_order: bool = True) -> list[list[int]]:
        """
        Judge the completion for the given prompts and return the ranks of each completion.

        Args:
            prompts (`list[str]`):
                List of prompts.
            completions (`list[list[str]]`):
                List of completions list, where each element is a list of completions for the corresponding prompt.
            shuffle_order (`bool`, *optional*, defaults to `True`):
                Whether to shuffle the order of the completions to avoid positional bias.

        Returns:
            `list[list[int]]`:
                List of lists of idxs, where each list contains the ranks of the completions for the corresponding
                prompt. E.g., `[1, 2, 0]` means that the second completion (`idx=1`) is the best, followed by the
                third, and then the first.
        """
        raise NotImplementedError("Judge subclasses must implement the `judge` method.")


class BasePairwiseJudge(BaseJudge):
    """
    Base class for pairwise judges.
    """

    @abstractmethod
    def judge(self, prompts: list[str], completions: list[list[str]], shuffle_order: bool = True) -> list[int]:
        """
        Judge the completion pairs for the given prompts.

        Args:
            prompts (`list[str]`):
                List of prompts.
            completions (`list[list[str]]`):
                List of completions pairs, where each element is a pair of completions for the corresponding prompt.
            shuffle_order (`bool`, *optional*, defaults to `True`):
                Whether to shuffle the order of the completions to avoid positional bias.

        Returns:
            `list[int]`:
                List of idxs, where each idx is the rank of the best completion for the corresponding prompt. E.g., `1`
                means that the second completion (`idx=1`) is the best.

        Note:
            If the judge returns `-1` for any prompt, it indicates that the inner process used to compute the
            preference has failed. For instance, this could occur if the underlying language model returned an invalid
            answer. In such cases, the caller should handle these invalid indices appropriately, possibly by
            implementing fallback logic or error handling.
        """
        raise NotImplementedError("Judge subclasses must implement the `judge` method.")


class BaseBinaryJudge(BaseJudge):
    """
    Base class for binary judges.
    """

    @abstractmethod
    def judge(
        self,
        prompts: list[str],
        completions: list[str],
        gold_completions: Optional[list[str]] = None,
        shuffle_order: bool = True,
    ) -> list[int]:
        """
        Judge the completion for a given prompt. Used to assess if a completion satisfies a constraint.

        This base class should be used to implement binary evaluations as done in section 4.1.4 of the [CGPO
        paper](https://huggingface.co/papers/2409.20370). It is relevant for assessing whether a prompt completion pair
        satisfies a specific contraint.

        Args:
            prompts (`list[str]`): List of prompts.
            completions (`list[str]`): List of completions.
            gold_completions (`list[str]`, `optional`): List of gold completions if it exists.
            shuffle_order (`bool`): Whether to shuffle the order of the completions to avoid positional bias.

        Returns:
            list[int]: A list of binary labels:
                - 1 indicates that the completion satisfies the evaluated constraint.
                - 0 indicates that the completion does not satisfy the evaluated constraint.

        Note:
            If the judge returns -1 for any prompt, it indicates that the inner process used to compute the preference
            has failed. For instance, this could occur if the underlying language model or rule based contraint
            returned an invalid answer. In such cases, the caller should handle these invalid indices appropriately,
            possibly by implementing fallback logic or error handling.
        """
        raise NotImplementedError("Judge subclasses must implement the `judge` method.")

import hashlib, json, re
from datetime import datetime
from typing import Dict

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
        json.dumps(result)  # 如果有问题会抛出异常
        return result
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")
    except Exception as e:
        raise ValueError(f"Error processing JSON string: {e}")
def extract_gt_retention(completion: str):
    """
    从模型输出中提取gt_retention字段值
    
    Args:
        completion: 模型的输出字符串
        
    Returns:
        gt_retention的值，如果解析失败返回None
    """
    
    json_data = None
    gt_retention, gt_retention_code, reason = -1, '', ''
    
    try:
        # 尝试从文本中提取JSON部分
        # 查找可能的JSON格式内容
        json_pattern = r'\{[^{}]*"gt_retention"[^{}]*\}'
        matches = re.findall(json_pattern, completion, re.DOTALL)
        for match in matches:
            try:
                json_data = json.loads(match)
                # if 'gt_retention' in json_data:
                #     return json_data['gt_retention']
                if json_data is not None and 'gt_retention' in json_data:
                    break
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    
    # 第一种解析方法
    try:
        if json_data is None or 'gt_retention' not in json_data or 'gt_retention_code' not in json_data or 'reason' not in json_data:
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
            json_data = sanitize_json_string(completion)
    except Exception as e:
        print(f'[Error-3] {datetime.now()} fail to convert completion to json_data "{json_data}": {e}')
    
    # 第四种 json 解析方法
    try:
        if json_data is None or 'gt_retention' not in json_data or 'gt_retention_code' not in json_data or 'reason' not in json_data:
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
        print(f"extract_gt_retention() exception: {e} completion={[completion]}")

    if gt_retention==-1:
        print(f'[Error-5] {datetime.now()} finally fail to convert completion "{[completion]}" to json_data="{json_data}"')
    else:
        print(f'extract_gt_retention(): extract successfully, gt_retention={gt_retention}')
    return gt_retention, gt_retention_code, reason
    
class CustomBinaryJudge(BaseBinaryJudge):
    """
    自定义的二进制评判器，根据prompt的SHA-1哈希值查找真值集并比较模型输出结果
    """
    
    def __init__(self, ground_truth_dict: Dict[str, Dict[str, any]]):
        """
        初始化评判器
        
        Args:
            ground_truth_dict: 真值集字典，key为prompt的SHA-1哈希值，value为期望的原始 item 字典
        """
        self.ground_truth_dict = ground_truth_dict
    
    def judge(
        self,
        prompts: list[str],
        completions: list[str],
        gold_completions: Optional[list[str]] = None,
        shuffle_order: bool = True,
    ) -> list[int]:
        """
        评判模型输出结果
        
        Args:
            prompts: 输入的prompt列表
            completions: 模型输出的completion列表
            gold_completions: 黄金标准completion列表（本实现中不使用）
            shuffle_order: 是否打乱顺序（本实现中不使用）
            
        Returns:
            评判结果列表，1表示正确，0表示错误（包括解析错误），-1表示评判失败
        """
        results = []
        for prompt, completion in zip(prompts, completions):
            try:
                # 1. 将prompt转换为SHA-1哈希值
                if prompt.startswith('User: '):
                    prompt=prompt[6:]
                if prompt.endswith('\n\n\n'):
                    prompt=prompt[:-2]
                prompt_hash = get_prompt_hash(prompt)
                
                # 2. 查找真值集中的期望值
                expected_value = self.ground_truth_dict.get(prompt_hash)
                
                if expected_value is None:
                    # 如果真值集中没有对应的key，返回-1表示无法评判
                    results.append(-1)
                    print(f'[Error] {datetime.now()} judge(): prompt_hash={prompt_hash} not in ground_truth_dict, prompts={prompts}')
                    continue
                
                # 3. 从模型输出中提取gt_retention值
                actual_value, _, _ = extract_gt_retention(completion)
                print(f'read actual_value={actual_value}, expected_value={expected_value}')
                
                # 4. 比较期望值和实际值
                if expected_value == actual_value:
                    results.append(1)  # 正确
                else:
                    results.append(0)  # 错误
                    
            except Exception as e:
                # 处理任何其他异常
                print(f"评判过程中出现异常: {e}")
                results.append(-1)

        return results

class CustomPairRMJudge(BasePairwiseJudge):
    """
    This judge uses the PairRM model to rank pairs of completions for given prompts. It's designed for pairwise
    comparison of language model outputs. 
    **Example**:
    ```python
    >>> pairrm_judge = PairRMJudge()
    >>> prompts = ["Translate 'hello' to French", "What's the capital of Japan?"]
    >>> completions = [["Bonjour", "Salut"], ["Kyoto", "Tokyo"]]
    >>> results = pairrm_judge.judge(prompts, completions)
    >>> print(results)  # [0, 1] (indicating the first completion is preferred for the first prompt)
    ```
    """

    def __init__(self, ground_truth_dict: Dict[str, Dict[str, any]]):
        self.ground_truth_dict = ground_truth_dict

    def judge(
        self,
        prompts: list[str],
        completions: list[list[str]],
        shuffle_order: bool = False,
    ) -> list[Union[int, float]]:
        """
        Judge the completion pairs for the given prompts using the PairRM model.

        Args:
            prompts (`list[str]`):
                List of prompts to judge.
            completions (`list[list[str]]`):
                List of completion pairs for each prompt.
            shuffle_order (`bool`, *optional*, defaults to `True`):
                Whether to shuffle the order of the completions to avoid positional bias.
            return_scores (`bool`, *optional*, defaults to `False`):
                If `True`, return probability scores of the first completion instead of ranks (i.e. a *soft-judge*).
        Returns:
            `Union[list[int, float]]`:
                If `return_scores` is `False`, returns a list of ranks (`0` or `1`) for each prompt, indicating which
                completion is preferred. If `return_scores` is `True`, returns softmax probabilities for the first
                completion.
        Raises:
            `ValueError`:
                If the number of completions per prompt is not exactly 2.
        """

        if len(completions[0]) != 2:
            raise ValueError("PairRM judge requires exactly 2 completions per prompt.")

        # Shuffle the order of the completions to avoid positional bias
        if shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(prompts))
            completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions)]

        # Rank the completions
        ranks = []
        for prompt, (c1,c2) in zip(prompts, completions):
            try:
                # 1. 将prompt转换为SHA-1哈希值
                if prompt.startswith('User: '):
                    prompt=prompt[6:]
                if prompt.endswith('\n\n\n'):
                    prompt=prompt[:-2]
                prompt_hash = get_prompt_hash(prompt)
                
                # 2. 从模型输出中提取gt_retention值
                g1, co1, r1  = extract_gt_retention(c1) # 状态码，推荐代码，原因
                g2, co2, r2 = extract_gt_retention(c2)
                
                # 3. 查找真值集中的期望值
                gt = self.ground_truth_dict.get(prompt_hash)
                # 真值读取失败
                if gt is None or 'gt_retention' not in gt or gt['gt_retention'] not in [0,1,2] or 'gt_retention_code' not in gt or 'reason' not in gt:
                    if g1==-1 and g2 in [0,1,2]: # 按流畅度评判
                        ranks.append(1)
                    elif g1 in [0,1,2] and g2==-1:
                        ranks.append(0)
                    else: # 真值集中没有对应的key，返回-1表示无法评判
                        ranks.append(-1)
                    print(f'[Error] {datetime.now()} judge(): read prompt_hash={prompt_hash} failed in ground_truth_dict, gt={gt}, g1={g1}, g2={g2}, prompts={prompts}')
                    continue
                
                gt_retention, gt_retention_code, gt_reason = gt['gt_retention'], gt['gt_retention_code'], gt['reason']
                if gt_retention==1 and gt_retention_code=='':
                    gt_retention_code = gt['retention_code']
                print(f'read actual_value_1={g1}, actual_value_2={g2}, expected_value={gt_retention}')
                
                # 4. 比较期望值和实际值
                if gt_retention==g1 and g1==g2: # 都是 0/1/2
                    if gt_retention==1:
                        if co1==gt_retention_code:
                            if co2!=gt_retention_code:
                                print(f'co1==gt_retention_code, co2!=gt_retention_code, judge 0')
                                ranks.append(0)
                            elif len(r1)>len(r2):
                                print(f'co1==gt_retention_code, co2==gt_retention_code, len(r1)>len(r2), judge 0')
                                ranks.append(0)
                            else:
                                print(f'co1==gt_retention_code, co2==gt_retention_code, len(r1)<=len(r2), judge 1')
                                ranks.append(1)
                        elif co2==gt_retention_code:
                            print(f'co2==gt_retention_code, co1!=gt_retention_code, judge 1')
                            ranks.append(1)
                        elif len(r1)>len(r2):
                            print(f'gt_retention_code!=co1!=co2, len(r1)>len(r2), judge 0')
                            ranks.append(0)
                        else:
                            print(f'gt_retention_code!=co1!=co2, len(r1)<=len(r2), judge 1')
                            ranks.append(1)
                    elif gt_retention==0: # 这里开始简略了
                        if co1==gt_retention_code:
                            ranks.append(1)
                        elif co2==gt_retention_code:
                            ranks.append(0)
                        elif len(r1)>len(r2):
                            ranks.append(0)
                        else:
                            ranks.append(1)
                    elif gt_retention==2:
                        if co1=='':
                            ranks.append(0)
                        elif co2=='':
                            ranks.append(1)
                        elif len(r1)>len(r2):
                            ranks.append(0)
                        else:
                            ranks.append(1)
                    else:
                        ranks.append(-1) # 不可能出现
                        
                elif gt_retention!=g1 and g1==g2: # 犯了相同的错误
                    print(f'Error: 犯了相同的错误，应该回退，gt_retention={gt_retention}, g1={g1}, g2={g2}')
                    tmp_dict = {
                        'gt_retention': gt_retention,
                        'gt_retention_code': gt_retention_code,
                        'reason': gt_reason
                    }
                    ranks.append(json.dumps(tmp_dict, ensure_ascii=False))
                
                elif gt_retention==g1 or g2==-1 or (gt_retention==0 and g1==2 and g2==1) or (gt_retention==1 and g1==2 and g2==0) or (gt_retention==2 and g1==1 and g2==0):
                    ranks.append(0)
                elif gt_retention==g2 or g1==-1 or (gt_retention==0 and g1==1 and g2==2) or (gt_retention==1 and g1==0 and g2==2) or (gt_retention==2 and g1==0 and g2==1):
                    ranks.append(1)
                else:
                    print(f'Error: 不可能出现，gt_retention={gt_retention}, g1={g1}, g2={g2}')
                    ranks.append(-1)
            except Exception as e:
                # 处理任何其他异常
                print(f"评判过程中出现异常: {e}")
                ranks.append(-1)
            
        
        # Flip back the ranks or scores to the original order if needed
        if shuffle_order:
            ranks = [ranks[i] if not flip else 1 - ranks[i] for i, flip in enumerate(flip_mask)]
        
        return ranks
        
class PairRMJudge(BasePairwiseJudge):
    """
    LLM judge based on the PairRM model from AllenAI.

    This judge uses the PairRM model to rank pairs of completions for given prompts. It's designed for pairwise
    comparison of language model outputs. The PairRM model is loaded using the llm-blender library and runs on the
    default Accelerator device.

    **Attributes**:

        blender (`llm_blender.Blender`):
            An instance of the Blender class from llm-blender.

    **Example**:
    ```python
    >>> pairrm_judge = PairRMJudge()
    >>> prompts = ["Translate 'hello' to French", "What's the capital of Japan?"]
    >>> completions = [["Bonjour", "Salut"], ["Kyoto", "Tokyo"]]
    >>> results = pairrm_judge.judge(prompts, completions)
    >>> print(results)  # [0, 1] (indicating the first completion is preferred for the first prompt and the second)
    ```

    <Tip>

    This class requires the llm-blender library to be installed. Install it with: `pip install llm-blender`.

    </Tip>
    """

    def __init__(self):
        if not is_llm_blender_available():
            raise ValueError("llm-blender is not installed. Please install it with `pip install llm-blender`.")
        self.blender = llm_blender.Blender()
        self.blender.loadranker("llm-blender/PairRM", device=Accelerator().device)

    def judge(
        self,
        prompts: list[str],
        completions: list[list[str]],
        shuffle_order: bool = True,
        return_scores: bool = False,
        temperature: float = 1.0,
    ) -> list[Union[int, float]]:
        """
        Judge the completion pairs for the given prompts using the PairRM model.

        Args:
            prompts (`list[str]`):
                List of prompts to judge.
            completions (`list[list[str]]`):
                List of completion pairs for each prompt.
            shuffle_order (`bool`, *optional*, defaults to `True`):
                Whether to shuffle the order of the completions to avoid positional bias.
            return_scores (`bool`, *optional*, defaults to `False`):
                If `True`, return probability scores of the first completion instead of ranks (i.e. a *soft-judge*).
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature for scaling logits if `return_scores` is True.

        Returns:
            `Union[list[int, float]]`:
                If `return_scores` is `False`, returns a list of ranks (`0` or `1`) for each prompt, indicating which
                completion is preferred. If `return_scores` is `True`, returns softmax probabilities for the first
                completion.

        Raises:
            `ValueError`:
                If the number of completions per prompt is not exactly 2.

        Note:
            Unlike llm-blender, ranks are 0-indexed (`0` means the first completion is preferred).
        """

        if len(completions[0]) != 2:
            raise ValueError("PairRM judge requires exactly 2 completions per prompt.")

        # Shuffle the order of the completions to avoid positional bias
        if shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(prompts))
            completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions)]

        # Rank the completions
        ranks = self.blender.rank(prompts, completions, return_scores=return_scores, disable_tqdm=True)
        if not return_scores:
            ranks -= 1  # PairRM rank is 1-indexed, so we subtract 1 to make it 0-indexed
        else:
            # scale the logits by temperature
            ranks /= temperature

        # Flip back the ranks or scores to the original order if needed
        if shuffle_order:
            ranks[flip_mask] = ranks[flip_mask][:, ::-1]

        # Return the ranks or score probability
        if return_scores:
            logit_max = np.amax(ranks, axis=-1, keepdims=True)
            exp_logit_shifted = np.exp(ranks - logit_max)
            probs = exp_logit_shifted / np.sum(exp_logit_shifted, axis=-1, keepdims=True)
            return probs[:, 0].tolist()
        else:
            return ranks[:, 0].tolist()


class HfPairwiseJudge(BasePairwiseJudge):
    """
    Pairwise judge based on the Hugging Face API with chat completion.

    This judge is relevant for assessing the quality chat models, where the completion is a response to a given prompt.

    Args:
        model (`str`, *optional*, defaults to `"meta-llama/Meta-Llama-3-70B-Instruct"`):
            Model to use for the judge.
        token (`str`, *optional*):
            Hugging Face API token to use for the [`huggingface_hub.InferenceClient`].
        system_prompt (`str` or `None`, *optional*, defaults to `None`):
            The system prompt to be used for the judge. If not provided, a default prompt is used. Note that the system
            prompt should contain the following placeholders: `{prompt}`, `{response0}`, and `{response1}`. Also, the
            inference is called with `max_tokens=1`, consequently the system prompt should ask for a single token
            response.
    """

    def __init__(
        self,
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        token: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.client = InferenceClient(model=model, token=token)
        self.system_prompt = system_prompt or DEFAULT_PAIRWISE_SYSTEM_PROMPT

    def judge(self, prompts: list[str], completions: list[list[str]], shuffle_order: bool = True) -> list[int]:
        # Shuffle the order of the completions to avoid positional bias
        if shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(prompts))
            completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions)]

        # Define a function to get the rank for a single prompt, will be called concurrently
        def get_rank(prompt, candidates):
            content = self.system_prompt.format(prompt=prompt, response0=candidates[0], response1=candidates[1])
            completion = self.client.chat_completion(messages=[{"role": "user", "content": content}], max_tokens=1)
            response = completion.choices[0].message.content
            if response in ["0", "1"]:
                return int(response)
            else:
                logging.debug(f"Invalid response from the judge model: '{response}'. Returning -1.")
                return -1

        # Call the completions concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            ranks = list(executor.map(get_rank, prompts, completions))

        # Flip back the ranks to the original order if needed
        if shuffle_order:
            ranks = [ranks[i] if not flip else 1 - ranks[i] for i, flip in enumerate(flip_mask)]

        # Return the ranks
        return ranks


class OpenAIPairwiseJudge(BasePairwiseJudge):
    """
    Judge based on the OpenAI API.

    This judge is relevant for assessing the quality chat models, where the completion is a response to a given prompt.

    Args:
        model (`str`, *optional*, defaults to `"gpt-4-turbo-preview"`):
            Model to use for the judge.
        system_prompt (`str` or `None`, *optional*, defaults to `None`):
            System prompt to be used for the judge. If not provided, a default prompt is used. Note that the system
            prompt should contain the following placeholders: `{prompt}`, `{response0}`, and `{response1}`. Also, the
            inference is called with `max_tokens=1`, consequently the system prompt should ask for a single token
            response.
        max_requests (`int` or `None`, *optional*, defaults to `1000`):
            Maximum number of requests to make to the OpenAI API. If set to `None`, there is no limit.
    """

    def __init__(
        self, model="gpt-4-turbo-preview", system_prompt: Optional[str] = None, max_requests: Union[int, None] = 1_000
    ):
        if not is_openai_available():
            raise ValueError("OpenAI client is not installed. Please install it with 'pip install openai'.")
        self.client = OpenAI()
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_PAIRWISE_SYSTEM_PROMPT
        self.max_requests = max_requests
        self.num_requests = 0
        self._warned = False

    def judge(self, prompts: list[str], completions: list[list[str]], shuffle_order: bool = True) -> list[int]:
        # Check if the limit of requests is reached, if so, use random choice instead
        if self.max_requests is not None and self.num_requests >= self.max_requests:
            if not self._warned:  # Print the warning only once
                logging.warning(
                    f"Reached the maximum number of requests ({self.max_requests}). From now on, returning -1 instead. "
                    " To increase the limit, set `max_requests` to a higher value, or to `None` for no limit."
                )
                self._warned = True
            return [-1] * len(prompts)

        # Shuffle the order of the completions to avoid positional bias
        if shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(prompts))
            completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions)]

        # Define a function to get the rank for a single prompt, will be called concurrently
        def get_rank(prompt, candidates):
            content = self.system_prompt.format(prompt=prompt, response0=candidates[0], response1=candidates[1])
            messages = [{"role": "user", "content": content}]
            completion = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=1)
            response = completion.choices[0].message.content
            if response in ["0", "1"]:
                return int(response)
            else:
                logging.debug(f"Invalid response from the judge model: '{response}'. Returning -1.")
                return -1

        # Call the completions concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            ranks = list(executor.map(get_rank, prompts, completions))

        # Flip back the ranks to the original order if needed
        if shuffle_order:
            ranks = [ranks[i] if not flip else 1 - ranks[i] for i, flip in enumerate(flip_mask)]

        # Update the number of requests
        self.num_requests += len(prompts)

        # Return the ranks
        return ranks


class AllTrueJudge(BaseBinaryJudge):
    """
    Unify the decision of multiple [`BaseBinaryJudge`] instances.

    Returns `1` only if all inner binary judges return `1`. If any judge returns `0`, it returns `0`. If any judge
    returns `-1`, indicating a failure in its process, this judge will also return `-1`.

    Implements the Mixture of Judges as described in the [CGPO paper](https://huggingface.co/papers/2409.20370).

    Args:
    judges (`list[BaseBinaryJudge]`): A list of [`BaseBinaryJudge`] instances whose decisions will be unified.
    """

    def __init__(self, judges: list[BaseBinaryJudge]):
        self.judges = judges

    def judge(
        self,
        prompts: list[str],
        completions: list[str],
        gold_completions: Optional[list[str]] = None,
        shuffle_order: bool = True,
    ) -> list[int]:
        all_binary_judgments = [
            judge.judge(prompts, completions, gold_completions, shuffle_order) for judge in self.judges
        ]
        output = []
        for binary_judgments in zip(*all_binary_judgments):
            # Check that all values are in {0, 1, -1}
            if any(binary_judgment not in {0, 1, -1} for binary_judgment in binary_judgments):
                raise ValueError(
                    f"Invalid binary judgment: {binary_judgments}, expected list of values in {{0, 1, -1}}."
                )

            # Unify the decision
            if -1 in binary_judgments:
                output.append(-1)
            elif all(binary_judgment == 1 for binary_judgment in binary_judgments):
                output.append(1)
            else:
                output.append(0)
        return output

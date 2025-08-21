from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B")

# 使用简单的模板
# simple_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
# tokenizer.chat_template = simple_template

tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"

# 确保有必要的特殊 token
if "<|im_start|>" not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>"]
    })

# 保存
tokenizer.save_pretrained("/home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-2")
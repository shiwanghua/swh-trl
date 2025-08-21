"""
2025.8.3
2025.8.4
4.53.3
0.21.0.dev0
__UNSLOTH_VERSIONING__
"""
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from trl.trainer.online_dpo_trainer import (Any, AutoModelForCausalLM, BaseImageProcessor, BasePairwiseJudge, Callable, CustomBinaryJudge, CustomPairRMJudge, DPODataCollatorWithPadding, DataCollator, DataLoader, Dataset, EvalPrediction, F, FeatureExtractionMixin, GenerationConfig, IterableDataset, LLM, OnlineDPOConfig, OnlineDPOTrainer, OptimizerNames, Optional, Path, PeftModel, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, SIMPLE_CHAT_TEMPLATE, SamplingParams, Trainer, TrainerCallback, Union, apply_chat_template, create_reference_model, datasets, disable_dropout_in_model, empty_cache, generate_model_card, get_comet_experiment_url, get_reward, is_conversational, is_peft_available, is_wandb_available, jinja2, logging, maybe_apply_chat_template, nn, os, prepare_deepspeed, seed_worker, textwrap, time, torch, truncate_right, unwrap_model_for_generation, version, wandb, warnings, wraps, F, is_conversational, os, torch, F, Optional, PeftModel, PreTrainedModel, Trainer, is_peft_available, os, torch)


import os
from typing import *
from dataclasses import dataclass, field
from packaging.version import Version
import torch
import numpy as np
from contextlib import nullcontext
from torch.nn import functional as F
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling as TransformersDataCollatorForLanguageModeling

torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : False,
    "shape_padding"     : True,
    "trace.enabled"     : False,
    "triton.cudagraphs" : False,
}

@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def chunked_selective_log_softmax(logits, index):
    # Split into 4 chunks only
    chunked_logits = torch.chunk(logits.reshape(-1, logits.shape[-1]), chunks = 4, dim = 0)
    chunked_index  = torch.chunk(index.reshape(-1), chunks = 4, dim = 0)
    all_per_token_logps = []
    # Below loop does the same as selective_log_softmax(chunk_logits, chunk_index)
    for chunk_logits, chunk_index in zip(chunked_logits, chunked_index):
        chunk_logits = chunk_logits.to(torch.float32)
        selected_logits = torch.gather(chunk_logits, dim = -1, index = chunk_index.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.logsumexp(chunk_logits, dim = -1)
        per_token_logps = selected_logits - logsumexp_values
        all_per_token_logps.append(per_token_logps)
    pass
    all_per_token_logps = torch.concat(all_per_token_logps)
    all_per_token_logps = all_per_token_logps.reshape((logits.shape[0], logits.shape[1]))
    return all_per_token_logps
def vLLMSamplingParams(**kwargs):
    from vllm import SamplingParams
    sampling_params = SamplingParams(**kwargs)
    sampling_params._set_kwargs = kwargs
    return sampling_params
@dataclass
class UnslothOnlineDPOConfig(OnlineDPOConfig):
    """
    
    Configuration class for the [`OnlineDPOTrainer`].

    This class includes only the parameters that are specific to Online DPO training. For a full list of training
    arguments, please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this
    class may differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        reward_model_path (`str` or `None`, *optional*, defaults to `None`):
            Path to the reward model. Either `judge` or `reward_model_path` must be set, but not both.
        judge (`str` or `None`, *optional*, defaults to `None`):
            Name of the judge to use. Either `judge` or `reward_model_path` must be set, but not both.
        max_new_tokens (`int`, *optional*, defaults to `64`):
            Maximum number of tokens to generate per completion.
        max_length (`int`, *optional*, defaults to `256`):
            Maximum total length of the sequence (prompt + completion) used to compute log probabilities. If the
            sequence exceeds this limit, the leftmost tokens will be truncated to preserve as much of the completion as
            possible.
        temperature (`float`, *optional*, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        missing_eos_penalty (`float` or `None`, *optional*, defaults to `None`):
            Penalty applied to the score when the model fails to generate an EOS token. This is useful to encourage to
            generate completions shorter than the maximum length (`max_new_tokens`). The penalty must be a positive
            value.
        beta (`float` or `list[float]`, *optional*, defaults to `0.1`):
            Parameter controlling the deviation from the reference model. Higher β means less deviation from the
            reference model. For the IPO loss (`loss_type="ipo"`), β is the regularization parameter denoted by τ in
            the [paper](https://huggingface.co/papers/2310.12036). If a list of floats is provided then the β is
            selected for each new epoch and the last β is used for the rest of the epochs.
        loss_type (`str`, *optional*, defaults to `"sigmoid"`):
            Type of loss to use. Possible values are:

                - `"sigmoid"`: sigmoid loss from the original [DPO](https://huggingface.co/papers/2305.18290) paper.
                - `"ipo"`: IPO loss from the [IPO](https://huggingface.co/papers/2310.12036) paper.

        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model and reference model.
        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions. Requires vLLM to be installed (`pip install vllm`).
        gpu_memory_utilization (`float`, *optional*, defaults to `0.55`):
            The vLLM memory utilization. The default value is 0.55.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation.
        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a
            string.
    
    """
    vllm_sampling_params: Optional[Any] = field(
        default = None,
        metadata = {'help': 'vLLM SamplingParams'},
    )
    unsloth_num_chunks : Optional[int] = field(
        default = -1,
        metadata = {'help': 'Chunk size to reduce memory usage. -1 is most efficient.'},
    )
    def __init__(
        self,
        output_dir = None,
        overwrite_output_dir = None,
        do_train = False,
        do_eval = False,
        do_predict = False,
        eval_strategy = 'no',
        prediction_loss_only = False,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
        per_gpu_train_batch_size = None,
        per_gpu_eval_batch_size = None,
        gradient_accumulation_steps = 2,
        eval_accumulation_steps = 2,
        eval_delay = 0,
        torch_empty_cache_steps = 250,
        learning_rate = 5e-05,
        weight_decay = 0.01,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-08,
        max_grad_norm = 1.0,
        num_train_epochs = 3.0,
        max_steps = -1,
        lr_scheduler_type = 'linear',
        warmup_ratio = 0.1,
        warmup_steps = 0,
        log_level = 'passive',
        log_level_replica = 'warning',
        log_on_each_node = True,
        logging_dir = None,
        logging_strategy = 'steps',
        logging_first_step = False,
        logging_steps = 1,
        logging_nan_inf_filter = False,
        save_strategy = 'steps',
        save_steps = 500,
        save_total_limit = None,
        save_safetensors = True,
        save_on_each_node = False,
        save_only_model = False,
        restore_callback_states_from_checkpoint = False,
        no_cuda = False,
        use_cpu = False,
        use_mps_device = False,
        seed = 3407,
        data_seed = 3407,
        jit_mode_eval = False,
        use_ipex = False,
        bf16 = False,
        fp16 = False,
        fp16_opt_level = 'O1',
        half_precision_backend = 'auto',
        bf16_full_eval = False,
        fp16_full_eval = False,
        tf32 = None,
        local_rank = -1,
        ddp_backend = None,
        tpu_num_cores = None,
        tpu_metrics_debug = False,
        debug = '',
        dataloader_drop_last = False,
        eval_steps = None,
        dataloader_num_workers = 0,
        dataloader_prefetch_factor = None,
        past_index = -1,
        run_name = None,
        disable_tqdm = None,
        remove_unused_columns = True,
        label_names = None,
        load_best_model_at_end = False,
        metric_for_best_model = None,
        greater_is_better = None,
        ignore_data_skip = False,
        fsdp = '',
        fsdp_min_num_params = 0,
        fsdp_config = None,
        fsdp_transformer_layer_cls_to_wrap = None,
        accelerator_config = None,
        deepspeed = None,
        label_smoothing_factor = 0.0,
        optim = 'adamw_8bit',
        optim_args = None,
        adafactor = False,
        group_by_length = False,
        length_column_name = 'length',
        report_to = None,
        ddp_find_unused_parameters = None,
        ddp_bucket_cap_mb = None,
        ddp_broadcast_buffers = None,
        dataloader_pin_memory = True,
        dataloader_persistent_workers = False,
        skip_memory_metrics = True,
        use_legacy_prediction_loop = False,
        push_to_hub = False,
        resume_from_checkpoint = None,
        hub_model_id = None,
        hub_strategy = 'every_save',
        hub_token = None,
        hub_private_repo = None,
        hub_always_push = False,
        hub_revision = None,
        gradient_checkpointing = False,
        gradient_checkpointing_kwargs = None,
        include_inputs_for_metrics = False,
        eval_do_concat_batches = True,
        fp16_backend = 'auto',
        push_to_hub_model_id = None,
        push_to_hub_organization = None,
        push_to_hub_token = None,
        mp_parameters = '',
        auto_find_batch_size = True,
        full_determinism = False,
        torchdynamo = None,
        ray_scope = 'last',
        ddp_timeout = 1800,
        torch_compile = False,
        torch_compile_backend = None,
        torch_compile_mode = None,
        include_tokens_per_second = False,
        include_num_input_tokens_seen = False,
        neftune_noise_alpha = None,
        optim_target_modules = None,
        batch_eval_metrics = False,
        eval_on_start = False,
        use_liger_kernel = False,
        liger_kernel_config = None,
        eval_use_gather_object = False,
        average_tokens_across_devices = False,
        reward_model_path = None,
        judge = None,
        max_new_tokens = 64,
        max_length = 512,
        temperature = 0.9,
        missing_eos_penalty = None,
        loss_type = 'sigmoid',
        dataset_num_proc = None,
        disable_dropout = True,
        use_vllm = False,
        gpu_memory_utilization = 0.55,
        ds3_gather_for_generation = True,
        model_init_kwargs = None,
        vllm_sampling_params = None,
        unsloth_num_chunks = -1,
        **kwargs,
    ):
        if learning_rate < 1e-7: raise FloatingPointError(f'Unsloth: Your learning rate of `{learning_rate}` is too small and less than 1e-7! Consider increasing it, otherwise gradient updates will be close to 0!')
        if learning_rate > 1: raise OverflowError(f'Unsloth: Your learning rate of `{learning_rate}` is way too larger > 1! Consider decreasing it to 1e-1, otherwise gradient updates will explode!')
        if output_dir is None and save_strategy == 'steps' and save_steps == 500:
            output_dir = 'unsloth_training_checkpoints'
            save_strategy = 'no'
        if dataset_num_proc is None:
            from multiprocessing import cpu_count
            dataset_num_proc = min(cpu_count()*2, 2)
        if temperature <= 0:
            raise MathError('Unsloth: Please set a positive non-zero temperature since your results will be wrong.')
        elif temperature >= 10:
            raise MathError('Unsloth: Please set a positive non-zero temperature less than 10, since sampling will be quite erratic.')
        
        
        super().__init__(
            output_dir = output_dir,
            overwrite_output_dir = overwrite_output_dir,
            do_train = do_train,
            do_eval = do_eval,
            do_predict = do_predict,
            eval_strategy = eval_strategy,
            prediction_loss_only = prediction_loss_only,
            per_device_train_batch_size = per_device_train_batch_size,
            per_device_eval_batch_size = per_device_eval_batch_size,
            per_gpu_train_batch_size = per_gpu_train_batch_size,
            per_gpu_eval_batch_size = per_gpu_eval_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            eval_accumulation_steps = eval_accumulation_steps,
            eval_delay = eval_delay,
            torch_empty_cache_steps = torch_empty_cache_steps,
            learning_rate = learning_rate,
            weight_decay = weight_decay,
            adam_beta1 = adam_beta1,
            adam_beta2 = adam_beta2,
            adam_epsilon = adam_epsilon,
            max_grad_norm = max_grad_norm,
            num_train_epochs = num_train_epochs,
            max_steps = max_steps,
            lr_scheduler_type = lr_scheduler_type,
            warmup_ratio = warmup_ratio,
            warmup_steps = warmup_steps,
            log_level = log_level,
            log_level_replica = log_level_replica,
            log_on_each_node = log_on_each_node,
            logging_dir = logging_dir,
            logging_strategy = logging_strategy,
            logging_first_step = logging_first_step,
            logging_steps = logging_steps,
            logging_nan_inf_filter = logging_nan_inf_filter,
            save_strategy = save_strategy,
            save_steps = save_steps,
            save_total_limit = save_total_limit,
            save_safetensors = save_safetensors,
            save_on_each_node = save_on_each_node,
            save_only_model = save_only_model,
            restore_callback_states_from_checkpoint = restore_callback_states_from_checkpoint,
            no_cuda = no_cuda,
            use_cpu = use_cpu,
            use_mps_device = use_mps_device,
            seed = seed,
            data_seed = data_seed,
            jit_mode_eval = jit_mode_eval,
            use_ipex = use_ipex,
            bf16 = bf16,
            fp16 = fp16,
            fp16_opt_level = fp16_opt_level,
            half_precision_backend = half_precision_backend,
            bf16_full_eval = bf16_full_eval,
            fp16_full_eval = fp16_full_eval,
            tf32 = tf32,
            local_rank = local_rank,
            ddp_backend = ddp_backend,
            tpu_num_cores = tpu_num_cores,
            tpu_metrics_debug = tpu_metrics_debug,
            debug = debug,
            dataloader_drop_last = dataloader_drop_last,
            eval_steps = eval_steps,
            dataloader_num_workers = dataloader_num_workers,
            dataloader_prefetch_factor = dataloader_prefetch_factor,
            past_index = past_index,
            run_name = run_name,
            disable_tqdm = disable_tqdm,
            remove_unused_columns = remove_unused_columns,
            label_names = label_names,
            load_best_model_at_end = load_best_model_at_end,
            metric_for_best_model = metric_for_best_model,
            greater_is_better = greater_is_better,
            ignore_data_skip = ignore_data_skip,
            fsdp = fsdp,
            fsdp_min_num_params = fsdp_min_num_params,
            fsdp_config = fsdp_config,
            fsdp_transformer_layer_cls_to_wrap = fsdp_transformer_layer_cls_to_wrap,
            accelerator_config = accelerator_config,
            deepspeed = deepspeed,
            label_smoothing_factor = label_smoothing_factor,
            optim = optim,
            optim_args = optim_args,
            adafactor = adafactor,
            group_by_length = group_by_length,
            length_column_name = length_column_name,
            report_to = report_to,
            ddp_find_unused_parameters = ddp_find_unused_parameters,
            ddp_bucket_cap_mb = ddp_bucket_cap_mb,
            ddp_broadcast_buffers = ddp_broadcast_buffers,
            dataloader_pin_memory = dataloader_pin_memory,
            dataloader_persistent_workers = dataloader_persistent_workers,
            skip_memory_metrics = skip_memory_metrics,
            use_legacy_prediction_loop = use_legacy_prediction_loop,
            push_to_hub = push_to_hub,
            resume_from_checkpoint = resume_from_checkpoint,
            hub_model_id = hub_model_id,
            hub_strategy = hub_strategy,
            hub_token = hub_token,
            hub_private_repo = hub_private_repo,
            hub_always_push = hub_always_push,
            hub_revision = hub_revision,
            gradient_checkpointing = gradient_checkpointing,
            gradient_checkpointing_kwargs = gradient_checkpointing_kwargs,
            include_inputs_for_metrics = include_inputs_for_metrics,
            eval_do_concat_batches = eval_do_concat_batches,
            fp16_backend = fp16_backend,
            push_to_hub_model_id = push_to_hub_model_id,
            push_to_hub_organization = push_to_hub_organization,
            push_to_hub_token = push_to_hub_token,
            mp_parameters = mp_parameters,
            auto_find_batch_size = auto_find_batch_size,
            full_determinism = full_determinism,
            torchdynamo = torchdynamo,
            ray_scope = ray_scope,
            ddp_timeout = ddp_timeout,
            torch_compile = torch_compile,
            torch_compile_backend = torch_compile_backend,
            torch_compile_mode = torch_compile_mode,
            include_tokens_per_second = include_tokens_per_second,
            include_num_input_tokens_seen = include_num_input_tokens_seen,
            neftune_noise_alpha = neftune_noise_alpha,
            optim_target_modules = optim_target_modules,
            batch_eval_metrics = batch_eval_metrics,
            eval_on_start = eval_on_start,
            use_liger_kernel = use_liger_kernel,
            liger_kernel_config = liger_kernel_config,
            eval_use_gather_object = eval_use_gather_object,
            average_tokens_across_devices = average_tokens_across_devices,
            reward_model_path = reward_model_path,
            judge = judge,
            max_new_tokens = max_new_tokens,
            max_length = max_length,
            temperature = temperature,
            missing_eos_penalty = missing_eos_penalty,
            loss_type = loss_type,
            dataset_num_proc = dataset_num_proc,
            disable_dropout = disable_dropout,
            use_vllm = use_vllm,
            gpu_memory_utilization = gpu_memory_utilization,
            ds3_gather_for_generation = ds3_gather_for_generation,
            model_init_kwargs = model_init_kwargs,**kwargs)
        self.vllm_sampling_params = vllm_sampling_params
        self.unsloth_num_chunks = unsloth_num_chunks
pass

class _UnslothOnlineDPOTrainer(Trainer):
    r""""""

    _tag_names = ["trl", "online-dpo"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str],
        ref_model: Union[PreTrainedModel, nn.Module, None] = None,
        val_file_path: str = '/home/ubisec/swh/codes/AssessModel/data/confirm_mark_completion_data_20250609/confirm_mark_c_cpp_completion_detail_20250609.json',
        reward_model: Union[PreTrainedModel, nn.Module, None] = None,
        judge: Optional[BasePairwiseJudge] = None,
        args: Optional[OnlineDPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset], "datasets.Dataset"]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        reward_processing_class: Optional[PreTrainedTokenizerBase] = None,
        peft_config: Optional[dict] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:

        if hasattr(model, 'vllm_engine') and hasattr(args, 'use_vllm'):
            if (getattr(args, 'use_vllm', False) == False):
                args.use_vllm = True
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, either omit the `ref_model` argument or pass `None`."
            )

        self.ref_model = ref_model
        print(f'peft_config={peft_config}, ref_model={ref_model}')
        if reward_model is not None and judge is not None:
            warnings.warn(
                "Both `reward_model` and `judge` are provided. Please choose provide only one of them. "
                "Ignoring `judge` and using `reward_model`.",
                UserWarning,
            )
            judge = None
        elif reward_model is None and judge is None:
            raise ValueError("Either `reward_model` or `judge` must be provided.")

        import json,hashlib
        self.validate_data = dict()
        it = 0
        for item in json.load(open(val_file_path, 'r', encoding='utf-8'))['RECORDS']:
            if item['confirm']==1 and item['gt_retention']!=3:
                # self.validate_data.add[item['code_id']+str[len[item['head']]*100000000+len[item['tail']]*10000+len[item['retention_code']]*100+len[item['reason']]]]
                lang = item['lang']
                head = item['head']
                retention_code = item['retention_code']
                tail = item['tail']
                clipboard_info = item['clipboard_info'][:4096] if item['clipboard_info'] is not None else ''
                code_diff = item['code_diff']
                code_file_path = item['file_path']
                prompt = self.construct_msg(lang, head, retention_code, tail, clipboard_info, code_diff, code_file_path)
                it+=1
                hash_prompt = hashlib.sha1(prompt.encode('utf-8')).hexdigest()
                prompt = [prompt]
                # if it==443:
                #     print[f'it={it}, hash_prompt={hash_prompt}, prompt={prompt}']
                self.validate_data[hash_prompt] = item
        print(f'[INFO] read {len(self.validate_data)} confirm data for validation.')
        self.reward_model = reward_model
        self.reward_processing_class = reward_processing_class
        self.judge = judge
        if judge=='one-answer-custom':
            self.judge_name = 'one-answer-custom'
            self.judge = CustomBinaryJudge(self.validate_data)
        elif judge=='two-answer-custom':
            self.judge_name = 'two-answer-custom'
            self.judge = CustomPairRMJudge(self.validate_data)
        self.is_encoder_decoder = model.config.is_encoder_decoder

        if args.missing_eos_penalty is not None and judge is not None:
            raise ValueError("`missing_eos_penalty` is not supported when `judge` is provided.")

        if args is None:
            raise ValueError("`args` must be provided.")
        
        # Check that the processing_class is provided
        if processing_class is None:
            raise ValueError("`processing_class` must be provided.")

        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model

            # Handle torch_dtype in model_init_kwargs
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass
            elif isinstance(torch_dtype, str):
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `OnlineDPOConfig`. Expected either 'auto' or a string "
                    f"representing a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )

            model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `OnlineDPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )
        self.is_encoder_decoder = model.config.is_encoder_decoder

        # Convert to PEFT model if peft_config is provided
        if False:
            # Check if PEFT is available
            if not is_peft_available():
                raise ImportError(
                    "PEFT is not available and passed `peft_config`. Please install PEFT with "
                    "`pip install peft` to use it."
                )

            # If the model is already a PeftModel, we need to merge and unload it.
            # Further information here: https://huggingface.co/docs/trl/dpo_trainer#reference-model-considerations-with-peft
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            # Get peft model with the given config
            model = model

        # Disable dropout in the model and reference model
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Handle the ref_model
        # Usually, the user wants the ref model to be the initial version of the model. When using PEFT, it's easy to
        # get the ref model, as it's just the model with a disabled adapter. When not using PEFT, we need to create
        # the ref model from the model by copying it and disable the gradients and set it in evaluation mode.
        if ref_model is None:  # No ref model provided, the most common case
            if False:
                self.ref_model = create_reference_model(model)  # copy, disable gradients, set eval mode
            else:
                self.ref_model = None  # we don't need a ref model here, we can just disable the adapter.
        else:  # rare case, the user provided a ref model
            self.ref_model = ref_model
            self.ref_model.eval()

        # Disable the gradient and set the reward model in eval mode
        if self.reward_model is not None:
            self.reward_model.eval()

        # Define the collator is not provided
        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(pad_token_id=processing_class.pad_token_id)

        self.max_length = args.max_length

        self.stats = {
            "objective/kl": [],
            "objective/entropy": [],
            "objective/non_score_reward": [],
            "rewards/chosen": [],
            "rewards/rejected": [],
            "rewards/accuracies": [],
            "rewards/margins": [],
            "logps/chosen": [],
            "logps/rejected": [],
            "val/contain_eos_token": [],
            "beta": [],
        }
        if self.reward_model is not None:
            self.stats["objective/rlhf_reward"] = []
            self.stats["objective/scores_margin"] = []
            self.stats["objective/scores"] = []

        if args.use_vllm:
            self.llm = model.vllm_engine; self._last_loaded_step = 0; self.generation_config = SamplingParams(
                n=n_generation,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=0.95,
                top_k=20,
                skip_special_tokens=False,
                detokenize=False,
                **getattr(getattr(args, 'vllm_sampling_params', vLLMSamplingParams()), '_set_kwargs', {}),
            )
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                # top_k=50,
                # top_p=1.0,
                do_sample=True,
                top_p=0.95,
                top_k=20,
                skip_special_tokens=False,
                use_cache=False if args.gradient_checkpointing else True,
            )

        # The trainer estimates the number of FLOPs [floating-point operations] using the number of elements in the
        # input tensor associated with the key "input_ids". However, in Online DPO, the sampled data does not include
        # the "input_ids" key. As a result, the trainer issues the warning: "Could not estimate the number of tokens
        # of the input, floating-point operations will not be computed." To suppress this warning, we set the
        # "estimate_tokens" key in the model's "warnings_issued" dictionary to True. This acts as a flag to indicate
        # that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self._train_batch_size = 1
        print(f'init(): force set self._train_batch_size to {self._train_batch_size}')
        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        self._beta = args.beta

        # Placed after the super[].__init__ because we need self.is_deepspeed_enabled and self.accelerator
        if self.is_deepspeed_enabled:
            if self.reward_model is not None:
                self.reward_model = prepare_deepspeed(
                    self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
            if self.ref_model is not None:
                self.ref_model = prepare_deepspeed(
                    self.ref_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
        else:
            # print[f'self.accelerator.device={self.accelerator.device}']
            if self.ref_model is not None:
                self.ref_model = self.ref_model.to(self.accelerator.device)
            if self.reward_model is not None:
                self.reward_model = self.reward_model.to(self.accelerator.device)
        
    def construct_msg(self, lang, head, retention_code, tail, clipboard_info, code_diff, code_file_path):
        return f"""你是一个编程代码评价工程师，专门检查实时补全的代码是否正确，识别出那些错误补全的情况以方便后续改进代码补全模型。
        
根据以下<context>标签之间的信息，判断其中补全的 {lang} 代码是否正确，<context> 标签内字段的每个值前后都加上 <begin> 和 <end> 标签，避免不知道起止边界。
        
<context>
上文: <begin>{head}<end>
        
补全代码: <begin>{retention_code}<end>
        
下文: <begin>{tail}<end>
        
剪切板信息: <begin>{clipboard_info}<end>
        
diff信息: <begin>{code_diff}<end>
        
代码文件路径: <begin>{code_file_path}<end>
<context>
        
只输出一个 json 格式的结果，不带其他内容，json 里只包含三个字段，即 "gt_retention", "gt_retention_code" 和 "reason"，赋值规则如下：

（1）如果补全的代码是错误的，则 "gt_retention" 的值设为 0，此时需要生成正确的补全代码，赋值给 "gt_retention_code"，并在 "reason" 的值里用中文给出错误原因和正确的补全逻辑。

（2）如果补全的代码是正确的，则 "gt_retention" 的值设为 1，"gt_retention_code" 可以给出更准确完整的补全代码，"reason" 里给出补全正确的原因，注意只要续写的代码逻辑合理，暂时存在一些语法错误、编译错误、无意义冗余现象等不完整代码导致的缺陷是可以容忍的，此时也算补全正确。

（3）如果根据所提供信息无法判断补全代码的正确性，有争议性，或者上下文代码量小且逻辑简单，作为一个补全案例没有什么参考价值，不适合作为代码补全模型的训练数据，则丢弃该样本，将 "gt_retention" 的值设为 2，"gt_retention_code" 设为空字符串，"reason" 里给出无法判断的原因。

（4）可能根据现有信息判断出待补全代码和业务逻辑相关，意图不明，此时应该主动放弃进行代码补全，如果补全代码非空，则判断为补全错误，"gt_retention" 应该赋为 0，"gt_retention_code" 为空字符串，"reason" 里说明此时不适合进行代码补全的原因。
"""

    @property
    def beta(self):
        if isinstance(self._beta, list):
            epoch = self.state.epoch
            return self._beta[epoch] if epoch < len(self._beta) else self._beta[-1]
        else:
            return self._beta

    @staticmethod
    def tokenize_row(feature, is_encoder_decoder: bool, tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
        """Tokenize a single row from a DPO specific dataset."""
        if not is_encoder_decoder:
            batch = tokenizer(feature["prompt"], add_special_tokens=False)
            # Add BOS token to head of prompt. Avoid adding if it's already there
            if tokenizer.bos_token_id is not None:
                prompt_len_input_ids = len(batch["input_ids"])
                if prompt_len_input_ids == 0 or tokenizer.bos_token_id != batch["input_ids"][0]:
                    batch["input_ids"] = [tokenizer.bos_token_id] + batch["input_ids"]
                    batch["attention_mask"] = [1] + batch["attention_mask"]
        else:
            batch = tokenizer(feature["prompt"], add_special_tokens=True)
        batch = {f"prompt_{key}": value for key, value in batch.items()}
        return batch

    # Same as Trainer.get_train_dataloader but skip the "remove_unused_columns".
    @wraps(Trainer.get_train_dataloader)
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    # Same as Trainer.get_eval_dataloader but skip the "remove_unused_columns".
    @wraps(Trainer.get_eval_dataloader)
    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)

    def _generate_vllm(self, model, prompts):
        eos_token_id = self.processing_class.eos_token_id
        pad_token_id = self.processing_class.pad_token_id

        # Load the latest weights

        pass

        pass

        if is_conversational({"prompt": prompts[0]}):
            outputs = self.llm.chat(prompts, self.generation_config, use_tqdm=False, lora_request = self.model.load_lora('online_dpo_trainer_lora_model', load_tensors = True))
        else:
            outputs = self.llm.generate(prompts, self.generation_config, use_tqdm=False, lora_request = self.model.load_lora('online_dpo_trainer_lora_model', load_tensors = True))

        completion_ids = [list(output.outputs[i].token_ids) for i in range(2) for output in outputs]
        prompt_ids = [list(output.prompt_token_ids) for _ in range(2) for output in outputs]

        # Create mask and pad the prompt and completion
        max_prompt_length = max(len(ids) for ids in prompt_ids)
        prompt_mask = [[0] * (max_prompt_length - len(ids)) + [1] * len(ids) for ids in prompt_ids]
        prompt_ids = [[pad_token_id] * (max_prompt_length - len(ids)) + ids for ids in prompt_ids]
        max_tokens = self.generation_config.max_tokens
        completion_mask = [[1] * len(ids) + [0] * (max_tokens - len(ids)) for ids in completion_ids]
        completion_ids = [
            ids + [eos_token_id] if ids[-1] != eos_token_id and len(ids) < max_tokens else ids
            for ids in completion_ids
        ]
        completion_ids = [ids + [pad_token_id] * (max_tokens - len(ids)) for ids in completion_ids]

        # Convert to tensors
        prompt_ids = torch.tensor(prompt_ids, device=self.accelerator.device)
        prompt_mask = torch.tensor(prompt_mask, device=self.accelerator.device)
        completion_ids = torch.tensor(completion_ids, device=self.accelerator.device)
        completion_mask = torch.tensor(completion_mask, device=self.accelerator.device)

        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def _generate(self, model, prompts):
        t1=time.time()
        eos_token_id = self.processing_class.eos_token_id
        pad_token_id = self.processing_class.pad_token_id

        # Apply chat template and tokenize the input. We do this on-the-fly to enable the use of reward models and
        # policies with different tokenizers / chat templates.
        inputs = [{"prompt": prompt} for prompt in prompts]
        inputs = [maybe_apply_chat_template(x, self.processing_class) for x in inputs]
        inputs = [self.tokenize_row(x, self.is_encoder_decoder, self.processing_class) for x in inputs]
        inputs = self.data_collator(inputs)

        # Sample 2 completions per prompt of size `max_new_tokens` from the model
        inputs = self._prepare_inputs(inputs)
        prompt_ids = inputs["prompt_input_ids"]
        prompt_mask = inputs["prompt_attention_mask"]
        if self.judge_name != 'one-answer-custom':
            prompt_ids = inputs["prompt_input_ids"].repeat(2, 1)
            prompt_mask = inputs["prompt_attention_mask"].repeat(2, 1)
        print(f'self.generation_config={self.generation_config}, prompt_ids.shape={prompt_ids.shape}')
        with unwrap_model_for_generation(
            model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            output = unwrapped_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=self.generation_config,
            )

        completion_ids = output[:, prompt_ids.size(1) :]
        completion_ids, completion_mask = truncate_right(completion_ids, eos_token_id, pad_token_id)
        print(f'_generate cost {time.time()-t1}s', flush=True)
        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def old_forward(self, model, prompt_ids, prompt_mask, completion_ids, completion_mask):
        # Get the number of tokens to truncate from prompt
        num_tokens_to_truncate = max(prompt_ids.size(1) + completion_ids.size(1) - self.max_length, 0)
        print(f'num_tokens_to_truncate={num_tokens_to_truncate}, prompt_ids.size(1)={prompt_ids.size(1)}, completion_ids.size(1)={completion_ids.size(1)}, self.max_length={self.max_length}')
        
        # Truncate left to avoid oom
        prompt_ids = prompt_ids[:, num_tokens_to_truncate:]
        prompt_mask = prompt_mask[:, num_tokens_to_truncate:]
        
        # 如果prompt被完全截断，进一步截断completion_ids
        if prompt_ids.size(1) == 0:
            if completion_ids.size(1) > self.max_length :
                completion_ids = completion_ids[:, -self.max_length:]
                completion_mask = completion_mask[:, -self.max_length:]
                print(f"old_forward(): 截断completion_ids到长度: {self.max_length }")
                
        # Concat the prompt and completion
        prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        prompt_completion_mask = torch.cat((prompt_mask, completion_mask), dim=1)

        # Get the logprobs of the completions from the model
        print(f'old_forward: prompt_completion_ids={prompt_completion_ids}')
        print(f'old_forward: prompt_completion_ids.shape={prompt_completion_ids.shape}')
        print(f'old_forward: prompt_completion_mask={prompt_completion_mask}')
        print(f'old_forward: prompt_completion_mask.shape={prompt_completion_mask.shape}')
        output = model(prompt_completion_ids, attention_mask=prompt_completion_mask)
        print(f'old_forward: output={output}')
        # outputs = []
        # for i in range(0, len(prompt_completion_ids), 1):
        #     batch_ids = prompt_completion_ids[i:i+1]
        #     batch_mask = prompt_completion_mask[i:i+1]
        #     with torch.no_grad():  # 如果不需要梯度
        #         batch_output = model(batch_ids, attention_mask=batch_mask)
        #         outputs.append(batch_output)
        # output = torch.cat(outputs, dim=0)
        print(f'prompt_completion_ids.shape={prompt_completion_ids.shape}, prompt_completion_mask.shape={prompt_completion_mask.shape}')
        print(f"output.logits.shape={output.logits.shape}")
        # There is 1 offset, because the model predict the next token
        logits = output.logits[:, prompt_ids.size(1) - 1 : -1]
        # 修复：确保切片索引有效
        prompt_length = prompt_ids.size(1)
        if prompt_length > 0:
            # 获取completion部分的logits
            logits = output.logits[:, prompt_length - 1 : -1]  # 原来只有这一行代码
        else:
            # 如果prompt被完全截断，直接使用所有logits
            print(f'prompt被完全截断, 直接使用所有logits, prompt_length={prompt_length}')
            logits = output.logits[:, 0:-1]
            
        # Take the completion tokens logprob
        logprobs = torch.take_along_dim(logits.log_softmax(dim=-1), completion_ids.unsqueeze(-1), dim=2).squeeze(-1)
        return logprobs
    
    # 会导致 backward 报错 element 0 of tensors does not require grad and does not have a grad_fn
    def _forward(self, model, prompt_ids, prompt_mask, completion_ids, completion_mask):
        """
        修复切片问题的版本，避免 batch 跑导致 OOM
        """
        def print_memory_usage(step_name):
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"{step_name}: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")
            
        # # 获取原始模型（如果是DataParallel包装的）
        # if hasattr(model, 'module'):
        #     original_model = model.module
        # else:
        #     original_model = model
        # # 启用梯度检查点
        # if hasattr(original_model, 'gradient_checkpointing_enable'):
        #     original_model.gradient_checkpointing_enable()
        #     print("已启用梯度检查点")
        # else:
        #     print("警告：模型不支持梯度检查点")
        
        batch_size = prompt_ids.size(0)
        all_logprobs = []
        print(f'_forward(): prompt_ids.shape={prompt_ids.shape}') # [2, 1537]
        for i in range(batch_size):
            # 获取单个样本
            single_prompt_ids = prompt_ids[i:i+1]
            single_prompt_mask = prompt_mask[i:i+1]
            single_completion_ids = completion_ids[i:i+1]
            single_completion_mask = completion_mask[i:i+1]
            
            # 计算截断
            num_tokens_to_truncate = max(single_prompt_ids.size(1) + single_completion_ids.size(1) - self.max_length, 0)
            print(f'num_tokens_to_truncate={num_tokens_to_truncate}, single_prompt_ids: {single_prompt_ids.shape}, single_prompt_mask: {single_prompt_mask.shape}; single_completion_ids: {single_completion_ids.shape},single_completion_mask: {single_completion_mask.shape}')
            
            # 截断prompt
            single_prompt_ids = single_prompt_ids[:, num_tokens_to_truncate:]
            single_prompt_mask = single_prompt_mask[:, num_tokens_to_truncate:]
            
            # 如果prompt被完全截断，进一步截断completion_ids
            if single_prompt_ids.size(1) == 0:
                if single_completion_ids.size(1) > self.max_length :
                    single_completion_ids = single_completion_ids[:, -self.max_length:]
                    single_completion_mask = single_completion_mask[:, -self.max_length:]
                    print(f"截断completion_ids到长度: {self.max_length }")
                    
            # 拼接
            prompt_completion_ids = torch.cat((single_prompt_ids, single_completion_ids), dim=1) # [1, 1235]
            prompt_completion_mask = torch.cat((single_prompt_mask, single_completion_mask), dim=1) # [1, 1235]
            print(f'拼接后 prompt_completion_ids.shape={prompt_completion_ids.shape}, prompt_completion_mask.shape={prompt_completion_mask.shape}')
            
            # 推理
            #with torch.no_grad():  # 关闭梯度会导致 backward 报错 element 0 of tensors does not require grad and does not have a grad_fn
            # with torch.cuda.amp.autocast():
            print(f'new_forward: prompt_completion_ids.shape={prompt_completion_ids.shape}, prompt_completion_ids={prompt_completion_ids}')
            print(f'new_forward: prompt_completion_mask.shape={prompt_completion_mask.shape}, prompt_completion_mask={prompt_completion_mask}')
            output = model(prompt_completion_ids, attention_mask=prompt_completion_mask)
            print_memory_usage(f"new_forward: output.logits.shape={output.logits.shape}, output.loss={output.loss}, past_key_values={output.past_key_values}, hidden_states={output.hidden_states}, 推理后-样本{i}")
            # 修复：确保切片索引有效
            prompt_length = single_prompt_ids.size(1)
            if prompt_length > 0:
                # 获取completion部分的logits
                logits = output.logits[:, prompt_length - 1 : -1]
            else:
                # 如果prompt被完全截断，直接使用所有logits
                logits = output.logits[:, 0:-1]
            # 确保logits和completion_ids的维度匹配
            if logits.size(1) != single_completion_ids.size(1):
                # 如果长度不匹配，截断到较短的长度
                min_length = min(logits.size(1), single_completion_ids.size(1))
                print(f'长度不匹配, logits.shape={logits.shape}, single_completion_ids.shape={single_completion_ids.shape}, min_length={min_length}')
                logits = logits[:, :min_length]
                single_completion_ids = single_completion_ids[:, :min_length]
            # 计算logprobs
            logprobs = torch.take_along_dim(
                logits.log_softmax(dim=-1), 
                single_completion_ids.unsqueeze(-1), 
                dim=2
            ).squeeze(-1)
            print(f'_forward(): logprobs.requires_grad={logprobs.requires_grad}, shape={logprobs.shape}')
            all_logprobs.append(logprobs)
            
            # 清理缓存
            # del output, logits
            # output.detach(), logits.detach()
            # torch.cuda.empty_cache()
            # # 可选：强制垃圾回收
            # import gc
            # gc.collect()
            # print_memory_usage(f"清理后-样本{i}")
        return torch.cat(all_logprobs, dim=0)

    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        model.train()

        prompts = inputs["prompt"][:1] # 强行把 batch_size 设为 1
        # batch_size = len(prompts)
        batch_size = 1
        print(f'[INFO] training_step(): batch_size={batch_size}, prompts_1={prompts}, num_items_in_batch={num_items_in_batch}', flush=True)

        if self.args.use_vllm:
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate_vllm(model, prompts)
        else:
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate(model, prompts)

        contain_eos_token = torch.any(completion_ids == self.processing_class.eos_token_id, dim=-1)

        if self.judge_name == 'one-answer-custom':
            logprobs = self.old_forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask)
        else:
            logprobs = self._forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask)
        with torch.no_grad():
            if self.ref_model is not None:
                ref_logprobs = self._forward(self.ref_model, prompt_ids, prompt_mask, completion_ids, completion_mask)
            else:  # peft case: we just need to disable the adapter
                with self.model.disable_adapter():
                    if self.judge_name == 'one-answer-custom':
                        ref_logprobs = self.old_forward(self.model, prompt_ids, prompt_mask, completion_ids, completion_mask)
                    else:
                        ref_logprobs = self._forward(self.model, prompt_ids, prompt_mask, completion_ids, completion_mask)
        print(f'logprobs={logprobs}, ref_logprobs={ref_logprobs}')
        
        # Decode the completions, and format them if the input is conversational
        device = logprobs.device
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational({"prompt": prompts[0]}):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
        print(f'[INFO] training_step(): completion_ids_len={len(completion_ids)}, completions_len={len(completions)}')
        
        # Get the reward from the reward model or judge
        if self.judge is not None:
            # Once formatted, conversational data may contain special tokens (such as <|im_start|>) that are not
            # directly understandable by the judge and could alter its judgment. To avoid this and make the judge
            # independent of the model's chat template, we use the raw conversation data, and apply our own chat
            # template to it.
            if is_conversational({"prompt": prompts[0]}):
                environment = jinja2.Environment()
                template = environment.from_string(SIMPLE_CHAT_TEMPLATE)
                prompts = [template.render(messages=prompt) for prompt in prompts]
                completions = [template.render(messages=completion) for completion in completions]
            
            content_tmp = []
            for completion in completions:
                if "</think>" in completion:
                    thinking, content = completion.rsplit("</think>", 1)
                    # thinking = thinking.replace("<|im_start|>think\n", "").strip()
                    content_tmp.append(content)
                else:
                    content_tmp.append(completion)
            completions = content_tmp
            print(f'completions_content={completions}')
            
            if self.judge_name == 'one-answer-custom':
                ranks_of_first_completion = self.judge.judge(prompts, completions)
            else:
                ranks_of_first_completion = self.judge.judge(
                    prompts, list(zip(completions[:batch_size], completions[batch_size:]))
                )
                if isinstance(ranks_of_first_completion[0], str): # 两个答案都是错误的，把第二个回答回退为人工标记的标注答案
                    correct_answer = ranks_of_first_completion[0]
                    print(f'两个答案都是错误的, 回退为人工标注的答案:{correct_answer}')
                    # completion_ids 是2维tensor, 参考 _generate() 把 correct_answer 作为第二个问题的回答 转换为 token id 赋值到 completion_ids[1]
                    
                    # 使用processing_class将正确回答转换为token ids
                    correct_tokens = self.processing_class(
                        correct_answer, 
                        add_special_tokens=False, 
                        return_tensors="pt"
                    )["input_ids"].squeeze(0).to(device)
                    
                    eos_token_id = self.processing_class.eos_token_id
                    
                    # 确保长度不超过max_new_tokens
                    target_token_num = len(completion_ids[1])
                    if len(correct_tokens) > target_token_num:
                        print(f'correct_tokens 长度 {len(correct_tokens)} 超过 target_token_num, 截断到 {target_token_num} 个 token')
                        correct_tokens = correct_tokens[:target_token_num-1]
                        correct_tokens = torch.cat([correct_tokens, torch.tensor([eos_token_id], device=device)])
                    
                    # 添加EOS token（如果还没有的话）
                    if correct_tokens[-1] != eos_token_id:
                        correct_tokens = torch.cat([correct_tokens, torch.tensor([eos_token_id], device=device)])
                    
                    # 填充到固定长度
                    useful_length = len(correct_tokens)
                    if len(correct_tokens) < target_token_num:
                        padding_length = target_token_num - len(correct_tokens)
                        correct_tokens = torch.cat([
                            correct_tokens, 
                            torch.full((padding_length,), self.processing_class.pad_token_id, device=device)
                        ])
                    
                    print(f'completion_ids.shape={completion_ids.shape}, correct_tokens.shape={correct_tokens.shape}, useful_length={useful_length}, target_token_num={target_token_num}')
                    # 更新completion_ids[1]和completion_mask[1]
                    completion_ids[1] = correct_tokens
                    
                    # 创建mask：前actual_length个位置为1，其余为0
                    completion_mask[1] = torch.cat([
                        torch.ones(useful_length, device=device),
                        torch.zeros(len(correct_tokens) - useful_length, device=device)
                    ])
                    
                    print(f'已更新completion_ids[1]为正确回答，长度: {len(correct_tokens)}, 开始重新计算 logits')
                    del logprobs, ref_logprobs
                    # output.detach(), logits.detach()
                    torch.cuda.empty_cache()
                    # import gc
                    # gc.collect()
                    logprobs = self._forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask)
                    with torch.no_grad():
                        with self.model.disable_adapter():
                            ref_logprobs = self._forward(self.model, prompt_ids, prompt_mask, completion_ids, completion_mask)
                    print(f'重新计算后：logprobs={logprobs}, ref_logprobs={ref_logprobs}')
                    
                    # 将ranks_of_first_completion[0]设置为1，表示第二个答案更好
                    ranks_of_first_completion[0] = 1
            print(f'ranks_of_first_completion={ranks_of_first_completion}')
            
            # convert ranks to a True/False mask:
            # when rank == 0, it means the first completion is the best
            # when rank == 1, it means the second completion is the best
            mask = torch.tensor([rank == 0 for rank in ranks_of_first_completion], device=device)
        else:
            # The reward model may not have the same chat template or tokenizer as the model, so we need to use the
            # raw data (string), apply the chat template (if needed), and tokenize it with the reward processing class.
            prompts = 2 * prompts  # repeat the prompt: [prompt0, prompt1] -> [prompt0, prompt1, prompt0, prompt1]
            if is_conversational({"prompt": prompts[0]}):
                examples = [{"prompt": p, "completion": c} for p, c in zip(prompts, completions)]
                examples = [apply_chat_template(example, self.reward_processing_class) for example in examples]
                prompts = [example["prompt"] for example in examples]
                completions = [example["completion"] for example in examples]

            # Tokenize the prompts
            prompts_ids = self.reward_processing_class(
                prompts, padding=True, return_tensors="pt", padding_side="left"
            )["input_ids"].to(device)
            context_length = prompts_ids.shape[1]

            # Tokenize the completions
            completions_ids = self.reward_processing_class(
                completions, padding=True, return_tensors="pt", padding_side="right"
            )["input_ids"].to(device)

            # Concatenate the prompts and completions and get the reward
            prompt_completion_ids = torch.cat((prompts_ids, completions_ids), dim=1)
            with torch.inference_mode():
                _, scores, _ = get_reward(
                    self.reward_model, prompt_completion_ids, self.reward_processing_class.pad_token_id, context_length
                )

                # Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty

            # Split the scores in 2 (the prompts of the first half are the same as the second half)
            first_half, second_half = scores.split(batch_size)

            # Get the indices of the chosen and rejected examples
            mask = first_half >= second_half

        if self.judge_name=='one-answer-custom':
            """
            处理单个样本的loss计算
            """
            # 在loss计算开始前添加
            print(f"logprobs.requires_grad: {logprobs.requires_grad}")
            print(f"logprobs.grad_fn: {logprobs.grad_fn}")
            print(f"ref_logprobs.requires_grad: {ref_logprobs.requires_grad}")
            
            # 当只有一个样本时，无法进行对比学习, 可以改为计算其他类型的loss
            padding_mask = (~completion_mask.bool())[:, -self.max_length:]
            logprobs_sum = (logprobs * ~padding_mask).sum(1)
            ref_logprobs_sum = (ref_logprobs * ~padding_mask).sum(1) # 计算单个样本的负对数似然
            logits = logprobs_sum - ref_logprobs_sum
            print(f"logits.requires_grad: {logits.requires_grad}")
            if self.args.loss_type == "single_sample":
                if mask.item():  # 如果样本被接受
                    losses = -logits  # 鼓励模型输出接近参考模型
                else:  # 如果样本被拒绝
                    losses = logits   # 惩罚模型输出偏离参考模型
            elif self.args.loss_type == "sigmoid":
                # 对于单个样本，使用简单的sigmoid loss
                if mask.item():  # 如果样本被接受
                    losses = -F.logsigmoid(self.beta * logits)  # 鼓励高概率
                else:  # 如果样本被拒绝
                    losses = -F.logsigmoid(-self.beta * logits)  # 鼓励低概率
            elif self.args.loss_type == "ipo":
                # 对于单个样本，使用IPO loss的变体
                if mask.item():  # 如果样本被接受
                    losses = (logits - 1 / (2 * self.beta)) ** 2  # 鼓励logits接近1/(2*beta)
                else:  # 如果样本被拒绝
                    losses = (logits + 1 / (2 * self.beta)) ** 2  # 鼓励logits接近-1/(2*beta)
            print(f"losses.requires_grad: {losses.requires_grad}")
            loss = losses.mean()
            print(f'losses={losses}, loss={loss}, self.beta={self.beta}, loss_type={self.args.loss_type}')
            print(f"loss.requires_grad: {loss.requires_grad}")
            print(f"loss.grad_fn: {loss.grad_fn}")
        else:
            batch_range = torch.arange(batch_size, device=device)
            # 如果是 True 说明选左边，不需要 +batch_size, 如果是 False 说明选右边，需要 +batch_size
            chosen_indices = batch_range + (~mask * batch_size)
            rejected_indices = batch_range + (mask * batch_size)
            print(f'mask={mask}, batch_range={batch_range}, chosen_indices={chosen_indices}, rejected_indices={rejected_indices}')

            # Build tensor so that the first half is the chosen examples and the second half the rejected examples
            cr_indices = torch.cat((chosen_indices, rejected_indices), dim=0)  # cr = chosen and rejected
            print(f'cr_indices={cr_indices}, logprobs={logprobs.shape}, ref_logprobs={ref_logprobs.shape}, completion_mask={completion_mask.shape}')
            cr_logprobs = logprobs[cr_indices]
            cr_ref_logprobs = ref_logprobs[cr_indices]

            # mask out the padding tokens
            padding_mask = ~completion_mask.bool()
            cr_padding_mask = padding_mask[cr_indices][:, -self.max_length:]

            cr_logprobs_sum = (cr_logprobs * ~cr_padding_mask).sum(1)
            cr_ref_logprobs_sum = (cr_ref_logprobs * ~cr_padding_mask).sum(1)
            print(f'padding_mask={padding_mask}, padding_mask.shape={padding_mask.shape}, \ncr_padding_mask={cr_padding_mask}, cr_padding_mask.shape={cr_padding_mask.shape}\ncr_logprobs_sum={cr_logprobs_sum}, cr_logprobs_sum.shape={cr_logprobs_sum.shape}, \ncr_ref_logprobs_sum={cr_ref_logprobs_sum}')
            # Split the chosen and rejected examples
            chosen_logprobs_sum, rejected_logprobs_sum = torch.split(cr_logprobs_sum, batch_size)
            chosen_ref_logprobs_sum, rejected_ref_logprobs_sum = torch.split(cr_ref_logprobs_sum, batch_size)
            pi_logratios = chosen_logprobs_sum - rejected_logprobs_sum
            ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum
            print(f'chosen_logprobs_sum={chosen_logprobs_sum}, rejected_logprobs_sum={rejected_logprobs_sum}, chosen_ref_logprobs_sum={chosen_ref_logprobs_sum}, rejected_ref_logprobs_sum={rejected_ref_logprobs_sum}')
            logits = pi_logratios - ref_logratios
            print(f'pi_logratios={pi_logratios}, ref_logratios={ref_logratios}, logits={logits}, self.beta={self.beta}')
            if self.args.loss_type == "sigmoid":
                losses = -F.logsigmoid(self.beta * logits)
            elif self.args.loss_type == "ipo":
                losses = (logits - 1 / (2 * self.beta)) ** 2
            else:
                raise NotImplementedError(f"invalid loss type {self.loss_type}") 

            loss = losses.mean()
            print(f'losses={losses}, loss={loss}, self.beta={self.beta}, loss_type={self.args.loss_type}')
            
            # Log everything
            if self.reward_model is not None:
                scores_margin = scores[chosen_indices] - scores[rejected_indices]
                self.stats["objective/scores_margin"].append(
                    self.accelerator.gather_for_metrics(scores_margin.mean()).mean().item()
                )
                self.stats["objective/scores"].append(self.accelerator.gather_for_metrics(scores.mean()).mean().item())
            self.stats["val/contain_eos_token"].append(contain_eos_token.float().mean().item())
            self.stats["logps/chosen"].append(self.accelerator.gather_for_metrics(chosen_logprobs_sum).mean().item())
            self.stats["logps/rejected"].append(self.accelerator.gather_for_metrics(rejected_logprobs_sum).mean().item())

            kl = logprobs - ref_logprobs
            mean_kl = kl.sum(1).mean()
            self.stats["objective/kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
            non_score_reward = (-self.beta * kl).sum(1)
            mean_non_score_reward = non_score_reward.mean()
            self.stats["objective/non_score_reward"].append(
                self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
            )
            if self.reward_model is not None:
                rlhf_reward = scores + non_score_reward
                self.stats["objective/rlhf_reward"].append(self.accelerator.gather_for_metrics(rlhf_reward).mean().item())
            mean_entropy = -logprobs.sum(1).mean()
            self.stats["objective/entropy"].append(self.accelerator.gather_for_metrics(mean_entropy).mean().item())
            chosen_rewards = self.beta * (chosen_logprobs_sum - chosen_ref_logprobs_sum)
            gathered_chosen_rewards = self.accelerator.gather_for_metrics(chosen_rewards)
            self.stats["rewards/chosen"].append(gathered_chosen_rewards.mean().item())
            rejected_rewards = self.beta * (rejected_logprobs_sum - rejected_ref_logprobs_sum)
            gathered_rejected_rewards = self.accelerator.gather_for_metrics(rejected_rewards)
            self.stats["rewards/rejected"].append(gathered_rejected_rewards.mean().item())
            margin = gathered_chosen_rewards - gathered_rejected_rewards
            self.stats["rewards/margins"].append(margin.mean().item())
            accuracy = margin > 0
            self.stats["rewards/accuracies"].append(accuracy.float().mean().item())
            self.stats["beta"].append(self.beta)

        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learning rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps

    # Same as Trainer._maybe_log_save_evaluate but log our metrics
    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            # Add our metrics
            for key, val in self.stats.items():
                logs[key] = sum(val) / max(len(val),1)
                if len(val)==0:
                    print(f'_maybe_log_save_evaluate(): key={key}, val={val}')
            self.stats = {key: [] for key in self.stats}  # reset stats

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == "best":
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            print(f'call _save_checkpoint(): trial={trial}')
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        tmp_quantization_config = None
        if hasattr(self.processing_class, 'init_kwargs') and 'quantization_config' in self.processing_class.init_kwargs:
            tmp_quantization_config = self.processing_class.init_kwargs['quantization_config']
            print(f'delete quantization_config from processing_class.init_kwargs, tmp_quantization_config={tmp_quantization_config}')
            del self.processing_class.init_kwargs['quantization_config']
            print(f'after delete, tmp_quantization_config={tmp_quantization_config}')
        super()._save_checkpoint(model, trial)
        if tmp_quantization_config:
            print("set quantization_config to self.processing_class.init_kwargs['quantization_config']")
            self.processing_class.init_kwargs['quantization_config'] = tmp_quantization_config

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        # normalize `tags` to a mutable set
        if tags is None:
            tags = set()
        elif isinstance(tags, str):
            tags = {tags}
        else:
            tags = set(tags)

        if hasattr(self.model.config, "unsloth_version"):
            tags.add("unsloth")

        tags.update(self._tag_names)

        citation = textwrap.dedent("""\
        @article{guo2024direct,
            title        = {{Direct Language Model Alignment from Online AI Feedback}},
            author       = {Shangmin Guo and Biao Zhang and Tianlin Liu and Tianqi Liu and Misha Khalman and Felipe Llinares and Alexandre Ram{\'{e}} and Thomas Mesnard and Yao Zhao and Bilal Piot and Johan Ferret and Mathieu Blondel},
            year         = 2024,
            eprint       = {arXiv:2402.04792}
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.url if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="Online DPO",
            trainer_citation=citation,
            paper_title="Direct Language Model Alignment from Online AI Feedback",
            paper_id="2402.04792",
        )
        model_card.save(os.path.join(self.args.output_dir, "README.md"))
class UnslothOnlineDPOTrainer(_UnslothOnlineDPOTrainer):
    """
    
    Initialize OnlineDPOTrainer.

    Args:
        model (`Union[str, nn.Module, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        ref_model (`transformers.PreTrainedModel` or `torch.nn.Module` or `None`):
            The reference model to use for training. If None is specified, the reference model will be created from the
            model.
        val_file_path: 验证文件路径，取代 reward_model
        reward_model (`transformers.PreTrainedModel` or `torch.nn.Module` or `None`):
            The reward model to score completions with, preferably an `AutoModelForSequenceClassification`.
        judge (`BasePairwiseJudge`):
            The judge to use for pairwise comparison of model completions.
        args (`OnlineDPOConfig`):
            The online DPO config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator
            (`DPODataCollatorWithPadding`) will be used which will pad the sequences to the maximum length of the
            sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.BaseImageProcessor`], [`~transformers.FeatureExtractionMixin`] or [`~transformers.ProcessorMixin`], *optional*, defaults to `None`):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
        peft_config (`dict`):
            The peft config to use for training.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return a dictionary string to
            metric values.
        callbacks (`list[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
    
    """
    def __init__(
        self,
        model,
        ref_model = None,
        val_file_path = '/home/ubisec/swh/codes/AssessModel/data/confirm_mark_completion_data_20250609/confirm_mark_c_cpp_completion_detail_20250609.json',
        reward_model = None,
        judge = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        processing_class = None,
        reward_processing_class = None,
        peft_config = None,
        compute_metrics = None,
        callbacks = None,
        preprocess_logits_for_metrics = None,
        **kwargs
    ):
        if args is None: args = UnslothOnlineDPOConfig()
        use_bf16 = getattr(args, 'bf16', False)
        if type(use_bf16) is not bool: use_bf16 = False
        use_fp16 = getattr(args, 'fp16', False)
        if type(use_fp16) is not bool: use_fp16 = False
        force_float32 = False
        if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1':
            print('Unsloth: Switching to float32 training since model cannot work with float16')
            force_float32 = True
        mixed_precision_dtype = os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32')
        dtype = getattr(model.config, 'torch_dtype', None)
        if dtype is None: dtype = model.get_input_embeddings().dtype
        from unsloth_zoo.utils import _get_dtype
        dtype = _get_dtype(dtype)
        float16 = dtype == torch.float16
        if not force_float32 and (float16 and use_bf16): raise TypeError('Unsloth: Model is in float16 precision but you want to use bfloat16 precision. Set fp16 to `True` and bf16 to `False`')
        if not force_float32 and (not float16 and use_fp16): raise TypeError('Unsloth: Model is in bfloat16 precision but you want to use float16 precision. Set fp16 to `False` and bf16 to `True`')
        if force_float32:
            args.fp16 = False
            args.bf16 = False
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'
        elif (not use_bf16 and not use_fp16) and mixed_precision_dtype == 'float32':
            args.fp16 = float16
            args.bf16 = not float16
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'fp16' if float16 else 'bf16'
        if getattr(args, 'eval_dataset', None) is not None and getattr(args, 'eval_strategy', 'no') == 'no':
            args.eval_strategy = 'steps'
            if getattr(args, 'eval_steps', None) is None: args.eval_steps = 0.1
        ga_steps = getattr(args, 'gradient_accumulation_steps', None)
        if ga_steps is not None and ga_steps > 1:
            from transformers import __version__ as transformers_version
            if Version(transformers_version) <= Version('4.45.2'):
                print('**** Unsloth: Please use our fixed gradient_accumulation_steps by updating transformers, TRL and Unsloth!\n'
                      '`pip install --upgrade --no-cache-dir --force-reinstall --no-deps unsloth transformers trl unsloth_zoo`')
        if getattr(args, 'eval_strategy', 'no') != 'no':
            eval_bsz = getattr(args, 'per_device_eval_batch_size', 8)
            if eval_bsz == 8 and args.per_device_train_batch_size < eval_bsz: args.per_device_eval_batch_size = args.per_device_train_batch_size
            if getattr(args, 'eval_accumulation_steps', None) is None and ga_steps is not None: args.eval_accumulation_steps = ga_steps
        fp16_full_eval = getattr(args, 'fp16_full_eval', False)
        if type(fp16_full_eval) is not bool: fp16_full_eval = False
        bf16_full_eval = getattr(args, 'bf16_full_eval', False)
        if type(bf16_full_eval) is not bool: bf16_full_eval = False
        if args.fp16 and bf16_full_eval: args.bf16_full_eval = False; args.fp16_full_eval = True
        if args.bf16 and fp16_full_eval: args.bf16_full_eval = True; args.fp16_full_eval = False
        if force_float32:
            args.bf16_full_eval = False
            args.fp16_full_eval = False
        elif os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32') == 'bfloat16':
            args.bf16_full_eval = True
            args.fp16_full_eval = False
        elif not bf16_full_eval and not fp16_full_eval:
            args.bf16_full_eval = args.bf16
            args.fp16_full_eval = args.fp16
        _output_logits = False
        if locals().get('compute_metrics', None) is not None: _output_logits = True
        if locals().get('preprocess_logits_for_metrics', None) is not None: _output_logits = True
        if _output_logits:
            os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        if 'max_seq_length' not in locals() and not hasattr(args, 'max_seq_length'):
            pass
        else:
            model_max_seq_length = getattr(model, 'max_seq_length', None)
            args_max_seq_length  = getattr(args,  'max_seq_length', None)
            if args_max_seq_length is None and model_max_seq_length is not None:
                max_seq_length = model.max_seq_length
                if hasattr(args, 'max_seq_length'): args.max_seq_length = max_seq_length
        if model is not None and hasattr(model, 'for_training'):
            model.for_training()
        if 'tokenizer' in locals() and hasattr(tokenizer, 'padding_side'): tokenizer.padding_side = 'right'
        if 'processing_class' in locals():
            if hasattr(processing_class, 'padding_side'): processing_class.padding_side = 'right'
            if hasattr(processing_class, 'tokenizer') and hasattr(processing_class.tokenizer, 'padding_side'): processing_class.tokenizer.padding_side = 'right'
        __tokenizer = processing_class if 'processing_class' in locals() else tokenizer
        from unsloth_zoo.vision_utils import UnslothVisionDataCollator
        if not isinstance(data_collator, UnslothVisionDataCollator):
            if isinstance(data_collator, DataCollatorForSeq2Seq) and 'labels' not in train_dataset.column_names:
                data_collator = TransformersDataCollatorForLanguageModeling(__tokenizer, mlm = False, mlm_probability = 0.0)
            elif isinstance(data_collator, TransformersDataCollatorForLanguageModeling) and 'labels' in train_dataset.column_names:
                data_collator = DataCollatorForSeq2Seq(__tokenizer)
        else:
            if hasattr(args, 'remove_unused_columns'): args.remove_unused_columns = False
            if hasattr(args, 'dataset_text_field'): args.dataset_text_field = ''
            if hasattr(args, 'dataset_kwargs'): args.dataset_kwargs = {'skip_prepare_dataset': True}
        if not isinstance(data_collator, UnslothVisionDataCollator):
            if not hasattr(__tokenizer, 'pad') and hasattr(__tokenizer, 'tokenizer'):
                if isinstance(data_collator, DataCollatorForSeq2Seq):
                    data_collator = DataCollatorForSeq2Seq(__tokenizer.tokenizer)
                else:
                    data_collator = TransformersDataCollatorForLanguageModeling(__tokenizer.tokenizer, mlm = False, mlm_probability = 0.0)
        other_metrics = []
        
        from unsloth_zoo.logging_utils import PatchRLStatistics
        PatchRLStatistics('online_dpo_trainer', other_metrics)
        
        super().__init__(
            model = model,
            ref_model = ref_model,
            val_file_path = val_file_path,
            reward_model = reward_model,
            judge = judge,
            args = args,
            data_collator = data_collator,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            processing_class = processing_class,
            reward_processing_class = reward_processing_class,
            peft_config = peft_config,
            compute_metrics = compute_metrics,
            callbacks = callbacks,
            preprocess_logits_for_metrics = preprocess_logits_for_metrics,**kwargs)
        if hasattr(self, 'neftune_hook_handle'):
            self.neftune_hook_handle.remove()
            if hasattr(self, 'neftune_hook_handle'): del self.neftune_hook_handle
        if getattr(args, 'neftune_noise_alpha', None) is not None:
            model.get_input_embeddings().neftune_noise_alpha = self.neftune_noise_alpha
        pass
        
pass

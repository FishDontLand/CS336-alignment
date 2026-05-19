import json
import math
from typing import Callable, Literal

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from cs336_alignment.SFT import init_vllm, tokenize_prompt_and_output, get_response_log_probs, log_generations
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def compute_group_normalized_rewards(
        reward_fn: Callable[[str, str], dict[str, float]],
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
        group_size: int,
        advantage_eps: float,
        normalize_by_std: bool
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    rewards = torch.tensor([reward_fn(response, truth)['reward'] for response, truth in zip(rollout_responses, repeated_ground_truths)])
    rewards = rewards.view(-1, group_size)
    group_mean_rewards = rewards.mean(dim=1, keepdim=True)
    normalized_rewards = rewards - group_mean_rewards
    meta_data = {
        'mean': rewards.mean().item(),
        'std': rewards.std().item(),
        'min': rewards.min().item(),
        'max': rewards.max().item(),
    }

    if normalize_by_std:
        group_reward_std = rewards.std(dim=1, keepdim=True)
        normalized_rewards = normalized_rewards / (group_reward_std + advantage_eps)

    return (normalized_rewards.view(-1), rewards.view(-1), meta_data)

def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor
) -> torch.Tensor:
    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rn_derivative = torch.exp(policy_log_probs - old_log_probs)
    clipped_derivative = torch.where(rn_derivative > 1 + cliprange, 1 + cliprange, rn_derivative)
    clipped_derivative = torch.where(rn_derivative < 1 - cliprange, 1 - cliprange, clipped_derivative)

    adjust_advantage = rn_derivative * advantages
    clipped_advantage = clipped_derivative * advantages

    is_clipped = adjust_advantage > clipped_advantage

    loss = -torch.where(is_clipped, clipped_advantage, adjust_advantage)
    meta_data = {
        'is_clipped': is_clipped.detach().cpu()
    }
    return loss, meta_data


def compute_policy_gradient_loss(
        policy_log_probs: torch.Tensor,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, dict()
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, dict()
    else:
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)

def masked_mean(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        dim: int | None = None,
) -> torch.Tensor:
    all_avg = (tensor * mask).mean(dim=dim)
    mask_numeric = torch.where(mask, 1.0, 0.0)
    mask_ratio = mask_numeric.mean(dim=dim)
    return all_avg / mask_ratio

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, meata_data = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    )

    per_example_loss = masked_mean(loss, response_mask) / gradient_accumulation_steps
    total_loss = per_example_loss.mean()

    total_loss.backward()
    example_loss_cpu = per_example_loss.cpu()

    return total_loss.detach().cpu(), meata_data | {
        'max_example_loss': example_loss_cpu.max(),
        'min_example_loss': example_loss_cpu.min(),
        'example_loss_std': example_loss_cpu.std(),
    }

def run_grpo(n_grpo_steps: int = 200, learning_rate: float = 1e-5, advantage_eps: float = 1e-6,
             rollout_batch_size: int = 256, group_size: int = 8, sampling_temperature: float = 1.0,
             sampling_min_tokens: int = 4, sampling_max_tokens: int = 1024, epochs_per_rollout_batch: int = 1,
             mini_batch_size: int = 2, gpu_memory_utilization: float = 0.85,
             loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline",
             use_std_normalization: bool = True, cliprange: float=0.2, grad_clip: float | None=1.0, eval_freq: int=5):

    train_batch_size = rollout_batch_size // epochs_per_rollout_batch
    assert rollout_batch_size % train_batch_size == 0, (
        "rollout_batch_size must be divisible by train_batch_size"
    )
    grad_acc_steps = train_batch_size // mini_batch_size
    assert train_batch_size % grad_acc_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )

    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )

    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )

    n_questions = rollout_batch_size / group_size
    train_data = []
    with open('./data/train.jsonl', 'r') as f:
        for line in f:
            train_data.append(json.loads(line))

    valid_data = []
    with open('./data/valid.jsonl', 'r') as f:
        for line in f:
            valid_data.append(json.loads(line))

    np.random.seed(42)
    valid_data = np.random.choice(valid_data, size=2048, replace=False)
    valid_prompts = [e['question'] for e in valid_data]
    valid_answers = [e['answer'] for e in valid_data]

    model_path = '/workspace/cs336/hf_cache/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots'
    model_hash = '4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2'
    full_path = model_path + '/' + model_hash

    policy_model = AutoModelForCausalLM.from_pretrained(full_path).to('cuda:0')
    old_params = policy_model.state_dict()
    tokenizer = AutoTokenizer.from_pretrained(full_path)
    llm = init_vllm(full_path, device='cuda:1', seed=12432, gpu_memory_utilization=gpu_memory_utilization)
    print('llm loaded')

    sampling_params = SamplingParams(
        temperature=sampling_temperature, top_p=1.0, max_tokens=sampling_max_tokens, min_tokens=sampling_min_tokens, seed=67346,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=group_size
    )

    valid_sampling_params = SamplingParams(
        temperature=sampling_temperature, top_p=1.0, max_tokens=sampling_max_tokens, min_tokens=sampling_min_tokens, seed=123123,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    optimizer.zero_grad()


    total_update_steps = 0
    for step in range(n_grpo_steps):
        selections = np.random.choice(train_data, size=n_questions, replace=False)
        sample_questions = [e['question'] for e in selections]
        sample_answers = [e['answer'] for e in selections]

        llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(old_params.items())
        response = llm.generate(sample_questions, sampling_params)
        rollout_answers = []
        rollout_responses = []
        rollout_questions = []
        for idx, r in enumerate(response):
            rollout_responses += [o.text for o in r.outputs]
            rollout_answers += [sample_answers[idx]] * group_size
            rollout_questions += [sample_questions[idx]] * group_size

        group_normalized_rewards, raw_rewards, reward_meta_data = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            rollout_responses,
            rollout_answers,
            group_size,
            advantage_eps,
            use_std_normalization
        )

        tokenized_results_by_epoch = []
        old_log_probs = [] if loss_type == 'grpo_clip' else None
        current_params = policy_model.state_dict()
        policy_model.load_state_dict(old_params)

        for epoch in range(epochs_per_rollout_batch):
            train_questions = rollout_questions[epoch * train_batch_size:(epoch + 1) * train_batch_size]
            train_responses = rollout_responses[epoch * train_batch_size:(epoch + 1) * train_batch_size]

            tokenized_results = tokenize_prompt_and_output(
                train_questions,
                train_responses,
                tokenizer
            )

            tokenized_results_by_epoch.append(tokenized_results)
            with torch.inference_mode():
                if loss_type == 'grpo_clip':
                    old_log_probs.append([])
                    for mini_batch_step in range(grad_acc_steps):
                        input_ids = tokenized_results['input_ids'][
                            mini_batch_step * mini_batch_size: (mini_batch_step + 1) * mini_batch_size]
                        response_mask = tokenized_results['response_mask'][
                            mini_batch_step * mini_batch_size: (mini_batch_step + 1) * mini_batch_size]

                        mini_batch_log_probs = get_response_log_probs(
                            policy_model,
                            input_ids.to('cuda:0'),
                            response_mask.to('cuda:0')
                        )['log_probs']
                        old_log_probs[-1].append(mini_batch_log_probs)

            policy_model.load_state_dict(current_params)

        for epoch in range(epochs_per_rollout_batch):
            train_rewards = raw_rewards[epoch * train_batch_size:(epoch + 1) * train_batch_size]
            advantages = group_normalized_rewards[epoch * train_batch_size:(epoch + 1) * train_batch_size]

            tokenized_results = tokenized_results_by_epoch[epoch]

            avg_loss = 0.0
            avg_entropy = 0.0
            avg_clip_fraction = 0.0
            for mini_batch_step in range(grad_acc_steps):
                if mini_batch_step < grad_acc_steps - 1:
                    input_ids = tokenized_results['input_ids'][mini_batch_step * mini_batch_size : (mini_batch_step + 1) * mini_batch_size]
                    response_mask = tokenized_results['response_mask'][mini_batch_step * mini_batch_size : (mini_batch_step + 1) * mini_batch_size]
                else:
                    input_ids = tokenized_results['input_ids'][mini_batch_step * mini_batch_size :]
                    response_mask = tokenized_results['response_mask'][mini_batch_step * mini_batch_size:]

                response_mask = response_mask.to('cuda:0')
                min_batch_results = get_response_log_probs(
                    policy_model,
                    input_ids.to('cuda:0'),
                    response_mask,
                    return_token_entropy=True,
                )
                mini_batch_log_probs = min_batch_results['log_probs']
                token_entropy = min_batch_results['token_entropy']
                avg_entropy += masked_mean(token_entropy, response_mask).detach().cpu().item() / grad_acc_steps

                loss, meta_data = grpo_microbatch_train_step(
                    mini_batch_log_probs,
                    response_mask,
                    grad_acc_steps,
                    loss_type,
                    raw_rewards=train_rewards.to('cuda:0'),
                    advantages=advantages.to('cuda:0'),
                    old_log_probs = old_log_probs[epoch][mini_batch_step] if loss_type == 'grpo_clip' else None,
                    cliprange=cliprange,
                )

                if loss_type == 'grpo_clip':
                    avg_clip_fraction += meta_data['clip_fraction']
                avg_loss += loss.detach().cpu().item()

            # gradient clip
            total_var = 0
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        total_var += (p.grad.data * p.grad.data).sum()

            if grad_clip is not None and total_var > grad_clip * grad_clip:
                scale = grad_clip / math.sqrt(total_var)

                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            p.grad = scale * p.grad.data

            train_summary = {
                'train/loss': avg_loss,
                'train/grad_norm': math.sqrt(total_var),
                'train/token_entropy': avg_entropy,
            }
            print(train_summary)

            optimizer.step()
            optimizer.zero_grad()
            total_update_steps += 1

            if total_update_steps % eval_freq == eval_freq - 1:
                with torch.inference_mode():
                    eval_summary = log_generations(valid_prompts, valid_answers, llm, valid_sampling_params, policy_model,
                                                   tokenizer, mini_batch_size)
                eval_summary = {'val/' + k: v for k, v in eval_summary.items()}
                eval_summary['val_step'] = step // eval_freq

                print(eval_summary)

        old_params = policy_model.state_dict()


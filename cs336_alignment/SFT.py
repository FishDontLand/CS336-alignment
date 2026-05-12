import json
import math
import os
from typing import Callable, List

import numpy as np
import torch
import wandb
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams, LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, extract_answer


def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer) -> dict[str, torch.Tensor]:
    prompt_tokens = [tokenizer.encode(prompt) for prompt in prompt_strs]
    output_tokens = [tokenizer.encode(output) for output in output_strs]

    prompt_and_output = []
    masks = []
    for i in range(len(prompt_tokens)):
        prompt_and_output.append(prompt_tokens[i] + output_tokens[i])
        masks.append([0] * len(prompt_tokens[i]) + [1] * len(output_tokens[i]))

    max_prompt_and_output_lens = max([len(tokens) for tokens in prompt_and_output])
    prompt_and_output_padded = []
    padded_data_masks = []
    for i in range(len(prompt_and_output)):
        num_to_pad = max_prompt_and_output_lens - len(prompt_and_output[i])
        prompt_and_output_padded.append(prompt_and_output[i] + [tokenizer.pad_token_id] * num_to_pad)
        padded_data_masks.append(masks[i] + [0] * num_to_pad)


    return {
        'input_ids': torch.tensor([d[:-1] for d in prompt_and_output_padded]),
        'labels': torch.tensor([d[1:] for d in prompt_and_output_padded]),
        'response_mask': torch.tensor([m[1:] for m in padded_data_masks])
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    logsums = torch.logsumexp(logits, dim=-1, keepdim=True)
    log_prob = logits - logsums
    prob = torch.exp(log_prob)
    entropy = -(prob * log_prob).sum(dim=-1)
    return entropy

def get_response_log_probs(
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
    log_prob = torch.gather(log_prob, dim=-1, index=labels.view(*labels.size(), 1))
    log_prob = log_prob.view(*log_prob.size()[:-1])
    if return_token_entropy:
        entropy = compute_entropy(logits)
        return {
            'log_probs': log_prob,
            'token_entropy': entropy
        }
    else:
        return {
            'log_probs': log_prob
        }


def masked_normalize(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        normalize_constant: float,
        dim: int | None = None
) -> torch.Tensor:
    return (tensor * mask).sum(dim=dim) / normalize_constant

def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    log_likelihood = masked_normalize(policy_log_probs, response_mask, normalize_constant, -1)
    loss = -log_likelihood.mean()  / gradient_accumulation_steps
    loss.backward()
    return (loss.detach().cpu(),
            {'log_likelihood': log_likelihood}
    )

def log_generations(
        prompts: list[str],
        truths: list[str],
        llm: LLM,
        sampling_params,
        policy_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        mini_batch_size: int
) -> dict[str, float]:

    all_example_info = evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        prompts,
        truths,
        sampling_params
    )

    responses = [entry['model_response'] for entry in all_example_info]

    input_label_mask = tokenize_prompt_and_output(
        prompts,
        responses,
        tokenizer
    )

    total_valid_steps = len(prompts) // mini_batch_size
    if total_valid_steps * mini_batch_size < len(prompts):
        total_valid_steps += 1

    all_entropy = []
    for j in range(total_valid_steps):
        batch_inputs = input_label_mask['input_ids'][j * mini_batch_size : (j + 1) * mini_batch_size]
        batch_labels = input_label_mask['labels'][j * mini_batch_size : (j + 1) * mini_batch_size]
        batch_mask = input_label_mask['response_mask'][j * mini_batch_size : (j + 1) * mini_batch_size]
        token_entropy = get_response_log_probs(policy_model, batch_inputs.to('cuda:0'),
                                               batch_labels.to('cuda:0'), True)['token_entropy']

        response_entropy = masked_normalize(token_entropy, batch_mask.to('cuda:0'),
                                            1.0,dim=-1)

        all_entropy.append(response_entropy)

    all_entropy = torch.concat(all_entropy)
    avg_entropy = all_entropy.mean().item()

    total_model_response_len = 0
    for i in range(len(prompts)):
        total_model_response_len += len(all_example_info[i]['model_response'])

    avg_model_response_len = total_model_response_len / len(prompts)

    total_correct_response_len = 0
    total_correct_response = 0
    for i in range(len(prompts)):
        total_correct_response_len += len(all_example_info[i]['model_response']) * all_example_info[i]['reward']
        total_correct_response += all_example_info[i]['reward']

    total_wrong_response_len = total_model_response_len - total_correct_response_len
    total_wrong_response = len(prompts) - total_correct_response

    avg_correct_response_len = total_correct_response_len / total_correct_response
    avg_wrong_response_len = total_wrong_response_len / total_wrong_response

    avg_rewards = total_correct_response / len(prompts)

    log_info ={
        'avg_entropy': avg_entropy,
        'avg_response_len': avg_model_response_len,
        'avg_correct_response_len': avg_correct_response_len,
        'avg_wrong_response_len': avg_wrong_response_len,
        'avg_reward': avg_rewards
    }

    return log_info


def evaluate_vllm(
        vllm_model: LLM,
        reward_fn: Callable[[str, str], dict[str, float]],
        prompts: List[str],
        ground_truth: List[str],
        eval_sampling_params: SamplingParams
) -> list[dict]:
    response = vllm_model.generate(prompts, eval_sampling_params)
    results = []
    for i in range(len(prompts)):
        rewards = reward_fn(response[i].outputs[0].text, ground_truth[i])
        results.append({
            'question': prompts[i],
            'answer': ground_truth[i],
            'model_response': response[i].outputs[0].text,
            **rewards
        })

    return results

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float=0.85):
    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model = model_id,
            device= device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def run_sft(prompts: list[str], responses: list[str], valid_prompts: list[str], valid_truths: list[str],
            policy_model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
            llm: LLM, n_steps: int, batch_size: int, mini_batch_size: int, lr: float, grad_clip: float|None, eval_freq: int, run: wandb.Run | None = None):

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # compute batch gradient using mini-batch
    grad_acc_steps = min(len(prompts), batch_size) // mini_batch_size
    if grad_acc_steps * mini_batch_size < min(len(prompts), batch_size):
        grad_acc_steps += 1


    optimizer = torch.optim.Adam(policy_model.parameters(), lr=lr)
    optimizer.zero_grad()

    np.random.seed(324234)

    for step in range(n_steps):
        batch_loss = 0.0
        train_idx = np.random.choice(len(prompts), batch_size, replace=False)
        train_prompts = [prompts[idx] for idx in train_idx]
        train_responses = [responses[idx] for idx in train_idx]

        # calculate gradients
        for i in range(grad_acc_steps):
            if i < grad_acc_steps - 1:
                mini_batch_prompts = train_prompts[i * mini_batch_size:(i + 1) * mini_batch_size]
                mini_batch_response = train_responses[i * mini_batch_size:(i + 1) * mini_batch_size]
            else:
                mini_batch_prompts = train_prompts[i * mini_batch_size:]
                mini_batch_response = train_responses[i * mini_batch_size:]

            output_dict = tokenize_prompt_and_output(mini_batch_prompts, mini_batch_response, tokenizer)

            log_prob_dict = get_response_log_probs(
                policy_model,
                output_dict['input_ids'].to('cuda:0'),
                output_dict['labels'].to('cuda:0'),
                True
            )
            log_probs = log_prob_dict['log_probs']

            loss = sft_microbatch_train_step(
                log_probs,
                output_dict['response_mask'].to('cuda:0'),
                grad_acc_steps,
                float(len(prompts)) / (len(mini_batch_prompts) * grad_acc_steps),
            )[0]

            batch_loss += loss.item()

        # gradient clip
        if grad_clip is not None:
            total_var = 0
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        total_var += (p.grad.data * p.grad.data).sum()

            if total_var > grad_clip * grad_clip:
                scale = grad_clip / math.sqrt(total_var)

                for group in optimizer.param_groups:
                    for p in group[('params')]:
                        if p.grad is not None:
                            p.grad = scale * p.grad.data

        # take optimization step
        optimizer.step()
        optimizer.zero_grad()

        base_info = {'train_step': step, 'train/loss': batch_loss}
        if run is not None:
            run.log(base_info)
        else:
            print(base_info)

        load_policy_into_vllm_instance(policy_model, llm)

        if step % eval_freq == eval_freq - 1:
            with torch.inference_mode():
                eval_summary = log_generations(valid_prompts, valid_truths, llm, sampling_params, policy_model, tokenizer, mini_batch_size)
            eval_summary = {'val/' + k : v for k, v in eval_summary.items()}
            eval_summary['val_step'] = step // eval_freq

            if run is not None:
                run.log(
                    eval_summary
                )
            else:
                print(eval_summary)

def run_expert_iteration(n_ei_steps: int, n_steps: int, batch_size: int, inner_batch_size: int, rollouts: int,
                         mini_batch_size: int, lr: float, eval_freq: int, eval_size: int, output_dir: str, grad_clip=1.0, run: wandb.Run = None) -> None:
    if run is not None:
        wandb.define_metric("train_step")
        wandb.define_metric("val_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("val/*", step_metric="val_step")

    model_path = '/workspace/cs336/hf_cache/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots'
    model_hash = '4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2'
    full_path = model_path + '/' + model_hash

    policy_model = AutoModelForCausalLM.from_pretrained(full_path).to('cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(full_path)
    llm = init_vllm(full_path, device='cuda:1', seed=12432)
    print('llm loaded')

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, min_tokens=4, seed=666,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=rollouts
    )

    train_data = []
    with open('./data/train.jsonl', 'r') as f:
        for line in f:
            train_data.append(json.loads(line))

    test_data = []
    with open('./data/test.jsonl', 'r') as f:
        for line in f:
            test_data.append(json.loads(line))

    np.random.seed(12468)
    test_data = np.random.choice(test_data, size=eval_size, replace=False)

    test_prompts = [entry['question'] for entry in test_data]
    test_truths = [entry['answer'] for entry in test_data]


    for step in range(n_ei_steps):
        print('iteration: ', step)
        select_idx = np.random.choice(len(train_data), size=batch_size, replace=False)
        train_batch = [train_data[idx] for idx in select_idx]
        train_questions = [entry['question'] for entry in train_batch]
        ground_truths = [entry['answer'] for entry in train_batch]

        response = llm.generate(train_questions, sampling_params)
        print('num responses: ', len(response))
        print('output per response: ', len(response[0].outputs))
        train_prompts = []
        train_responses = []
        train_truths = []
        for i in range(batch_size):
            rewards = [r1_zero_reward_fn(o.text, ground_truths[i]) for o in response[i].outputs]
            for j in range(len(rewards)):
                if abs(rewards[j]['reward'] - 1.0) < 1e-10:
                    train_prompts.append(train_questions[i])
                    train_responses.append(response[i].outputs[j].text)
                    train_truths.append(ground_truths[i])

        print("expert batch data generated, size: ", len(train_prompts))

        run_sft(train_prompts, train_responses, test_prompts, test_truths, policy_model, tokenizer, llm, n_steps, inner_batch_size,
                mini_batch_size, lr, grad_clip, eval_freq, run)

    policy_model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)


if __name__ == "__main__":
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    print("torch.cuda.device_count() =", torch.cuda.device_count())
    run_expert_iteration(
        n_ei_steps=5,
        n_steps=10,
        batch_size=2048,
        inner_batch_size=32,
        rollouts=4,
        mini_batch_size=4,
        lr=1e-5,
        eval_freq=5,
        eval_size=2048,
        output_dir='./outputs/experiments1'
    )



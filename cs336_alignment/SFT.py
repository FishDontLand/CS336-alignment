from typing import Callable, List

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from vllm import SamplingParams, LLM

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
    return (loss.detach(),
            {'log_likelihood': log_likelihood}
    )

def log_generations(
        prompts: list[str],
        responses: list[str],
        ground_truths: list[str],
        model: LLM,
):

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"],
        include_stop_str_in_output=True
    )

    log_info = evaluate_vllm(
        model,
        r1_zero_reward_fn,
        prompts,
        ground_truths,
        sampling_params
    )

    underly_model = model.llm_engine.model_executor.driver_worker.model_runner.model

    input_label_mask = tokenize_prompt_and_output(
        prompts,
        responses,
        model.get_tokenizer()
    )

    token_entropy = get_response_log_probs(underly_model,input_label_mask['input_ids'], input_label_mask['labels'], True)['token_entropy']

    avg_token_entropy = masked_normalize(token_entropy, input_label_mask['response_mask'], dim=-1).mean().detach().cpu().item()

    total_model_response_len = 0
    for i in range(len(prompts)):
        total_model_response_len += len(log_info[i]['model_response'])
    avg_model_response_len = total_model_response_len / len(prompts)

    total_correct_response_len = 0
    total_correct_response = 0
    for i in range(len(prompts)):
        total_correct_response_len += len(log_info[i]['model_response']) * log_info[i]['reward']
        total_correct_response += log_info[i]['reward']

    total_wrong_response_len = total_model_response_len - total_correct_response_len
    total_wrong_response = len(prompts) - total_correct_response

    avg_correct_response_len = total_correct_response_len / total_correct_response
    avg_wrong_response_len = total_wrong_response_len / total_wrong_response

    log_info.append({
        'avg_token_entropy': avg_token_entropy,
        'avg_response_len': avg_model_response_len,
        'avg_correct_response_len': avg_correct_response_len,
        'avg_wrong_response_len': avg_wrong_response_len,
    })

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

if __name__ == "__main__":
    model_folder = '/workspace/cs336/hf_cache/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots'
    model_hash = '4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2'
    model_path = model_folder + '/' + model_hash
    llm = LLM(model=model_path)



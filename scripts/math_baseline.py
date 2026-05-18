import json
import os
import re
from math_verify import parse, verify, LatexExtractionConfig
from typing import Callable, List

from vllm import LLM, SamplingParams
from datasets import load_dataset, concatenate_datasets

from cs336_alignment.SFT import evaluate_vllm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, extract_boxed_answer


def zero_shot_evaluation(output_file, model_path=None):
    all_subjects = [
        'algebra',
        'counting_and_probability',
        'geometry',
        'intermediate_algebra',
        'number_theory',
        'prealgebra',
        'precalculus'
    ]

    questions = []
    answers = []

    for subject in all_subjects:
        ds = load_dataset(
            'EleutherAI/hendrycks_math',
            subject,
            cache_dir="/workspace/cs336/hf_cache/datasets"
        )
        sampled_ds = ds['test']
        questions = questions + list(sampled_ds['problem'])
        answers = answers + list(sampled_ds['solution'])

    # use the default Qwen2.5 model if model_path is missing
    if model_path is None:
        model_folder = '/workspace/cs336/hf_cache/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots'
        model_hash = '4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2'
        model_path = model_folder + '/' + model_hash
    llm = LLM(model=model_path)

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"],
        include_stop_str_in_output=True
    )

    with open('./cs336_alignment/prompts/r1_zero.prompt', 'r') as f:
        prompt = f.read()

    questions = [prompt.replace('{question}', q) for q in questions]

    results = evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        questions,
        answers,
        sampling_params,
    )

    with open(output_file, 'w') as f:
        for entry in results:
            f.write(json.dumps(entry) + '\n')

def analyze_eval_results(result_file_path):
    results = []
    with open(result_file_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    format_1_answer_1 = 0

    for r in results:
        if abs(r['format_reward'] - 1.0) < 1e-12 and abs(r['answer_reward'] - 1.0) < 1e-12:
            format_1_answer_1 += 1

    format_1_answer_0 = 0
    for r in results:
        if abs(r['format_reward'] - 1.0) < 1e-12 and abs(r['answer_reward'] - 0.0) < 1e-12:
            format_1_answer_0 += 1

    format_0_answer_0 = 0
    for r in results:
        if abs(r['format_reward'] - 0.0) < 1e-12 and abs(r['answer_reward'] - 0.0) < 1e-12:
            format_0_answer_0 += 1

    print("Frequency count")
    print("format reward = 1.0, answer reward = 1.0: ", format_1_answer_1)
    print("format reward = 1.0, answer reward = 0.0: ", format_1_answer_0)
    print("format reward = 0.0, answer_reward = 0.0: ", format_0_answer_0)

    format_reward_0_examples = []
    for r in results:
        if abs(r['format_reward'] - 0.0) < 1e-12:
           format_reward_0_examples.append(r)

        if len(format_reward_0_examples) == 10:
            break

    format_1_answer_0_examples = []
    for r in results:
        if abs(r['format_reward'] - 1.0) < 1e-12 and abs(r['answer_reward'] - 1.0) < 1e-12:
            format_1_answer_0_examples.append(r)

        if len(format_1_answer_0_examples) == 10:
            break

    print("Examples where format reward = 0.0 ")
    print(format_reward_0_examples)

    print("Examples where format reward = 1.0 but answer reward = 0.0")
    print(format_1_answer_0_examples)


if __name__ == '__main__':
    zero_shot_evaluation(output_file='./evaluations/results_without_sft.jsonl')
    zero_shot_evaluation(output_file='./evaluations/experiments1/results_after_expert_iteration.jsonl',
                         model_path='./outputs/experiments1')
    analyze_eval_results('./evaluations/results_without_sft.jsonl')
    analyze_eval_results('./evaluations/experiments1/results_after_expert_iteration.jsonl')





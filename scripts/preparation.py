import json

from datasets import load_dataset, concatenate_datasets
from huggingface_hub import snapshot_download

from cs336_alignment.drgrpo_grader import extract_answer


def prepare_hendrycks_math_data():
    all_subjects = [
        'algebra',
        'counting_and_probability',
        'geometry',
        'intermediate_algebra',
        'number_theory',
        'prealgebra',
        'precalculus'
    ]

    with open('./cs336_alignment/prompts/r1_zero.prompt', 'r') as f:
        prompt = f.read()

    for data_type in ['train', 'test']:
        qas = []
        for subject in all_subjects:
            ds = load_dataset(
                'EleutherAI/hendrycks_math',
                subject,
                cache_dir="/workspace/cs336/hf_cache/datasets"
            )
            data = ds[data_type]
            questions = list(data['problem'])
            solutions = list(data['solution'])
            for i in range(len(questions)):
                q = prompt.replace('{question}', questions[i])
                qas.append({
                    'question': q,
                    'answer': solutions[i]
                })

        with open(f'./data/{data_type}.jsonl', 'w') as f:
            for entry in qas:
                f.write(json.dumps(entry) + '\n')


if __name__ == '__main__':
    prepare_hendrycks_math_data()

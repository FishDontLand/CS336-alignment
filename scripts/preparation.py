from datasets import load_dataset, concatenate_datasets
from huggingface_hub import snapshot_download

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

    for subject in all_subjects:
        ds = load_dataset(
            'EleutherAI/hendrycks_math',
            subject,
            cache_dir="/workspace/cs336/hf_cache/datasets"
        )


if __name__ == '__main__':
    prepare_hendrycks_math_data()

import typer
import wandb

from cs336_alignment.RL import run_rl


def baseline_learning_rate():
    learning_rate_options = [1e-6, 1e-5, 1e-4]
    for learning_rate in learning_rate_options:
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally youdfr team name).
            entity="dyu68-georgia-institute-of-technology",
            # Set the wandb project where this run will be logged.
            project="RLForLM",
            # Track hyperparameters and run metadata.
            name=f"lr={learning_rate}",
            config={
                "architecture": "Qwen2.5-Math-1.5B",
                "dataset": '/workspace/CS336-alignment/data/train.jsonl',
                "n_grpo_steps": 200,
                "rollout_batch_size": 256,
                "group_size": 8,
                "sampling_temperature": 1.0,
                "sampling_min_tokens": 4,
                "sampling_max_tokens": 1024,
                "epochs_per_rollout_batch": 1,
                "mini_batch_size": 2,
                "gpu_memory_utilization": 0.85,
                "clipped_norm": 1,
                "optimizer": "AdamW"
            },
            reinit=True,
        )

        run_rl(learning_rate=learning_rate, run=run)
        run.finish()

def main(experiment_name: str):
    if experiment_name == 'grpo_learning_rate':
        baseline_learning_rate()
    else:
        raise ValueError("Unknown experiment name")

if __name__ == "__main__":
    typer.run(main)








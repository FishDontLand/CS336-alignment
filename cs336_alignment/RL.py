from typing import Callable, Literal

import torch


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

    return (normalized_rewards.view(-1), rewards, meta_data)

def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor
) -> torch.Tensor:
    return raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rn_derivative = policy_log_probs / old_log_probs
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


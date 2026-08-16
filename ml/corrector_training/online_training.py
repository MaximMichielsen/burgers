import logging
from collections import deque
from pathlib import Path
from typing import NamedTuple, cast

from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ml.corrector_training.SAC import SACAgent, SACConfig, ReplayBuffer
from ml.corrector_training.rl_environment import AVCEnvironment
from ml.ml_agents.corrector import save_corrector, AVCTrainingConfig, Transition

logger = logging.getLogger(__name__)








@dataclass
class TrainingStats:
    """Per-episode and per-update diagnostics."""

    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    critic_losses: list[float] = field(default_factory=list)
    actor_losses: list[float] = field(default_factory=list)
    alpha_temp_values: list[float] = field(default_factory=list)
    total_env_steps: int = 0


class OnlineAVCTrainer:
    """Online training loop for the AVC.

    Parameters
    ----------
    environment:
        The AVC solver environment.
    agent:
        RL agent / training algorithm.
    output_dir:
        Directory for policy checkpoints."""

    def __init__(
        self,
        environment: AVCEnvironment,
        agent: SACAgent,
        agent_config: SACConfig,
        training_config: AVCTrainingConfig,
        output_dir: Path,
    ) -> None:
        self.environment = environment
        self.agent = agent
        self.agent_config = agent_config
        self.training_config = training_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.replay_buffer = ReplayBuffer(capacity=agent_config.replay_capacity)
        self.stats = TrainingStats()

    def train(self, n_episodes: int, checkpoint_every: int = 50) -> TrainingStats:
        """Run n_episodes of online RL training."""

        print(
            f"\nOnline SAC training (AVC - {self.environment.correction_mode}) — {n_episodes} episodes "
            f"| warmup: {self.training_config.n_warmup_steps} steps "
            f"| Nₛₖᵢₚ: {self.environment.avc_config.n_skip_steps} "
            f"| DNS ref: {'time-varying (projection)'}"
        )
        print("-" * 64)
        episode_idx = 0
        while episode_idx < n_episodes:
            state_current = self.environment.reset()
            episode_reward_total = 0.0
            episode_step_count = 0
            done_flag = False

            while not done_flag:
                if self.stats.total_env_steps < self.training_config.n_warmup_steps:
                    if self.environment.correction_mode == "global":
                        random_scalar = float(
                            np.random.uniform(
                                0, self.training_config.exploration_bound_upper
                            )
                        )
                        alpha_action = random_scalar
                    else:
                        raise ValueError(
                            f"Correction mode is not being passed on correctly, got {self.environment.correction_mode}"
                        )

                else:
                    alpha_action = self.agent.select_action(state_current)

                next_state_array, reward_val, done_flag = self.environment.step(
                    alpha_action
                )

                self.replay_buffer.push(
                    Transition(
                        state=state_current,
                        action=alpha_action,
                        reward=reward_val,
                        next_state=next_state_array,
                        done=done_flag,
                    )
                )

                state_current = next_state_array
                episode_reward_total += reward_val
                episode_step_count += 1
                self.stats.total_env_steps += 1

                enough_samples = (
                    len(self.replay_buffer) >= self.training_config.batch_size
                )
                past_warmup = (
                    self.stats.total_env_steps >= self.training_config.n_warmup_steps
                )
                update_due = (
                    self.stats.total_env_steps % self.training_config.update_every == 0
                )

                if enough_samples and past_warmup and update_due:
                    for _ in range(self.training_config.updates_per_step):
                        critic_loss_val, actor_loss_val, alpha_temp_val = (
                            self.agent.update(self.replay_buffer)
                        )
                        self.stats.critic_losses.append(critic_loss_val)
                        self.stats.actor_losses.append(actor_loss_val)
                        self.stats.alpha_temp_values.append(alpha_temp_val)

            self.stats.episode_rewards.append(episode_reward_total)
            self.stats.episode_lengths.append(episode_step_count)

            if episode_idx % 10 == 0 or episode_idx == n_episodes - 1:
                recent_mean = float(np.mean(self.stats.episode_rewards[-10:]))
                print(
                    f"Episode {episode_idx:04d} | "
                    f"Return: {episode_reward_total:+.2f} | "
                    f"Mean(10): {recent_mean:+.2f} | "
                    f"Control steps: {episode_step_count} | "
                    f"Buffer: {len(self.replay_buffer)}"
                )

            if (episode_idx + 1) % checkpoint_every == 0:
                self.save_checkpoint(episode_idx=episode_idx)

            episode_idx += 1

        self.save_checkpoint(episode_idx=n_episodes - 1, tag="final")
        print(f"\nTraining complete. Checkpoints saved to '{self.output_dir}'.")
        return self.stats

    def save_checkpoint(self, episode_idx: int, tag: str | None = None) -> None:
        """Save policy weights to a .pt file."""
        suffix = tag if tag else f"ep{episode_idx:04d}"
        checkpoint_path = self.output_dir / f"av_corrector_{suffix}.pt"
        save_corrector(self.agent.policy, checkpoint_path)
        logger.info("Checkpoint saved to %s", checkpoint_path)

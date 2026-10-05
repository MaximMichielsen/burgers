"""Diagnostics for ANN training. Able to handle both regular training and proof training."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib import pyplot as plt

from ml.environment import EnvironmentForcingDNS
from proofing.ann_config import ProofMode
from proofing.environment import EnvironmentProof

if TYPE_CHECKING:
    from ml.td3 import TD3Trainer


def moving_average(data: np.ndarray | list[float], window_size: int = 50) -> np.ndarray:
    """Calculate moving average with constant NaN padding to retain input length."""
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) < window_size or window_size <= 1:
        return arr

    ma = np.convolve(arr, np.ones(window_size) / window_size, mode="valid")
    pad_left = (window_size - 1) // 2
    pad_right = window_size - 1 - pad_left
    return np.pad(ma, (pad_left, pad_right), mode="constant", constant_values=np.nan)


def safe_legend(ax: plt.Axes, loc: str = "upper right", **kwargs: Any) -> None:
    """Add a legend to an axis only if labeled handles exist."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc=loc, **kwargs)


def plot_diagnostic_metrics(trainer: TD3Trainer, save_dir: Path | None = None) -> None:
    """Plot comprehensive step-level and episode-level diagnostic metrics with noise visualization."""
    if save_dir is None:
        save_dir = Path(trainer.master_path) / "diagnostics"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Safely retrieve environment reference
    env: EnvironmentForcingDNS | EnvironmentProof | None = getattr(trainer, "env", None)
    if env is None:
        raise TypeError("Environment is not retrieved from the trainer correctly.")

    episodes = np.arange(1, len(getattr(trainer, "mean_actions", [])) + 1)
    ann_config = getattr(trainer, "ann_config", None)
    proof_mode = getattr(ann_config, "proof_mode", None)

    # =========================================================================
    # 1. Main Diagnostic Dashboard (2x2)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    fig.suptitle("TD3 Diagnostic Dashboard", fontsize=16, fontweight="bold")

    # --- Subplot [0, 0]: Policy Action Noise & Std Dev Visualization ---
    ax = axes[0, 0]
    action_mean_hist = getattr(env, "action_mean_history", [])

    if proof_mode == "d" and len(action_mean_hist) > 0:
        action_means = np.asarray(action_mean_hist)
        target_mean = np.asarray(env.target_action_mean(), dtype=np.float64)
        num_episodes = max(len(episodes), 1)
        steps_actions = np.linspace(0, num_episodes, num=len(action_means))

        ax.plot(
            steps_actions,
            action_means[:, 0],
            color="dodgerblue",
            alpha=0.35,
            linewidth=0.8,
            label="Action mean 1",
        )
        ax.plot(
            steps_actions,
            moving_average(action_means[:, 0]),
            color="navy",
            linewidth=1.5,
            label="Action mean 1 (trend)",
        )
        ax.plot(
            steps_actions,
            action_means[:, 1],
            color="darkorange",
            alpha=0.35,
            linewidth=0.8,
            label="Action mean 2",
        )
        ax.plot(
            steps_actions,
            moving_average(action_means[:, 1]),
            color="chocolate",
            linewidth=1.5,
            label="Action mean 2 (trend)",
        )

        if len(target_mean) >= 2:
            ax.axhline(
                target_mean[0],
                color="royalblue",
                linestyle="--",
                linewidth=1.5,
                label=f"Target mean 1 ({target_mean[0]:.2f})",
            )
            ax.axhline(
                target_mean[1],
                color="darkorange",
                linestyle="--",
                linewidth=1.5,
                label=f"Target mean 2 ({target_mean[1]:.2f})",
            )

    elif len(episodes) > 0:
        mean_a = np.array(trainer.mean_actions)
        std_a = np.array(trainer.action_deviation_history)

        ax.plot(
            episodes, mean_a, label=r"Mean Action $\mu_a$", color="navy", linewidth=2
        )
        ax.fill_between(
            episodes,
            mean_a - std_a,
            mean_a + std_a,
            color="dodgerblue",
            alpha=0.25,
            label=r"Action Noise ($\mu_a \pm 1\sigma_a$)",
        )
        ax.axhline(
            1.0,
            color="slategrey",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=r"Baseline ($a=1.0$)",
        )

        if proof_mode in ("b", ProofMode.b):
            target_vec = getattr(env, "target_actions_current", None)
            if target_vec is not None:
                target_arr = np.asarray(target_vec, dtype=np.float64).ravel()
                if len(target_arr) >= 1:
                    ax.axhline(
                        target_arr[0],
                        color="royalblue",
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.9,
                        label=f"Target $a_1$ ({target_arr[0]:.3f})",
                    )
                if len(target_arr) >= 2:
                    ax.axhline(
                        target_arr[1],
                        color="darkorange",
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.9,
                        label=f"Target $a_2$ ({target_arr[1]:.3f})",
                    )

        elif proof_mode in ("c", ProofMode.c):
            target_curr_hist = getattr(env, "target_actions_current_history", None)
            if target_curr_hist is not None and len(target_curr_hist) > 0:
                target_hist = np.asarray(target_curr_hist)
                x_steps = np.linspace(1.0, max(len(episodes), 1), num=len(target_hist))

                if target_hist.ndim > 1 and target_hist.shape[1] >= 2:
                    ax.plot(
                        x_steps,
                        target_hist[:, 0],
                        color="royalblue",
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.75,
                        label="Target $a_1$ History",
                    )
                    ax.plot(
                        x_steps,
                        target_hist[:, 1],
                        color="darkorange",
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.75,
                        label="Target $a_2$ History",
                    )

                    target_mean_history = np.mean(target_hist, axis=1)
                    ax.plot(
                        x_steps,
                        target_mean_history,
                        color="crimson",
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.9,
                        label="Target Mean History",
                    )
                else:
                    target_history = target_hist.ravel()
                    ax.plot(
                        x_steps,
                        target_history,
                        color="darkorange",
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.8,
                        label="Target History",
                    )

        stochastic_steps = getattr(
            getattr(trainer, "hp", None), "stochastic_timesteps", 0
        )
        n_agent_steps = getattr(ann_config, "n_agent_steps_per_episode", 1)
        cut_off_step = int(stochastic_steps / max(n_agent_steps, 1))

        episodes_ran = getattr(trainer, "episodes_ran", 0)
        if 0 < cut_off_step < episodes_ran:
            ax.axvline(
                cut_off_step,
                color="slategrey",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label=rf"Exploration Cutoff ($\mathrm{{step}}={cut_off_step:.1f}$)",
            )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Action Magnitude")
    ax.grid(True, linestyle="--", alpha=0.6)
    safe_legend(ax, loc="upper left")

    # --- Subplot [0, 1]: Exploration Noise & Critic Q-Value Sensitivity ---
    ax = axes[0, 1]
    q_hist = getattr(trainer, "q_sensitivity_history", [])
    if len(q_hist) > 0:
        ax.plot(
            np.arange(1, len(q_hist) + 1),
            q_hist,
            color="chocolate",
            linewidth=2,
            marker="o",
            markersize=3,
            label=r"$\Delta Q = \vert{}Q(s_0, 1.1) - Q(s_0, 1.0)\vert{}$",
        )
        ax.set_title(r"Critic Q-Value Sensitivity ($\Delta Q$)")
    else:
        ax.text(
            0.5,
            0.5,
            "No Q-Sensitivity Data Logged\n(Steps < stochastic_timesteps)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(r"Critic Sensitivity $\Delta Q$")
    ax.set_xlabel("Episode")
    ax.set_ylabel(r"$\Delta Q$")
    ax.grid(True, linestyle="--", alpha=0.6)
    safe_legend(ax)

    # --- Subplot [1, 0]: Critic & Actor Losses ---
    ax = axes[1, 0]
    c_loss = getattr(trainer, "critic_loss_history", [])
    a_loss = getattr(trainer, "actor_loss_history", [])
    has_loss = False

    if len(c_loss) > 0:
        ax.plot(c_loss, label="Critic Loss", color="darkorange", alpha=0.85)
        has_loss = True
    if len(a_loss) > 0:
        ax.plot(a_loss, label="Actor Loss", color="royalblue", alpha=0.85)
        has_loss = True

    if not has_loss:
        ax.text(
            0.5,
            0.5,
            "No Gradient Steps Executed Yet",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        ax.set_yscale("log")

    ax.set_title("TD3 Optimization Losses")
    ax.set_xlabel("Training Gradient Steps")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.6)
    safe_legend(ax)

    # --- Subplot [1, 1]: Total Scaled vs Unscaled Reward Trajectory ---
    ax = axes[1, 1]
    ep_rewards = getattr(trainer, "episode_reward_history", [])
    if len(ep_rewards) > 0:
        ax.plot(
            np.arange(1, len(ep_rewards) + 1),
            ep_rewards,
            color="navy",
            linewidth=2,
            label="Clipped Episode Reward",
        )
    ax.set_title("Episode Reward Trajectory")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, linestyle="--", alpha=0.6)
    safe_legend(ax, loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    dashboard_path = save_dir / "diagnostic_dashboard.png"
    plt.savefig(dashboard_path, dpi=300)
    plt.close(fig)

    print(f"[DIAGNOSTICS] Diagnostic dashboard plot saved to: {dashboard_path}")

    # =========================================================================
    # 2. Detailed Corrections Plot (Proof Modes)
    # =========================================================================
    _plot_actions_history(
        env=env,
        episodes=episodes,
        proof_mode=proof_mode,
        save_dir=save_dir,
    )


def _plot_actions_history(
    env: EnvironmentForcingDNS | EnvironmentProof,
    episodes: np.ndarray,
    proof_mode: str | None,
    save_dir: Path,
) -> None:
    """Internal helper to plot diagnostic corrections for proof modes b, c, or d."""
    if proof_mode not in ("b", "c", "d"):
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    if proof_mode in ("b", "c"):
        applied_actions = np.asarray(getattr(env, "applied_actions_history", []))
        if len(applied_actions) == 0:
            plt.close(fig)
            print('[DIAGNOSTICS] NO APPLIED ACTIONS FOUND')
            return

        # Check for history first (mode 'c'), fall back to static target vector (mode 'b')
        target_curr_hist = getattr(env, "target_actions_current_history", [])
        if len(target_curr_hist) > 0:
            target_actions = np.asarray(target_curr_hist)
            is_target_static = False
        else:
            static_target = getattr(env, "target_actions_current", None)
            is_target_static = True
            if static_target is None:
                plt.close(fig)
                print('[DIAGNOSTICS] NO TARGET ACTIONS FOUND')
                return

            target_vec = np.asarray(static_target, dtype=np.float64).ravel()
            # Broadcast the static target across all step iterations
            target_actions = np.tile(target_vec, (len(applied_actions), 1))

        target_action_1, target_action_2 = target_actions[:, 0], target_actions[:, 1]
        applied_actions_1, applied_actions_2 = (
            applied_actions[:, 0],
            applied_actions[:, 1],
        )

        target_mean_1 = float(np.mean(target_action_1))
        target_mean_2 = float(np.mean(target_action_2))

        num_episodes = len(episodes)
        steps = np.linspace(start=0, stop=num_episodes, num=len(target_action_1))
        steps_actions = np.linspace(
            start=0, stop=num_episodes, num=len(applied_actions_1)
        )

        # --- Subplot 1: Action Dimension 1 (Blue Palette) ---
        ax1.plot(
            steps_actions,
            applied_actions_1,
            color="dodgerblue",
            alpha=0.35,
            linewidth=0.8,
            label="Applied Action (Raw)",
        )
        ax1.plot(
            steps_actions,
            moving_average(applied_actions_1),
            color="navy",
            linewidth=1.5,
            label="Applied Action (Trend)",
        )
        ax1.plot(
            steps,
            target_action_1,
            color="royalblue",
            linestyle="--",
            linewidth=1.8,
            label="Target Action",
        )
        if not is_target_static:
            ax1.axhline(
                target_mean_1,
                color="deepskyblue",
                linestyle=":",
                linewidth=1.5,
                label=f"Target Mean ({target_mean_1:.2f})",
            )
        ax1.set_ylabel("Value", fontweight="bold")
        ax1.set_title(
            "Diagnostic: Corrections vs Target Actions (Action 1)",
            fontsize=11,
            fontweight="bold",
        )

        # --- Subplot 2: Action Dimension 2 (Orange Palette) ---
        ax2.plot(
            steps_actions,
            applied_actions_2,
            color="sandybrown",
            alpha=0.4,
            linewidth=0.8,
            label="Applied Action (Raw)",
        )
        ax2.plot(
            steps_actions,
            moving_average(applied_actions_2),
            color="chocolate",
            linewidth=1.5,
            label="Applied Action (Trend)",
        )
        ax2.plot(
            steps,
            target_action_2,
            color="darkorange",
            linestyle="--",
            linewidth=1.8,
            label="Target Action",
        )
        if not is_target_static:
            ax2.axhline(
                target_mean_2,
                color="darkamber" if "darkamber" in plt.colormaps() else "goldenrod",
                linestyle=":",
                linewidth=1.5,
                label=f"Target Mean ({target_mean_2:.2f})",
            )
        ax2.set_xlabel("Episodes / Steps", fontweight="bold")
        ax2.set_ylabel("Value", fontweight="bold")
        ax2.set_title(
            "Diagnostic: Corrections vs Target Actions (Action 2)",
            fontsize=11,
            fontweight="bold",
        )
    elif proof_mode == "d":
        target_mean_func = getattr(env, "target_action_mean", None)
        moving_action_mean_hist = getattr(env, "action_mean_history", [])
        action_history = np.asarray(getattr(env, "applied_actions_history", []))

        if target_mean_func is None or len(moving_action_mean_hist) == 0:
            plt.close(fig)
            return

        target_mean_1, target_mean_2 = target_mean_func()

        moving_action_means = np.asarray(moving_action_mean_hist)
        moving_action_mean_1 = moving_action_means[:, 0]
        moving_action_mean_2 = moving_action_means[:, 1]

        if len(action_history) > 0:
            true_action_mean = np.mean(action_history, axis=0)
            true_action_mean_1 = true_action_mean[0]
            true_action_mean_2 = true_action_mean[1]
        else:
            true_action_mean_1, true_action_mean_2 = 0.0, 0.0

        steps_actions = np.linspace(start=0, stop=len(episodes), num=len(moving_action_mean_1))

        # --- Subplot 1: Action Dimension 1 ---
        ax1.plot(
            steps_actions,
            moving_action_mean_1,
            color="dodgerblue",
            label="Moving Action Mean",
        )
        ax1.axhline(
            true_action_mean_1,
            color="darkblue",
            linestyle="--",
            label=f"True Action Mean ({true_action_mean_1:.2f})",
        )
        ax1.axhline(
            target_mean_1,
            color="royalblue",
            linestyle="--",
            linewidth=1.5,
            label=f"Target Mean ({target_mean_1:.2f})",
        )
        ax1.set_ylabel("Value", fontweight="bold")
        ax1.set_title(
            "Diagnostic: Correction Mean vs Target Mean (Action 1)",
            fontsize=11,
            fontweight="bold",
        )

        # --- Subplot 2: Action Dimension 2 ---
        ax2.plot(
            steps_actions,
            moving_action_mean_2,
            color="sandybrown",
            label="Moving Action Mean",
        )
        ax2.axhline(
            true_action_mean_2,
            color="saddlebrown",
            linestyle="--",
            label=f"True Action Mean ({true_action_mean_2:.2f})",
        )
        ax2.axhline(
            target_mean_2,
            color="darkorange",
            linestyle="--",
            linewidth=1.5,
            label=f"Target Mean ({target_mean_2:.2f})",
        )
        ax2.set_xlabel("Episodes / Steps", fontweight="bold")
        ax2.set_ylabel("Value", fontweight="bold")
        ax2.set_title(
            "Diagnostic: Correction Mean vs Target Mean (Action 2)",
            fontsize=11,
            fontweight="bold",
        )

    for ax in (ax1, ax2):
        ax.grid(True, linestyle="--", alpha=0.5)
        # Places the legend cleanly outside the plot area
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    plt.tight_layout()
    corrections_path = save_dir / "diagnostic_corrections.png"
    plt.savefig(corrections_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[DIAGNOSTICS] Corrections plot saved to: {corrections_path}")
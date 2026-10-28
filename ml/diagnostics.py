"""Diagnostics for ANN training. Able to handle both regular training and proof training."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from ml.original.environment import EnvironmentForcingDNS
from proofing.action_based.environment import EnvironmentProof
from proofing.contextual_bandit_reformulation.environment import (
    EnvironmentBanditTraining,
)
from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    from ml.original.training import TD3Trainer
    from ml.bandit.training import BanditTrainer

plt.rcParams.update(
    {
        "font.family": "serif",  # Match your thesis font (serif or sans-serif)
        "font.size": 10,  # Base font size
        "axes.labelsize": 11,  # Axis label size
        "axes.titlesize": 11,  # Title size (if using titles)
        "xtick.labelsize": 9,  # Tick label size
        "ytick.labelsize": 9,  # Tick label size
        "legend.fontsize": 9,  # Legend font size
        "xtick.direction": "in",  # Ticks pointing inward
        "ytick.direction": "in",
        "xtick.top": True,  # Ticks on all four sides for a clean box frame
        "ytick.right": True,
        "axes.linewidth": 0.8,  # Frame line width
        "figure.autolayout": True,  # Tight layout adjustment
    }
)


def moving_average(data: np.ndarray | list[float], window_size: int = 50) -> np.ndarray:
    """Calculate moving average with constant NaN padding to retain input length."""
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) < window_size or window_size <= 1:
        return arr

    ma = np.convolve(arr, np.ones(window_size) / window_size, mode="valid")
    pad_left = (window_size - 1) // 2
    pad_right = window_size - 1 - pad_left
    return np.pad(ma, (pad_left, pad_right), mode="constant", constant_values=np.nan)


def safe_legend(ax: plt.Axes, loc: str = "best", **kwargs: Any) -> None:
    """Add a legend to an axis only if labeled handles exist."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc=loc, **kwargs)


def _save_figure(fig: plt.Figure, base_path: Path, dpi: int = 300) -> None:
    """Save a figure as both PNG and PDF formats."""
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")


# =========================================================================
# Subplot Routines (Standalone + Dashboard Rendering)
# =========================================================================


def plot_actions(
    trainer: TD3Trainer | BanditTrainer,
    ax: plt.Axes | None = None,
    save_dir: Path | None = None,
) -> None:
    """Plot independent lines for each action dimension from trainer.action_history."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 5))

    action_history = getattr(trainer, "action_history", [])
    if len(action_history) == 0:
        env = getattr(trainer, "env", None)
        action_history = getattr(env, "action_history", [])

    random_exploration_episodes = trainer.ann_config.n_random_episodes

    if random_exploration_episodes != 0:
        ax.axvline(
            x=random_exploration_episodes,
            color="gray",
            alpha=0.8,
            linestyle="--",
            linewidth=0.8,
            label=f"End Exploration Phase (Ep. {random_exploration_episodes})",
        )

    actions = np.asarray(action_history)
    episodes = np.arange(1, len(actions) + 1)
    action_labels = ["Advection", "Viscosity", "third term", "fourht therm"]

    if len(actions) > 0:
        if actions.ndim == 1:
            ax.step(
                episodes,
                actions,
                color="dodgerblue",
                linewidth=1.0,
                alpha=0.6,  # Soft horizontal/step line
                where="post",
            )
            ax.plot(
                episodes,
                actions,
                color="dodgerblue",
                marker="o",
                markersize=4,
                linestyle="None",  # Hard marker at each episode
                label=action_labels[0],
            )
        else:
            num_dims = actions.shape[1]
            colors = ["dodgerblue", "darkorange", "forestgreen", "purple"]
            for i in range(num_dims):
                color = colors[i % len(colors)]
                label = (
                    action_labels[i] if i < len(action_labels) else f"Action {i + 1}"
                )
                # Soft step line for the trend
                ax.step(
                    episodes,
                    actions[:, i],
                    color=color,
                    linewidth=1.0,
                    alpha=0.5,
                    where="post",
                )
                # Hard marker for exact value reading per episode
                ax.plot(
                    episodes,
                    actions[:, i],
                    color=color,
                    marker="o",
                    markersize=4,
                    linestyle="None",
                    label=label,
                )
    else:
        ax.text(
            0.5,
            0.5,
            "No Action History Logged",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.minorticks_on()

    ax.set_title("(a) Policy Actions", loc="left")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Action Value")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces integer ticks
    ax.tick_params(which="major", length=6, width=0.8)
    ax.tick_params(which="minor", length=3, width=0.6)
    ax.set_ylim(bottom=trainer.ann_config.min_action, top=None)
    ax.grid(False)
    safe_legend(ax)

    if standalone:
        if save_dir is None:
            save_dir = Path(trainer.master_path) / "diagnostics"
        save_dir.mkdir(parents=True, exist_ok=True)
        _save_figure(fig, save_dir / "diagnostic_actions")
        plt.close(fig)


def plot_critic_sensitivity(
    trainer: TD3Trainer | BanditTrainer,
    ax: plt.Axes | None = None,
    save_dir: Path | None = None,
) -> None:
    """Plot exploration noise and critic Q-value sensitivity."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 5))

    n_random_episodes = trainer.ann_config.n_random_episodes

    q_hist = getattr(trainer, "q_sensitivity_history", [])
    if len(q_hist) > 0:
        ax.plot(
            np.arange(1 + n_random_episodes, len(q_hist) + 1 + n_random_episodes),
            q_hist,
            color="chocolate",
            linewidth=2,
            label=r"$\Delta Q = \vert{}Q(s_0, 1.1) - Q(s_0, 1.0)\vert{}$",
        )
        ax.set_title(r"(b) Critic Q-Value Sensitivity ($\Delta Q$)", loc="left")
    else:
        ax.text(
            0.5,
            0.5,
            "No Q-Sensitivity Data Logged\n(Steps < stochastic_timesteps)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(r"(b) Critic Sensitivity $\Delta Q$", loc="left")
    ax.set_xlabel("Episode")
    ax.set_ylabel(r"$\Delta Q$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces integer ticks
    ax.grid(False)
    safe_legend(ax)

    if standalone:
        if save_dir is None:
            save_dir = Path(trainer.master_path) / "diagnostics"
        save_dir.mkdir(parents=True, exist_ok=True)
        _save_figure(fig, save_dir / "diagnostic_critic_sensitivity")
        plt.close(fig)


def plot_optimization_losses(
    trainer: TD3Trainer | BanditTrainer,
    ax: plt.Axes | None = None,
    save_dir: Path | None = None,
) -> None:
    """Plot critic and actor gradient optimization losses."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 5))

    c_loss = getattr(trainer, "critic_loss_history", [])
    a_loss = getattr(trainer, "actor_loss_history", [])
    has_loss = False

    if len(c_loss) > 0:
        ax.plot(c_loss, label="Critic Loss", color="#B9314F", alpha=0.80)
        has_loss = True
    if len(a_loss) > 0:
        ax.plot(a_loss, label="Actor Loss", color="#47329A", alpha=0.80)
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

    ax.set_title("(c) TD3 Optimization Losses", loc="left")
    ax.set_xlabel("Training Gradient Steps")
    ax.set_ylabel("Loss")
    ax.grid(False)
    safe_legend(ax)

    if standalone:
        if save_dir is None:
            save_dir = Path(trainer.master_path) / "diagnostics"
        save_dir.mkdir(parents=True, exist_ok=True)
        _save_figure(fig, save_dir / "diagnostic_optimization_losses")
        plt.close(fig)


def plot_reward_trajectory(
    trainer: TD3Trainer | BanditTrainer,
    ax: plt.Axes | None = None,
    save_dir: Path | None = None,
) -> None:
    """Plot episode reward trajectory."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 5))

    ep_rewards = getattr(trainer, "reward_history", [])
    if len(ep_rewards) > 0:
        ax.plot(
            np.arange(1, len(ep_rewards) + 1),
            ep_rewards,
            color="navy",
            linewidth=2,
            label="Clipped Episode Reward",
        )
    ax.set_title("(d) Episode Reward Trajectory", loc="left")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces integer ticks
    ax.grid(False)
    safe_legend(ax, loc="best")

    if standalone:
        if save_dir is None:
            save_dir = Path(trainer.master_path) / "diagnostics"
        save_dir.mkdir(parents=True, exist_ok=True)
        _save_figure(fig, save_dir / "diagnostic_reward_trajectory")
        plt.close(fig)


# =========================================================================
# Main Orchestrator & Detailed Corrections
# =========================================================================


def plot_diagnostic_metrics(
    trainer: TD3Trainer | BanditTrainer, save_dir: Path | None = None
) -> None:
    """Plot comprehensive step-level and episode-level diagnostic metrics with noise visualization."""
    if save_dir is None:
        save_dir = Path(trainer.master_path) / "diagnostics"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Safely retrieve environment reference
    env: EnvironmentForcingDNS | EnvironmentProof | None = getattr(trainer, "env", None)
    if env is None:
        raise TypeError("Environment is not retrieved from the trainer correctly.")

    episodes = np.arange(1, getattr(trainer, "episodes_ran", 0) + 1)
    ann_config = getattr(trainer, "ann_config", None)
    proof_mode = getattr(ann_config, "proof_mode", None)

    # 1. Generate standalone individual plots
    plot_actions(trainer, save_dir=save_dir)
    plot_critic_sensitivity(trainer, save_dir=save_dir)
    plot_optimization_losses(trainer, save_dir=save_dir)
    plot_reward_trajectory(trainer, save_dir=save_dir)

    # 2. Main Diagnostic Dashboard (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=False)

    plot_actions(trainer, ax=axes[0, 0])
    plot_critic_sensitivity(trainer, ax=axes[0, 1])
    plot_optimization_losses(trainer, ax=axes[1, 0])
    plot_reward_trajectory(trainer, ax=axes[1, 1])

    plt.tight_layout()
    dashboard_path = save_dir / "diagnostics"
    _save_figure(fig, dashboard_path)
    plt.close(fig)

    print(
        f"[DIAGNOSTICS] Diagnostic dashboard plot saved to: {dashboard_path}.png / .pdf"
    )

    # 3. Detailed Corrections Plot (Proof Modes)
    _plot_actions_history(
        env=env,
        episodes=episodes,
        proof_mode=proof_mode,
        save_dir=save_dir,
    )


def _plot_actions_history(
    env: EnvironmentForcingDNS | EnvironmentProof | EnvironmentBanditTraining,
    episodes: NDArray,
    proof_mode: str | None,
    save_dir: Path,
) -> None:
    """Internal helper to plot diagnostic corrections for proof modes b, c, or d."""
    if proof_mode not in ("a", "b", "c", "d"):
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    if proof_mode == "a":
        applied_actions = np.asarray(getattr(env, "action_history", []))
        if len(applied_actions) == 0:
            plt.close(fig)
            print("[DIAGNOSTICS] NO APPLIED ACTIONS FOUND")
            return

        actions_1, actions_2 = applied_actions[:, 0], applied_actions[:, 1]

        num_episodes = len(episodes)
        episodes = np.linspace(start=1, stop=num_episodes + 1, num=len(episodes))
        steps_actions = np.linspace(start=1, stop=num_episodes, num=len(actions_1))

        # --- Subplot 1: Action Dimension 1 (Blue Palette) ---
        ax1.plot(
            steps_actions,
            actions_1,
            color="dodgerblue",
        )

        ax1.set_ylabel("Value", fontweight="bold")
        ax1.set_title(
            "(a) Diagnostic: Corrections vs Target Actions (Action 1)",
            fontsize=11,
            fontweight="bold",
        )

        # --- Subplot 2: Action Dimension 2 (Orange Palette) ---
        ax2.plot(
            steps_actions,
            actions_2,
            color="sandybrown",
        )
        ax2.set_xlabel("Episodes / Steps", fontweight="bold")
        ax2.set_ylabel("Value", fontweight="bold")
        ax2.set_title(
            "(b) Diagnostic: Corrections vs Target Actions (Action 2)",
            fontsize=11,
            fontweight="bold",
        )

    if proof_mode in ("b", "c"):
        applied_actions = np.asarray(getattr(env, "applied_actions_history", []))
        if len(applied_actions) == 0:
            plt.close(fig)
            print("[DIAGNOSTICS] NO APPLIED ACTIONS FOUND")
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
                print("[DIAGNOSTICS] NO TARGET ACTIONS FOUND")
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
            "(a) Diagnostic: Corrections vs Target Actions (Action 1)",
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
            "(b) Diagnostic: Corrections vs Target Actions (Action 2)",
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

        steps_actions = np.linspace(
            start=0, stop=len(episodes), num=len(moving_action_mean_1)
        )

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
            "(a) Diagnostic: Correction Mean vs Target Mean (Action 1)",
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
            "(b) Diagnostic: Correction Mean vs Target Mean (Action 2)",
            fontsize=11,
            fontweight="bold",
        )

    for ax in (ax1, ax2):
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    corrections_path = save_dir / "diagnostic_corrections"
    _save_figure(fig, corrections_path)
    plt.close(fig)

    print(f"[DIAGNOSTICS] Corrections plot saved to: {corrections_path}.png / .pdf")

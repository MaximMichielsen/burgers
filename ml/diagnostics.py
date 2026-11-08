"""Diagnostics for ANN training. Able to handle both regular training and proof training."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy
from numpy.typing import NDArray

from ml.bandit.ann import BanditConfig
from ml.original.environment import EnvironmentForcingDNS
from proofing.action_based.environment import EnvironmentProof
from proofing.contextual_bandit_reformulation.environment import (
    EnvironmentBanditTraining,
)
from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    from ml.original.training import TD3Trainer
    from ml.bandit.training import BanditTrainer

# NOTE: "figure.autolayout" was removed. It conflicts with constrained_layout,
# which is what the profile plot uses.
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 0.8,
    }
)

# Size of one panel; matches a single panel of the 1x2 profile figure.
_PANEL_SIZE = (5, 3.5)
_DPI = 300
# Distance (points) from the bottom of the axes to the legend; must clear tick labels + xlabel.
_LEGEND_GAP_PT = 29

# Colours for the actions plot (colour-blind safe, Okabe-Ito).
_ACTION_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
_ACTION_LABELS = ["Advection", "Viscosity", "Third term", "Fourth term"]


def moving_average(data: np.ndarray | list[float], window_size: int = 50) -> np.ndarray:
    """Calculate moving average with constant NaN padding to retain input length."""
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) < window_size or window_size <= 1:
        return arr

    ma = np.convolve(arr, np.ones(window_size) / window_size, mode="valid")
    pad_left = (window_size - 1) // 2
    pad_right = window_size - 1 - pad_left
    return np.pad(ma, (pad_left, pad_right), mode="constant", constant_values=np.nan)


def rolling_mean_std(
    data: np.ndarray | list[float], window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Centered rolling mean and std; the window shrinks at the edges (no NaN padding)."""
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    half = max(1, window // 2)
    mean = np.empty(n)
    std = np.empty(n)
    for i in range(n):
        seg = arr[max(0, i - half) : min(n, i + half + 1)]
        mean[i] = seg.mean()
        std[i] = seg.std()
    return mean, std


# =========================================================================
# Style helpers (same look as the profile comparison plot)
# =========================================================================


def _style_axes(ax: plt.Axes) -> None:
    """Ticks, minor ticks and spines as in the profile plot."""
    ax.minorticks_on()
    ax.tick_params(
        which="both", direction="in", top=True, right=True, length=4, width=0.8
    )
    ax.tick_params(which="minor", length=2, width=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.grid(False)


def _set_title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, fontsize=10, pad=6, loc="left")


def _set_labels(ax: plt.Axes, xlabel: str | None, ylabel: str | None) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=10)


def _zero_line(ax: plt.Axes) -> None:
    """Dotted zero line, only when zero lies inside a linear y-range."""
    y0, y1 = ax.get_ylim()
    if ax.get_yscale() == "linear" and y0 < 0 < y1:
        ax.axhline(0, color="black", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)


def safe_legend(ax: plt.Axes, loc: str = "best", **kwargs: Any) -> None:
    """Add a frameless legend to an axis only if labeled handles exist."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        kwargs.setdefault("frameon", False)
        ax.legend(loc=loc, **kwargs)


def _legend_below(ax: plt.Axes, handles: list | None = None, ncol: int = 3) -> None:
    """Frameless legend centered below the axes (same placement for every panel)."""
    if handles is None:
        handles, _ = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        bbox_transform=offset_copy(
            ax.transAxes, fig=ax.figure, y=-_LEGEND_GAP_PT, units="points"
        ),
        borderaxespad=0.0,
        ncol=min(ncol, len(handles)),
        frameon=False,
        fontsize=8,
        columnspacing=1.0,
        handlelength=1.6,
    )


def _save_figure(fig: plt.Figure, base_path: Path, dpi: int = _DPI) -> None:
    """Save a figure as both PNG and PDF formats."""
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")


def _standalone_fig(size: tuple[float, float] = _PANEL_SIZE):
    return plt.subplots(figsize=size, dpi=_DPI, constrained_layout=True)


# =========================================================================
# Subplot Routines (Standalone + Dashboard Rendering)
# =========================================================================


def plot_actions(
    trainer: TD3Trainer | BanditTrainer,
    ax: plt.Axes | None = None,
    save_dir: Path | None = None,
    window: int | None = None,
) -> None:
    """Plot the action trend per dimension.

    Raw actions are drawn as faint dots; the trend is a centered rolling mean
    with a +/- 1 std band. The exploration phase is shaded.
    `window` defaults to ~10% of the history (minimum 5).
    """
    standalone = ax is None
    if standalone:
        fig, ax = _standalone_fig((_PANEL_SIZE[0], 4.0))

    action_history = getattr(trainer, "action_history", [])
    if len(action_history) == 0:
        env = getattr(trainer, "env", None)
        action_history = getattr(env, "action_history", [])

    random_exploration_episodes = trainer.ann_config.n_random_episodes
    actions = np.asarray(action_history, dtype=np.float64)

    if len(actions) > 0:
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)
        n = len(actions)
        episodes = np.arange(1, n + 1)
        win = window if window is not None else max(5, n // 10)

        if random_exploration_episodes != 0:
            ax.axvspan(
                0,
                random_exploration_episodes
                * trainer.ann_config.n_agent_steps_per_episode,
                color="gray",
                alpha=0.10,
                linewidth=0,
            )
            ax.axvline(
                x=random_exploration_episodes
                * trainer.ann_config.n_agent_steps_per_episode,
                color="gray",
                alpha=0.8,
                linestyle="--",
                linewidth=0.8,
            )

        handles: list = []
        for i in range(actions.shape[1]):
            color = _ACTION_COLORS[i % len(_ACTION_COLORS)]
            label = _ACTION_LABELS[i] if i < len(_ACTION_LABELS) else f"Action {i + 1}"
            mean, std = rolling_mean_std(actions[:, i], win)
            ax.scatter(
                episodes,
                actions[:, i],
                s=8,
                color=color,
                alpha=0.5,
                linewidths=0,
                zorder=2,
                clip_on=False,
            )
            ax.fill_between(
                episodes,
                mean - std,
                mean + std,
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=1,
            )
            (line,) = ax.plot(
                episodes, mean, color=color, linewidth=1.8, zorder=3, label=label
            )
            handles.append(line)

        handles.append(
            Line2D(
                [],
                [],
                color="gray",
                marker="o",
                markersize=3,
                linestyle="None",
                alpha=0.6,
                label="Raw action",
            )
        )
        handles.append(
            Patch(
                facecolor="gray",
                alpha=0.25,
                linewidth=0,
                label=rf"Rolling mean $\pm 1\sigma$ ({win})",
            )
        )
        if random_exploration_episodes != 0:
            handles.append(
                Line2D(
                    [],
                    [],
                    color="gray",
                    linestyle="--",
                    linewidth=0.8,
                    label=f"End exploration (Ep. {random_exploration_episodes})",
                )
            )
        if n > 1:
            ax.set_xlim(1, n)
        _legend_below(ax, handles, ncol=3)
    else:
        ax.text(
            0.5,
            0.5,
            "No Action History Logged",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    if not standalone:
        _set_title(ax, "(a) Policy Actions")
    x_label = "Episode" if isinstance(trainer.ann_config, BanditConfig) else "Step"
    _set_labels(ax, x_label, "Action Value")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
    ax.set_ylim(bottom=trainer.ann_config.min_action, top=None)
    _style_axes(ax)

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
        fig, ax = _standalone_fig()

    n_random_episodes = trainer.ann_config.n_random_episodes

    q_hist = getattr(trainer, "q_sensitivity_history", [])
    if len(q_hist) > 0:
        x = np.arange(1 + n_random_episodes, len(q_hist) + 1 + n_random_episodes)
        ax.plot(
            x,
            q_hist,
            color="chocolate",
            linewidth=2,
            label=r"$\Delta Q = \vert{}Q(s_0, 1.1) - Q(s_0, 1.0)\vert{}$",
        )
        if len(x) > 1:
            ax.set_xlim(x[0], x[-1])
        if not standalone:
            _set_title(ax, r"(b) Critic Q-Value Sensitivity ($\Delta Q$)")
    else:
        ax.text(
            0.5,
            0.5,
            "No Q-Sensitivity Data Logged\n(Steps < stochastic_timesteps)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        if not standalone:
            _set_title(ax, r"(b) Critic Sensitivity $\Delta Q$")
    _set_labels(ax, "Episode", r"$\Delta Q$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
    _legend_below(ax)
    _style_axes(ax)

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
        fig, ax = _standalone_fig()

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
        n_max = max(len(c_loss), len(a_loss))
        if n_max > 1:
            ax.set_xlim(0, n_max - 1)

    if not standalone:
        _set_title(ax, "(c) TD3 Optimization Losses")
    _set_labels(ax, "Training Gradient Steps", "Loss")
    _legend_below(ax)
    _style_axes(ax)

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
        fig, ax = _standalone_fig()

    ep_rewards = getattr(trainer, "reward_history", [])
    if len(ep_rewards) > 0:
        ax.plot(
            np.arange(1, len(ep_rewards) + 1),
            ep_rewards,
            color="navy",
            linewidth=2,
            label="Clipped Episode Reward",
        )
        if len(ep_rewards) > 1:
            ax.set_xlim(1, len(ep_rewards))
    if not standalone:
        _set_title(ax, "(d) Episode Reward Trajectory")
    x_label = "Episode" if isinstance(trainer.ann_config, BanditConfig) else "Step"
    _set_labels(ax, x_label, "Reward")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
    _zero_line(ax)
    _legend_below(ax)
    _style_axes(ax)

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
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 7), dpi=_DPI, constrained_layout=True, sharex=False
    )

    plot_actions(trainer, ax=axes[0, 0])
    plot_critic_sensitivity(trainer, ax=axes[0, 1])
    plot_optimization_losses(trainer, ax=axes[1, 0])
    plot_reward_trajectory(trainer, ax=axes[1, 1])

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
    """Internal helper to plot diagnostic corrections for proof modes a, b, c, or d."""
    if proof_mode not in ("a", "b", "c", "d"):
        return

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), dpi=_DPI, sharex=True, constrained_layout=True
    )

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

        _set_labels(ax1, None, "Value")
        _set_title(ax1, "(a) Diagnostic: Corrections vs Target Actions (Action 1)")

        # --- Subplot 2: Action Dimension 2 (Orange Palette) ---
        ax2.plot(
            steps_actions,
            actions_2,
            color="sandybrown",
        )
        _set_labels(ax2, "Episodes / Steps", "Value")
        _set_title(ax2, "(b) Diagnostic: Corrections vs Target Actions (Action 2)")

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
            alpha=0.7,
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
        _set_labels(ax1, None, "Value")
        _set_title(ax1, "(a) Diagnostic: Corrections vs Target Actions (Action 1)")

        # --- Subplot 2: Action Dimension 2 (Orange Palette) ---
        ax2.plot(
            steps_actions,
            applied_actions_2,
            color="sandybrown",
            alpha=0.7,
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
        _set_labels(ax2, "Episodes / Steps", "Value")
        _set_title(ax2, "(b) Diagnostic: Corrections vs Target Actions (Action 2)")
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
        _set_labels(ax1, None, "Value")
        _set_title(ax1, "(a) Diagnostic: Correction Mean vs Target Mean (Action 1)")

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
        _set_labels(ax2, "Episodes / Steps", "Value")
        _set_title(ax2, "(b) Diagnostic: Correction Mean vs Target Mean (Action 2)")

    for ax in (ax1, ax2):
        _style_axes(ax)
        safe_legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    corrections_path = save_dir / "diagnostic_corrections"
    _save_figure(fig, corrections_path)
    plt.close(fig)

    print(f"[DIAGNOSTICS] Corrections plot saved to: {corrections_path}.png / .pdf")

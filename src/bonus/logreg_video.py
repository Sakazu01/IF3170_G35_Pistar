from pathlib import Path
from subprocess import CalledProcessError
from tempfile import TemporaryDirectory

import numpy as np

from src.models.logistic_regression import (
    LogisticRegressionScratch,
    binary_data,
    finite_scalar,
    positive_int,
)


def make_log_loss_video(
    X, y, path, feature_index=0, learning_rate=0.2, max_iter=1000,
    random_state=42, fps=15, max_frames=100, grid_points=65,
):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    X, y = binary_data(X, y)
    if (
        isinstance(feature_index, (bool, np.bool_))
        or not isinstance(feature_index, (int, np.integer))
        or not 0 <= feature_index < X.shape[1]
    ):
        raise ValueError("feature_index must identify one input feature")
    fps = finite_scalar(fps, "fps", positive=True)
    max_frames = positive_int(max_frames, "max_frames")
    grid_points = positive_int(grid_points, "grid_points")
    if max_frames < 2 or grid_points < 8:
        raise ValueError("max_frames must be at least 2 and grid_points at least 8")
    path = Path(path)
    if path.suffix.lower() not in (".mp4", ".gif"):
        raise ValueError("path must end in .mp4 or .gif")
    single_X = X[:, [feature_index]]
    model = LogisticRegressionScratch(
        learning_rate=learning_rate, max_iter=max_iter, random_state=random_state,
        store_parameters=True,
    ).fit(single_X, y)
    trajectory = model.parameter_history_
    losses = np.array([
        model.binary_cross_entropy(y, theta[0] + single_X[:, 0] * theta[1])
        for theta in trajectory
    ])
    if not np.allclose(losses, model.history_["train_log_loss"], rtol=1e-10, atol=1e-12):
        raise ValueError("Recorded trajectory does not match the contour training objective")
    low, high = trajectory.min(axis=0), trajectory.max(axis=0)
    margin = np.maximum((high - low) * 0.3, 0.7)
    theta0, theta1 = np.meshgrid(
        np.linspace(low[0] - margin[0], high[0] + margin[0], grid_points),
        np.linspace(low[1] - margin[1], high[1] + margin[1], grid_points),
    )
    pairs = np.column_stack([theta0.ravel(), theta1.ravel()])
    surface = np.empty(len(pairs))
    for start in range(0, len(pairs), 128):
        block = pairs[start:start + 128]
        logits = block[:, :1] + block[:, 1:] * single_X[:, 0]
        surface[start:start + len(block)] = np.logaddexp(0.0, (1 - 2 * y) * logits).mean(axis=1)
    surface = surface.reshape(theta0.shape)
    fig, (ax, loss_ax) = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    levels = np.linspace(np.sqrt(surface.min()), np.sqrt(surface.max()), 19)**2
    filled = ax.contourf(theta0, theta1, surface, levels=levels, cmap="viridis")
    lines = ax.contour(theta0, theta1, surface, levels=levels[::2], colors="white", linewidths=0.5, alpha=0.6)
    ax.clabel(lines, inline=True, fontsize=7, fmt="%.2f")
    fig.colorbar(filled, ax=ax, label="Mean training log-loss")
    ax.scatter(*trajectory[0], c="white", edgecolors="black", s=65, label="Start", zorder=5)
    ax.scatter(*trajectory[-1], c="#ffd166", edgecolors="black", marker="*", s=240, label="Final", zorder=7)
    trail, = ax.plot([], [], color="#ff5876", linewidth=2, label="Actual training path", zorder=4)
    current, = ax.plot([], [], "o", markerfacecolor="none", markeredgecolor="#ff5876", markeredgewidth=2, markersize=9, label="Current", zorder=6)
    ax.set(xlabel=r"$\theta_0$ (intercept)", ylabel=rf"$\theta_1$ (feature {feature_index})")
    ax.legend(fontsize=8)
    loss_ax.plot(model.history_["epoch"], losses, color="#ddd5e8", label="Full history")
    loss_trace, = loss_ax.plot([], [], color="#3b6ea5", linewidth=2, label="Observed so far")
    loss_ax.set(xlabel="Epoch", ylabel="Mean training log-loss", xlim=(0, max(model.n_iter_, 1)))
    loss_ax.grid(alpha=0.2)
    loss_ax.legend(fontsize=8)
    title = fig.suptitle("Actual two-parameter Logistic Regression training")
    frames = np.unique(np.linspace(0, model.n_iter_, min(max_frames, len(trajectory)), dtype=int))

    def update(epoch):
        trail.set_data(trajectory[:epoch + 1, 0], trajectory[:epoch + 1, 1])
        current.set_data([trajectory[epoch, 0]], [trajectory[epoch, 1]])
        loss_trace.set_data(np.arange(epoch + 1), losses[:epoch + 1])
        title.set_text(f"Log-loss contour and actual training path | epoch {epoch} | loss {losses[epoch]:.5f}")
        return trail, current, loss_trace, title

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=False)
    fallback_reason = None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(prefix=".lr_video_", dir=path.parent) as temporary:
            temporary = Path(temporary)
            rendered = temporary / path.name
            if path.suffix.lower() == ".mp4":
                try:
                    import imageio_ffmpeg

                    with matplotlib.rc_context({"animation.ffmpeg_path": imageio_ffmpeg.get_ffmpeg_exe()}):
                        animation.save(str(rendered), writer=FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]), dpi=100)
                except (ImportError, OSError, RuntimeError, CalledProcessError) as error:
                    fallback_reason = str(error)
                    path = path.with_suffix(".gif")
                    rendered = temporary / path.name
                    animation.save(str(rendered), writer=PillowWriter(fps=fps), dpi=100)
            else:
                animation.save(str(rendered), writer=PillowWriter(fps=fps), dpi=100)
            rendered.replace(path)
    finally:
        plt.close(fig)
    return {
        "path": path, "model": model, "feature_index": int(feature_index),
        "frames": len(frames), "frame_epochs": frames.tolist(), "fps": fps,
        "fallback_reason": fallback_reason,
    }

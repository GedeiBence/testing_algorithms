# ==================== helper.py ====================
import matplotlib.pyplot as plt
from IPython import display
import numpy as np

# Turn on interactive mode
plt.ion()

def plot(scores, mean_scores, window=50):
    """
    Live plot of episode scores and moving average.
    Works in Jupyter and in regular Python scripts.
    """
    if len(scores) == 0:
        return

    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    plt.title("PPO Training - 1DOF System")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    # Raw scores
    plt.plot(scores, label="Episode Reward", alpha=0.6)

    # Moving average
    if len(mean_scores) > 0:
        plt.plot(mean_scores, label=f"Moving Avg ({window})", linewidth=2)

    # Show last values
    plt.text(len(scores) - 1, scores[-1], f"{scores[-1]:.1f}", fontsize=9)
    if len(mean_scores) > 0:
        plt.text(len(mean_scores) - 1, mean_scores[-1], f"{mean_scores[-1]:.1f}", fontsize=9)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.01)


def compute_moving_average(scores, window=50):
    """Helper to compute moving average of scores."""
    if len(scores) < window:
        return [np.mean(scores)] * len(scores)
    return np.convolve(scores, np.ones(window)/window, mode='valid').tolist()


def save_plot(scores, mean_scores, filename="training_curve.png"):
    """Save the current training curve to a file."""
    plt.figure(figsize=(10, 5))
    plt.title("PPO Training - 1DOF System")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.plot(scores, label="Episode Reward", alpha=0.6)
    if len(mean_scores) > 0:
        plt.plot(mean_scores, label="Moving Average", linewidth=2)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Plot saved to {filename}")
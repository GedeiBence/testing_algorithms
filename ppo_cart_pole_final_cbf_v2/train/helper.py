# ==================== helper.py ====================
import matplotlib
matplotlib.use("TkAgg")   # or "Agg" if you only want to save, not show
import matplotlib.pyplot as plt
import numpy as np

# rest of the file stays exactly the same
# Create the figure once
plt.ion()
fig, ax = plt.subplots(figsize=(8, 5))
line1, = ax.plot([], [], label="Episode Reward", alpha=0.6)
line2, = ax.plot([], [], label="Moving Avg", linewidth=2)
text1 = ax.text(0, 0, "", fontsize=9)
text2 = ax.text(0, 0, "", fontsize=9)
ax.set_title("PPO Training - Cart-Pole System")
ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=False)

def plot(scores, mean_scores, window=50):
    if len(scores) == 0:
        return

    line1.set_data(range(len(scores)), scores)
    if len(mean_scores) > 0:
        line2.set_data(range(len(mean_scores)), mean_scores)

    # Update last-value texts
    text1.set_position((len(scores) - 1, scores[-1]))
    text1.set_text(f"{scores[-1]:.1f}")
    if len(mean_scores) > 0:
        text2.set_position((len(mean_scores) - 1, mean_scores[-1]))
        text2.set_text(f"{mean_scores[-1]:.1f}")

    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)


def compute_moving_average(scores, window=50):
    if len(scores) < window:
        return [np.mean(scores)] * len(scores)
    return np.convolve(scores, np.ones(window)/window, mode='valid').tolist()


def save_plot(scores, mean_scores, filename="training_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.title("PPO Training - Cart-Pole System")
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
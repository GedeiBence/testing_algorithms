# ==================== test.py ====================
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from pendulum_sim import Pendulum
from models import ActorNetwork

def run_continuous_and_save_first(actor, episodes=10, save_first=True):
    """
    Run the pendulum continuously (like test_pendulum_gui).
    Saves the first episode's fi vs time to disk and plots it.
    """
    actor.eval()
    pend = Pendulum()
    pend.gui()

    print("\n" + "="*50)
    print("CONTINUOUS TEST + SAVE FIRST EPISODE")
    print("="*50)

    first_times = []
    first_fis = []
    first_saved = False

    for ep in range(episodes):
        state = pend.reset_2()
        center_deg = np.rad2deg(pend.wrap_angle(pend.barrier_center))
        width_deg = np.rad2deg(pend.barrier_width)
        print(f"Barrier: center = {center_deg:6.1f}° | width = {width_deg:5.1f}°")
        total_reward = 0.0
        last_log_time = 0.0
        episode_times = []
        episode_fis = []

        print(f"\n--- Episode {ep+1} ---")
        print(f"{'Time':>6} | {'fi (deg)':>8} | {'fi_dot':>8} | {'Torque':>8} | {'Reward':>8}")

        while True:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _ = actor.get_action(state_t, deterministic=True)
            torque = action.item()

            state, reward, done, _ = pend.step(torque)
            total_reward += reward
            pend.update_gui()
            time.sleep(0.02)          # slow down for visuals

            # Record for this episode
            episode_times.append(pend.time)
            episode_fis.append(np.rad2deg(pend.fi))

            # Log every 0.1 s
            if pend.time - last_log_time >= 0.1 or done:
                fi_deg = np.rad2deg(pend.fi)
                print(f"{pend.time:6.2f} | {fi_deg:8.2f} | {pend.fi_dot:8.3f} | {torque:8.2f} | {reward:8.2f}")
                last_log_time = pend.time

            if done or pend.stop_training:
                print(f"Episode ended | Time: {pend.time:.2f}s | Total Reward: {total_reward:.1f}")
                break

        # Save the first episode
        if save_first and not first_saved and ep == 0:
            first_times = episode_times
            first_fis = episode_fis
            first_saved = True
            np.savez("first_episode.npz", times=first_times, fis=first_fis)
            print("First episode saved to first_episode.npz")

        if pend.stop_training:
            break

    # Plot the first episode
    if first_saved:
        plt.figure(figsize=(10, 4))
        plt.plot(first_times, first_fis, label="fi (deg)", color="orange")
        plt.axhline(180, color="gray", linestyle="--", label="upright")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle fi (degrees)")
        plt.title("First episode – angle vs time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("first_episode_fi.png", dpi=150)
        plt.show()
        print("Plot saved to first_episode_fi.png")

    actor.train()
    print("Test finished.")


if __name__ == "__main__":
    # Load the trained actor
    actor = ActorNetwork(state_dim=4, action_dim=1, hidden_size=128)
    actor.load_state_dict(torch.load("pendulum_actor_v2.pth", map_location="cpu"))
    actor.eval()

    run_continuous_and_save_first(actor, episodes=10) 
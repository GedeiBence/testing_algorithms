# ==================== test_policy.py ====================
import torch
import numpy as np
import time
from train.models import ActorNetwork
from train.scara_sim import SCARA

def test_policy(episodes=5):
    # Load trained actor
    actor = ActorNetwork(state_dim=6, action_dim=2, hidden_size=128)
    actor.load_state_dict(torch.load("scara_actor.pth", map_location="cpu"))
    actor.eval()

    scara = SCARA()
    scara.gui()

    print("\n" + "="*50)
    print("TESTING TRAINED POLICY")
    print("="*50)

    for ep in range(episodes):
        state = scara.reset()
        total_reward = 0.0
        print(f"\n--- Episode {ep+1} ---")
        print(f"{'Time':>6} | {'α (deg)':>8} | {'β (deg)':>8} | {'Reward':>8}")

        while True:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _ = actor.get_action(state_t, deterministic=True)
            action = action.squeeze(0).numpy()

            next_state, reward, done, _ = scara.step(action)
            total_reward += reward

            scara.update_gui()
            time.sleep(0.02)

            if int(scara.time * 10) % 5 == 0 or done:
                print(f"{scara.time:6.2f} | {np.rad2deg(scara.alpha):8.1f} | {np.rad2deg(scara.beta):8.1f} | {reward:8.2f}")

            state = next_state
            if done or scara.stop_training:
                print(f"Episode ended | Time: {scara.time:.2f}s | Total Reward: {total_reward:.1f}")
                break

        if scara.stop_training:
            break

    print("GUI test finished.")
    actor.train()

if __name__ == "__main__":
    test_policy(episodes=5)
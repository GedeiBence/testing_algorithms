# ==================== test.py ====================
import torch
import numpy as np
from sim import SystemSim
from models import ActorNetwork

def test_policy(actor, episodes=5, max_steps=120, desired_x=1.0):
    """
    Run the trained policy and print the trajectory.
    Resets automatically when the episode ends.
    """
    sim = SystemSim(desired_x=desired_x)
    actor.eval()

    print("\n" + "="*50)
    print("TESTING TRAINED POLICY")
    print("="*50)

    for ep in range(episodes):
        state = sim.reset()
        total_reward = 0.0
        positions = []

        print(f"\n--- Episode {ep+1} ---")
        for step in range(max_steps):
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _ = actor.get_action(state_t, deterministic=True)
            action = action.item()

            next_state, reward, done, _ = sim.step(action)
            total_reward += reward
            positions.append(sim.x)

            # Print every 20 steps so you can see the movement
            if step % 20 == 0 or done:
                print(f"  t={sim.current_time:.2f}s | x={sim.x:.3f} | x_dot={sim.x_dot:.3f} | F={action:.2f}")

            state = next_state
            if done:
                print(f"  → Episode ended | Total reward: {total_reward:.2f}")
                break

        if not done:
            print(f"  → Max steps reached | Total reward: {total_reward:.2f}")

    actor.train()
    print("\n" + "="*50)
# ==================== test_policy.py ====================
import torch
import time
import numpy as np
from models import ActorNetwork
from cart_pole_sim import CartPole


def test_policy(actor, episodes=10):
    actor.eval()
    pole = CartPole()
    pole.gui()

    # Test boundaries (override sim defaults)
    MAX_X       = 3.4
    MAX_X_DOT   = 15.0
    MAX_FI      = np.deg2rad(180)#pole.max_angle      # currently 12°
    MAX_FI_DOT  = 15.0
    MAX_TIME    = 10.0

    print("\n" + "="*50)
    print("TESTING TRAINED POLICY")
    print(f"Boundaries: |x|≤{MAX_X}, |fi|≤{np.rad2deg(MAX_FI):.0f}°, t≤{MAX_TIME}s")
    print("="*50)

    for ep in range(episodes):
        state = pole.reset_test()
        total_reward = 0.0
        last_log = 0.0

        print(f"\n--- Episode {ep+1} ---")
        print("  Time |      x | fi (deg) |   fi_dot |    Force")

        # Apply the test boundaries
        pole.max_x = MAX_X
        pole.max_angle = MAX_FI
        pole.max_time = MAX_TIME

        while True:
            if (abs(pole.x) >= MAX_X or
                abs(pole.x_dot) > MAX_X_DOT or
                abs(pole.err()) > MAX_FI or
                abs(pole.fi_dot) > MAX_FI_DOT or
                pole.time >= MAX_TIME or
                pole.stop_training):
                break

            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _ = actor.get_action(state_t, deterministic=True)
            force = np.clip(action.item(), -pole.max_force, pole.max_force)

            state, reward, done, _ = pole.step(force)
            total_reward += reward

            pole.update_gui()
            time.sleep(0.01)

            if pole.time - last_log >= 0.1:
                print(f"  {pole.time:5.2f} | {pole.x:6.2f} | {np.rad2deg(pole.fi):7.2f} | {pole.fi_dot:8.3f} | {force:8.2f}")
                last_log = pole.time

            if done:
                break

        print(f"Ended | Time: {pole.time:.2f}s | Total reward: {total_reward:.1f}")

        if pole.stop_training:
            break

    print("Test finished.")
    actor.train()


if __name__ == "__main__":
   

    actor = ActorNetwork(state_dim=5, action_dim=1, hidden_size=128)

    # STAGE 1.
    #actor.load_state_dict(torch.load("cartpole_actor.pth", map_location="cpu"))

    # STAGE 2.
    #actor.load_state_dict(torch.load("actor_stage2.pth", map_location="cpu"))

    # STAGE 3.
    #actor.load_state_dict(torch.load("actor_stage3.pth", map_location="cpu"))

    # STAGE 4.
    #actor.load_state_dict(torch.load("actor_stage4.pth", map_location="cpu"))

    # Ultimate
    actor.load_state_dict(torch.load("actor_ultimate2.pth", map_location="cpu"))

    actor.eval()

    print("Policy loaded. Starting test...")
    test_policy(actor, episodes=20)
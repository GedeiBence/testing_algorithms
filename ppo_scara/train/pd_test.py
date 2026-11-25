# ==================== pd_test.py ====================
import numpy as np
import time
import matplotlib.pyplot as plt
from scara_sim import SCARA

def pd_control(scara, alpha_des, beta_des, kp=80.0, kd=15.0):
    t1 = kp * (alpha_des - scara.alpha) - kd * scara.alpha_dot
    t2 = kp * (beta_des - scara.beta) - kd * scara.beta_dot
    return np.array([t1, t2])

def run_pd_test(kp=80.0, kd=15.0, steps=300):
    scara = SCARA()
    scara.gui()

    alpha_des = np.deg2rad(40)
    beta_des = np.deg2rad(-25)

    times = []
    alpha_hist = []
    beta_hist = []

    print(f"Testing Kp={kp}, Kd={kd}")

    for step in range(steps):
        torque = pd_control(scara, alpha_des, beta_des, kp, kd)
        scara.step(torque)

        times.append(scara.time)
        alpha_hist.append(np.rad2deg(scara.alpha))
        beta_hist.append(np.rad2deg(scara.beta))

        scara.update_gui()
        time.sleep(0.01)

        if scara.stop_training:
            break

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(times, alpha_hist, label=f"α (target={np.rad2deg(alpha_des):.1f}°)")
    plt.plot(times, beta_hist, label=f"β (target={np.rad2deg(beta_des):.1f}°)")
    plt.axhline(np.rad2deg(alpha_des), color='C0', linestyle='--', alpha=0.6)
    plt.axhline(np.rad2deg(beta_des), color='C1', linestyle='--', alpha=0.6)
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [deg]")
    plt.title(f"PD Response  |  Kp={kp}  Kd={kd}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Try different values here
    run_pd_test(kp=90, kd=78)

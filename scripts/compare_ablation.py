import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from robot_simulator.envs import RobotNavEnv
from robot_simulator.obstacles import CircularObstacle
from robot_simulator.simulation import render_arena


def run_ablation_comparison():
    print("=================================================================")
    print("Étude d'Ablation : Modèle Baseline vs Modèle Lissé (Smooth PPO)")
    print("=================================================================")

    # 1. Mission setup
    goal = (4.5, 4.0)
    start_x, start_y = -4.0, -3.5
    init_theta = float(np.arctan2(goal[1] - start_y, goal[0] - start_x))
    start = (start_x, start_y, init_theta)

    # 6 obstacles arranged in a diagonal slalom corridor along the direct path
    obstacles = [
        CircularObstacle(x=-2.5, y=-1.5, radius=0.75),
        CircularObstacle(x=-1.2, y=-2.8, radius=0.70),
        CircularObstacle(x=-0.2, y=-0.8, radius=0.75),
        CircularObstacle(x=0.8, y=1.2, radius=0.80),
        CircularObstacle(x=2.0, y=-0.2, radius=0.75),
        CircularObstacle(x=2.8, y=2.2, radius=0.80),
    ]

    models_info = [
        {
            "name": "Baseline (PPO Standard)",
            "path": "models/ppo_robot_nav",
            "color": "crimson",
            "style": "--",
        },
        {
            "name": "Physically Smooth (Avec Pénalités)",
            "path": "models/ppo_robot_nav_smooth",
            "color": "dodgerblue",
            "style": "-",
        },
    ]

    results = {}

    for m_info in models_info:
        name = m_info["name"]
        path = m_info["path"]
        print(f"\nSimulation en cours pour : {name}...")

        model = PPO.load(path, device="cpu")
        env = RobotNavEnv(max_steps=250, goal=goal, obstacles=obstacles, randomize=False)
        obs, _ = env.reset()
        env.robot_x, env.robot_y, env.robot_theta = start
        env.prev_distance = np.hypot(goal[0] - env.robot_x, goal[1] - env.robot_y)
        obs = env._get_obs()

        x_hist = [env.robot_x]
        y_hist = [env.robot_y]
        v_hist = []
        w_hist = []
        d_hist = [env.prev_distance]

        for step in range(250):
            action, _ = model.predict(obs, deterministic=True)
            v, omega = float(action[0]), float(action[1])
            v_hist.append(v)
            w_hist.append(omega)

            obs, reward, terminated, truncated, info = env.step(action)
            x_hist.append(env.robot_x)
            y_hist.append(env.robot_y)
            d_hist.append(np.hypot(goal[0] - env.robot_x, goal[1] - env.robot_y))

            if terminated or truncated:
                break

        results[name] = {
            "x": x_hist,
            "y": y_hist,
            "v": v_hist,
            "w": w_hist,
            "d": d_hist,
            "steps": len(v_hist),
            "final_v": v_hist[-1] if v_hist else 0.0,
            "avg_v": float(np.mean(v_hist)) if v_hist else 0.0,
            "color": m_info["color"],
            "style": m_info["style"],
            "success": info.get("is_success", False),
        }
        print(f"-> {name} : {len(v_hist)} pas | Vitesse finale : {v_hist[-1]:.2f} m/s | Succès : {info.get('is_success', False)}")

    # 2. Plotting (3-panel academic comparison)
    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0])

    ax_traj = fig.add_subplot(gs[:, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_w = fig.add_subplot(gs[1, 1])

    # Trajectory Plot
    render_arena(ax_traj, obstacles=obstacles, goal=goal, start=(start[0], start[1]), bounds=(-6, 6, -5, 6))
    ax_traj.plot([start[0], goal[0]], [start[1], goal[1]], "k:", alpha=0.4, label="Ligne droite directe")
    for name, data in results.items():
        ax_traj.plot(data["x"], data["y"], color=data["color"], linestyle=data["style"], linewidth=2.3, label=name)

    ax_traj.set_title("1. Comparaison des Trajectoires", fontsize=11, fontweight="bold")
    ax_traj.legend(loc="upper left")

    # Linear Velocity Profile v(t)
    for name, data in results.items():
        time_steps = np.arange(len(data["v"])) * 0.1  # In seconds (dt = 0.1s)
        ax_v.plot(time_steps, data["v"], color=data["color"], linestyle=data["style"], linewidth=2.0, label=f"{name} (Fin: {data['final_v']:.2f} m/s)")

    ax_v.set_title("2. Profil de Vitesse Linéaire v(t) [Freinage à l'Arrivée]", fontsize=11, fontweight="bold")
    ax_v.set_xlabel("Temps (s)")
    ax_v.set_ylabel("Vitesse (m/s)")
    ax_v.grid(True, linestyle="--", alpha=0.55)
    ax_v.legend(loc="lower left")

    # Angular Velocity Profile omega(t)
    for name, data in results.items():
        time_steps = np.arange(len(data["w"])) * 0.1
        ax_w.plot(time_steps, data["w"], color=data["color"], linestyle=data["style"], linewidth=1.8, label=name)

    ax_w.set_title("3. Profil de Vitesse Angulaire ω(t) [Fluidité des Virages]", fontsize=11, fontweight="bold")
    ax_w.set_xlabel("Temps (s)")
    ax_w.set_ylabel("Vitesse angulaire (rad/s)")
    ax_w.grid(True, linestyle="--", alpha=0.55)
    ax_w.legend(loc="upper right")

    plt.suptitle("Étude d'Ablation : Impact des Pénalités Physiques sur la Navigation RL", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    os.makedirs("assets", exist_ok=True)
    out_path = os.path.join("assets", "ablation_velocity_comparison.png")
    plt.savefig(out_path, dpi=300)
    print(f"\nGraphique sauvegardé dans {out_path}")
    plt.show()


if __name__ == "__main__":
    run_ablation_comparison()

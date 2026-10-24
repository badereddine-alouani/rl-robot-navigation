import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from robot_simulator.envs import RobotNavEnv
from robot_simulator.obstacles import CircularObstacle
from robot_simulator.lidar import simulate_lidar


def run_waypoint_following():
    # 1. Define reference path (sequence of waypoints)
    waypoints = [
        (3.5, 2.0),    # W1
        (6.0, 6.0),    # W2
        (1.0, 7.0),    # W3
        (-3.0, 4.0),   # W4 (Final Goal)
    ]

    # 2. Place obstacles along the corridors to test reactive obstacle avoidance
    obstacles = [
        CircularObstacle(x=2.0, y=0.5, radius=0.7),   # Near corridor to W1
        CircularObstacle(x=5.0, y=3.8, radius=0.7),   # Between W1 and W2
        CircularObstacle(x=3.5, y=7.0, radius=0.7),   # Between W2 and W3
    ]

    # Initialize environment with first waypoint as active goal
    env = RobotNavEnv(
        max_steps=200,
        goal=waypoints[0],
        obstacles=obstacles,
        randomize=False
    )

    model = PPO.load("models/ppo_robot_nav", device="cpu")
    obs, info = env.reset()

    # Track histories
    x_history = [env.robot_x]
    y_history = [env.robot_y]

    current_wp_idx = 0
    total_steps = 0
    max_total_steps = 800

    print("=================================================================")
    print("Suivi de chemin avec évitement d'obstacles par PPO")
    print("=================================================================")
    print(f"-> Suivi du Waypoint 1/{len(waypoints)} : {waypoints[0]}")

    while total_steps < max_total_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        x_history.append(env.robot_x)
        y_history.append(env.robot_y)
        total_steps += 1

        # Check collision
        if reward <= -50.0:
            print(f"Collision détectée à l'étape {total_steps} !")
            break

        # Check if current waypoint reached (capture tolerance)
        dist_to_current = np.hypot(env.goal[0] - env.robot_x, env.goal[1] - env.robot_y)
        if dist_to_current <= env.goal_tolerance + 0.15:
            print(f"   [V] Waypoint {current_wp_idx + 1} atteint ({env.robot_x:.2f}, {env.robot_y:.2f}) à l'étape {total_steps} !")
            current_wp_idx += 1

            if current_wp_idx >= len(waypoints):
                print("=================================================================")
                print("TOUS LES WAYPOINTS ONT ÉTÉ ATTEINTS AVEC SUCCÈS !")
                print("=================================================================")
                break

            # Switch target to next waypoint
            next_wp = waypoints[current_wp_idx]
            print(f"-> Suivi du Waypoint {current_wp_idx + 1}/{len(waypoints)} : {next_wp}")
            env.goal = next_wp
            env.prev_distance = np.hypot(env.goal[0] - env.robot_x, env.goal[1] - env.robot_y)
            env.current_step = 0  # Reset step horizon for next segment
            obs = env._get_obs()

    # 3. Visualization
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_aspect("equal")
    ax.set_xlim(-5, 8)
    ax.set_ylim(-2, 9)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Reference Path (Chemin de référence)
    path_x = [0.0] + [wp[0] for wp in waypoints]
    path_y = [0.0] + [wp[1] for wp in waypoints]
    ax.plot(path_x, path_y, "k--", linewidth=1.8, alpha=0.7, label="Chemin de référence (Waypoints)")

    # Plot Waypoints
    for i, (wx, wy) in enumerate(waypoints):
        ax.scatter(wx, wy, c="green", s=110, zorder=5)
        ax.text(wx + 0.25, wy + 0.15, f"W{i+1}", fontsize=11, fontweight="bold", color="darkgreen")

    # Plot Obstacles
    for obs_item in obstacles:
        circle = plt.Circle((obs_item.x, obs_item.y), obs_item.radius, color="gray", alpha=0.55)
        ax.add_patch(circle)

    # Plot Robot Trajectory
    ax.plot(x_history, y_history, "b-", linewidth=2.4, label="Trajectoire réelle (PPO)")
    ax.scatter(x_history[0], y_history[0], c="blue", s=90, zorder=6, label="Départ (0,0)")
    ax.scatter(x_history[-1], y_history[-1], c="orange", s=90, zorder=6, label="Arrivée")

    # Final LiDAR beams
    scan = simulate_lidar(env.robot_x, env.robot_y, env.robot_theta, obstacles, env.num_beams, env.max_range, env.fov)
    angles = np.linspace(0, env.fov, env.num_beams, endpoint=False) + env.robot_theta
    for dist, angle in zip(scan, angles):
        x_end = env.robot_x + dist * np.cos(angle)
        y_end = env.robot_y + dist * np.sin(angle)
        ax.plot([env.robot_x, x_end], [env.robot_y, y_end], "r:", alpha=0.4)

    ax.set_title("Planification et suivi de chemin d'un robot mobile à roues par apprentissage par renforcement", fontsize=11)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper left")

    os.makedirs("assets", exist_ok=True)
    out_path = os.path.join("assets", "trajectory_waypoints.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Graphique sauvegardé dans {out_path}")
    plt.show()


if __name__ == "__main__":
    run_waypoint_following()

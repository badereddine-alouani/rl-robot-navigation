import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from robot_simulator.envs import RobotNavEnv
from robot_simulator.obstacles import CircularObstacle
from robot_simulator.a_star import AStarPlanner
from robot_simulator.lidar import simulate_lidar


def main():
    print("=================================================================")
    print("Navigation Hiérarchique : Planification Globale (A*) + Suivi Local (PPO)")
    print("=================================================================")

    # 1. Mission setup
    start = (-4.0, -3.5)
    goal = (4.5, 4.0)

    # 2. Obstacles in the environment (Diagonal Barrier)
    obstacles = [
        CircularObstacle(x=-2.5, y=-1.5, radius=0.75),
        CircularObstacle(x=-1.2, y=-2.8, radius=0.70),
        CircularObstacle(x=-0.2, y=-0.8, radius=0.75),
        CircularObstacle(x=0.8, y=1.2, radius=0.80),
        CircularObstacle(x=2.0, y=-0.2, radius=0.75),
        CircularObstacle(x=2.8, y=2.2, radius=0.80),
    ]

    # 3. Global Planning with A*
    print("\n[Étape 1] Calcul du chemin global avec A*...")
    planner = AStarPlanner(
        obstacles=obstacles,
        bounds=(-8.0, 8.0, -8.0, 8.0),
        resolution=0.35,
        robot_radius=0.2,
        safety_margin=0.6,
    )
    
    # A* generates the waypoints automatically
    waypoints = planner.plan(start, goal)
    if waypoints is None:
        print("Erreur : Aucun chemin trouvé par A* !")
        return

    print(f"-> A* a calculé avec succès {len(waypoints)} waypoints de référence :")
    for i, wp in enumerate(waypoints):
        print(f"   W{i} : ({wp[0]:.2f}, {wp[1]:.2f})")

    # 4. Local Execution with PPO (The Driver)
    print("\n[Étape 2] Suivi de chemin réactif avec l'agent PPO...")
    # Initialize environment at start position with first waypoint as goal
    env = RobotNavEnv(
        max_steps=500,
        goal=waypoints[1],  # First target after start
        obstacles=obstacles,
        randomize=False
    )
    env.reset()
    env.robot_x = start[0]
    env.robot_y = start[1]
    
    # Initial heading oriented towards the final goal
    env.robot_theta = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
    env.prev_distance = np.hypot(waypoints[1][0] - env.robot_x, waypoints[1][1] - env.robot_y)
    obs = env._get_obs()

    model = PPO.load("models/ppo_robot_nav_smooth", device="cpu")

    x_history = [env.robot_x]
    y_history = [env.robot_y]

    current_wp_idx = 1
    total_steps = 0
    max_total_steps = 1000

    while total_steps < max_total_steps and current_wp_idx < len(waypoints):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        x_history.append(env.robot_x)
        y_history.append(env.robot_y)
        total_steps += 1

        if reward <= -50.0:
            print(f"Collision détectée à l'étape {total_steps} !")
            break

        dist_to_current = np.hypot(env.goal[0] - env.robot_x, env.goal[1] - env.robot_y)
        # Acceptance radius: 0.75m for intermediate waypoints (smooth corner cutting),
        # and goal_tolerance for the final mission destination.
        acceptance_radius = env.goal_tolerance if current_wp_idx == len(waypoints) - 1 else 0.75

        if dist_to_current <= acceptance_radius:
            print(f"   [V] Waypoint {current_wp_idx} atteint à ({env.robot_x:.2f}, {env.robot_y:.2f}) à l'étape {total_steps}")
            current_wp_idx += 1

            if current_wp_idx >= len(waypoints):
                print("\n=================================================================")
                print(f"MISSION ACCOMPLIE : Objectif final atteint en {total_steps} étapes !")
                print("=================================================================")
                break

            # Switch target to next waypoint
            next_wp = waypoints[current_wp_idx]
            env.goal = next_wp
            env.prev_distance = np.hypot(env.goal[0] - env.robot_x, env.goal[1] - env.robot_y)
            env.current_step = 0
            obs = env._get_obs()

    # 5. Visualization
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_aspect("equal")
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-6.5, 6.5)
    ax.grid(True, linestyle="--", alpha=0.55)

    # Plot A* Global Path
    astar_x = [wp[0] for wp in waypoints]
    astar_y = [wp[1] for wp in waypoints]
    ax.plot(astar_x, astar_y, "g--", linewidth=2.0, alpha=0.75, label="Chemin global (A*)")

    for i, (wx, wy) in enumerate(waypoints):
        ax.scatter(wx, wy, c="darkgreen", s=80, zorder=5)
        ax.text(wx + 0.25, wy + 0.15, f"W{i}", fontsize=10, fontweight="bold", color="darkgreen")

    # Plot Obstacles
    for obs_item in obstacles:
        circle = plt.Circle((obs_item.x, obs_item.y), obs_item.radius, color="gray", alpha=0.55)
        ax.add_patch(circle)

    # Plot Actual Trajectory (PPO)
    ax.plot(x_history, y_history, "b-", linewidth=2.5, label="Trajectoire réelle du robot (PPO)")
    ax.scatter(x_history[0], y_history[0], c="blue", s=110, zorder=6, label=f"Départ {start}")
    ax.scatter(x_history[-1], y_history[-1], c="orange", s=110, zorder=6, label=f"Arrivée ({x_history[-1]:.2f}, {y_history[-1]:.2f})")

    # Plot Final LiDAR beams
    scan = simulate_lidar(env.robot_x, env.robot_y, env.robot_theta, obstacles, env.num_beams, env.max_range, env.fov)
    angles = np.linspace(0, env.fov, env.num_beams, endpoint=False) + env.robot_theta
    for dist, angle in zip(scan, angles):
        x_end = env.robot_x + dist * np.cos(angle)
        y_end = env.robot_y + dist * np.sin(angle)
        ax.plot([env.robot_x, x_end], [env.robot_y, y_end], "r:", alpha=0.35)

    ax.set_title("Navigation Hiérarchique : Planification Globale (A*) + Suivi Local Réactif (PPO)", fontsize=11)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper left")

    os.makedirs("assets", exist_ok=True)
    out_path = os.path.join("assets", "trajectory_hierarchical.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Graphique sauvegardé dans {out_path}")
    plt.show()


if __name__ == "__main__":
    main()

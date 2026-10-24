import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from robot_simulator.envs import RobotNavEnv
from robot_simulator.obstacles import CircularObstacle
from robot_simulator.a_star import AStarPlanner
from robot_simulator.lidar import simulate_lidar


def run_comparison():
    print("=================================================================")
    print("Comparaison : Navigation Réactive (Sans A*) vs Hiérarchique (Avec A*)")
    print("=================================================================")

    start = (-5.0, -4.0)
    goal = (6.0, 5.0)

    obstacles = [
        CircularObstacle(x=-2.0, y=-2.0, radius=1.0),
        CircularObstacle(x=1.0, y=-0.5, radius=1.1),
        CircularObstacle(x=-1.0, y=2.0, radius=0.9),
        CircularObstacle(x=3.5, y=3.0, radius=1.0),
    ]

    model = PPO.load("models/ppo_robot_nav", device="cpu")

    # -------------------------------------------------------------
    # 1. RUN WITHOUT PATH PLANNING (Purely Reactive PPO)
    # -------------------------------------------------------------
    print("\n[1] Exécution SANS planification (PPO réactif pur vers le but)...")
    env_reactive = RobotNavEnv(max_steps=500, goal=goal, obstacles=obstacles, randomize=False)
    env_reactive.reset()
    env_reactive.robot_x, env_reactive.robot_y = start
    env_reactive.robot_theta = 0.0
    env_reactive.prev_distance = np.hypot(goal[0] - env_reactive.robot_x, goal[1] - env_reactive.robot_y)
    obs_reactive = env_reactive._get_obs()

    hist_no_a_x, hist_no_a_y = [start[0]], [start[1]]
    for step in range(500):
        action, _ = model.predict(obs_reactive, deterministic=True)
        obs_reactive, reward, terminated, truncated, _ = env_reactive.step(action)
        hist_no_a_x.append(env_reactive.robot_x)
        hist_no_a_y.append(env_reactive.robot_y)

        if reward <= -50.0:
            print(f"   Collision SANS A* à l'étape {step} !")
            break
        if np.hypot(goal[0] - env_reactive.robot_x, goal[1] - env_reactive.robot_y) <= env_reactive.goal_tolerance:
            print(f"   [V] But atteint SANS A* en {step} étapes !")
            break

    # -------------------------------------------------------------
    # 2. RUN WITH PATH PLANNING (A* Global Planner + PPO Tracker)
    # -------------------------------------------------------------
    print("\n[2] Exécution AVEC planification (A* Global + PPO Local)...")
    planner = AStarPlanner(obstacles=obstacles, bounds=(-8.0, 8.0, -8.0, 8.0), resolution=0.35, safety_margin=0.6)
    waypoints = planner.plan(start, goal)

    env_hierarchical = RobotNavEnv(max_steps=200, goal=waypoints[1], obstacles=obstacles, randomize=False)
    env_hierarchical.reset()
    env_hierarchical.robot_x, env_hierarchical.robot_y = start
    env_hierarchical.robot_theta = 0.0
    env_hierarchical.prev_distance = np.hypot(waypoints[1][0] - env_hierarchical.robot_x, waypoints[1][1] - env_hierarchical.robot_y)
    obs_hierarchical = env_hierarchical._get_obs()

    hist_a_x, hist_a_y = [start[0]], [start[1]]
    wp_idx = 1
    for step in range(800):
        action, _ = model.predict(obs_hierarchical, deterministic=True)
        obs_hierarchical, reward, terminated, truncated, _ = env_hierarchical.step(action)
        hist_a_x.append(env_hierarchical.robot_x)
        hist_a_y.append(env_hierarchical.robot_y)

        if reward <= -50.0:
            print(f"   Collision AVEC A* à l'étape {step} !")
            break

        dist = np.hypot(env_hierarchical.goal[0] - env_hierarchical.robot_x, env_hierarchical.goal[1] - env_hierarchical.robot_y)
        if dist <= env_hierarchical.goal_tolerance + 0.25:
            wp_idx += 1
            if wp_idx >= len(waypoints):
                print(f"   [V] But final atteint AVEC A* en {step} étapes !")
                break
            env_hierarchical.goal = waypoints[wp_idx]
            env_hierarchical.prev_distance = np.hypot(env_hierarchical.goal[0] - env_hierarchical.robot_x, env_hierarchical.goal[1] - env_hierarchical.robot_y)
            env_hierarchical.current_step = 0
            obs_hierarchical = env_hierarchical._get_obs()

    # Calculate metrics
    dist_no_a = np.sum(np.hypot(np.diff(hist_no_a_x), np.diff(hist_no_a_y)))
    dist_with_a = np.sum(np.hypot(np.diff(hist_a_x), np.diff(hist_a_y)))

    print("\n-----------------------------------------------------------------")
    print(f"RÉSULTATS COMPARATIFS :")
    print(f"- SANS A* (Réactif pur) : {len(hist_no_a_x)} étapes | Longueur : {dist_no_a:.2f} m")
    print(f"- AVEC A* (Hiérarchique) : {len(hist_a_x)} étapes | Longueur : {dist_with_a:.2f} m")
    print("-----------------------------------------------------------------")

    # -------------------------------------------------------------
    # 3. SIDE-BY-SIDE VISUALIZATION
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5))

    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.set_xlim(-8, 8)
        ax.set_ylim(-7, 8)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        for obs in obstacles:
            ax.add_patch(plt.Circle((obs.x, obs.y), obs.radius, color="gray", alpha=0.55))

    # Subplot 1: SANS A*
    ax1.plot([start[0], goal[0]], [start[1], goal[1]], "k:", alpha=0.4, label="Ligne droite (Non planifiée)")
    ax1.plot(hist_no_a_x, hist_no_a_y, color="crimson", linewidth=2.5, label="Trajectoire Réactive (PPO pur)")
    ax1.scatter(start[0], start[1], c="blue", s=90, zorder=6, label=f"Départ {start}")
    ax1.scatter(goal[0], goal[1], c="green", s=100, zorder=6, label=f"But {goal}")
    ax1.set_title(f"1. SANS Planification (Réactif pur)\nLongueur : {dist_no_a:.2f} m | {len(hist_no_a_x)} pas", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left")

    # Subplot 2: AVEC A*
    astar_x = [wp[0] for wp in waypoints]
    astar_y = [wp[1] for wp in waypoints]
    ax2.plot(astar_x, astar_y, "g--", linewidth=1.8, alpha=0.7, label="Chemin global (A*)")
    for i, (wx, wy) in enumerate(waypoints):
        ax2.scatter(wx, wy, c="darkgreen", s=70, zorder=5)
        ax2.text(wx + 0.25, wy + 0.15, f"W{i}", fontsize=9, fontweight="bold", color="darkgreen")
    ax2.plot(hist_a_x, hist_a_y, color="dodgerblue", linewidth=2.5, label="Trajectoire Guidée (A* + PPO)")
    ax2.scatter(start[0], start[1], c="blue", s=90, zorder=6, label=f"Départ {start}")
    ax2.scatter(goal[0], goal[1], c="green", s=100, zorder=6, label=f"But {goal}")
    ax2.set_title(f"2. AVEC Planification (A* + PPO)\nLongueur : {dist_with_a:.2f} m | {len(hist_a_x)} pas", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left")

    plt.suptitle("Impact de la Planification Globale sur la Navigation Autonome", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig("comparison_with_and_without_planning.png", dpi=300)
    print("Graphique comparatif sauvegardé dans comparison_with_and_without_planning.png")
    plt.show()


if __name__ == "__main__":
    run_comparison()

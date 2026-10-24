import json
import numpy as np
from stable_baselines3 import PPO
from robot_simulator.envs import RobotNavEnv
from robot_simulator.obstacles import CircularObstacle


def generate_scenario(rng, num_obstacles, robot_radius=0.2, goal_tolerance=0.2):
    """Generates valid, non-overlapping start, goal, and obstacles with safe clearance."""
    rx = float(rng.uniform(-3.5, 3.5))
    ry = float(rng.uniform(-3.5, 3.5))
    r_theta = float(rng.uniform(-np.pi, np.pi))

    while True:
        gx = float(rng.uniform(-5.5, 5.5))
        gy = float(rng.uniform(-5.5, 5.5))
        if np.hypot(gx - rx, gy - ry) >= 3.5:
            goal = (gx, gy)
            break

    obstacles = []
    for _ in range(num_obstacles):
        for _ in range(200):  # Rejection attempts
            ox = float(rng.uniform(-5.0, 5.0))
            oy = float(rng.uniform(-5.0, 5.0))
            r = float(rng.uniform(0.6, 0.9))

            dist_robot = np.hypot(ox - rx, oy - ry)
            dist_goal = np.hypot(ox - gx, oy - gy)

            if dist_robot >= (r + robot_radius + 0.6) and dist_goal >= (r + goal_tolerance + 0.6):
                obstacles.append(CircularObstacle(ox, oy, r))
                break

    return (rx, ry, r_theta), goal, obstacles


def run_benchmark(num_episodes_per_level=25, seed=4):
    print("=========================================================================")
    print("  ÉVALUATION ET BENCHMARK QUANTITATIF DE LA NAVIGATION AUTONOME PPO     ")
    print("=========================================================================")
    print(f"Modèle : models/ppo_robot_nav.zip | Seed : {seed}")
    print(f"Épisodes par niveau : {num_episodes_per_level} (Total : {num_episodes_per_level * 4} runs)\n")

    model = PPO.load("models/ppo_robot_nav", device="cpu")
    rng = np.random.default_rng(seed)

    difficulty_levels = [
        {"name": "1 Obstacle (Facile)", "num_obs": 1},
        {"name": "2 Obstacles (Moyen)", "num_obs": 2},
        {"name": "3 Obstacles (Nominal)", "num_obs": 3},
        {"name": "5 Obstacles (Haute Densité)", "num_obs": 5},
    ]

    benchmark_summary = {}

    for level in difficulty_levels:
        name = level["name"]
        num_obs = level["num_obs"]
        print(f"-> Test en cours : {name}...")

        success_count = 0
        collision_count = 0
        timeout_count = 0
        steps_list = []
        path_efficiencies = []
        path_lengths = []

        for ep in range(num_episodes_per_level):
            start_pose, goal, obstacles = generate_scenario(rng, num_obs)

            env = RobotNavEnv(
                max_steps=200,
                goal=goal,
                obstacles=obstacles,
                randomize=False
            )
            obs, _ = env.reset()
            env.robot_x, env.robot_y, env.robot_theta = start_pose
            env.prev_distance = np.hypot(goal[0] - env.robot_x, goal[1] - env.robot_y)
            obs = env._get_obs()

            straight_dist = env.prev_distance
            x_hist = [env.robot_x]
            y_hist = [env.robot_y]

            terminated = False
            truncated = False
            info = {}
            ep_steps = 0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                x_hist.append(env.robot_x)
                y_hist.append(env.robot_y)
                ep_steps += 1

            if info.get("is_success", False):
                success_count += 1
                actual_length = float(np.sum(np.hypot(np.diff(x_hist), np.diff(y_hist))))
                efficiency = float(straight_dist / max(actual_length, 1e-4))
                steps_list.append(ep_steps)
                path_lengths.append(actual_length)
                path_efficiencies.append(min(efficiency, 1.0))
            elif info.get("is_collision", False):
                collision_count += 1
            else:  # Truncated without collision or reaching goal (timeout)
                timeout_count += 1

        success_rate = (success_count / num_episodes_per_level) * 100
        collision_rate = (collision_count / num_episodes_per_level) * 100
        timeout_rate = (timeout_count / num_episodes_per_level) * 100
        mean_steps = float(np.mean(steps_list)) if steps_list else 0.0
        mean_eff = float(np.mean(path_efficiencies)) * 100 if path_efficiencies else 0.0
        mean_len = float(np.mean(path_lengths)) if path_lengths else 0.0

        benchmark_summary[name] = {
            "obstacles": num_obs,
            "episodes": num_episodes_per_level,
            "success_rate_pct": success_rate,
            "collision_rate_pct": collision_rate,
            "timeout_rate_pct": timeout_rate,
            "mean_steps": mean_steps,
            "mean_path_length_m": mean_len,
            "mean_path_efficiency_pct": mean_eff,
        }

    # Print ASCII Report
    print("\n" + "=" * 80)
    print("                    RAPPORT FINAL DU BENCHMARK QUANTITATIF                      ")
    print("=" * 80)
    header = f"{'Difficulté':<26} | {'Succès':<8} | {'Collision':<9} | {'Timeout':<7} | {'Pas Moy.':<8} | {'Efficacité'}"
    print(header)
    print("-" * 80)

    for name, stats in benchmark_summary.items():
        print(
            f"{name:<26} | "
            f"{stats['success_rate_pct']:>6.1f}% | "
            f"{stats['collision_rate_pct']:>8.1f}% | "
            f"{stats['timeout_rate_pct']:>6.1f}% | "
            f"{stats['mean_steps']:>8.1f} | "
            f"{stats['mean_path_efficiency_pct']:>9.1f}%"
        )
    print("=" * 80)

    # Save to JSON
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=4, ensure_ascii=False)
    print("\nDonnées brutes sauvegardées dans benchmark_results.json")


if __name__ == "__main__":
    run_benchmark()

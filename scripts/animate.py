import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from stable_baselines3 import PPO
from robot_simulator.envs import RobotNavEnv
from robot_simulator.obstacles import CircularObstacle
from robot_simulator.lidar import simulate_lidar


def record_episode(env, model, max_steps=250, start_pose=(-4.0, -3.5, 0.0)):
    """Runs a single episode and records all state histories for animation."""
    obs, info = env.reset()
    if start_pose is not None:
        env.robot_x, env.robot_y, env.robot_theta = start_pose
        env.prev_distance = np.hypot(env.goal[0] - env.robot_x, env.goal[1] - env.robot_y)
        obs = env._get_obs()
    
    frames = []
    total_reward = 0.0
    
    for step in range(max_steps):
        # 1. Compute current LiDAR beams for visualization
        scan = simulate_lidar(
            env.robot_x, env.robot_y, env.robot_theta,
            env.obstacles, env.num_beams, env.max_range, env.fov
        )
        
        # 2. Predict action
        action, _ = model.predict(obs, deterministic=True)
        v, omega = float(action[0]), float(action[1])
        
        # 3. Store frame snapshot
        dist_to_goal = np.hypot(env.goal[0] - env.robot_x, env.goal[1] - env.robot_y)
        frame_data = {
            "step": step,
            "x": env.robot_x,
            "y": env.robot_y,
            "theta": env.robot_theta,
            "v": v,
            "omega": omega,
            "scan": list(scan),
            "dist": dist_to_goal,
            "goal": env.goal,
            "obstacles": list(env.obstacles),
        }
        frames.append(frame_data)
        
        # 4. Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            # Append final state
            final_scan = simulate_lidar(
                env.robot_x, env.robot_y, env.robot_theta,
                env.obstacles, env.num_beams, env.max_range, env.fov
            )
            frames.append({
                "step": step + 1,
                "x": env.robot_x,
                "y": env.robot_y,
                "theta": env.robot_theta,
                "v": 0.0,
                "omega": 0.0,
                "scan": list(final_scan),
                "dist": np.hypot(env.goal[0] - env.robot_x, env.goal[1] - env.robot_y),
                "goal": env.goal,
                "obstacles": list(env.obstacles),
            })
            break
            
    success = env.prev_distance <= env.goal_tolerance
    return frames, success, total_reward


def create_animation(frames, save_gif=True, gif_filename="robot_navigation.gif"):
    """Generates an interactive real-time 2D animation with live LiDAR beams and telemetry."""
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.set_aspect("equal")
    ax.set_xlim(-7.0, 7.0)
    ax.set_ylim(-7.0, 7.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title("Navigation Autonome Réactive par PPO - Animation Temps Réel", fontsize=12, fontweight="bold")

    initial_frame = frames[0]
    goal = initial_frame["goal"]
    obstacles = initial_frame["obstacles"]

    # 1. Plot static obstacles
    for obs in obstacles:
        circle = Circle((obs.x, obs.y), obs.radius, color="gray", alpha=0.6, zorder=2)
        ax.add_patch(circle)
        # Add subtle contour ring
        ring = Circle((obs.x, obs.y), obs.radius + 0.2, color="gray", fill=False, linestyle=":", alpha=0.4)
        ax.add_patch(ring)

    # 2. Plot Goal
    goal_circle = Circle(goal, 0.2, color="limegreen", alpha=0.3, zorder=3)
    ax.add_patch(goal_circle)
    ax.scatter(goal[0], goal[1], c="green", s=120, zorder=5, label=f"Objectif ({goal[0]:.1f}, {goal[1]:.1f})")

    # 3. Dynamic Elements
    # Trajectory path line
    (traj_line,) = ax.plot([], [], "b-", linewidth=2.2, alpha=0.85, label="Trajectoire parcourue", zorder=4)
    
    # Robot body
    robot_radius = 0.2
    robot_body = Circle((initial_frame["x"], initial_frame["y"]), robot_radius, color="dodgerblue", ec="black", lw=1.5, zorder=6, label="Robot mobile")
    ax.add_patch(robot_body)
    
    # Robot heading indicator (front nose)
    (heading_line,) = ax.plot([], [], "k-", linewidth=2.5, zorder=7)

    # LiDAR beam lines (16 beams)
    lidar_lines = [ax.plot([], [], "r:", linewidth=1.1, alpha=0.65, zorder=3)[0] for _ in range(16)]

    # HUD Telemetry Text box
    hud_template = (
        "Étape: {:03d}\n"
        "Vitesse: {:.2f} m/s\n"
        "Rotation: {:+.2f} rad/s\n"
        "Dist. But: {:.2f} m\n"
        "Statut: {}"
    )
    hud_box = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
        zorder=10
    )

    ax.legend(loc="upper right", framealpha=0.9)

    # Cumulative trajectory arrays
    traj_x = []
    traj_y = []

    fov = 2 * np.pi
    num_beams = 16
    beam_angles_rel = np.linspace(-fov / 2, fov / 2, num_beams, endpoint=False)

    def init():
        traj_line.set_data([], [])
        heading_line.set_data([], [])
        for line in lidar_lines:
            line.set_data([], [])
        hud_box.set_text("")
        return [traj_line, robot_body, heading_line, hud_box] + lidar_lines

    def update(frame_idx):
        f = frames[frame_idx]
        rx, ry, theta = f["x"], f["y"], f["theta"]
        
        # Update trajectory history
        traj_x.append(rx)
        traj_y.append(ry)
        traj_line.set_data(traj_x, traj_y)

        # Update robot circle position
        robot_body.set_center((rx, ry))

        # Update heading vector
        hx = [rx, rx + robot_radius * 1.5 * np.cos(theta)]
        hy = [ry, ry + robot_radius * 1.5 * np.sin(theta)]
        heading_line.set_data(hx, hy)

        # Update 16 LiDAR beams
        scan = f["scan"]
        for i, (dist, angle_rel) in enumerate(zip(scan, beam_angles_rel)):
            global_angle = theta + angle_rel
            bx = [rx, rx + dist * np.cos(global_angle)]
            by = [ry, ry + dist * np.sin(global_angle)]
            lidar_lines[i].set_data(bx, by)

        # Determine status
        is_last = (frame_idx == len(frames) - 1)
        if is_last and f["dist"] <= 0.25:
            status = "BUT ATTEINT !"
            robot_body.set_facecolor("limegreen")
        elif is_last:
            status = "TERMINÉ"
        else:
            status = "EN NAVIGATION"

        # Update Telemetry HUD
        hud_box.set_text(hud_template.format(f["step"], f["v"], f["omega"], f["dist"], status))

        return [traj_line, robot_body, heading_line, hud_box] + lidar_lines

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), init_func=init,
        interval=70, blit=True, repeat=True
    )

    if save_gif:
        print(f"Génération du GIF animé ({gif_filename})...")
        anim.save(gif_filename, writer="pillow", fps=15)
        print(f"GIF sauvegardé avec succès : {gif_filename}")

    plt.tight_layout()
    plt.show()


def main():
    print("=================================================================")
    print("Enregistrement d'un épisode de navigation PPO avec LiDAR animé")
    print("=================================================================")

    # Initialize environment with custom obstacles
    obstacles = [
        CircularObstacle(x=-1.0, y=-0.8, radius=0.9),
        CircularObstacle(x=1.8, y=1.2, radius=0.9),
        CircularObstacle(x=-1.5, y=2.0, radius=0.8),
    ]

    start_pose = (-4.0, -3.5, 0.0)
    goal = (4.5, 4.0)

    env = RobotNavEnv(
        max_steps=200,
        goal=goal,
        obstacles=obstacles,
        randomize=False
    )

    model = PPO.load("models/ppo_robot_nav", device="cpu")

    print(f"Départ : ({start_pose[0]}, {start_pose[1]}) -> Objectif : {goal}")
    print("Exécution de la politique neuronale...")
    frames, success, reward = record_episode(env, model, start_pose=start_pose)

    print(f"Épisode terminé en {len(frames)} étapes. Succès : {success} | Récompense totale : {reward:.2f}")
    
    # Launch animation and generate GIF
    create_animation(frames, save_gif=True, gif_filename="robot_navigation.gif")


if __name__ == "__main__":
    main()

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from stable_baselines3 import PPO
from robot_simulator.envs import RobotNavEnv
from robot_simulator.obstacles import CircularObstacle
from robot_simulator.a_star import AStarPlanner
from robot_simulator.lidar import simulate_lidar


def record_hierarchical_episode(
    env, model, waypoints, max_steps=1000, start_pose=(-4.0, -3.5, 0.0)
):
    """
    Runs an episode guided by sequential A* waypoints and records
    all state histories for smooth real-time animation.
    """
    env.reset()
    env.robot_x, env.robot_y, env.robot_theta = start_pose
    env.goal = waypoints[1]
    env.prev_distance = np.hypot(waypoints[1][0] - env.robot_x, waypoints[1][1] - env.robot_y)
    obs = env._get_obs()

    frames = []
    total_reward = 0.0
    current_wp_idx = 1

    for step in range(max_steps):
        # 1. Compute current LiDAR beams
        scan = simulate_lidar(
            env.robot_x,
            env.robot_y,
            env.robot_theta,
            env.obstacles,
            env.num_beams,
            env.max_range,
            env.fov,
        )

        # 2. Predict action
        action, _ = model.predict(obs, deterministic=True)
        v, omega = float(action[0]), float(action[1])

        dist_to_wp = np.hypot(env.goal[0] - env.robot_x, env.goal[1] - env.robot_y)
        is_final_wp = current_wp_idx == len(waypoints) - 1

        # 3. Store frame snapshot
        frame_data = {
            "step": step,
            "x": env.robot_x,
            "y": env.robot_y,
            "theta": env.robot_theta,
            "v": v,
            "omega": omega,
            "scan": list(scan),
            "dist": dist_to_wp,
            "goal": env.goal,
            "obstacles": list(env.obstacles),
            "wp_idx": current_wp_idx,
            "total_wps": len(waypoints),
            "target_name": f"W{current_wp_idx} (Final)" if is_final_wp else f"W{current_wp_idx}",
        }
        frames.append(frame_data)

        # 4. Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if reward <= -50.0:
            print(f"Collision détectée à l'étape {step} !")
            break

        # Waypoint acceptance: 0.75m for intermediate waypoints, fine tolerance for final goal
        acceptance_radius = env.goal_tolerance if is_final_wp else 0.75
        dist_now = np.hypot(env.goal[0] - env.robot_x, env.goal[1] - env.robot_y)

        if dist_now <= acceptance_radius:
            current_wp_idx += 1
            if current_wp_idx >= len(waypoints):
                # Final goal reached! Append terminal frame
                final_scan = simulate_lidar(
                    env.robot_x,
                    env.robot_y,
                    env.robot_theta,
                    env.obstacles,
                    env.num_beams,
                    env.max_range,
                    env.fov,
                )
                frames.append({
                    "step": step + 1,
                    "x": env.robot_x,
                    "y": env.robot_y,
                    "theta": env.robot_theta,
                    "v": 0.0,
                    "omega": 0.0,
                    "scan": list(final_scan),
                    "dist": 0.0,
                    "goal": env.goal,
                    "obstacles": list(env.obstacles),
                    "wp_idx": current_wp_idx - 1,
                    "total_wps": len(waypoints),
                    "target_name": "Objectif Atteint",
                })
                break

            # Switch target to next waypoint
            env.goal = waypoints[current_wp_idx]
            env.prev_distance = np.hypot(env.goal[0] - env.robot_x, env.goal[1] - env.robot_y)
            env.current_step = 0
            obs = env._get_obs()

    success = current_wp_idx >= len(waypoints)
    return frames, success, total_reward


def create_hierarchical_animation(
    frames, waypoints, save_gif=True, gif_filename="assets/robot_navigation_hierarchical.gif"
):
    """
    Generates an interactive real-time 2D animation showing:
    - Obstacle barrier
    - A* reference waypoints and path (green dashed)
    - PPO robot moving with dynamic LiDAR rays and heading
    - Real-time Telemetry HUD (Step, Velocity, Rotation, Active Waypoint, Status)
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-6.5, 6.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title(
        "Navigation Hiérarchique (A* + PPO) - Simulation Dynamique",
        fontsize=12,
        fontweight="bold",
    )

    initial_frame = frames[0]
    obstacles = initial_frame["obstacles"]

    # 1. Plot static obstacles
    for obs in obstacles:
        circle = Circle((obs.x, obs.y), obs.radius, color="gray", alpha=0.55, zorder=2)
        ax.add_patch(circle)
        # Add subtle clearance ring
        ring = Circle(
            (obs.x, obs.y),
            obs.radius + 0.2,
            color="gray",
            fill=False,
            linestyle=":",
            alpha=0.35,
        )
        ax.add_patch(ring)

    # 2. Plot A* Global Path & Reference Waypoints
    astar_x = [wp[0] for wp in waypoints]
    astar_y = [wp[1] for wp in waypoints]
    ax.plot(
        astar_x,
        astar_y,
        "g--",
        linewidth=2.0,
        alpha=0.75,
        label="Chemin global (A*)",
        zorder=3,
    )

    for i, (wx, wy) in enumerate(waypoints):
        ax.scatter(wx, wy, c="darkgreen", s=70, zorder=4)
        ax.text(
            wx + 0.2,
            wy + 0.15,
            f"W{i}",
            fontsize=9,
            fontweight="bold",
            color="darkgreen",
            zorder=4,
        )

    # Final Goal target marker
    final_goal = waypoints[-1]
    ax.scatter(
        final_goal[0],
        final_goal[1],
        c="gold",
        edgecolors="darkorange",
        s=140,
        zorder=5,
        label=f"But final ({final_goal[0]:.1f}, {final_goal[1]:.1f})",
    )

    # 3. Dynamic Elements
    # Trajectory path line
    (traj_line,) = ax.plot(
        [],
        [],
        "b-",
        linewidth=2.4,
        alpha=0.85,
        label="Trajectoire réelle (PPO)",
        zorder=4,
    )

    # Robot body
    robot_radius = 0.2
    robot_body = Circle(
        (initial_frame["x"], initial_frame["y"]),
        robot_radius,
        color="dodgerblue",
        ec="black",
        lw=1.5,
        zorder=6,
        label="Robot mobile",
    )
    ax.add_patch(robot_body)

    # Robot heading indicator (front nose)
    (heading_line,) = ax.plot([], [], "k-", linewidth=2.5, zorder=7)

    # LiDAR beam lines (16 beams)
    lidar_lines = [
        ax.plot([], [], "r:", linewidth=1.1, alpha=0.55, zorder=3)[0]
        for _ in range(16)
    ]

    # HUD Telemetry Text box
    hud_template = (
        "Étape: {:03d}\n"
        "Vitesse: {:.2f} m/s\n"
        "Rotation: {:+.2f} rad/s\n"
        "Balise active: {}\n"
        "Dist. Balise: {:.2f} m\n"
        "Statut: {}"
    )
    hud_box = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
        zorder=10,
    )

    ax.legend(loc="lower left", framealpha=0.9)

    # Cumulative trajectory arrays
    traj_x = []
    traj_y = []

    fov = 2 * np.pi
    num_beams = 16
    beam_angles_rel = np.linspace(0, fov, num_beams, endpoint=False)

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
        is_last = frame_idx == len(frames) - 1
        if is_last and f["dist"] <= 0.35:
            status = "MISSION ACCOMPLIE !"
            robot_body.set_facecolor("limegreen")
        elif is_last:
            status = "TERMINÉ"
        else:
            status = "SUIVI DE CHEMIN"

        # Update Telemetry HUD
        hud_box.set_text(
            hud_template.format(
                f["step"], f["v"], f["omega"], f["target_name"], f["dist"], status
            )
        )

        return [traj_line, robot_body, heading_line, hud_box] + lidar_lines

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        init_func=init,
        interval=35,
        blit=True,
        repeat=True,
    )

    if save_gif:
        os.makedirs(os.path.dirname(gif_filename) or ".", exist_ok=True)
        print(f"Génération du GIF animé ({gif_filename})...")
        anim.save(gif_filename, writer="pillow", fps=15)
        print(f"GIF sauvegardé avec succès : {gif_filename}")

    plt.tight_layout()
    plt.show()


def main():
    print("=================================================================")
    print("Animation Dynamique : Navigation Hiérarchique (A* + PPO)")
    print("=================================================================")

    # 1. Obstacles (Diagonal Barrier)
    obstacles = [
        CircularObstacle(x=-2.5, y=-1.5, radius=0.75),
        CircularObstacle(x=-1.2, y=-2.8, radius=0.70),
        CircularObstacle(x=-0.2, y=-0.8, radius=0.75),
        CircularObstacle(x=0.8, y=1.2, radius=0.80),
        CircularObstacle(x=2.0, y=-0.2, radius=0.75),
        CircularObstacle(x=2.8, y=2.2, radius=0.80),
    ]

    # 2. Mission coordinates
    start_x, start_y = -4.0, -3.5
    goal = (4.5, 4.0)

    # Initial heading pointing straight towards the final destination
    init_theta = float(np.arctan2(goal[1] - start_y, goal[0] - start_x))
    start_pose = (start_x, start_y, init_theta)

    # 3. Global Planning with A*
    print("\n[Étape 1] Calcul du chemin macroscopique par A*...")
    planner = AStarPlanner(
        obstacles=obstacles,
        bounds=(-8.0, 8.0, -8.0, 8.0),
        resolution=0.35,
        robot_radius=0.2,
        safety_margin=0.6,
    )
    waypoints = planner.plan((start_x, start_y), goal)
    if waypoints is None:
        print("Erreur : Aucun chemin trouvé par A* !")
        return

    print(f"-> A* a calculé {len(waypoints)} waypoints :")
    for i, wp in enumerate(waypoints):
        print(f"   W{i} : ({wp[0]:.2f}, {wp[1]:.2f})")

    # 4. Simulation Environment & Model
    env = RobotNavEnv(
        max_steps=500,
        goal=waypoints[1],
        obstacles=obstacles,
        randomize=False,
    )

    model = PPO.load("models/ppo_robot_nav_smooth", device="cpu")

    print(f"\n[Étape 2] Enregistrement de l'épisode (Départ: {start_pose[:2]} -> But: {goal})...")
    frames, success, reward = record_hierarchical_episode(
        env, model, waypoints, start_pose=start_pose
    )
    print(f"Épisode terminé en {len(frames)} étapes. Succès : {success} | Récompense : {reward:.2f}")

    # 5. Launch interactive animation & save GIF
    create_hierarchical_animation(
        frames,
        waypoints,
        save_gif=True,
        gif_filename="assets/robot_navigation_hierarchical.gif",
    )


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from robot_simulator.controllers import calculate_omega, calculate_speed
from robot_simulator.kinematics import update_robot, normalize_angle
from robot_simulator.obstacles import check_collision
from robot_simulator.lidar import simulate_lidar


def simulate(x, y, theta, steps, goal, Kv, Kp, max_speed, delta, goal_tolerance=0.05, obstacles=None, robot_radius=0.0):
    if obstacles is None:
        obstacles = []
    x_history = [x]
    y_history = [y]
    outcome = "timeout"


    for _ in range(steps):
        v = calculate_speed(x, y, goal, Kv, max_speed)
        omega = calculate_omega(x, y, theta, goal, Kp)

        x, y, theta = update_robot(x, y, theta, v, omega, delta)
        x_history.append(x)
        y_history.append(y)

        if check_collision(x, y, obstacles, robot_radius):
            outcome = "collision"
            break

        d = np.hypot(goal[0] - x, goal[1] - y)
        if d <= goal_tolerance:
            outcome = "success"
            break

    return x, y, theta, x_history, y_history, outcome


def render_arena(ax, obstacles=None, goal=None, start=None, bounds=(-8.0, 8.0, -8.0, 8.0)):
    """Standardized helper to draw the 2D arena, boundaries, and circular obstacles."""
    if obstacles is None:
        obstacles = []

    ax.set_aspect("equal")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    for obs in obstacles:
        circle = Circle((obs.x, obs.y), obs.radius, color="gray", alpha=0.55, zorder=2)
        ax.add_patch(circle)

    if start is not None:
        ax.scatter(start[0], start[1], c="blue", s=90, zorder=5, label=f"Départ ({start[0]:.1f}, {start[1]:.1f})")

    if goal is not None:
        ax.scatter(goal[0], goal[1], c="green", s=100, zorder=5, label=f"Objectif ({goal[0]:.1f}, {goal[1]:.1f})")


def plot_trajectory(x_history, y_history, goal, obstacles=None, theta=None, num_beams=16, fov=2*np.pi, max_range=5.0):
    if obstacles is None:
        obstacles = []
    fig, ax = plt.subplots(figsize=(8, 8))
    x, y = x_history[-1], y_history[-1]

    render_arena(ax, obstacles=obstacles, goal=goal, start=(x_history[0], y_history[0]), bounds=(-10.0, 10.0, -10.0, 10.0))

    ax.plot(x_history, y_history, "b-", linewidth=2.0, label="Trajectoire réelle")
    ax.scatter(x, y, c="orange", s=90, zorder=6, label="Position finale")

    if theta is not None:
        scan = simulate_lidar(x, y, theta, obstacles, num_beams=num_beams, max_range=max_range, fov=fov)
        angles = np.linspace(0, fov, num_beams, endpoint=False) + theta
        for angle, d in zip(angles, scan):
            x_end = x + d * np.cos(angle)
            y_end = y + d * np.sin(angle)
            ax.plot([x, x_end], [y, y_end], color="red", linestyle=":", alpha=0.45)

    ax.legend()
    plt.show()


if __name__ == '__main__':
    from robot_simulator.obstacles import CircularObstacle
    x, y, theta = 0, 0, np.pi
    Kv, Kp = 0.5, 2
    max_speed = 10
    delta = 0.1
    goal = (4, 4)
    steps = 100
    obstacles = [CircularObstacle(2, 2, 0.8)]

    x, y, theta, x_history, y_history, outcome = simulate(
        x, y, theta, steps, goal, Kv, Kp, max_speed, delta, obstacles=obstacles
    )

    print(f"Simulation finished. Outcome: {outcome}")
    plot_trajectory(x_history, y_history, goal, obstacles, theta)

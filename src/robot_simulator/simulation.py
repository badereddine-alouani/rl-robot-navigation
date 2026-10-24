import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from robot_simulator.controllers import calculate_omega, calculate_speed
from robot_simulator.kinematics import update_robot
from robot_simulator.obstacles import check_collision


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


def plot_trajectory(x_history, y_history, goal, obstacles=None):
    if obstacles is None:
        obstacles = []
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.plot(x_history, y_history, label="Robot trajectory")
    ax.scatter(x_history[0], y_history[0], label="Start")
    ax.scatter(x_history[-1], y_history[-1], label="End")
    ax.scatter(goal[0], goal[1], label="Goal")
    for obs in obstacles:
        circle = Circle((obs.x, obs.y), obs.radius, color="gray", alpha=0.6)
        ax.add_patch(circle)

    ax.grid(True)
    ax.legend()
    plt.show()


if __name__ == '__main__':
    from robot_simulator.obstacles import CircularObstacle
    x, y, theta = 0, 0, np.pi / 4
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
    plot_trajectory(x_history, y_history, goal, obstacles)

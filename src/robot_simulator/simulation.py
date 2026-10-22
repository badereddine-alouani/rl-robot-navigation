import numpy as np
import matplotlib.pyplot as plt
from robot_simulator.controllers import calculate_omega, calculate_speed
from robot_simulator.kinematics import update_robot


def simulate(x, y, theta, steps, goal, Kv, Kp, max_speed, delta, goal_tolerance=0.05):
    x_history = [x]
    y_history = [y]
    success = False

    for _ in range(steps):
        v = calculate_speed(x, y, goal, Kv, max_speed)
        omega = calculate_omega(x, y, theta, goal, Kp)

        x, y, theta = update_robot(x, y, theta, v, omega, delta)
        x_history.append(x)
        y_history.append(y)

        d = np.hypot(goal[0] - x, goal[1] - y)
        if d <= goal_tolerance:
            success = True
            break

    return x, y, theta, x_history, y_history, success


def plot_trajectory(x_history, y_history, goal):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.plot(x_history, y_history, label="Robot trajectory")
    ax.scatter(x_history[0], y_history[0], label="Start")
    ax.scatter(x_history[-1], y_history[-1], label="End")
    ax.scatter(goal[0], goal[1], label="Goal")

    ax.grid(True)
    ax.legend()
    plt.show()


if __name__ == '__main__':
    x, y, theta = 0, 0, np.pi
    Kv, Kp = 0.5, 2
    max_speed = 10
    delta = 0.1
    goal = (5, -5)
    steps = 100

    x, y, theta, x_history, y_history, success = simulate(
        x, y, theta, steps, goal, Kv, Kp, max_speed, delta
    )

    print(f"Simulation finished. Goal reached: {success}")
    plot_trajectory(x_history, y_history, goal)

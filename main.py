import numpy as np
import matplotlib.pyplot as plt


def calculate_heading_error(x, y, theta, goal):
    dx, dy = goal[0] - x, goal[1] - y
    theta_goal = np.arctan2(dy, dx)
    error = (theta_goal - theta + np.pi) % (2 * np.pi) - np.pi

    return error

def calculate_omega(x, y, theta, goal, Kp):
    error = calculate_heading_error(x, y, theta, goal)
    omega = Kp * error

    return omega

def calculate_speed(x, y, goal, Kv, max_speed):
    d = np.hypot(goal[0] - x, goal[1] - y)
    v = min(max_speed, Kv * d)

    return v



def update_robot(x, y, theta, v, omega, delta):
    x += v * np.cos(theta) * delta
    y += v * np.sin(theta) * delta
    theta += omega * delta

    return x, y, theta

def simulate(x, y, theta, steps, goal, Kv, Kp, max_speed, delta):
    x_history = [x]
    y_history = [y]
    for _ in range(steps):
        v = calculate_speed(x, y, goal, Kv, max_speed)
        omega = calculate_omega(x, y, theta, goal, Kp)

        x, y, theta = update_robot(x, y, theta, v, omega, delta)
        x_history.append(x)
        y_history.append(y)

        d = np.hypot(goal[0] - x, goal[1] - y)
        if d <= 0.05:
            break

    return x, y, theta, x_history, y_history
        


if __name__ == '__main__':
    x, y, theta = 0, 0, np.pi
    v, omega = 0, 0
    Kv, Kp = 0.5, 2
    max_speed = 10
    delta = 0.1
    goal = (5, -5)
    steps = 100

    x, y, theta, x_history, y_history = simulate(x, y, theta, steps, goal, Kv, Kp, max_speed, delta)

    fig, ax = plt.subplots(figsize=(8,5))

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
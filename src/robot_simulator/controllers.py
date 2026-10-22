import numpy as np
from .kinematics import normalize_angle

def calculate_heading_error(x, y, theta, goal):
    dx, dy = goal[0] - x, goal[1] - y
    theta_goal = np.arctan2(dy, dx)
    error = normalize_angle(theta_goal - theta)

    return error

def calculate_omega(x, y, theta, goal, Kp):
    error = calculate_heading_error(x, y, theta, goal)
    omega = Kp * error

    return omega

def calculate_speed(x, y, goal, Kv, max_speed):
    d = np.hypot(goal[0] - x, goal[1] - y)
    v = min(max_speed, Kv * d)

    return v
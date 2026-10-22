import numpy as np

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def update_robot(x, y, theta, v, omega, delta):
    x += v * np.cos(theta) * delta
    y += v * np.sin(theta) * delta
    theta += omega * delta
    theta = normalize_angle(theta)

    return x, y, theta
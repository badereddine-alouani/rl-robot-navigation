import numpy as np

def ray_circle_intersection(ray_x, ray_y, angle, obstacle):
    dx = ray_x - obstacle.x
    dy = ray_y - obstacle.y
    B = 2 * (dx * np.cos(angle) + dy * np.sin(angle))
    C = dx ** 2 + dy ** 2 - obstacle.radius ** 2
    discriminant = B ** 2 - 4 * C

    if discriminant < 0:
        return

    t1 = (-B - np.sqrt(discriminant)) / 2
    t2 = (-B + np.sqrt(discriminant)) / 2

    if t1 >= 0:
        return t1
    elif t2 >= 0:
        return t2


def simulate_lidar(robot_x, robot_y, robot_theta, obstacles, num_beams=16, max_range=5.0, fov=2*np.pi):
    if obstacles is None:
        obstacles = []
    angles = np.linspace(0, fov, num_beams, endpoint=False)
    scan = [max_range] * num_beams

    for i, a in enumerate(angles):
        for obs in obstacles:
            angle = a + robot_theta
            d = ray_circle_intersection(robot_x, robot_y, angle, obs)
            if d is not None:
                scan[i] = min(scan[i], d)
                
    return scan

    
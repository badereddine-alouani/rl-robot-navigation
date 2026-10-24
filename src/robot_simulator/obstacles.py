from dataclasses import dataclass
import numpy as np

@dataclass
class CircularObstacle:
    x: float
    y: float
    radius: float

def check_collision(robot_x, robot_y, obstacles, robot_radius=0.0):
    for obs in obstacles:
        d = np.hypot(obs.x - robot_x, obs.y - robot_y)
        if d <= obs.radius + robot_radius:
            return True
        
    return False
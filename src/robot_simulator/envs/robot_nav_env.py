import numpy as np
import gymnasium as gym
from gymnasium import spaces

from robot_simulator.kinematics import update_robot
from robot_simulator.controllers import calculate_heading_error
from robot_simulator.obstacles import CircularObstacle, check_collision
from robot_simulator.lidar import simulate_lidar


class RobotNavEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_steps=200,
        delta=0.1,
        max_speed=1.0,
        max_omega=1.0,
        goal_tolerance=0.2,
        robot_radius=0.2,
        num_beams=16,
        max_range=5.0,
        fov=2 * np.pi,
        obstacles=None,
        goal=(4.0, 4.0),
    ):
        super().__init__()
        self.max_steps = max_steps
        self.delta = delta
        self.max_speed = max_speed
        self.max_omega = max_omega
        self.goal_tolerance = goal_tolerance
        self.robot_radius = robot_radius
        self.num_beams = num_beams
        self.max_range = max_range
        self.fov = fov
        self.goal = goal
        self.obstacles = obstacles if obstacles is not None else [CircularObstacle(2.0, 2.0, 0.8)]

        # Action Space: [v, omega]
        self.action_space = spaces.Box(
            low=np.array([0.0, -self.max_omega], dtype=np.float32),
            high=np.array([self.max_speed, self.max_omega], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation Space: [distance, heading_error, 16 lidar beams]
        low_obs = np.array([0.0, -np.pi] + [0.0] * self.num_beams, dtype=np.float32)
        high_obs = np.array([50.0, np.pi] + [self.max_range] * self.num_beams, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # State tracking
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.current_step = 0
        self.prev_distance = 0.0

    def _get_obs(self):
        d = np.hypot(self.goal[0] - self.robot_x, self.goal[1] - self.robot_y)
        heading_err = calculate_heading_error(self.robot_x, self.robot_y, self.robot_theta, self.goal)
        scan = simulate_lidar(self.robot_x, self.robot_y, self.robot_theta, self.obstacles, self.num_beams, self.max_range, self.fov)

        return np.array([d, heading_err] + list(scan), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.current_step = 0
        self.prev_distance = np.hypot(self.goal[0] - self.robot_x, self.goal[1] - self.robot_y)

        return self._get_obs(), {}

    def step(self, action):
        v, omega = action
        self.robot_x, self.robot_y, self.robot_theta = update_robot(self.robot_x, self.robot_y, self.robot_theta, v, omega, self.delta)
        current_distance = np.hypot(self.goal[0] - self.robot_x, self.goal[1] - self.robot_y)
        is_collision = check_collision(self.robot_x, self.robot_y, self.obstacles, self.robot_radius)
        is_goal = current_distance <= self.goal_tolerance
        progress = self.prev_distance - current_distance
        reward = progress - 0.01

        if is_goal:
            reward += 100
        elif is_collision:
            reward -= 100

        self.prev_distance = current_distance
        terminated = bool(is_collision or is_goal)
        self.current_step += 1
        truncated = bool(self.current_step >= self.max_steps)

        return self._get_obs(), float(reward), terminated, truncated, {}

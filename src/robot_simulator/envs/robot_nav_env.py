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
        randomize=False
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
        self.initial_goal = goal
        self.initial_obstacles = list(obstacles) if obstacles is not None else [CircularObstacle(2.0, 2.0, 0.8)]
        self.goal = self.initial_goal
        self.obstacles = list(self.initial_obstacles)
        self.randomize = randomize

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

    def _reset_defaults(self):
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.goal = self.initial_goal
        self.obstacles = list(self.initial_obstacles)

    def _sample_goal(self, min_distance=3.0):
        while True:
            gx = float(self.np_random.uniform(-6.0, 6.0))
            gy = float(self.np_random.uniform(-6.0, 6.0))
            if np.hypot(gx - self.robot_x, gy - self.robot_y) >= min_distance:
                return (gx, gy)

    def _sample_obstacles(self):
        new_obstacles = []
        num_obstacles = int(self.np_random.integers(1, 4))
        for _ in range(num_obstacles):
            while True:
                ox = float(self.np_random.uniform(-5.0, 5.0))
                oy = float(self.np_random.uniform(-5.0, 5.0))
                r = float(self.np_random.uniform(0.6, 0.9))
                
                dist_to_robot = np.hypot(ox - self.robot_x, oy - self.robot_y)
                dist_to_goal = np.hypot(ox - self.goal[0], oy - self.goal[1])
                
                # Keep clear of robot and goal
                if dist_to_robot >= (r + self.robot_radius + 0.5) and dist_to_goal >= (r + self.goal_tolerance + 0.5):
                    new_obstacles.append(CircularObstacle(ox, oy, r))
                    break
        return new_obstacles

    def _randomize_environment(self):
        self.robot_x = float(self.np_random.uniform(-3.0, 3.0))
        self.robot_y = float(self.np_random.uniform(-3.0, 3.0))
        self.robot_theta = float(self.np_random.uniform(-np.pi, np.pi))
        self.goal = self._sample_goal()
        self.obstacles = self._sample_obstacles()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.randomize:
            self._randomize_environment()
        else:
            self._reset_defaults()

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

        info = {
            "is_success": bool(is_goal),
            "is_collision": bool(is_collision),
        }

        return self._get_obs(), float(reward), terminated, truncated, info

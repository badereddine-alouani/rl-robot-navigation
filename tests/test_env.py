import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from robot_simulator.envs import RobotNavEnv
from robot_simulator.obstacles import CircularObstacle


def test_gymnasium_check_env():
    # Official Gymnasium environment compliance check
    env = RobotNavEnv()
    check_env(env)


def test_env_reset():
    env = RobotNavEnv()
    obs, info = env.reset()

    # Observation must match space shape (18,) and dtype float32
    assert obs.shape == (18,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)

    # Initial distance to goal (4, 4) from (0, 0) should be approx sqrt(32) ~= 5.6568
    assert obs[0] == pytest.approx(np.hypot(4.0, 4.0), abs=1e-3)
    assert env.current_step == 0


def test_env_step_structure():
    env = RobotNavEnv()
    env.reset()

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (18,)
    assert obs.dtype == np.float32
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert env.current_step == 1


def test_env_goal_reached():
    # Place goal close at (1, 0) with tolerance 0.2
    env = RobotNavEnv(goal=(1.0, 0.0), goal_tolerance=0.2)
    env.reset()

    # Place robot inside goal tolerance
    env.robot_x = 0.95
    env.robot_y = 0.0

    # Step with zero velocity
    _, reward, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert reward >= 90.0  # Must include +100 goal bonus


def test_env_collision():
    # Obstacle at (1, 0) with radius 0.5, robot radius 0.2
    obstacle = CircularObstacle(x=1.0, y=0.0, radius=0.5)
    env = RobotNavEnv(obstacles=[obstacle], robot_radius=0.2)
    env.reset()

    # Place robot inside obstacle collision zone (distance 0.1 <= 0.7)
    env.robot_x = 0.9
    env.robot_y = 0.0

    _, reward, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert reward <= -90.0  # Must include -100 collision penalty


def test_env_truncation():
    # Max steps set to 5
    env = RobotNavEnv(max_steps=5)
    env.reset()

    action = np.array([0.1, 0.0], dtype=np.float32)
    for step in range(1, 5):
        _, _, terminated, truncated, _ = env.step(action)
        assert truncated is False

    # 5th step should trigger truncation
    _, _, terminated, truncated, _ = env.step(action)
    assert truncated is True


def test_env_randomize():
    env = RobotNavEnv(randomize=True)
    check_env(env)

    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    # Seeding should reproduce identical observations
    np.testing.assert_allclose(obs1, obs2)

    # Different seeds should produce different configurations
    obs3, _ = env.reset(seed=99)
    assert not np.allclose(obs1, obs3)

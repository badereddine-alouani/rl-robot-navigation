import pytest
from robot_simulator.obstacles import CircularObstacle, check_collision


def test_collision_empty_world():
    # An empty world with no obstacles should never trigger a collision
    assert not check_collision(0.0, 0.0, [])


@pytest.mark.parametrize(
    ("robot_x", "robot_y", "robot_radius", "obstacles", "expected"),
    [
        # Clear separation: robot is far from the obstacle
        (0.0, 0.0, 0.0, [CircularObstacle(5.0, 5.0, 1.0)], False),
        # Direct hit: robot center is inside the obstacle
        (0.0, 0.0, 0.0, [CircularObstacle(0.5, 0.0, 1.0)], True),
        # Touching boundary: distance (2.0) equals r_obs + r_robot (1.5 + 0.5)
        (0.0, 0.0, 0.5, [CircularObstacle(2.0, 0.0, 1.5)], True),
        # Just outside: distance (2.01) is slightly greater than 1.5 + 0.5
        (0.0, 0.0, 0.5, [CircularObstacle(2.01, 0.0, 1.5)], False),
        # Multiple obstacles: first obstacle is far, second is a collision
        (
            0.0,
            0.0,
            0.0,
            [CircularObstacle(10.0, 10.0, 1.0), CircularObstacle(0.2, 0.0, 0.5)],
            True,
        ),
    ],
)
def test_check_collision(robot_x, robot_y, robot_radius, obstacles, expected):
    assert check_collision(robot_x, robot_y, obstacles, robot_radius) == expected

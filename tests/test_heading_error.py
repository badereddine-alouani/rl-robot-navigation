import math
import pytest
from robot_simulator.controllers import calculate_heading_error



@pytest.mark.parametrize(
    ("x", "y", "theta", "goal", "expected"),
    [
        # Already facing the goal
        (0, 0, 0, (5, 0), 0),

        # Goal directly above
        (0, 0, 0, (0, 5), math.pi / 2),

        # Robot needs to turn clockwise
        (0, 0, 3 * math.pi / 4, (5, 0), -3 * math.pi / 4),

        # Robot needs to turn counterclockwise
        (0, 0, -3 * math.pi / 4, (5, 0), 3 * math.pi / 4),

        # Goal directly behind: normalization returns -pi
        (0, 0, 0, (-5, 0), -math.pi),

        # Angle wrapping: the shortest turn is small and positive
        (0, 0, 3.1, (-1, -0.0416), 0.0832),

        # Robot is not at the origin
        (2, 1, 0, (2, 6), math.pi / 2),
    ],
)
def test_calculate_heading_error(x, y, theta, goal, expected):
    assert calculate_heading_error(x, y, theta, goal) == pytest.approx(
        expected, abs=1e-3
    )

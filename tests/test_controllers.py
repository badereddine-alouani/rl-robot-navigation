import math
import pytest
from robot_simulator.controllers import calculate_speed, calculate_omega


@pytest.mark.parametrize(
    ("x", "y", "goal", "Kv", "max_speed", "expected"),
    [
        # Far from goal: speed is capped by max_speed
        (0, 0, (10, 0), 1.0, 5.0, 5.0),
        # Close to goal: speed scales proportionally (Kv * d = 1.0 * 2.0)
        (0, 0, (2, 0), 1.0, 5.0, 2.0),
        # Exactly at goal: speed is 0
        (4, 4, (4, 4), 1.0, 5.0, 0.0),
    ],
)
def test_calculate_speed(x, y, goal, Kv, max_speed, expected):
    assert calculate_speed(x, y, goal, Kv, max_speed) == pytest.approx(
        expected, abs=1e-3
    )


def test_calculate_omega():
    # Facing east (theta = 0), goal is north at (0, 5) -> heading error = pi / 2
    # With Kp = 2.0, expected omega = Kp * error = 2.0 * (pi / 2) = pi
    omega = calculate_omega(0, 0, 0, (0, 5), Kp=2.0)
    assert omega == pytest.approx(math.pi, abs=1e-3)

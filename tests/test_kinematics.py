import pytest
import math
from robot_simulator.kinematics import update_robot



@pytest.mark.parametrize(
    ("x", "y", "theta", "v", "omega", "delta", "expected"),
    [
        (0, 0, 0, 1, 0, 1, (1, 0, 0)),
        (0, 0, math.pi / 2, 2, 0, 0.5, (0, 1, math.pi / 2)),
        (0, 0, 0, 0, math.pi / 2, 1, (0, 0, math.pi / 2)),
        (0, 0, 3 * math.pi / 4, 0, math.pi / 2, 1, (0, 0, -3 * math.pi / 4))
    ]
)
def test_update_robot(x, y, theta, v, omega, delta, expected):
    assert update_robot(x, y, theta, v, omega, delta) == pytest.approx(expected, abs=1e-3)





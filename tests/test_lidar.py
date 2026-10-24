import math
import pytest
from robot_simulator.lidar import ray_circle_intersection, simulate_lidar
from robot_simulator.obstacles import CircularObstacle


# ==============================================================================
# Unit Tests: ray_circle_intersection
# ==============================================================================

def test_ray_circle_direct_hit():
    # Ray at origin aimed along positive x-axis (angle = 0)
    # Obstacle center at (5, 0) with radius 1.0 -> should hit front edge at distance 4.0
    obstacle = CircularObstacle(x=5.0, y=0.0, radius=1.0)
    hit_dist = ray_circle_intersection(0.0, 0.0, 0.0, obstacle)
    assert hit_dist == pytest.approx(4.0, abs=1e-3)


def test_ray_circle_miss():
    # Ray along x-axis, obstacle is far to the side at (5, 5) with radius 1.0
    obstacle = CircularObstacle(x=5.0, y=5.0, radius=1.0)
    hit_dist = ray_circle_intersection(0.0, 0.0, 0.0, obstacle)
    assert hit_dist is None


def test_ray_circle_behind():
    # Ray aimed East (angle = 0), but obstacle is West at (-5, 0)
    obstacle = CircularObstacle(x=-5.0, y=0.0, radius=1.0)
    hit_dist = ray_circle_intersection(0.0, 0.0, 0.0, obstacle)
    assert hit_dist is None


def test_ray_inside_obstacle():
    # Ray starts at origin, obstacle centered at origin with radius 2.0
    # Beam exits at distance 2.0
    obstacle = CircularObstacle(x=0.0, y=0.0, radius=2.0)
    hit_dist = ray_circle_intersection(0.0, 0.0, 0.0, obstacle)
    assert hit_dist == pytest.approx(2.0, abs=1e-3)


# ==============================================================================
# Unit Tests: simulate_lidar
# ==============================================================================

def test_lidar_empty_world():
    # In an empty world, all beams should read max_range
    scan = simulate_lidar(0.0, 0.0, 0.0, obstacles=[], num_beams=8, max_range=5.0)
    assert len(scan) == 8
    assert all(math.isclose(d, 5.0, abs_tol=1e-3) for d in scan)


def test_lidar_detects_closest_obstacle():
    # Beam 0 points East (theta = 0, relative angle = 0)
    # Two obstacles along East: one at 4m (radius 0.5 -> edge at 3.5m)
    # and another at 8m (radius 0.5 -> edge at 7.5m).
    # The LiDAR must register the closest one (3.5m), not the farther one.
    obs1 = CircularObstacle(x=4.0, y=0.0, radius=0.5)
    obs2 = CircularObstacle(x=8.0, y=0.0, radius=0.5)

    scan = simulate_lidar(0.0, 0.0, 0.0, obstacles=[obs1, obs2], num_beams=8, max_range=10.0)
    assert scan[0] == pytest.approx(3.5, abs=1e-3)


def test_lidar_rotates_with_robot():
    # Obstacle is at (0, 5) with radius 1.0 (North of the origin)
    # Case 1: Robot faces East (theta = 0)
    # With 4 beams (0, pi/2, pi, 3pi/2):
    # Beam 0 is East, Beam 1 is North (pi/2) -> Beam 1 should detect it at distance 4.0
    obs = CircularObstacle(x=0.0, y=5.0, radius=1.0)
    scan_east = simulate_lidar(0.0, 0.0, 0.0, obstacles=[obs], num_beams=4, max_range=10.0)
    assert scan_east[1] == pytest.approx(4.0, abs=1e-3)

    # Case 2: Robot turns to face North (theta = pi/2)
    # Now Beam 0 (straight ahead relative to robot) points North -> Beam 0 should detect it!
    scan_north = simulate_lidar(0.0, 0.0, math.pi / 2, obstacles=[obs], num_beams=4, max_range=10.0)
    assert scan_north[0] == pytest.approx(4.0, abs=1e-3)


def test_lidar_beyond_max_range():
    # Obstacle is at 15m, but max_range is 10.0m -> beam should still read 10.0m
    obs = CircularObstacle(x=15.0, y=0.0, radius=1.0)
    scan = simulate_lidar(0.0, 0.0, 0.0, obstacles=[obs], num_beams=8, max_range=10.0)
    assert scan[0] == pytest.approx(10.0, abs=1e-3)

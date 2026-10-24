import numpy as np
import pytest
from robot_simulator.a_star import AStarPlanner
from robot_simulator.obstacles import CircularObstacle


def test_a_star_straight_line():
    # In an open arena with no obstacles, A* should prune directly to [start, goal]
    planner = AStarPlanner(obstacles=[], bounds=(-5, 5, -5, 5), resolution=0.5)
    start = (0.0, 0.0)
    goal = (4.0, 4.0)

    path = planner.plan(start, goal)
    assert path is not None
    assert len(path) == 2
    assert path[0] == start
    assert path[-1] == goal


def test_a_star_around_obstacle():
    # Obstacle directly centered between start (0,0) and goal (4,0)
    obstacle = CircularObstacle(x=2.0, y=0.0, radius=0.8)
    planner = AStarPlanner(obstacles=[obstacle], bounds=(-5, 5, -5, 5), resolution=0.3)
    start = (0.0, 0.0)
    goal = (4.0, 0.0)

    path = planner.plan(start, goal)
    assert path is not None
    assert len(path) >= 3  # Start, at least one corner detour waypoint, and Goal
    assert path[0] == start
    assert path[-1] == goal

    # Make sure detoured waypoints are not inside the obstacle collision zone
    for wx, wy in path:
        dist = np.hypot(wx - obstacle.x, wy - obstacle.y)
        if (wx, wy) != start and (wx, wy) != goal:
            assert dist >= obstacle.radius + 0.2  # Must be clear of obstacle

import heapq
import numpy as np


class AStarPlanner:
    def __init__(
        self,
        obstacles,
        bounds=(-10.0, 10.0, -10.0, 10.0),
        resolution=0.4,
        robot_radius=0.2,
        safety_margin=0.35,
    ):
        self.obstacles = obstacles
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin

        self.width = int(np.ceil((self.x_max - self.x_min) / resolution))
        self.height = int(np.ceil((self.y_max - self.y_min) / resolution))

        # Precompute occupancy grid
        self.grid = self._create_grid()

    def _create_grid(self):
        """Creates a boolean 2D occupancy grid where True represents blocked/obstacle cells."""
        grid = np.zeros((self.height, self.width), dtype=bool)

        for row in range(self.height):
            for col in range(self.width):
                x, y = self.grid_to_world(col, row)
                for obs in self.obstacles:
                    # Inflate obstacle by robot radius + safety margin
                    effective_radius = obs.radius + self.robot_radius + self.safety_margin
                    if np.hypot(x - obs.x, y - obs.y) <= effective_radius:
                        grid[row, col] = True
                        break
        return grid

    def world_to_grid(self, x, y):
        col = int((x - self.x_min) / self.resolution)
        row = int((y - self.y_min) / self.resolution)
        col = np.clip(col, 0, self.width - 1)
        row = np.clip(row, 0, self.height - 1)
        return int(col), int(row)

    def grid_to_world(self, col, row):
        x = self.x_min + (col + 0.5) * self.resolution
        y = self.y_min + (row + 0.5) * self.resolution
        return float(x), float(y)

    def is_line_clear(self, p1, p2):
        """Checks if a straight line between two world points is collision-free."""
        dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = int(np.ceil(dist / (self.resolution * 0.5)))
        for s in range(steps + 1):
            t = s / max(steps, 1)
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            for obs in self.obstacles:
                if np.hypot(x - obs.x, y - obs.y) <= (obs.radius + self.robot_radius + self.safety_margin):
                    return False
        return True

    def prune_path(self, raw_path):
        """Reduces a sequence of grid cells into essential corner waypoints using line-of-sight checks."""
        if len(raw_path) <= 2:
            return raw_path

        pruned = [raw_path[0]]
        current_idx = 0

        while current_idx < len(raw_path) - 1:
            next_idx = len(raw_path) - 1
            # Look ahead from furthest to nearest
            while next_idx > current_idx + 1:
                if self.is_line_clear(raw_path[current_idx], raw_path[next_idx]):
                    break
                next_idx -= 1
            pruned.append(raw_path[next_idx])
            current_idx = next_idx

        return pruned

    def plan(self, start, goal):
        """Plans a collision-free path from start (x, y) to goal (x, y). Returns list of waypoints."""
        start_cell = self.world_to_grid(*start)
        goal_cell = self.world_to_grid(*goal)

        if self.grid[start_cell[1], start_cell[0]]:
            # If start is inside an inflated margin, unblock start cell
            self.grid[start_cell[1], start_cell[0]] = False

        if self.grid[goal_cell[1], goal_cell[0]]:
            self.grid[goal_cell[1], goal_cell[0]] = False

        # Priority queue stores: (f_score, counter, current_cell)
        counter = 0
        open_set = []
        heapq.heappush(open_set, (0.0, counter, start_cell))

        came_from = {}
        g_score = {start_cell: 0.0}

        # 8-connectivity motion: (d_col, d_row, cost)
        moves = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (-1, -1, np.sqrt(2))
        ]

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal_cell:
                # Reconstruct path
                path = [goal]
                curr = current
                while curr in came_from:
                    curr = came_from[curr]
                    path.append(self.grid_to_world(*curr))
                path.reverse()
                path[0] = start  # Ensure exact start coordinate

                # Prune into minimal corner waypoints
                return self.prune_path(path)

            for dc, dr, move_cost in moves:
                neighbor = (current[0] + dc, current[1] + dr)

                # Bounds check
                if not (0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height):
                    continue

                # Obstacle check
                if self.grid[neighbor[1], neighbor[0]]:
                    continue

                tentative_g = g_score[current] + move_cost * self.resolution

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    # Euclidean heuristic
                    h = np.hypot(
                        (neighbor[0] - goal_cell[0]) * self.resolution,
                        (neighbor[1] - goal_cell[1]) * self.resolution
                    )
                    counter += 1
                    heapq.heappush(open_set, (tentative_g + h, counter, neighbor))

        # No path found
        return None

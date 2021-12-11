# coding: utf-8
# scan_matcher_branch_bound.py

import heapq
import math

from typing import Any, List, Tuple

from grid_2d import Grid2D, compute_coarse_map
from point_2d import Point2D
from pose_2d import Pose2D, project_point_2d, wrap_angle

class ScanMatcherBranchBound(object):
    """`ScanMatcherBranchBound` class implements branch-and-bound scan matching,
    which aligns a LiDAR scan and a 2D occupancy grid map. \n
    For more details, please refer to the following paper: \n
    Wolfgang Hess, Damon Kohler, Holger Rapp, and Daniel Andor.
    "Real-time Loop Closure in 2D LiDAR SLAM," in Proceedings of the IEEE
    International Conference on Robotics and Automation (ICRA), 2016.
    """

    class Result(object):
        def __init__(self, win: Tuple[int, int, int],
                     step: Tuple[float, float, float],
                     best: Tuple[int, int, int], best_score: float,
                     initial_pose: Pose2D, best_pose: Pose2D,
                     num_solutions: int, num_processed: int) -> None:
            self.win_x, self.win_y, self.win_theta = win
            self.step_x, self.step_y, self.step_theta = step
            self.best_x, self.best_y, self.best_theta = best
            self.best_score = best_score
            self.initial_pose = initial_pose
            self.best_pose = best_pose
            self.num_solutions = num_solutions
            self.num_processed = num_processed

        def dump(self) -> None:
            msg = "Statistics for branch-and-bound scan matcher\n" \
                  "Window size: ({}, {}, {})\n" \
                  "Step size: ({:.6f} m, {:.6f} m, {:.6f} rad)\n" \
                  "Best solution: ({}, {}, {})\n" \
                  "Best score: {}\n" \
                  "Initial pose: ({:.6f} m, {:.6f} m, {:.6f} rad)\n" \
                  "Final pose: ({:.6f} m, {:.6f} m, {:.6f} rad)\n" \
                  "# of possible solutions: {}\n" \
                  "# of solutions processed: {}\n"
            print(msg.format(self.win_x, self.win_y, self.win_theta,
                             self.step_x, self.step_y, self.step_theta,
                             self.best_x, self.best_y, self.best_theta,
                             self.best_score,
                             self.initial_pose.x, self.initial_pose.y,
                             self.initial_pose.theta,
                             self.best_pose.x, self.best_pose.y,
                             self.best_pose.theta,
                             self.num_solutions, self.num_processed))

    def __init__(self, window_size_x: float, window_size_y: float,
                 window_size_theta: float, min_step_theta: float,
                 max_node_height: int) -> None:
        self.window_size_x = window_size_x
        self.window_size_y = window_size_y
        self.window_size_theta = window_size_theta
        self.min_step_theta = min_step_theta
        self.max_node_height = max_node_height
        self.max_stride = 1 << self.max_node_height

    def compute_step(self, grid_map: Grid2D, scan: List[Point2D]):
        ranges = [(p.x ** 2 + p.y ** 2) for p in scan]
        theta = grid_map.resolution() / max(ranges)

        step_x = grid_map.resolution()
        step_y = grid_map.resolution()
        step_theta = math.acos(1.0 - 0.5 * (theta ** 2))
        step_theta = max(step_theta, self.min_step_theta)

        return step_x, step_y, step_theta

    def compute_window(self, step_x: float, step_y: float, step_theta: float):
        return int(math.ceil(0.5 * self.window_size_x / step_x)), \
               int(math.ceil(0.5 * self.window_size_y / step_y)), \
               int(math.ceil(0.5 * self.window_size_theta / step_theta))

    def compute_score(self, grid_map: Grid2D, indices: List[Tuple[int, int]],
                      offset: Tuple[int, int]) -> int:
        score = 0
        for idx_base in indices:
            idx = (idx_base[0] + offset[0], idx_base[1] + offset[1])
            if grid_map.is_index_inside(*idx):
                score += grid_map.get_index_value(*idx)
        return score

    def match_scan(self, initial_pose: Pose2D, grid_map: Grid2D,
                   scan: List[Point2D]) -> Tuple[Pose2D, Any]:
        # Determine the search step and window size
        step = self.compute_step(grid_map, scan)
        win = self.compute_window(*step)
        step_x, step_y, step_theta = step
        win_x, win_y, win_theta = win
        best_x, best_y, best_theta, best_score = 0, 0, 0, 0

        num_solutions = (2 * win_x) * (2 * win_y) * (2 * win_theta)
        num_processed = 0

        # Compute coarse grid maps, which are analogous to image pyramids
        grid_maps = [grid_map]

        for i in range(self.max_node_height):
            grid_maps.append(compute_coarse_map(grid_map, 2 << i))

        # `node_queue` is a priority queue of nodes, where each node represents
        # a subregion in the 3D search window along x, y, and theta axes
        node_queue = []

        def append_node(x: int, y: int, theta: int,
                        height: int, indices: List[Tuple[int, int]]):
            score = self.compute_score(grid_maps[height], indices, (x, y))

            if score > best_score:
                # Store the negative score as a key, since `heapq.heappop()`
                # returns a node with the smallest key
                heapq.heappush(node_queue,
                    (-score, (x, y, theta, height, indices, score)))

        # Initialize a priority queue
        for t in range(-win_theta, win_theta):
            pose = Pose2D(initial_pose.x, initial_pose.y,
                          initial_pose.theta + t * step_theta)
            points = [project_point_2d(pose, p) for p in scan]
            indices = [grid_map.point_to_index(p) for p in points]

            for y in range(-win_y, win_y, self.max_stride):
                for x in range(-win_x, win_x, self.max_stride):
                    append_node(x, y, t, self.max_node_height, indices)

        while node_queue:
            # Get the node from the priority queue
            _, (x, y, t, height, indices, score) = heapq.heappop(node_queue)
            num_processed += 1

            if score <= best_score:
                # Skip the node if the score is below the best score so far
                continue

            if height == 0:
                # If the current node is a left, then update the solution
                best_x, best_y, best_theta, best_score = x, y, t, score
            else:
                # Otherwise, split the current node into four new nodes
                new_x, new_y, new_height = x, y, height - 1
                # Compute a new stride
                s = 1 << new_height
                append_node(new_x, new_y, t, new_height, indices)
                append_node(new_x + s, new_y, t, new_height, indices)
                append_node(new_x, new_y + s, t, new_height, indices)
                append_node(new_x + s, new_y + s, t, new_height, indices)

        best_pose = Pose2D(initial_pose.x + best_x * step_x,
                           initial_pose.y + best_y * step_y,
                           initial_pose.theta + best_theta * step_theta)
        best_pose.theta = wrap_angle(best_pose.theta)

        result = ScanMatcherBranchBound.Result(
            win, step, (best_x, best_y, best_theta), best_score,
            initial_pose, best_pose, num_solutions, num_processed)

        return best_pose, result

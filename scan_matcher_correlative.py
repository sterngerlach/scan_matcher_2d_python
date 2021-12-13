# coding: utf-8
# scan_matcher_correlative.py

import math

from typing import Any, List, Tuple

from PIL import Image

from grid_2d import Grid2D, compute_coarse_map
from point_2d import Point2D
from pose_2d import Pose2D, project_point_2d, wrap_angle

class ScanMatcherCorrelative(object):
    """`ScanMatcherCorrelative` class implements real-time correlative scan
    matching, which aligns a LiDAR scan and a 2D occupancy grid map. \n
    For more details, please refer to the following paper: \n
    Edwin B. Olson. "Real-time Correlative Scan Matching," in Proceedings of the
    IEEE International Conference on Robotics and Automation (ICRA), 2009.
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
            msg = "Statistics for correlative scan matcher\n" \
                  "Window size: ({}, {}, {})\n" \
                  "Step size: ({:.6f} m, {:.6f} m, {:.6f} rad)\n" \
                  "Best solution: ({}, {}, {})\n" \
                  "Best score: {}\n" \
                  "Initial pose: ({:.6f} m, {:.6f} m, {:.6f} rad)\n" \
                  "Final pose: ({:.6f} m, {:.6f} m, {:.6f} rad)\n" \
                  "# of possible solutions: {}\n" \
                  "# of solutions processed: {}\n"
            print(msg.format(self.win_x * 2, self.win_y * 2, self.win_theta * 2,
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
                 stride: int) -> None:
        self.window_size_x = window_size_x
        self.window_size_y = window_size_y
        self.window_size_theta = window_size_theta
        self.min_step_theta = min_step_theta
        self.stride = stride

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

        # Compute a coarse grid map
        coarse_map = compute_coarse_map(grid_map, self.stride)

        def fine_match(grid_map: Grid2D, indices: List[Tuple[int, int]],
                       offset: Tuple[int, int, int], num_processed: int,
                       best: Tuple[int, int, int], best_score: int):
            offset_x, offset_y, theta = offset

            for y in range(offset_y, offset_y + self.stride):
                for x in range(offset_x, offset_x + self.stride):
                    score = self.compute_score(grid_map, indices, (x, y))
                    num_processed += 1

                    if score > best_score:
                        best, best_score = (x, y, theta), score

            return num_processed, best, best_score

        for t in range(-win_theta, win_theta):
            # Under the fixed angle, the projected scan points `indices` are
            # related by the pure translation for the x and y search directions
            pose = Pose2D(initial_pose.x, initial_pose.y,
                          initial_pose.theta + t * step_theta)
            points = [project_point_2d(pose, p) for p in scan]
            indices = [grid_map.point_to_index(p) for p in points]

            # Perform the coarse match
            for y in range(-win_y, win_y, self.stride):
                for x in range(-win_x, win_x, self.stride):
                    score = self.compute_score(coarse_map, indices, (x, y))
                    num_processed += 1

                    if score <= best_score:
                        continue

                    # Perform the fine match
                    num_processed, (best_x, best_y, best_theta), best_score = \
                        fine_match(grid_map, indices, (x, y, t), num_processed,
                                   (best_x, best_y, best_theta), best_score)

        best_pose = Pose2D(initial_pose.x + best_x * step_x,
                           initial_pose.y + best_y * step_y,
                           initial_pose.theta + best_theta * step_theta)
        best_pose.theta = wrap_angle(best_pose.theta)

        result = ScanMatcherCorrelative.Result(
            win, step, (best_x, best_y, best_theta), best_score,
            initial_pose, best_pose, num_solutions, num_processed)

        return best_pose, result

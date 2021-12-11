# coding: utf-8
# scan_matcher_gauss_newton.py

import math
import numpy as np

from typing import List, Tuple

from grid_2d import Grid2D
from point_2d import Point2D
from pose_2d import Pose2D, project_point_2d, wrap_angle

class ScanMatcherGaussNewton(object):
    """`ScanMatcherGaussNewton` class implements Gauss-Newton based iterative
    scan matching, which aligns a LiDAR scan and a 2D occupancy grid map. \n
    For more details, please refer to the following paper: \n
    Stefan Kohlbrecher, Oskar von Stryk, Johannes Meyer, and Uwe Klingauf.
    "A Flexible and Scalable SLAM System with Full 3D Motion Estimation,"
    in Proceedings of the IEEE International Symposium on Safety, Security,
    and Rescue Robotics (SSRR), 2011.
    """

    def __init__(self) -> None:
        self.max_iterations = 10
        self.convergence_threshold = 1e-4
        self.damping_factor = 1e-6

    def compute_map_grad(self, resolution: float,
                         idx_offset: Tuple[float, float],
                         map_values: Tuple[float, float, float, float]):
        """Compute a gradient of the smoothed occupancy probability
        with respect to the map coordinate"""
        dx, dy = idx_offset
        m00, m01, m10, m11 = map_values
        scale = 1.0 / resolution
        grad_x = scale * (dy * (m11 - m01) + (1.0 - dy) * (m10 - m00))
        grad_y = scale * (dx * (m11 - m10) + (1.0 - dx) * (m01 - m00))
        return np.array([grad_x, grad_y]).reshape(1, -1)

    def compute_scan_point_jacobian(self, pose: Pose2D, point: Point2D):
        """Compute a Jacobian of the scan point with respect to the pose"""
        sin_theta = math.sin(pose.theta)
        cos_theta = math.cos(pose.theta)
        rotated_x = -point.x * sin_theta - point.y * cos_theta
        rotated_y =  point.x * cos_theta - point.y * sin_theta
        return np.array([[1.0, 0.0, rotated_x],
                         [0.0, 1.0, rotated_y]])

    def optimize_step(self, pose: Pose2D, grid_map: Grid2D,
                      scan: List[Point2D]) -> Pose2D:
        hessian = np.zeros((3, 3))
        residual = np.zeros((3, 1))

        for point in scan:
            hit_point = project_point_2d(pose, point)
            _, idx_offset, map_values, interpolated = \
                grid_map.get_neighbor_probabilities(hit_point)
            resolution = grid_map.resolution()

            hit_point_jacobian = self.compute_scan_point_jacobian(pose, point)
            map_grad = self.compute_map_grad(resolution, idx_offset, map_values)
            map_grad_pose = map_grad.dot(hit_point_jacobian).reshape(1, -1)

            hessian += map_grad_pose.transpose() @ map_grad_pose
            residual += (1.0 - interpolated) * map_grad_pose.transpose()

        hessian[np.diag_indices_from(hessian)] += self.damping_factor
        pose_update = np.linalg.solve(hessian, residual)

        return Pose2D(pose.x + pose_update[0],
                      pose.y + pose_update[1],
                      wrap_angle(pose.theta + pose_update[2]))

    def compute_cost(self, pose: Pose2D, grid_map: Grid2D,
                     scan: List[Point2D]) -> float:
        points = [project_point_2d(pose, p) for p in scan]
        values = [grid_map.get_interpolated_probability(p) for p in points]
        residuals = [(1.0 - x) ** 2.0 for x in values]
        return sum(residuals)

    def match_scan(self, initial_pose: Pose2D, grid_map: Grid2D,
                   scan: List[Point2D]) -> Pose2D:
        prev_cost = self.compute_cost(initial_pose, grid_map, scan)
        best_pose = initial_pose

        for _ in range(self.max_iterations):
            best_pose = self.optimize_step(best_pose, grid_map, scan)
            cost = self.compute_cost(best_pose, grid_map, scan)

            if abs(prev_cost - cost) < self.convergence_threshold:
                break
            else:
                prev_cost = cost

        return best_pose

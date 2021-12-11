# coding: utf-8
# pose_2d.py

import math

from point_2d import Point2D

class Pose2D(object):
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def __init__(self, x: float, y: float, theta: float) -> None:
        self.x = x
        self.y = y
        self.theta = theta

def wrap_angle(x: float) -> float:
    """Wrap the angle between -pi to pi"""
    x = math.fmod(x, 2.0 * math.pi)
    return x - 2.0 * math.pi if x > math.pi else \
           x + 2.0 * math.pi if x < -math.pi else x

def angle_difference(start: float, end: float) -> float:
    """Compute the angle difference (between -pi to pi)"""
    diff = wrap_angle(end) - wrap_angle(start)
    return diff - 2.0 * math.pi if diff > math.pi else \
           diff + 2.0 * math.pi if diff < -math.pi else diff

def compute_end_pose_2d(start_pose: Pose2D, diff_pose: Pose2D) -> Pose2D:
    """Compute a composition of 2D poses `start_pose` and `diff_pose`"""
    sin_theta = math.sin(start_pose.theta)
    cos_theta = math.cos(start_pose.theta)

    x = start_pose.x + cos_theta * diff_pose.x - sin_theta * diff_pose.y
    y = start_pose.y + sin_theta * diff_pose.x + cos_theta * diff_pose.y
    theta = wrap_angle(start_pose.theta + diff_pose.theta)

    return Pose2D(x, y, theta)

def compute_diff_pose_2d(start_pose: Pose2D, end_pose: Pose2D) -> Pose2D:
    """Compute a difference between 2D poses `start_pose` and `end_pose`"""
    sin_theta = math.sin(start_pose.theta)
    cos_theta = math.cos(start_pose.theta)

    dx = end_pose.x - start_pose.x
    dy = end_pose.y - start_pose.y
    dtheta = angle_difference(start_pose.theta, end_pose.theta)

    x =  cos_theta * dx + sin_theta * dy
    y = -sin_theta * dx + cos_theta * dy
    theta = dtheta

    return Pose2D(x, y, theta)

def compute_start_pose_2d(end_pose: Pose2D, diff_pose: Pose2D) -> Pose2D:
    """Compute a 2D pose `start_pose` such that the composition of `start_pose`
    and `diff_pose` is `end_pose`"""
    theta = angle_difference(diff_pose.theta, end_pose.theta)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    x = end_pose.x - cos_theta * diff_pose.x + sin_theta * diff_pose.y
    y = end_pose.y - sin_theta * diff_pose.x - cos_theta * diff_pose.y

    return Pose2D(x, y, theta)

def project_point_2d(pose: Pose2D, point: Point2D) -> Point2D:
    """Project a 2D point using a 2D pose"""
    sin_theta = math.sin(pose.theta)
    cos_theta = math.cos(pose.theta)

    x = pose.x + cos_theta * point.x - sin_theta * point.y
    y = pose.y + sin_theta * point.x + cos_theta * point.y

    return Point2D(x, y)

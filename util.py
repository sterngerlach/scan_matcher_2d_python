# coding: utf-8
# util.py

import math
from typing import List, Tuple

from point_2d import Point2D

def interpolate_scan(scan: List[Point2D], dist_scans: float,
                     dist_free_space: float) -> List[Point2D]:
    interpolated = [scan[0]]

    prev_point = scan[0]
    accum_dist = 0.0
    idx = 1

    while idx < len(scan):
        point = scan[idx]
        dist = math.hypot(point.x - prev_point.x, point.y - prev_point.y)

        if accum_dist + dist < dist_scans:
            # Do not interpolate the scan point, since adjacent scan point
            # `point` is too close from the current scan point
            accum_dist += dist
            prev_point = point
        elif accum_dist + dist >= dist_free_space:
            # Do not interpolate the scan point, since adjacent scan point
            # `point` is too far from the current scan point
            interpolated.append(point)
            accum_dist = 0.0
            prev_point = point
        else:
            # Interpolate the scan point
            ratio = (dist_scans - accum_dist) / dist
            x = prev_point.x + (point.x - prev_point.x) * ratio
            y = prev_point.y + (point.y - prev_point.y) * ratio
            new_point = Point2D(x, y)
            interpolated.append(new_point)
            accum_dist = 0.0
            prev_point = new_point
            idx -= 1

        idx += 1

    return interpolated

def bresenham(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    indices = []

    delta_x = abs((x1 - x0) * 2)
    delta_y = abs((y1 - y0) * 2)
    step_x = -1 if x1 - x0 < 0 else 1
    step_y = -1 if y1 - y0 < 0 else 1
    next_x, next_y = x0, y0

    indices.append((next_x, next_y))

    if delta_x > delta_y:
        err = delta_y - delta_x / 2

        while next_x != x1:
            if err >= 0:
                next_y += step_y
                err -= delta_x
            next_x += step_x
            err += delta_y
            indices.append((next_x, next_y))
    else:
        err = delta_x - delta_y / 2

        while next_y != y1:
            if err >= 0:
                next_x += step_x
                err -= delta_y
            next_y += step_y
            err += delta_x
            indices.append((next_x, next_y))

    return indices

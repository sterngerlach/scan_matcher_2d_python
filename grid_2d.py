# coding: utf-8
# grid_2d.py

import math

from collections import deque
from typing import Callable, List, Tuple

from point_2d import Point2D
from pose_2d import Pose2D, project_point_2d
from util import bresenham

def probability_to_value(x: float) -> int:
    return int(x * 65536.0)

def value_to_probability(x: int) -> float:
    return float(x) / 65536.0

def probability_to_log_odds(x: float) -> float:
    return math.log(x / (1.0 - x))

def log_odds_to_probability(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def probability_to_odds(x: float) -> float:
    return x / (1.0 - x)

def odds_to_probability(x: float) -> float:
    return x / (1.0 + x)

def log_odds_to_value(x: float) -> int:
    return probability_to_value(log_odds_to_probability(x))

def value_to_log_odds(x: int) -> float:
    return probability_to_log_odds(value_to_probability(x))

def probability_to_grayscale(x: float) -> int:
    return 255 - int(x * 255.0)

def value_to_grayscale(x: int) -> int:
    return 255 - (x >> 8)

class Grid2D(object):
    """Grid2D class represents a simple 2D grid map. Each grid cell contains
    a quantized occupancy probability, which is incrementally updated by
    Binary Bayes Filter."""

    VALUE_MIN = 1
    VALUE_MAX = 65535
    PROBABILITY_MIN = value_to_probability(VALUE_MIN)
    PROBABILITY_MAX = value_to_probability(VALUE_MAX)
    LOG_ODDS_MIN = probability_to_log_odds(PROBABILITY_MIN)
    LOG_ODDS_MAX = probability_to_log_odds(PROBABILITY_MAX)

    def __init__(self, resolution: float, center_x: float, center_y: float,
                 num_of_cells_x: int, num_of_cells_y: int,
                 log_odds_hit: float, log_odds_miss: float) -> None:
        assert num_of_cells_x > 0
        assert num_of_cells_y > 0
        assert Grid2D.LOG_ODDS_MIN <= log_odds_miss <= Grid2D.LOG_ODDS_MAX
        assert Grid2D.LOG_ODDS_MIN <= log_odds_hit <= Grid2D.LOG_ODDS_MAX

        self.__resolution = resolution
        self.__min_x = center_x - resolution * num_of_cells_x / 2.0
        self.__min_y = center_y - resolution * num_of_cells_y / 2.0
        self.__center_x = center_x
        self.__center_y = center_y
        self.__num_of_cells_x = num_of_cells_x
        self.__num_of_cells_y = num_of_cells_y
        self.__log_odds_miss = log_odds_miss
        self.__log_odds_hit = log_odds_hit

        num_of_cells = num_of_cells_x * num_of_cells_y
        self.__values = [Grid2D.VALUE_MIN for _ in range(num_of_cells)]

    def resolution(self) -> float:
        return self.__resolution

    def num_of_cells_x(self) -> int:
        return self.__num_of_cells_x

    def num_of_cells_y(self) -> int:
        return self.__num_of_cells_y

    def log_odds_miss(self) -> float:
        return self.__log_odds_miss

    def log_odds_hit(self) -> float:
        return self.__log_odds_hit

    def shape(self) -> Tuple[int, int]:
        return self.__num_of_cells_x, self.__num_of_cells_y

    def min_pos(self) -> Tuple[float, float]:
        return self.__min_x, self.__min_y

    def center_pos(self) -> Tuple[float, float]:
        return self.__center_x, self.__center_y

    def flat_index(self, x: int, y: int) -> int:
        return y * self.__num_of_cells_x + x

    def is_index_inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.__num_of_cells_x and \
               0 <= y < self.__num_of_cells_y

    def is_point_inside(self, p: Point2D) -> bool:
        return self.is_index_inside(*self.point_to_index(p))

    def index_to_point(self, x: int, y: int) -> Point2D:
        """Compute the minimum coordinates in the map frame"""
        return Point2D(self.__min_x + x * self.__resolution,
                       self.__min_y + y * self.__resolution)

    def point_to_index(self, p: Point2D) -> Tuple[int, int]:
        """Compute the index from the coordinates in the map frame"""
        idx_x = int((p.x - self.__min_x) // self.__resolution)
        idx_y = int((p.y - self.__min_y) // self.__resolution)
        return idx_x if p.x >= self.__min_x else idx_x - 1, \
               idx_y if p.y >= self.__min_y else idx_y - 1

    def point_to_index_float(self, p: Point2D) -> Tuple[float, float]:
        """Compute the index in the floating point in the map frame"""
        idx_x = (p.x - self.__min_x) / self.__resolution
        idx_y = (p.y - self.__min_y) / self.__resolution
        return idx_x, idx_y

    def get_index_value(self, x: int, y: int) -> int:
        assert self.is_index_inside(x, y)
        return self.__values[self.flat_index(x, y)]

    def get_index_probability(self, x: int, y: int) -> float:
        assert self.is_index_inside(x, y)
        return value_to_probability(self.__values[self.flat_index(x, y)])

    def get_point_value(self, p: Point2D) -> int:
        return self.get_index_value(*self.point_to_index(p))

    def get_point_probability(self, p: Point2D) -> float:
        return self.get_index_probability(*self.point_to_index(p))

    def get_neighbor_probabilities(self, p: Point2D):
        x, y = self.point_to_index_float(p)
        x0, y0 = int(x), int(y)
        x1, y1 = x0 + 1, y0 + 1
        dx, dy = x - x0, y - y0

        x0_valid = 0 <= x0 < self.__num_of_cells_x
        x1_valid = 0 <= x1 < self.__num_of_cells_x
        y0_valid = 0 <= y0 < self.__num_of_cells_y
        y1_valid = 0 <= y1 < self.__num_of_cells_y

        m00 = self.get_index_probability(x0, y0) \
              if x0_valid and y0_valid else 0.5
        m01 = self.get_index_probability(x0, y1) \
              if x0_valid and y1_valid else 0.5
        m10 = self.get_index_probability(x1, y0) \
              if x1_valid and y0_valid else 0.5
        m11 = self.get_index_probability(x1, y1) \
              if x1_valid and y1_valid else 0.5

        interpolated = dy * (dx * m11 + (1.0 - dx) * m01) + \
                       (1.0 - dy) * (dx * m10 + (1.0 - dx) * m00)

        return (x0, y0), (dx, dy), (m00, m01, m10, m11), interpolated

    def get_interpolated_probability(self, p: Point2D) -> float:
        return self.get_neighbor_probabilities(p)[3]

    def set_index_value(self, x: int, y: int, value: int) -> None:
        assert self.is_index_inside(x, y)
        self.__values[self.flat_index(x, y)] = self.__clamp_value(value)

    def set_index_probability(self, x: int, y: int, value: float) -> None:
        assert self.is_index_inside(x, y)
        self.__values[self.flat_index(x, y)] = \
             self.__clamp_value(probability_to_value(value))

    def set_point_value(self, p: Point2D, value: int) -> None:
        self.set_index_value(*self.point_to_index(p), value)

    def set_point_probability(self, p: Point2D, value: float) -> None:
        self.set_index_probability(*self.point_to_index(p), value)

    def __update_index_log_odds(self, x: int, y: int, value: float) -> None:
        assert self.is_index_inside(x, y)
        idx = self.flat_index(x, y)
        log_odds = value_to_log_odds(self.__values[idx]) + value
        self.__values[idx] = self.__clamp_value(log_odds_to_value(log_odds))

    def update_index_hit(self, x: int, y: int) -> None:
        self.__update_index_log_odds(x, y, self.__log_odds_hit)

    def update_index_miss(self, x: int, y: int) -> None:
        self.__update_index_log_odds(x, y, self.__log_odds_miss)

    def update_point_hit(self, p: Point2D) -> None:
        self.update_index_hit(*self.point_to_index(p))

    def update_point_miss(self, p: Point2D) -> None:
        self.update_index_miss(*self.point_to_index(p))

    def update_from_scans(self, pose: Pose2D, scan: List[Point2D]) -> None:
        pose_idx = self.point_to_index(Point2D(pose.x, pose.y))
        points = [project_point_2d(pose, p) for p in scan]
        scan_indices = [self.point_to_index(p) for p in points]
        ray_indices = [bresenham(*pose_idx, *idx) for idx in scan_indices]
        # hit_indices = [indices[-1] for indices in ray_indices]

        for indices in ray_indices:
            for missed_idx in indices[:-1]:
                if self.is_index_inside(*missed_idx):
                    self.update_index_miss(*missed_idx)

            if self.is_index_inside(*indices[-1]):
                self.update_index_hit(*indices[-1])

    def __clamp_value(self, x: int) -> int:
        return max(Grid2D.VALUE_MIN, min(Grid2D.VALUE_MAX, x))

    def to_bytes(self) -> bytes:
        return bytes([value_to_grayscale(x) for x in self.__values])

def sliding_window_max(in_func: Callable[[int], int],
                       out_func: Callable[[int, int], None],
                       num_of_elements: int, win_size: int) -> None:
    idx_queue, idx_in, idx_out = deque(), 0, 0

    # Process the first `win_size` elements (first window)
    while idx_in < win_size:
        # Previous smaller elements are useless so remove them from `idx_queue`
        while idx_queue and in_func(idx_in) >= in_func(idx_queue[-1]):
            idx_queue.pop()
        idx_queue.append(idx_in)
        idx_in += 1

    # `idx_queue[0]` contains index of the maximum element in the first window
    # Process the rest of the elements here
    while idx_in < num_of_elements:
        # The element pointed by the `idx_queue[0]` is the maximum element of
        # the previous window
        out_func(idx_out, in_func(idx_queue[0]))
        idx_out += 1

        # Remove the elements that are out of the current window
        while idx_queue and idx_queue[0] <= idx_in - win_size:
            idx_queue.popleft()

        # Remove all elements smaller than the current element
        while idx_queue and in_func(idx_in) >= in_func(idx_queue[-1]):
            idx_queue.pop()

        # Append the current element to `idx_queue`
        idx_queue.append(idx_in)
        idx_in += 1

    # Repeat the last elements
    while idx_out < num_of_elements:
        out_func(idx_out, in_func(idx_queue[0]))
        idx_out += 1

def compute_coarse_map_slow(grid_map: Grid2D, win_size: int) -> Grid2D:
    num_of_cells = grid_map.num_of_cells_x() * grid_map.num_of_cells_y()
    buffer = [0 for _ in range(num_of_cells)]
    coarse_map = Grid2D(grid_map.resolution(), *grid_map.center_pos(),
                        grid_map.num_of_cells_x(), grid_map.num_of_cells_y(),
                        grid_map.log_odds_hit(), grid_map.log_odds_miss())

    def get_buffer(idx_x: int, idx_y: int) -> int:
        return buffer[idx_y * grid_map.num_of_cells_x() + idx_x]

    def set_buffer(idx_x: int, idx_y: int, value: int) -> None:
        buffer[idx_y * grid_map.num_of_cells_x() + idx_x] = value

    def set_coarse_map(idx_x: int, idx_y: int, value: int) -> None:
        coarse_map.set_index_value(idx_x, idx_y, value)

    # Sliding window maximum for y-axis
    for x in range(grid_map.num_of_cells_x()):
        in_func_row = lambda y: grid_map.get_index_value(x, y)
        out_func_row = lambda y, max_value: set_buffer(x, y, max_value)
        sliding_window_max(in_func_row, out_func_row, \
                           grid_map.num_of_cells_y(), win_size)

    # Sliding window maximum for x-axis
    for y in range(grid_map.num_of_cells_y()):
        in_func_col = lambda x: get_buffer(x, y)
        out_func_col = lambda x, max_value: set_coarse_map(x, y, max_value)
        sliding_window_max(in_func_col, out_func_col, \
                           grid_map.num_of_cells_x(), win_size)

    return coarse_map

def compute_coarse_map_fast(grid_map: Grid2D, win_size: int) -> Grid2D:
    num_of_cells_x = grid_map.num_of_cells_x()
    num_of_cells_y = grid_map.num_of_cells_y()
    num_of_cells = num_of_cells_x * num_of_cells_y
    buffer = [0 for _ in range(num_of_cells)]
    coarse_map = Grid2D(grid_map.resolution(), *grid_map.center_pos(),
                        grid_map.num_of_cells_x(), grid_map.num_of_cells_y(),
                        grid_map.log_odds_hit(), grid_map.log_odds_miss())

    # Sliding window maximum for y-axis
    for x in range(num_of_cells_x):
        idx_queue, idx_in, idx_out = deque(), 0, 0

        while idx_in < win_size:
            while idx_queue and grid_map.get_index_value(x, idx_in) >= \
                grid_map.get_index_value(x, idx_queue[-1]):
                idx_queue.pop()
            idx_queue.append(idx_in)
            idx_in += 1

        while idx_in < num_of_cells_y:
            buffer[idx_out * num_of_cells_x + x] = \
                grid_map.get_index_value(x, idx_queue[0])
            idx_out += 1

            while idx_queue and idx_queue[0] <= idx_in - win_size:
                idx_queue.popleft()
            while idx_queue and grid_map.get_index_value(x, idx_in) >= \
                grid_map.get_index_value(x, idx_queue[-1]):
                idx_queue.pop()

            idx_queue.append(idx_in)
            idx_in += 1

        while idx_out < num_of_cells_y:
            buffer[idx_out * num_of_cells_x + x] = \
                grid_map.get_index_value(x, idx_queue[0])
            idx_out += 1

    # Sliding window maximum for x-axis
    for y in range(num_of_cells_y):
        idx_queue, idx_in, idx_out = deque(), 0, 0
        idx_offset = y * num_of_cells_x

        while idx_in < win_size:
            while idx_queue and buffer[idx_offset + idx_in] >= \
                buffer[idx_offset + idx_queue[-1]]:
                idx_queue.pop()
            idx_queue.append(idx_in)
            idx_in += 1

        while idx_in < num_of_cells_x:
            coarse_map.set_index_value(
                idx_out, y, buffer[idx_offset + idx_queue[0]])
            idx_out += 1

            while idx_queue and idx_queue[0] <= idx_in - win_size:
                idx_queue.popleft()
            while idx_queue and buffer[idx_offset + idx_in] >= \
                buffer[idx_offset + idx_queue[-1]]:
                idx_queue.pop()

            idx_queue.append(idx_in)
            idx_in += 1

        while idx_out < num_of_cells_x:
            coarse_map.set_index_value(
                idx_out, y, buffer[idx_offset + idx_queue[0]])
            idx_out += 1

    return coarse_map

def compute_coarse_map(grid_map: Grid2D, win_size: int) -> Grid2D:
    return compute_coarse_map_fast(grid_map, win_size)

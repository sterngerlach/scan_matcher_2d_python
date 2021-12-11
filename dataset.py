# coding: utf-8
# dataset.py

import json
import math
import os
import random
import sys
from typing import List, Tuple

from grid_2d import Grid2D, probability_to_log_odds
from point_2d import Point2D
from pose_2d import Pose2D, wrap_angle

def load_dataset(data_idx: int, perturb_x: float, perturb_y: float,
                 perturb_theta: float) \
    -> Tuple[Grid2D, List[Point2D], Pose2D, Pose2D]:
    grid_map_file_name = "./dataset/{}-map.json".format(data_idx)
    scan_file_name = "./dataset/{}-scan.json".format(data_idx)

    if not os.path.isfile(grid_map_file_name):
        print("Grid map file does not exist: {}".format(grid_map_file_name))
        sys.exit(1)
    if not os.path.isfile(scan_file_name):
        print("Scan file does not exist: {}".format(scan_file_name))
        sys.exit(1)

    # Load an occupancy grid map from JSON file
    grid_map_file = open(grid_map_file_name, "r", encoding="utf-8")
    grid_map_json = json.load(grid_map_file)
    grid_map_file.close()

    resolution = float(grid_map_json["Resolution"])
    num_of_cells_x = int(grid_map_json["NumOfCellsX"])
    num_of_cells_y = int(grid_map_json["NumOfCellsY"])
    center_x = float(grid_map_json["CenterX"])
    center_y = float(grid_map_json["CenterY"])
    probabilities = list(map(float, grid_map_json["Probabilities"].split()))
    assert len(probabilities) == num_of_cells_x * num_of_cells_y

    grid_map = Grid2D(resolution=resolution,
                      center_x=center_x, center_y=center_y,
                      num_of_cells_x=num_of_cells_x,
                      num_of_cells_y=num_of_cells_y,
                      log_odds_hit=probability_to_log_odds(0.62),
                      log_odds_miss=probability_to_log_odds(0.46))

    for y in range(num_of_cells_y):
        for x in range(num_of_cells_x):
            grid_map.set_index_probability(
                x, y, probabilities[y * num_of_cells_x + x])

    # Load a LiDAR scan from JSON file
    scan_file = open(scan_file_name, "r", encoding="utf-8")
    scan_json = json.load(scan_file)
    scan_file.close()

    num_of_scans = int(scan_json["NumOfScans"])
    ranges = list(map(float, scan_json["Ranges"].split()))
    angles = list(map(float, scan_json["Angles"].split()))
    assert len(ranges) == num_of_scans
    assert len(angles) == num_of_scans

    scan = [Point2D(r * math.cos(t), r * math.sin(t)) \
            for r, t in zip(ranges, angles)]

    # Load the final scan pose
    final_pose = Pose2D(float(scan_json["FinalPoseX"]),
                        float(scan_json["FinalPoseY"]),
                        float(scan_json["FinalPoseTheta"]))

    # Perturb the final scan pose and get the initial scan pose
    rand_x = random.random() * perturb_x - perturb_x / 2.0
    rand_y = random.random() * perturb_y - perturb_y / 2.0
    rand_theta = random.random() * perturb_theta - perturb_theta / 2.0
    initial_pose = Pose2D(final_pose.x + rand_x,
                          final_pose.y + rand_y,
                          wrap_angle(final_pose.theta + rand_theta))

    return grid_map, scan, initial_pose, final_pose

#!/usr/bin/env python3
# coding: utf-8
# main.py

from PIL import Image, ImageDraw
from typing import List, Tuple

from dataset import load_dataset
from grid_2d import Grid2D
from point_2d import Point2D
from pose_2d import Pose2D, project_point_2d
from scan_matcher_branch_bound import ScanMatcherBranchBound
from scan_matcher_correlative import ScanMatcherCorrelative

def draw_scan(image: Image, grid_map: Grid2D,
              pose: Pose2D, scan: List[Point2D],
              color=Tuple[float, float, float, float]) -> None:
    draw = ImageDraw.Draw(image)

    for point in scan:
        hit_point = project_point_2d(pose, point)
        hit_idx = grid_map.point_to_index(hit_point)
        if grid_map.is_index_inside(*hit_idx):
            rect = [(hit_idx[0], hit_idx[1]),
                    (hit_idx[0] + 1, hit_idx[1] + 1)]
            draw.rectangle(rect, color)

def main():
    # Load the dataset
    data_idx = 7
    perturb_x = 10.0
    perturb_y = 10.0
    perturb_theta = 0.2
    grid_map, scan, initial_pose, _ = \
        load_dataset(data_idx, perturb_x, perturb_y, perturb_theta)

    # Perform branch-and-bound scan matching
    scan_matcher_branch = ScanMatcherBranchBound(
        window_size_x=perturb_x, window_size_y=perturb_y,
        window_size_theta=perturb_theta, min_step_theta=0.0025,
        max_node_height=6)
    pose_branch, stats_branch = \
        scan_matcher_branch.match_scan(initial_pose, grid_map, scan)
    stats_branch.dump()

    # Perform correlative scan matching
    # scan_matcher_correlative = ScanMatcherCorrelative(
    #     window_size_x=perturb_x, window_size_y=perturb_y,
    #     window_size_theta=perturb_theta, min_step_theta=0.0025, stride=8)
    # pose_correlative, stats_correlative = \
    #     scan_matcher_correlative.match_scan(initial_pose, grid_map, scan)
    # stats_correlative.dump()

    # Visualize the grid map
    image = Image.frombytes("L", grid_map.shape(), grid_map.to_bytes())
    image = image.convert("RGB")

    draw_scan(image, grid_map, initial_pose, scan, (0, 0, 255, 0))

    image_before = image.copy()
    image_before = image_before.transpose(Image.FLIP_TOP_BOTTOM)
    image_before.save("before-{}.png".format(data_idx))
    image_before.show()

    draw_scan(image, grid_map, pose_branch, scan, (255, 0, 0, 0))
    # draw_scan(image, grid_map, pose_correlative, scan, (0, 255, 0, 0))

    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save("after-{}.png".format(data_idx))
    image.show()

if __name__ == "__main__":
    main()

# coding: utf-8
# visualize_data.py

from PIL import Image, ImageDraw

from pose_2d import project_point_2d
from dataset import load_dataset

def main():
    # Load the dataset
    grid_map, scan, initial_pose, final_pose = load_dataset(
        data_idx=7, perturb_x=10.0, perturb_y=10.0, perturb_theta=1.0)
    draw_initial_scan = True
    draw_final_scan = False

    # Visualize the grid map
    image = Image.frombytes("L", grid_map.shape(), grid_map.to_bytes())
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    for point in scan:
        if draw_initial_scan:
            initial_hit_point = project_point_2d(initial_pose, point)
            initial_hit_idx = grid_map.point_to_index(initial_hit_point)
            if grid_map.is_index_inside(*initial_hit_idx):
                rect = [(initial_hit_idx[0], initial_hit_idx[1]),
                        (initial_hit_idx[0] + 1, initial_hit_idx[1] + 1)]
                draw.rectangle(rect, (0, 0, 255, 0))

        if draw_final_scan:
            hit_point = project_point_2d(final_pose, point)
            hit_idx = grid_map.point_to_index(hit_point)
            if grid_map.is_index_inside(*hit_idx):
                rect = [(hit_idx[0], hit_idx[1]),
                        (hit_idx[0] + 1, hit_idx[1] + 1)]
                draw.rectangle(rect, (255, 0, 0, 0))

    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.show()

if __name__ == "__main__":
    main()

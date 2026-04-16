import pandas as pd
import json
import numpy as np
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R
import os

def _cv_to_gl(cv):
    # convert to GL convention used in iNGP
    gl = cv * np.array([1, -1, -1, 1])
    return gl

def find_closest_point(p1, d1, p2, d2):
    # Calculate the direction vectors of the lines
    d1_norm = d1 / np.linalg.norm(d1)
    d2_norm = d2 / np.linalg.norm(d2)

    # Create the coefficient matrix A and the constant vector b
    A = np.vstack((d1_norm, -d2_norm)).T
    b = p2 - p1

    # Solve the linear system to find the parameters t1 and t2
    t1, t2 = np.linalg.lstsq(A, b, rcond=None)[0]

    # Calculate the closest point on each line
    closest_point1 = p1 + d1_norm * t1
    closest_point2 = p2 + d2_norm * t2

    # Calculate the average of the two closest points
    closest_point = 0.5 * (closest_point1 + closest_point2)

    return closest_point

def bound_by_pose(poses):

    center = np.array([0.0, 0.0, 0.0])
    for f in poses:
        src_frame = f[0:3, :]
        for g in poses:
            tgt_frame = g[0:3, :]
            p = find_closest_point(src_frame[:, 3], src_frame[:, 2], tgt_frame[:, 3], tgt_frame[:, 2])
            center += p
    center /= len(poses) ** 2

    radius = 0.0
    for f in poses:
        radius += np.linalg.norm(f[0:3, 3])
    radius /= len(poses)
    bounding_box = [
        [center[0] - radius, center[0] + radius],
        [center[1] - radius, center[1] + radius],
        [center[2] - radius, center[2] + radius],
    ]
    return center, radius, bounding_box

def bound_by_points(xyzs):
    center = xyzs.mean(axis=0)
    std = xyzs.std(axis=0)
    radius = float(std.max() * 2)  # use 2*std to define the region, equivalent to 95% percentile
    bounding_box = [
        [center[0] - std[0] * 3, center[0] + std[0] * 3],
        [center[1] - std[1] * 3, center[1] + std[1] * 3],
        [center[2] - std[2] * 3, center[2] + std[2] * 3],
    ]
    return center, radius, bounding_box

def capture_stage_to_json(args):

    # Read the calibration and aabbs data
    with open(args.calibration_dir) as file:
        calibration_json = json.load(file)

    cameras = calibration_json["cameras"]

    json_data = {
        "is_fisheye": False,  # TODO:
        "frames": [],
    }

    camera_locations = []

    for camera in cameras:
        w, h = camera["intrinsics"]["resolution"]
        json_data["w"] = w
        json_data["h"] = h
        name = camera["camera_id"]
        if not os.path.exists(f"datasets/vci_capture_mohammed/{args.frame_num}/rgb_masked/{name}.png"):
            continue
        w2c = np.array(camera["extrinsics"]["view_matrix"]).reshape((4,4))
        c2w = _cv_to_gl(np.linalg.inv(w2c))
        intrinsics = np.array(camera["intrinsics"]["camera_matrix"]).reshape((3,3))
        fx = intrinsics[0][0]
        fy = intrinsics[1][1]
        cx = intrinsics[0][2]
        cy = intrinsics[1][2]

        angle_x = np.arctan(w / (fx * 2)) * 2
        angle_y = np.arctan(h / (fy * 2)) * 2

        frame = {"file_path": os.path.join(args.frame_num, f"rgb_masked/{name}.png"), "transform_matrix": c2w.tolist(), "camera_angle_x": angle_x, "camera_angle_y": angle_y, "fx": fx, "fy": fy, "cx": cx, "py": cy}
        
        json_data["frames"].append(frame)

        camera_locations.append(w2c[:-1,-1].reshape(-1,3))

    camera_locations = np.concatenate(camera_locations, axis=0)

    center, radius, bounding_box = bound_by_points(camera_locations)
    radius = 2.0

    json_data["aabb_scale"] = np.exp2(np.rint(np.log2(radius)))
    json_data["aabb_range"] = bounding_box
    json_data["sphere_center"] = center.tolist()
    json_data["sphere_radius"] = radius

    with open(args.out_dir, "w") as outputfile:
        json.dump(json_data, outputfile, indent=2)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--calibration_dir", type=str, help="Path to calibration.csv")
    parser.add_argument("--out_dir", type=str, help="Path to the output transforms.json")
    parser.add_argument("--frame_num", type=str)

    args = parser.parse_args()
    capture_stage_to_json(args)

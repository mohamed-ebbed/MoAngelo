import pandas as pd
import json
import numpy as np
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R
import os
import shutil
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

def actorshq_to_json(args):

    # Read the calibration and aabbs data
    calibration_dir = os.path.join(args.dataroot,"calibration.csv")
    aabbs_dir = os.path.join(args.dataroot, "../aabbs.csv")
    calibration_df = pd.read_csv(calibration_dir)
    aabbs_df = pd.read_csv(aabbs_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "masks"), exist_ok=True)

    if args.aspect_ratio == "height_larger":
        calibration_df = calibration_df[calibration_df["h"] > calibration_df["w"]]
    else:
        calibration_df = calibration_df[[calibration_df["w"] > calibration_df["h"]]]

    fx, fy, px, py = calibration_df["fx"].to_numpy(), calibration_df["fy"].to_numpy(), calibration_df["px"].to_numpy(), calibration_df["py"].to_numpy()

    w, h = calibration_df["w"].to_numpy(), calibration_df["h"].to_numpy()

    fx *= w
    fy *= h
    px *= w
    py *= h

    camera_names = list(calibration_df["name"])

    rx, ry, rz = calibration_df["rx"].to_numpy(), calibration_df["ry"].to_numpy(), calibration_df["rz"].to_numpy()
    tx, ty, tz = calibration_df["tx"].to_numpy(), calibration_df["ty"].to_numpy(), calibration_df["tz"].to_numpy()


    rot_mat = R.from_rotvec(np.array([rx,ry,rz]).transpose()).as_matrix()

    transformation_matrix = np.zeros((rot_mat.shape[0], 4, 4))
    transformation_matrix[:,:3,:3] = rot_mat
    transformation_matrix[:,0,3] = tx
    transformation_matrix[:,1,3] = ty
    transformation_matrix[:,2,3] = tz
    transformation_matrix[:,3,3] = 1

    angle_x = np.arctan(w / (fx * 2)) * 2
    angle_y = np.arctan(h / (fy * 2)) * 2

    #frame_bounds_data = aabbs_df[aabbs_df["frame_number"] == 0]
    frame_bounds_data = aabbs_df[args.frame_start:args.frame_end+1]
    bounds = np.array([[frame_bounds_data["aabb_min_x"].min(), frame_bounds_data["aabb_max_x"].max()],
    [frame_bounds_data["aabb_min_y"].min(), frame_bounds_data["aabb_max_y"].max()],
    [frame_bounds_data["aabb_min_z"].min(), frame_bounds_data["aabb_max_z"].max()]]).squeeze()

    min_x, max_x = bounds[0]
    min_y, max_y = bounds[1]
    min_z, max_z = bounds[2]

    bbox = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])

    center = bbox.mean(axis=0)

    bbox_centered = bbox - center

    radius = np.linalg.norm(bbox_centered, axis=-1).max()

    print(radius)
    
    json_data = {
        "is_fisheye": False,  # TODO: not supporting fish eye camera
        "w": int(w[0]),
        "h": int(h[0]),
        "frames": [],
    }

    points = []

    for i in range(len(camera_names)):
        c2w = transformation_matrix[i]
        c2w = _cv_to_gl(c2w)  # convert to GL convention used in iNGP
        points.append(np.expand_dims(c2w[:3,-1],0))

        time_step = 0

        for j in range(args.frame_start, args.frame_end+1):
            time_step += 1
            image_dir = os.path.join(args.output_dir, "images", f"{camera_names[i]}_rgb{j:06d}.png")
            shutil.copyfile(os.path.join(args.dataroot, "rgbs", camera_names[i], f"{camera_names[i]}_rgb{j:06d}.jpg"), image_dir)
            shutil.copyfile(os.path.join(args.dataroot, "masks", camera_names[i], f"{camera_names[i]}_mask{j:06d}.png"), os.path.join(args.output_dir, "masks", f"{camera_names[i]}_mask{j:06d}.png"))
            frame = {"file_path": os.path.join("images", f"{camera_names[i]}_rgb{j:06d}.png"), "transform_matrix": c2w.tolist(), "camera_angle_x": angle_x[i], "camera_angle_y": angle_y[i], "fx": fx[i], "fy": fy[i], "cx": px[i], "py": py[i], "time_step": time_step}
            json_data["frames"].append(frame)
    json_data["num_frames"] = args.frame_end - args.frame_start + 1
    points = np.concatenate(points,axis=0)

    json_data["num_cameras"] = len(camera_names)

    print(json_data["num_frames"])
    
    #center, radius, bounding_box = bound_by_points(points)

    # "aabb_scale": np.exp2(np.rint(np.log2(radius))),  # power of two, for INGP resolution computation
    #     "aabb_range": bounding_box,
    #     "sphere_center": center,
    #     "sphere_radius": radius,

    json_data["aabb_scale"] = np.exp2(np.rint(np.log2(radius)))
    json_data["aabb_range"] = bounds.tolist()
    json_data["sphere_center"] = center.tolist()
    json_data["sphere_radius"] = radius

    with open(os.path.join(args.output_dir, "transforms.json"), "w") as outputfile:
        json.dump(json_data, outputfile, indent=2)
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--frame_start", type=int, default=0, help = "start frame")
    parser.add_argument("--frame_end", type=int, default=0, help = "end frame")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to data")
    parser.add_argument("--dataroot", type=str, help = "dataset_directory")    
    parser.add_argument("--aspect_ratio", type=str, help = "select from height_larger, width_larger or both")

    args = parser.parse_args()
    actorshq_to_json(args)

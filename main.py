import os
import math
import logging
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL                         import Image
from Matching.matcher            import Matcher
from utils.config                import load_config
from motion_estimation.estimator import MotionEstimator
from Extraction.extractor        import FeatureExtractor
from utils.image_utils           import convert_pil_to_cv
from utils.dataset               import AdverCityDataset, process_dataset
from utils.plotting              import  plot_trajectories_3d,compare_trajectories_3d

#function to convert euler angles to rotation matrix
def euler_to_matrix(roll, pitch, yaw, degrees=True):
    """Convert Euler angles (roll, pitch, yaw) to 3x3 rotation."""
    if degrees:
        roll  = math.radians(roll)
        pitch = math.radians(pitch)
        yaw   = math.radians(yaw)
    Rx = np.array([
        [1,           0,            0],
        [0,  math.cos(roll), -math.sin(roll)],
        [0,  math.sin(roll),  math.cos(roll)]
    ])
    Ry = np.array([
        [ math.cos(pitch),  0, math.sin(pitch)],
        [             0,    1,              0],
        [-math.sin(pitch),  0, math.cos(pitch)]
    ])
    Rz = np.array([
        [ math.cos(yaw), -math.sin(yaw), 0],
        [ math.sin(yaw),  math.cos(yaw), 0],
        [           0,               0, 1]
    ])
    # typical Rz * Ry * Rx
    return Rz @ Ry @ Rx

def main():
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config_path = os.path.join('configs', 'config.yaml')
    config      = load_config(config_path)
    root        = r'/Users/moezrashed/Documents/Programming/Python/QUARRG/ui_cd_s'
    # Dataset
    dataset_cam = AdverCityDataset(root, cam=0, car=0)

    # Initialize camera data
    camera_data = {
        'camera0': {
            'images': [],
            'intrinsics': [],
            'extrinsics': [],
            'vehicle_pos': []
        }
    }
    cam_refs = {
        'camera0': []
    }
    
    process_dataset(dataset_cam, "camera0", camera_data, cam_refs)
    
    camera_intrinsics = camera_data['camera0']['intrinsics'][0]
    camera_extrinsics = camera_data['camera0']['extrinsics'][0]
    ground_truth_3d   = [gt[:3] for gt in cam_refs['camera0']]
    x, y, z = ground_truth_3d[0]
    roll, yaw , pitch = cam_refs['camera0'][0][3:6]
    speedometer       = [gt[6] for gt in cam_refs['camera0']]
    speedometer       = [speed * 1000 / 3600 for speed in speedometer] # Convert to m/s

    R = euler_to_matrix(roll, -pitch, -yaw, degrees=True)
    extrinsics_0 = np.eye(4)
    # extrinsics_0[:3,:3] = R
    extrinsics_0[:3, 3] = [z, -x, -y]

    # Initialize pipeline
    extractor         = FeatureExtractor(config)
    matcher           = Matcher(config)
    motion_estimator  = MotionEstimator(camera_matrix=np.array(camera_intrinsics),extrinsics=extrinsics_0)
    
    ground_truth_3d = [[z,-x,-y] for x,y,z in ground_truth_3d]

    # Initialize estimated trajectory
    estimated_3d = [ground_truth_3d[0].copy()]

    # estimated_3d      = [[0,0,0]]
    dt                = 0.1  # 0.1-second timestep
    previous_features = None

    for idx, image in enumerate(camera_data["camera0"]["images"]):
        logging.info(f"Processing image {idx+1}/{len(camera_data['camera0']['images'])}")

        # Extract features
        keypoints, descriptors, _ = extractor(image)

        if previous_features is not None:
            prev_keypoints, prev_descriptors, _ = previous_features

            # Match
            matches = matcher(prev_descriptors, descriptors)
            logging.info(f"Found {len(matches)} matches between frames {idx} and {idx+1}.")

            if len(matches) < 10:
                # Maintain continuity
                estimated_3d.append(estimated_3d[-1].copy())
            else:
                # Estimate relative pose
                pose = motion_estimator.estimate_motion(prev_keypoints, keypoints, matches)
                
                # Extract 3D translation
                translation_3d = pose[:3, 3].tolist()
                # Magnitude
                mag_3d = np.linalg.norm(translation_3d)
                # If we have speed for this index
                gt_speed = speedometer[idx] if idx < len(speedometer) else 0.0

                if mag_3d > 0:
                    # Distance expected
                    expected_dist = gt_speed * dt
                    scale_factor  = expected_dist / mag_3d
                    logging.info(f"Scale factor for frame {idx+1} = {scale_factor:.4f}")

                    scaled_3d = [comp * scale_factor for comp in translation_3d]
                else:
                    logging.warning("Zero magnitude translation. Skipping scaling.")
                    scaled_3d = [0.0, 0.0, 0.0]

                new_est_3d = [
                    estimated_3d[-1][0] + scaled_3d[0],
                    estimated_3d[-1][1] + scaled_3d[1],
                    estimated_3d[-1][2] + scaled_3d[2]
                ]
                
                logging.info(f"New 3D position: {new_est_3d}")
                estimated_3d.append(new_est_3d)

        else:
            logging.info("No previous features, re-using initial position.")
            # Maintain continuity
            estimated_3d.append(estimated_3d[-1].copy())

        previous_features = (keypoints, descriptors, _)
    
    # ground_truth_3d = np.array(ground_truth_3d)
    # logging.info(f"Ground truth 3D: {ground_truth_3d.shape}")
    # ground_truth_3d = np.flip(ground_truth_3d, (2,0))
    # ground_truth_3d = np.flip(ground_truth_3d, (1,2))
    # ground_truth_3d[:,[1,2]] *= -1
    # estimated_3d_transformed = [ground_truth_3d[0].copy()]

    # axis_fix = np.array([
    #         [ 0, 0,  1],
    #         [ -1, 0,  0],
    #         [ 0, 1 ,  0]
    #     ], dtype=float)
    # for pt in estimated_3d:
    #     pt = np.array(pt)
    #     pt = axis_fix @ pt
    #     pt += ground_truth_3d[0]
    #     estimated_3d_transformed.append(pt.tolist())
    #     logging.info(f"Transformed 3D position: {pt}")
    
    logging.info(f"My 3D trajectory: {estimated_3d}")
    logging.info(f"Ground truth 3D: {ground_truth_3d}")
   
    results_folder = 'Results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        logging.info(f"Created folder '{results_folder}' for storing results.")
    else:
        logging.info(f"Folder '{results_folder}' already exists.")

    if estimated_3d and ground_truth_3d:
        # Trim 3D
        length_3d = min(len(estimated_3d), len(ground_truth_3d))
        est_3d_trim = estimated_3d[:length_3d]
        gt_3d_trim  = ground_truth_3d[:length_3d]

        # Plot & Compare 3D
        plot_trajectories_3d(gt_3d_trim, estimated_3d, title="3D Trajectory Comparison")
        plt.savefig(os.path.join(results_folder, 'trajectory_comparison_3d.png'))
        logging.info("Saved 3D trajectory plot.")

    else:
        logging.error("No 3D data for plotting.")

if __name__ == "__main__":
    main()
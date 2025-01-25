import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_trajectories_3d(gt_coords, est_coords, title="3D Trajectory Comparison"):
    """
    Plots two 3D trajectories (ground truth and estimated).
    Expects arrays of shape (N, 3).
    """
    gt  = np.array(gt_coords)
    est = np.array(est_coords)

    if gt.shape[1] != 3 or est.shape[1] != 3:
        raise ValueError("gt_coords and est_coords must be Nx3 for 3D plotting.")

    x_gt,  y_gt,  z_gt  = gt[:, 0],  gt[:, 1],  gt[:, 2]
    x_est, y_est, z_est = est[:, 0], est[:, 1], est[:, 2]

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')

    ax.plot(x_gt, y_gt, z_gt, 'r-o', label='GT')
    ax.plot(x_est,y_est,z_est,'b--^',label='Estimated')

    # Mark start & end
    ax.scatter(x_gt[0],  y_gt[0],  z_gt[0],  color='green',  s=50, label='GT Start')
    ax.scatter(x_gt[-1], y_gt[-1], z_gt[-1], color='black',  s=50, label='GT End')
    ax.scatter(x_est[0], y_est[0], z_est[0], color='green',  marker='^', s=50, label='Est Start')
    ax.scatter(x_est[-1],y_est[-1],z_est[-1], color='black',  marker='^', s=50, label='Est End')

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()
    plt.tight_layout()
    plt.show()


def compare_trajectories_3d(est_coords, gt_coords, title="3D Error Over Time"):
    """
    Plots the 3D Euclidean error (per frame).
    Expects arrays of shape (N, 3).
    """
    est = np.array(est_coords)
    gt  = np.array(gt_coords)

    if est.shape != gt.shape:
        raise ValueError("Estimated and ground_truth must have same shape for 3D error.")

    errors = np.linalg.norm(est - gt, axis=1)

    plt.figure(figsize=(8,5))
    plt.plot(errors, 'b-', label='3D Error (m)')
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Error (m)")
    plt.grid(True)
    plt.legend()
    plt.show()

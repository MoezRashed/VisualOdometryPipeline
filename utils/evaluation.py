import numpy as np

def compute_rmse(estimated, ground_truth):
    """
    Computes the Root Mean Square Error between estimated and ground truth trajectories.

    Parameters:
    - estimated (list of lists): Estimated trajectory points [[x1, y1], [x2, y2], ...]
    - ground_truth (list of lists): Ground truth trajectory points [[x1, y1], [x2, y2], ...]

    Returns:
    - rmse (float): Root Mean Square Error in meters.
    """
    est = np.array(estimated)
    gt = np.array(ground_truth)
    mse = np.mean((est - gt) ** 2)
    rmse = np.sqrt(mse)
    return rmse
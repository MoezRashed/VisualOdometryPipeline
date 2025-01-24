import logging 
import cv2   as cv
import numpy as np


class MotionEstimator:
    def __init__ (self, camera_matrix: np.ndarray,extrinsics: np.ndarray, dist_coeffs: np.ndarray = None ):
        """
        Initializes the MotionEstimator with camera intrinsic parameters.

        Parameters:
            camera_matrix (np.ndarray): The intrinsic camera matrix.
            dist_coeffs (np.ndarray, optional): Distortion coefficients. Defaults to None.
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs   = dist_coeffs if dist_coeffs is not None else np.zeros((4,1))
        self.current_pose  = extrinsics
      
    
    def estimate_motion(self, kp1, kp2, matches):

        if len(matches) < 5:
            logging.warning("Not enough matches to estimate motion")
            return self.current_pose
        
        pts1      = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2      = np.float32([kp2[m.trainIdx].pt for m in matches])

        E, mask   = cv.findEssentialMat(pts1, pts2, self.camera_matrix, method=cv.RANSAC, prob=0.999, threshold=1.0)

        if E is None:
            logging.error("Essential matrix computation failed.")
            return self.current_pose
        
        _, R, t, mask_pose = cv.recoverPose(E, pts1, pts2, self.camera_matrix)

        T = np.eye(4)
        T[:3, :3] = R.T #transpose of Rotation matrix
        T[:3, 3]  = (R.T @ t).flatten()
        self.current_pose = self.current_pose @ T

        # logging.info(f"T-matrix:{T} \n current-pose:{self.current_pose}")

        # logging.info(f"Recovered Pose:\nRotation:\n{R}\nTranslation:\n{t}")

        return self.current_pose
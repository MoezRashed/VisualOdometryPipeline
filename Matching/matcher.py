import os
import logging
import cv2   as cv
import numpy as np
from PIL                  import Image
from utils.config         import load_config
from utils.dataset        import AdverCityDataset
from Extraction.extractor import FeatureExtractor


class Matcher:

    def __init__(self, config: dict):

        self.config          = config.get('matcher',{})
        self.use_knn         = self.config.get("strategy",{}).get("use_knn",False)
        self.k               = self.config.get("strategy",{}).get("k",2)
        self.ratio_thresh    = self.config.get("strategy", {}).get("ratio",0.75)
        self.distance_thresh = self.config.get("strategy", {}).get("distance",None)

        logging.info("Feature Extractor Configuration:")
        logging.info(self.config)

        match_type           = self.config.get('type')


        if match_type == "BF":

            logging.info("Creating BF matcher...")

            #For Orb [Binary Descriptors]
            if self.config["BF"]["HAMMING"]:
                norm = cv.NORM_HAMMING
            #For SIFT [Floating point Descriptors]
            else                           :
                norm = cv.NORM_L2
            
            cross_check     = self.config["BF"].get("cross_check", True)
            if self.use_knn:
                logging.info("k-NN matching enabled. Setting crossCheck to False.")
                cross_check = False

            self.matcher    = cv.BFMatcher(norm, crossCheck=cross_check)

            logging.info(f"Initialized Brute-Force with norm type: {norm}")

        elif match_type == "FLANN" :

            index_params       = {
                                "algorithm" : self.config["FLANN"]["algorithm"], 
                                "trees"     : self.config["FLANN"]["kdTrees"]
            }
            search_params      = {
                                "checks"    : self.config["FLANN"]["searchChecks"]
            }
            self.matcher       = cv.FlannBasedMatcher(index_params , search_params)

            logging.info("Initialized FLANN-Based Matcher")
        else                                : 
            raise ValueError(f"The following matcher type is not implemented : {self.config["type"]}")
     

        
    def match_features (self, desc1: np.ndarray , desc2: np.ndarray):
        
        if self.config["type"] == "FLANN":
            if desc1.dtype != np.float32:
                desc1 = desc1.astype(np.float32)
            if desc2.dtype != np.float32:
                desc2 = desc2.astype(np.float32)
        
        good_matches = []

        if self.use_knn:
            logging.info("Performing KNN Matching...")
            matches = self.matcher.knnMatch(desc1, desc2, k=self.k)
            for m,n in matches:
                if m.distance < self.ratio_thresh * n.distance:
                    good_matches.append(m)
            logging.info(f"Found {len(good_matches)} good matches")
        else:
            logging.info("Performing Brute-Force Matching...")
            matches = self.matcher.match(desc1, desc2)
            matches = sorted(matches, key=lambda x: x.distance)
            if self.distance_thresh:
                good_matches = [m for m in matches if m.distance < self.distance_thresh]
                logging.info(f"Good matches after distance thresholding: {len(good_matches)}")
            else:
                good_matches = matches
                logging.info(f"Found {len(good_matches)} good matches")
        return good_matches
    
    def draw_matches(self, img1, kp1, img2, kp2, matches, max_matches=50):
        """
        Draws matches between two images.

        Parameters:
            img1 (np.ndarray): First image.
            kp1 (list of cv2.KeyPoint): Keypoints from the first image.
            img2 (np.ndarray): Second image.
            kp2 (list of cv2.KeyPoint): Keypoints from the second image.
            matches (list of cv2.DMatch): Matches to draw.
            max_matches (int): Maximum number of matches to display.

        Returns:
            np.ndarray: Image with matches drawn.
        """
        if len(matches) > max_matches:
            matches = matches[:max_matches]
        matched_img = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return matched_img
    
    def __call__(self, desc1: np.ndarray , desc2: np.ndarray): 
        return self.match_features(desc1, desc2)
    
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    root = r'/Users/moezrashed/Documents/Programming/Python/QUARRG/ui_cd_s'
    dataset = AdverCityDataset(root, cam= 0 , car= 0)
    _,_,camimg1 = dataset[0]
    _,_,camimg2 = dataset[1] 
    config = config_path = os.path.join('configs', 'config.yaml')
    config = load_config(config_path)
    extractor = FeatureExtractor(config)
    kpts1 ,desc1,Img1 = extractor(camimg1)
    kpts2 ,desc2,Img2 = extractor(camimg2)
    matcher = Matcher(config)
    matches = matcher(desc1, desc2)
    matched_img = matcher.draw_matches(Img1, kpts1, Img2, kpts2, matches)
    cv.imshow("Matches", matched_img)
    cv.waitKey(0)
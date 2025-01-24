import os
import logging
import cv2   as cv
import numpy as np
from PIL               import Image
from utils.config      import load_config
from utils.dataset     import AdverCityDataset
from utils.image_utils import convert_pil_to_cv , scale_image

class FeatureExtractor:

    # Will try different extractors ["ORB","SIFT"] for traditional pipeline.
    def __init__(self, config: dict):
        """
        Initializes the FeatureExtractor with the specified configuration.

        Parameters:
            config (dict): Configuration dictionary.
        """
        self.config = config.get('feature_extractor',{})
        logging.info("Feature extractor configuration:")
        logging.info(self.config)

        detector_type = self.config.get('type', 'SIFT')

        if   detector_type == "ORB"  : 

            logging.info("Creating ORB detector..")
            
            self.detector= cv.ORB_create(
            nfeatures    =self.config["ORB"]["nfeatures"],
            scaleFactor  =self.config["ORB"]["scaleFactor"],
            nlevels      =self.config["ORB"]["nLevels"],
            edgeThreshold=self.config["ORB"]["edgeThreshold"],
            firstLevel   =self.config["ORB"]["firstLevel"],
            WTA_K        =self.config["ORB"]["WTA_K"],
            patchSize    =self.config["ORB"]["patchSize"],
            fastThreshold=self.config["ORB"]["fastThreshold"])
                            
        elif detector_type == "SIFT"  : 

            logging.info("Creating SIFT detector..")

            self.detector = cv.SIFT_create(
            nfeatures        =self.config["SIFT"]["nfeatures"],
            nOctaveLayers    =self.config["SIFT"]["nOctaveLayers"],
            contrastThreshold=self.config["SIFT"]["contrastThreshold"],
            edgeThreshold    =self.config["SIFT"]["edgeThreshold"],
            sigma            =self.config["SIFT"]["sigma"])
        else                                : 
            raise ValueError(f"The following detector type is not implemented : {self.config["type"]}")
    

    
    def extract_features(self,image):
        """
        Extracts keypoints and descriptors from the given image.

        Parameters:
            image (PIL.Image.Image or np.ndarray): The input image.

        Returns:
            tuple: A tuple containing keypoints, descriptors, and the processed image.
        """
        image = convert_pil_to_cv(image)

        logging.info("Computing Descriptors...")
        keypoints , descriptor       = self.detector.detectAndCompute(image, None)
        return keypoints , descriptor, image
    
    def __call__ (self, image):
        """
        Extracts keypoints and descriptors from the given image.

        Parameters:
            image (PIL.Image.Image or np.ndarray): The input image.

        Returns:
            tuple: A tuple containing keypoints, descriptors, and the processed image.
        """
        return self.extract_features(image)
         
# python -m Extraction.extractor [To run module]
if __name__ == "__main__":
        
        #old, need to be updated after config.yaml
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        #change this:
        root = r'/Users/moezrashed/Documents/Programming/Python/QUARRG/ui_cd_s'
        dataset = AdverCityDataset(root, cam= 0 , car= 0)

        cam_ref, cam_intrinsics, cam_image = dataset[0]
        config_path  = os.path.join('configs', 'config.yaml')
        config       = load_config(config_path)
        extractor = FeatureExtractor (config)

        keypoints, descriptors, processed_image = extractor.extract_features(cam_image)

        image_with_keypoints = cv.drawKeypoints(
        processed_image,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv.DrawMatchesFlags_DEFAULT
        )
        image_with_keypoints_pil = Image.fromarray(cv.cvtColor(image_with_keypoints, cv.COLOR_BGR2RGB))

        image_with_keypoints_pil.show()

        
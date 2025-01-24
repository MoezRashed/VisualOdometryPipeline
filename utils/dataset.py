import os
import yaml
import numpy as np
from PIL import Image


def list_folders(directory):
    ''' Get all folders inside a directory.
    '''
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    return folders

def process_dataset(dataset, cam_id, camera_data, cam_refs):
        
    for car_pos, cam_ext, cam_ref, cam_intrinsics, cam_image in dataset:
        camera_data[cam_id]['images'].append(cam_image)
        camera_data[cam_id]['intrinsics'].append(cam_intrinsics)
        camera_data[cam_id]['extrinsics'].append(cam_ext)
        camera_data[cam_id]['vehicle_pos'].append(car_pos)
        cam_refs[cam_id].append(cam_ref)
    return camera_data , cam_refs


class AdverCityDataset:
    # will do a config for it in the future? do i need it?
    config = {
        "root" : r'/Users/moezrashed/Documents/Programming/Python/QUARRG/ui_cd_s',
        "car"  : 0,
        "cam"  : 0
    }
    def __init__(self, root='', car=0, cam=0):
        # Check if root is a valid path
        if not os.path.exists(root):
            raise Exception("'root' is not a valid path.") 

        self._root = root
        self._car  = car
        self._cam  = cam

        # Getting all agents available in the scenario
        self._agents = sorted([folder for folder in list_folders(self._root)])
        self._rsus = self._agents[:-2]
        self._cars = self._agents[2:]
        
        # Getting all timestamps
        self._timestamps = sorted([f.split('.')[0] for f in os.listdir(os.path.join(self._root, self._cars[self._car])) if f.endswith(('.yaml', '.yml')) and not 'gnss' in f])
        
    def __len__(self):
        return len(self._timestamps)

    def __getitem__(self, index):
        # Getting camera reference position
        with open(os.path.join(self._root, self._cars[self._car], self._timestamps[index]+'.yaml'), 'r') as file:
            data  = yaml.safe_load(file)
        car_speed = data.get('ego_speed')
        cam_ref   = data.get('camera'+str(self._cam), {}).get('cords')
        cam_ext   = data.get('camera'+str(self._cam), {}).get('extrinsic')
        car_pos   = data.get('true_ego_pos')
        # cam_ext = np.array(cam_ext)[:3,:3]
        cam_ext   = np.array(cam_ext)
        cam_ref   = [cam_ref[0], cam_ref[1], cam_ref[2], cam_ref[3], cam_ref[4], cam_ref[5], car_speed]
        cam_intrinsics = data.get('camera'+str(self._cam), {}).get('intrinsic')
        cam_image = Image.open(os.path.join(self._root, self._cars[self._car], self._timestamps[index] + '_camera' + str(self._cam) + '.png'))
        return (car_pos, cam_ext, cam_ref, cam_intrinsics, cam_image)

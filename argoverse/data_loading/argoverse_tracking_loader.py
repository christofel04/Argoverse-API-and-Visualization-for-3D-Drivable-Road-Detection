# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import glob
import logging
import os
from typing import Dict, Iterator, List, Optional, Union, cast

import numpy as np

import argoverse.data_loading.object_label_record as object_label
from argoverse.data_loading.object_label_record import ObjectLabelRecord
from argoverse.data_loading.pose_loader import get_city_SE3_egovehicle_at_sensor_t, read_city_name
from argoverse.data_loading.synchronization_database import SynchronizationDB
from argoverse.utils.calibration import Calibration, load_calib, load_image
from argoverse.utils.camera_stats import CAMERA_LIST, RING_CAMERA_LIST, STEREO_CAMERA_LIST
from argoverse.utils.ply_loader import load_ply
from argoverse.utils.se3 import SE3

# For making drivable area label
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageOps

from argoverse.map_representation.map_api import ArgoverseMap

import copy
import math


logger = logging.getLogger(__name__)


class ArgoverseTrackingLoader:
    def __init__(self, root_dir: str , av_hd_map_dir = "/home/ofel04/argoverse-api/map_files/" , max_number_of_log = None) -> None:
        # initialize class member
        self.CAMERA_LIST = CAMERA_LIST
        self._log_list: Optional[List[str]] = None
        self._image_list: Optional[Dict[str, Dict[str, List[str]]]] = None
        self._image_list_sync: Optional[Dict[str, Dict[str, List[np.ndarray]]]] = None
        self._lidar_list: Optional[Dict[str, List[str]]] = None
        self._image_timestamp_list: Optional[Dict[str, Dict[str, List[int]]]] = None
        self._timestamp_image_dict: Optional[Dict[str, Dict[str, Dict[int, str]]]] = None
        self._image_timestamp_list_sync: Optional[Dict[str, Dict[str, List[int]]]] = None
        self._lidar_timestamp_list: Optional[Dict[str, List[int]]] = None
        self._timestamp_lidar_dict: Optional[Dict[str, Dict[int, str]]] = None
        self._label_list: Optional[Dict[str, List[str]]] = None
        self._calib: Optional[Dict[str, Dict[str, Calibration]]] = None  # { log_name: { camera_name: Calibration } }
        self._city_name = None
        self.counter: int = 0
        
        self.max_number_of_log = max_number_of_log

        self.image_count: int = 0
        self.lidar_count: int = 0

        self.root_dir: str = root_dir

        self.current_log = self.log_list[self.counter]

        assert self.image_list is not None
        assert self.lidar_list is not None
        assert self.label_list is not None

        # load calibration file
        self.calib_filename: str = os.path.join(self.root_dir, self.current_log, "vehicle_calibration_info.json")

        # lidar @10hz, ring camera @30hz, stereo camera @5hz
        self.num_lidar_frame: int = len(self.lidar_timestamp_list)
        self.num_ring_camera_frame: int = len(self.image_timestamp_list[RING_CAMERA_LIST[0]])
        self.num_stereo_camera_frame: int = len(self.image_timestamp_list[STEREO_CAMERA_LIST[0]])

        self.sync: SynchronizationDB = SynchronizationDB(root_dir)

        # Get drivable area map for labelling

        avmap = ArgoverseMap( root = av_hd_map_dir )

        self.current_city_name = self.city_name

        self.drivable_rasterized_map = avmap.get_rasterized_driveable_area( self.current_city_name )[0][ : , : ]
        self.drivable_rasterized_map_rotation_matrix = np.array( avmap.get_rasterized_driveable_area( self.current_city_name )[1][ : , : ])

        assert self.image_list_sync is not None
        assert self.calib is not None

    @property
    def city_name(self) -> str:
        """get city name of the current log

        Returns:
            city_name: city name of the current log, either 'PIT' or 'MIA'
        """
        return read_city_name(os.path.join(self.root_dir, self.current_log, "city_info.json"))

    @property
    def calib(self) -> Dict[str, Calibration]:
        """get calibration dict for current log

        Returns:
            calib: Calibration object for the current log
        """
        self._ensure_calib_is_populated()
        assert self._calib is not None
        return self._calib[self.current_log]

    def _ensure_calib_is_populated(self) -> None:
        """load up calibration object for all logs

        Returns:
            None
        """
        if self._calib is None:
            self._calib = {}
            for log in self.log_list:
                calib_filename = os.path.join(self.root_dir, log, "vehicle_calibration_info.json")
                self._calib[log] = load_calib(calib_filename)

    @property
    def log_list(self) -> List[str]:
        """return list of log (str) in the current dataset directory

        Returns:
            log_list: list of string representing log id
        """
        if self._log_list is None:

            def valid_log(log: str) -> bool:
                return os.path.exists(os.path.join(self.root_dir, log, "vehicle_calibration_info.json"))

            self._log_list = [x for x in os.listdir(self.root_dir) if valid_log(x)]
            
            if self.max_number_of_log :
            
            	self._log_list = self._log_list[ : self.max_number_of_log ]

        return self._log_list

    @property
    def image_list(self) -> Dict[str, List[str]]:
        """return list of all image path (str) for all cameras for the current log

        Returns:
            image_list: dictionary of list of image, with camera name as key
        """
        if self._image_list is None:
            self._image_list = {}
            for log in self.log_list:
                self._image_list[log] = {}
                for camera in CAMERA_LIST:
                    self._image_list[log][camera] = sorted(
                        glob.glob((os.path.join(self.root_dir, log, camera, "*.jpg")))
                    )
                    self.image_count += len(self._image_list[log][camera])
        return self._image_list[self.current_log]

    @property
    def image_list_sync(self) -> Dict[str, List[np.ndarray]]:
        """return list of image path (str) for all cameras for the current log.

        The different between image_list and image_list_sync is that image_list_sync
        syncronizes the image to lidar frame.

        Returns:
            image_list_sync: dictionary of list of image, with camera name as key. Each camera will have the same
                             number of images as #lidar frame.
        """
        logging.info("syncronizing camera and lidar sensor...")
        self._ensure_lidar_timestamp_list_populated()
        assert self._lidar_timestamp_list is not None

        if self._image_list_sync is None:

            self._image_list_sync = {}
            self._image_timestamp_list_sync = {}
            for log in self.log_list:

                self._image_list_sync[log] = {}
                self._image_timestamp_list_sync[log] = {}

                for camera in CAMERA_LIST:
                    self._image_timestamp_list_sync[log][camera] = cast(
                        List[int],
                        list(
                            filter(
                                lambda x: x is not None,
                                (
                                    self.sync.get_closest_cam_channel_timestamp(x, camera, log)
                                    for x in self._lidar_timestamp_list[log]
                                ),
                            )
                        ),
                    )

                    self._image_list_sync[log][camera] = [
                        self.get_image_at_timestamp(x, camera=camera, log_id=log, load=False)
                        for x in self._image_timestamp_list_sync[log][camera]
                    ]

        return self._image_list_sync[self.current_log]

    @property
    def image_timestamp_list_sync(self) -> Dict[str, List[int]]:
        """return list of image timestamp (str) for all cameras for the current log.

        The different between image_timestamp and image_timestamp_list_sync is that image_timestamp_list_sync
        synchronizes the image to the lidar frame.

        Returns:
            image_timestamp_list_sync: dictionary of list of image timestamp, with camera name as key.
                                       Each camera will have the same number of image timestamps as #lidar frame.
        """
        assert self.image_list_sync is not None
        assert self._image_timestamp_list_sync is not None
        return self._image_timestamp_list_sync[self.current_log]

    @property
    def lidar_list(self) -> List[str]:
        """return list of lidar path (str) of the current log

        Returns:
            lidar_list: list of lidar path for the current log
        """
        if self._lidar_list is None:
            self._lidar_list = {}
            for log in self.log_list:
                self._lidar_list[log] = sorted(glob.glob(os.path.join(self.root_dir, log, "lidar", "*.ply")))

                self.lidar_count += len(self._lidar_list[log])
        return self._lidar_list[self.current_log]

    @property
    def label_list(self) -> List[str]:
        """return list of label path (str) of the current log

        Returns:
            label: list of label path for the current log
        """
        if self._label_list is None:
            self._label_list = {}
            for log in self.log_list:
                self._label_list[log] = sorted(
                    glob.glob(os.path.join(self.root_dir, log, "per_sweep_annotations_amodal", "*.json"))
                )

        return self._label_list[self.current_log]

    @property
    def image_timestamp_list(self) -> Dict[str, List[int]]:
        """return dict of list of image timestamp (str) for all cameras for the current log.

        Returns:
            image_timestamp_list: dictionary of list of image timestamp for all cameras
        """
        assert self.image_list is not None
        assert self._image_list is not None

        if self._image_timestamp_list is None:
            self._image_timestamp_list = {}
            for log in self.log_list:
                self._image_timestamp_list[log] = {}
                for camera in CAMERA_LIST:
                    self._image_timestamp_list[log][camera] = [
                        int(x.split("/")[-1][:-4].split("_")[-1]) for x in self._image_list[log][camera]
                    ]

        return self._image_timestamp_list[self.current_log]

    @property
    def timestamp_image_dict(self) -> Dict[str, Dict[int, str]]:
        """return dict of list of image path (str) for all cameras for the current log, index by timestamp.

        Returns:
            timestamp_image_dict: dictionary of list of image path for all cameras, with timestamp as key
        """
        if self._timestamp_image_dict is None:
            assert self.image_timestamp_list is not None
            assert self._image_timestamp_list is not None
            assert self.image_list is not None
            assert self._image_list is not None

            self._timestamp_image_dict = {}

            for log in self.log_list:
                self._timestamp_image_dict[log] = {}
                for camera in CAMERA_LIST:
                    self._timestamp_image_dict[log][camera] = {
                        self._image_timestamp_list[log][camera][i]: self._image_list[log][camera][i]
                        for i in range(len(self._image_timestamp_list[log][camera]))
                    }

        return self._timestamp_image_dict[self.current_log]

    @property
    def timestamp_lidar_dict(self) -> Dict[int, str]:
        """return dict of list of lidar path (str) for the current log, index by timestamp.

        Returns:
            timestamp_lidar_dict: dictionary of list of lidar path, with timestamp as key
        """
        if self._timestamp_lidar_dict is None:
            assert self._lidar_timestamp_list is not None
            assert self._lidar_list is not None

            self._timestamp_lidar_dict = {}

            for log in self.log_list:
                self._timestamp_lidar_dict[log] = {
                    self._lidar_timestamp_list[log][i]: self._lidar_list[log][i]
                    for i in range(len(self._lidar_timestamp_list[log]))
                }

        return self._timestamp_lidar_dict[self.current_log]

    @property
    def lidar_timestamp_list(self) -> List[int]:
        """return list of lidar timestamp

        Returns:
            lidar_timestamp_list: list of lidar timestamp (at 10hz)
        """
        self._ensure_lidar_timestamp_list_populated()
        assert self._lidar_timestamp_list is not None
        return self._lidar_timestamp_list[self.current_log]

    def _ensure_lidar_timestamp_list_populated(self) -> None:
        """load up lidar timestamp for all logs

        Returns:
            None
        """
        assert self.lidar_list is not None
        assert self._lidar_list is not None

        if self._lidar_timestamp_list is None:
            self._lidar_timestamp_list = {}
            for log in self.log_list:
                self._lidar_timestamp_list[log] = [
                    int(x.split("/")[-1][:-4].split("_")[-1]) for x in self._lidar_list[log]
                ]

    def __iter__(self) -> Iterator["ArgoverseTrackingLoader"]:
        self.counter = -1

        return self

    def __next__(self , av_hd_map_dir = "/home/ofel04/argoverse-api/map_files/" ) -> "ArgoverseTrackingLoader":
        self.counter += 1
        #print( "Current City is : " + str( self.city_name ))

        if self.counter >= len(self):
            raise StopIteration
        else:
            self.current_log = self.log_list[self.counter]
            self.num_lidar_frame = len(self.lidar_timestamp_list)
            self.num_ring_camera_frame = len(self.image_timestamp_list[RING_CAMERA_LIST[0]])
            self.num_stereo_camera_frame = len(self.image_timestamp_list[STEREO_CAMERA_LIST[0]])
            
            avmap = ArgoverseMap( root = av_hd_map_dir )
            
            self.current_city_name = self.city_name
            print( "Current City is : " + str( self.current_city_name ))
            self.drivable_rasterized_map = avmap.get_rasterized_driveable_area( self.current_city_name )[0][ : , : ]
            self.drivable_rasterized_map_rotation_matrix = np.array( avmap.get_rasterized_driveable_area( self.current_city_name )[1][ : , : ])
            
            print( "Self is : " + str( self ))
            return self

    def __len__(self) -> int:
        return len(self.log_list)

    def __str__(self) -> str:
        frame_lidar = self.num_lidar_frame
        frame_image_ring = self.num_ring_camera_frame
        frame_image_stereo = self.num_stereo_camera_frame

        num_images = [len(self.image_list[cam]) for cam in CAMERA_LIST]

        num_annotations = [len(object_label.read_label(label)) for label in self.label_list]

        start_time = self.lidar_timestamp_list[0]
        end_time = self.lidar_timestamp_list[-1]

        time_in_sec = (end_time - start_time) * 10 ** (-9)
        return f"""
--------------------------------------------------------------------
------Log id: {self.current_log}
--------------------------------------------------------------------
Time: {time_in_sec} sec
# frame lidar (@10hz): {frame_lidar}
# frame ring camera (@30hz): {frame_image_ring}
# frame stereo camera (@5hz): {frame_image_stereo}

Total images: {sum(num_images)}
Total bounding box: {sum(num_annotations)}
        """
    
    # Function to take BEV Drivable Label from Drivable Rasterized Map

    def get_rasterized_drivabel_area_label( self , key : int , map_range : list = [ -50 , -50 , 50 , 70] , grid_size : list = [0.2, 0.2]) -> np.array :

        # Get matrix rotation of Ego Vehicle Coordinate to City Coordinate
    
        city_to_egovehicle_se3 = self.get_pose(key)

        # Get ego vehicle city coordinate and rotation

        x,y,_ = city_to_egovehicle_se3.translation

        #print( "Location of ego- vehicle : x = {}, y = {}".format( x , y ) )

        ego_vehicle_rotation = city_to_egovehicle_se3.rotation

        pose_rotation_matrix_to_yaw = R.from_matrix( ego_vehicle_rotation )

        yaw_angle = pose_rotation_matrix_to_yaw.as_euler( "zyx" , degrees = True )[0]


        # Get Drivable area in Raster Map

        x_raster_map_coordinate = x + self.drivable_rasterized_map_rotation_matrix[0][2]
        y_raster_map_coordinate = y + self.drivable_rasterized_map_rotation_matrix[1][2]

        img = Image.fromarray( self.drivable_rasterized_map ).crop( (( x_raster_map_coordinate-120) , ( y_raster_map_coordinate -120 ) , ( x_raster_map_coordinate + 120 ) , ( y_raster_map_coordinate +  120 )))#.rotate( -yaw_angle/ math.pi * 180 + 180)

        img = img.resize((int( 120 * 1/grid_size[0]), int( 120* 1/grid_size[1])), resample=Image.BOX)

        #img_with_ground_height = Image.fromarray( ground_heigh_raster_map_with_color ).crop( (( x_raster_map_coordinate-80) , ( y_raster_map_coordinate -80 ) , ( x_raster_map_coordinate + 80 ) , ( y_raster_map_coordinate +  80 ))).rotate( -yaw_angle/ math.pi * 180 + 180)

        img = np.array( img )

        img = ImageOps.mirror( Image.fromarray( img ).rotate( yaw_angle + 90))

        rotated_bev_image_shape = np.array( img ).shape

        img = img.crop( ( rotated_bev_image_shape[0]/2 + map_range[0] * 1/grid_size[0] , rotated_bev_image_shape[1]/2 - map_range[3] * 1/grid_size[1] , rotated_bev_image_shape[0]/2 + map_range[2] * 1/grid_size[0]  , rotated_bev_image_shape[1]/2 - map_range[1] * 1/grid_size[1]))

        img = img.resize((int( 100 * 1/grid_size[0]), int( 120 * 1/grid_size[1])), resample=Image.BOX)

        return np.array( img )

    def get_drivable_area_label_from_pickle( self , idx : int ) -> str :
    
    	name_of_drivabel_area_label_pickle = self.root_dir + "/" + str( self.current_log ) + "/BEV_drivable_area_label/" + self._lidar_list[ self.current_log ][idx].split( "/" )[-1].replace(".ply", "" ) + ".pickle"
    	
    	return str( name_of_drivabel_area_label_pickle )

    def __getitem__(self, key: int) -> "ArgoverseTrackingLoader":
        self.counter = key
        self.current_log = self.log_list[self.counter]
        self.num_lidar_frame = len(self.lidar_timestamp_list)
        self.num_ring_camera_frame = len(self.image_timestamp_list[RING_CAMERA_LIST[0]])
        self.num_stereo_camera_frame = len(self.image_timestamp_list[STEREO_CAMERA_LIST[0]])
        self.bev_map_drivable_area_label = self.get_rasterized_drivabel_area_label( key ) # Get Drivable Area Label for map
        
        return self

    def get(self, log_id: str) -> "ArgoverseTrackingLoader":
        """get ArgoverseTrackingLoader object with current_log set to specified log_id

        Args:
            log_id: log id
        Returns:
            ArgoverseTrackingLoader: with current_log set to log_id
        """
        self.current_log = log_id
        self.num_lidar_frame = len(self.lidar_timestamp_list)
        self.num_ring_camera_frame = len(self.image_timestamp_list[RING_CAMERA_LIST[0]])
        self.num_stereo_camera_frame = len(self.image_timestamp_list[STEREO_CAMERA_LIST[0]])
        return self

    def get_image_list(self, camera: str, log_id: Optional[str] = None, load: bool = False) -> List[str]:
        """Get list of image/or image path

        Args:
            camera: camera based on camera_stats.CAMERA_LIST
            log_id: log_id, if not specified will use self.current_log
            load: whether to return image array (True) or image path (False)

        Returns:
            np.array: list of image path (str or np.array)),
        """
        assert self.image_list is not None
        assert self._image_list is not None

        if log_id is None:
            log_id = self.current_log
        if load:
            return [self.get_image(i, camera) for i in range(len(self._image_list[log_id][camera]))]

        return self._image_list[log_id][camera]

    def get_image_list_sync(self, camera: str, log_id: Optional[str] = None, load: bool = False) -> List[str]:
        """Get list of image/or image path in lidar index

        Args:
            camera: camera based on camera_stats.CAMERA_LIST
            log_id: log_id, if not specified will use self.current_log
            load: whether to return image array (True) or image path (False)

        Returns:
            np.array: list of image path (str or np.array)),
        """
        assert self.image_list_sync is not None
        assert self._image_list_sync is not None

        if log_id is None:
            log_id = self.current_log

        if load:
            return [self.get_image_sync(i, camera) for i in range(len(self._image_list_sync[log_id][camera]))]

        return self._image_list_sync[log_id][camera]

    def get_image_at_timestamp(
        self,
        timestamp: int,
        camera: str,
        log_id: Optional[str] = None,
        load: bool = True,
    ) -> Optional[Union[str, np.ndarray]]:
        """get image or image path at a specific timestamp

        Args:
            timestamp: timestamp
            camera: camera based on camera_stats.CAMERA_LIST
            log_id: log_id, if not specified will use self.current_log
            load: whether to return image array (True) or image path (False)

        Returns:
            np.array: list of image path (str or np.array)),
        """
        assert self.timestamp_image_dict is not None
        assert self._timestamp_image_dict is not None

        if log_id is None:
            log_id = self.current_log
        assert self.timestamp_image_dict is not None
        try:
            image_path = self._timestamp_image_dict[log_id][camera][timestamp]
        except KeyError:
            logging.error(f"Cannot find {camera} image at timestamp {timestamp} in log {log_id}")
            return None

        if load:
            return load_image(image_path)
        return image_path

    def get_image(
        self, idx: int, camera: str, log_id: Optional[str] = None, load: bool = True
    ) -> Union[str, np.ndarray]:
        """get image or image path at a specific index (in image index)

        Args:
            idx: image based 0-index
            camera: camera based on camera_stats.CAMERA_LIST
            log_id: log_id, if not specified will use self.current_log
            load: whether to return image array (True) or image path (False)

        Returns:
            np.array: list of image path (str or np.array)),
        """
        assert self.image_timestamp_list is not None
        assert self._image_timestamp_list is not None
        assert self.image_list is not None
        assert self._image_list is not None

        if log_id is None:
            log_id = self.current_log

        assert idx < len(self._image_timestamp_list[log_id][camera])
        image_path = self._image_list[log_id][camera][idx]

        if load:
            return load_image(image_path)
        return image_path

    def get_image_sync(
        self, idx: int, camera: str, log_id: Optional[str] = None, load: bool = True
    ) -> Union[str, np.ndarray]:
        """get image or image path at a specific index (in lidar index)

        Args:
            idx: lidar based 0-index
            camera: camera based on camera_stats.CAMERA_LIST
            log_id: log_id, if not specified will use self.current_log
            load: whether to return image array (True) or image path (False)

        Returns:
            np.array: list of image path (str or np.array)),
        """
        assert self.image_timestamp_list_sync is not None
        assert self._image_timestamp_list_sync is not None
        assert self.image_list_sync is not None
        assert self._image_list_sync is not None

        if log_id is None:
            log_id = self.current_log

        assert idx < len(self._image_timestamp_list_sync[log_id][camera])
        image_path = self._image_list_sync[log_id][camera][idx]

        if load:
            return load_image(image_path)
        return image_path

    def get_lidar(self, idx: int, log_id: Optional[str] = None, load: bool = True , is_return_intensity = False) -> Union[str, np.ndarray]:
        """Get lidar corresponding to frame index idx (in lidar frame).

        Args:
            idx: Lidar frame index
            log_id: ID of log to search (default: current log)
            load: whether to load up the data, will return path to the lidar file if set to false

        Returns:
            Either path to lidar at a specific index, or point cloud data if load is set to True
        """
        assert self.lidar_timestamp_list is not None
        assert self._lidar_timestamp_list is not None
        assert self.lidar_list is not None
        assert self._lidar_list is not None

        if log_id is None:
            log_id = self.current_log

        assert idx < len(self._lidar_timestamp_list[log_id])

        if load:
                        
            return load_ply(self._lidar_list[log_id][idx] , is_return_intensity = is_return_intensity)
        return self._lidar_list[log_id][idx]
    
    def get_lidar_in_rasterized_map_coordinate( self , idx : int , log_id: Optional[str] = None, load: bool = True , is_return_intensity = True) -> np.ndarray :
        # Function to get LiDAR points in a frame in City Coordinate

        lidar_points_in_ego_vehicle_coordinate = self.get_lidar( idx , log_id , load , is_return_intensity=is_return_intensity )

        city_to_egovehicle_se3 = self.get_pose(idx)

        x,y,_ = city_to_egovehicle_se3.translation

        lidar_pts_for_bev = copy.deepcopy(lidar_points_in_ego_vehicle_coordinate)[ : , :3 ]
        lidar_pts_in_city_coordinate = city_to_egovehicle_se3.transform_point_cloud(
                lidar_pts_for_bev
            )  # put into city coords
        
        #print( "Shape of LiDAR points in City Coordinate : " + str( lidar_pts_in_city_coordinate.shape ))
        #print( "Shape of LiDAR points in Ego Vehicle Coordinate : " + str( lidar_points_in_ego_vehicle_coordinate.shape ))
        
        if lidar_points_in_ego_vehicle_coordinate.shape[1] == 4 : 
            lidar_pts_in_city_coordinate = np.concatenate( [ lidar_pts_in_city_coordinate[ : ] , lidar_points_in_ego_vehicle_coordinate[ : , 3 : ] ] , axis = 1 )
        
        if lidar_pts_in_city_coordinate.shape[1] == 4 :

            lidar_pts_in_city_coordinate = lidar_pts_in_city_coordinate - [[ x , y , 0 , 0] for i in range( lidar_pts_in_city_coordinate.shape[0]) ]
        else :

            lidar_pts_in_city_coordinate = lidar_pts_in_city_coordinate - [[ x , y , 0] for i in range( lidar_pts_in_city_coordinate.shape[0]) ]


        # Rotate LiDAR point in ego Vehicle orientation
            
        ego_vehicle_rotation = city_to_egovehicle_se3.rotation

        pose_rotation_matrix_to_yaw = R.from_matrix( ego_vehicle_rotation )

        yaw_angle = pose_rotation_matrix_to_yaw.as_euler( "zyx" , degrees = False )[0]

        yaw_angle = yaw_angle + 0.5 * math.pi

        lidar_pts_new_coordinate = lidar_pts_in_city_coordinate[ : , 0 : 2].dot( np.array( [[math.cos( yaw_angle ) , -math.sin( yaw_angle )] , [ math.sin( yaw_angle ) , math.cos( yaw_angle )]]))

        #lidar_pts_new_coordinate[ : , 2 : ] = lidar_pts_in_city_coordinate[ : , 2 : ]

        #print( "Shape of LiDAR Point in City Coordinate : " + str( lidar_pts_new_coordinate.shape ))

        lidar_pts_new_coordinate = np.concatenate( [ np.array( lidar_pts_new_coordinate ) , np.array( lidar_pts_in_city_coordinate )[ : , 2 : ]]  , axis = 1 )

        # Mirror LiDAR points in y- axis to make LiDAR BEV map coordinate same exactly with ego- vehicle coordinate
            
        lidar_pts_new_coordinate = np.concatenate( [ -1* lidar_pts_new_coordinate[ : , 0 ].reshape(-1, 1) , -1* lidar_pts_new_coordinate[ : , 1 ].reshape(-1, 1) , lidar_pts_new_coordinate[ : , 2 : ] ] , axis = 1 )

        return lidar_pts_new_coordinate

    def get_bev_tensor_lidar_from_pickle( self , idx : int ) -> str :
    
    	name_of_bev_tensor_file = self.root_dir + "/" + str( self.current_log ) + "/BEV_tensor_folder/" + self._lidar_list[ self.current_log ][idx].split( "/" )[-1].replace(".ply", "" ) + ".pickle"
    	
    	#f = open( name_of_bev_tensor_file , "rb+")
    	
    	return name_of_bev_tensor_file
    	
    def get_label_object(self, idx: int, log_id: Optional[str] = None) -> List[ObjectLabelRecord]:
        """Get label corresponding to frame index idx (in lidar frame).

        Args:
            idx: Lidar frame index
            log_id: ID of log to search (default: current log)

        Returns:
            List of ObjectLabelRecord info for a particular index
        """
        assert self.lidar_timestamp_list is not None
        assert self._lidar_timestamp_list is not None
        assert self.label_list is not None
        assert self._label_list is not None

        if log_id is None:
            log_id = self.current_log

        assert idx < len(self._lidar_timestamp_list[log_id])

        return object_label.read_label(self._label_list[log_id][idx])
        

    def get_calibration(self, camera: str, log_id: Optional[str] = None) -> Calibration:
        """Get calibration corresponding to the camera.

        Args:
            camera: name of the camera; one of::

               ["ring_front_center",
                "ring_front_left",
                "ring_front_right",
                "ring_rear_left",
                "ring_rear_right",
                "ring_side_left",
                "ring_side_right",
                "stereo_front_left",
                "stereo_front_right"]

            log_id: ID of log to search (default: current log)

        Returns:
            Calibration info for a particular index
        """
        self._ensure_calib_is_populated()
        assert self._calib is not None

        if log_id is None:
            log_id = self.current_log

        return self._calib[log_id][camera]

    def get_pose(self, idx: int, log_id: Optional[str] = None) -> Optional[SE3]:
        """Get pose corresponding to an index in a particular log_id.

        Args:
            idx: Lidar frame index
            log_id: ID of log to search (default: current log)

        Returns:
            Pose for a particular index
        """
        if log_id is None:
            log_id = self.current_log
        self._ensure_lidar_timestamp_list_populated()
        assert self._lidar_timestamp_list is not None

        timestamp = self._lidar_timestamp_list[log_id][idx]

        return get_city_SE3_egovehicle_at_sensor_t(timestamp, self.root_dir, log_id)

    def get_idx_from_timestamp(self, timestamp: int, log_id: Optional[str] = None) -> Optional[int]:
        """Get index corresponding to a timestamp in a particular log_id.

        Args:
            timestamp: Timestamp to search for
            log_id: ID of log to search (default: current log)

        Returns:
            Index in the log if found, or None if not found.
        """
        if log_id is None:
            log_id = self.current_log
        self._ensure_lidar_timestamp_list_populated()
        assert self._lidar_timestamp_list is not None

        for i in range(len(self._lidar_timestamp_list[log_id])):
            if self._lidar_timestamp_list[log_id][i] == timestamp:
                return i
        return None

    def print_all(self) -> None:
        assert self.image_timestamp_list is not None
        assert self.lidar_timestamp_list is not None
        print("#images:", self.image_count)
        print("#lidar:", self.lidar_count)

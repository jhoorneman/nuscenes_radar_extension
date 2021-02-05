# General imports
import logging
import os
from os import path as osp
from typing import List
import laspy as las
import numpy as np
from pyquaternion import Quaternion
# Nuscenes imports
from nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud, Box
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.map_mask import MapMask


class RadarPC(RadarPointCloud):
    """
    Extended version of NuScenes RadarPointCloud class. Main data is stored in the points matrix. The matrix of the base
    RadarPointCloud class has 18 rows containing the following fields:
    x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
    These fields are elaborated on in RadarPointCloud.from_file()
    The RadarPC class adds 7 more rows containing for each row index:
    18: ground truth labels. 1 is boundary point, 0 non boundary point
    19: sweep number. The number of sweep a point comes from. 0 is the current one, 1 means one sweep in the past etc
    20: compensated velocity. It is sqrt(vx_comp^2_vy_comp^2) only gets stored after first request
    21: normalized x. Only gets stored after first request
    22: normalized y. Only gets stored after first request
    23: normalized z. Not used at the moment since the z values is ignored in thesis
    24: normalized compensated velocity. Only gets stored after first request
    25: distance in xy-plane to ego location. Only gets stored after first request
    """
    def __init__(self, points: np.ndarray, map_resolution: float = None, filename: str = None, nsweeps: int = 1,
                 sample_data_token: str = None, nusc: NuScenes = None):
        super().__init__(points)
        self.map_resolution = map_resolution
        self.sample_data_token = sample_data_token
        self.filename = filename
        self.nsweeps = nsweeps
        self.ego_position = None
        self.ego_direction = None
        self.map_token = None
        if sample_data_token is not None and nusc is not None:
            self.__set_fields_from_sd_token(sample_data_token, nusc)

    def __repr__(self):
        return 'RadarPC: {} points. {} sweeps. Sample data token: {}'\
            .format(self.nbr_points(), self.nsweeps, self.sample_data_token)

    @classmethod
    def from_radar_point_cloud(cls, radar_pc: RadarPointCloud, map_resolution: float = 0.1) -> 'RadarPC':
        """The nuScenes radar points all have a map resolution of 0.1 so this is hardcoded"""
        extra_dims = RadarPC.nbr_dims() - RadarPointCloud.nbr_dims()
        return cls(np.vstack((radar_pc.points, np.zeros((extra_dims, radar_pc.nbr_points())))), map_resolution)

    @classmethod
    def from_las(cls, filename: str, nusc: NuScenes = None, nsweeps: int = 0, do_logging: bool = True) \
            -> 'RadarPC':
        """
        Load radar pc from las file
        :param filename:
        :param nusc: nuScenes database, if given the nuScenes metadata will be included in the RadarPC
        :param nsweeps: number of sweeps to get from las file. If nsweeps = 0 all available sweeps will be included
        :param do_logging: will print warning messages if true
        """
        assert filename[-4:] == '.las', "Filename should end in .las. Filename was: {}".format(filename)
        if osp.exists(filename):
            file = las.file.File(filename, mode='r')
            num_points = file.header.point_records_count
            sample_data_token = file.header.software_id
            nsweeps_file = file.header.file_source_id
            points = np.zeros((cls.nbr_dims(), num_points))
            points[0] = file.x
            points[1] = file.y
            points[2] = file.z
            points[3] = file.user_data  # dynprop
            points[8] = file.x_t    # compensated x velocity
            points[9] = file.y_t    # compensated y velocity
            points[11] = file.wave_packet_desc_index    # ambig state
            points[14] = file.waveform_packet_size  # invalid state
            points[18] = file.classification
            points[19] = file.pt_src_id # sweep number
            file.close()

            # Filter points with a sweep number less than nsweeps
            if nsweeps != 0:
                points = points[:, points[19] < nsweeps]
                if nsweeps_file < nsweeps:
                    if do_logging:
                        logging.warning('Not enough sweeps available in las file. {} were requested, but only {} are '
                                        'stored in the file. File name: {}'.format(nsweeps, nsweeps_file, filename))
                    nsweeps = nsweeps_file
            else:
                # if the nsweep argument is set to 0, all sweeps are included. Change nsweeps to the number saved in
                # file so it gets stored correctly in the RadarPC object
                nsweeps = nsweeps_file

            # Filter points with an invalid state.
            valid = [p in RadarPointCloud.invalid_states for p in points[14, :]]
            points = points[:, valid]

            # Filter by dynProp.
            valid = [p in RadarPointCloud.dynprop_states for p in points[3, :]]
            points = points[:, valid]

            # Filter by ambig_state.
            valid = [p in RadarPointCloud.ambig_states for p in points[11, :]]
            points = points[:, valid]
        else:
            points = np.zeros((cls.nbr_dims(), 0))
            sample_data_token = None
            nsweeps_file = 0
            if do_logging:
                logging.warning('Empty radar point cloud was created because no file exists at: ' + filename)

        assert sample_data_token != ' '*32, "las file had no sample_data_token defined. File name: {}".format(filename)
        radar_pc = cls(points, map_resolution=0.1, filename=filename, nsweeps=nsweeps,
                       sample_data_token=sample_data_token, nusc=nusc)

        return radar_pc

    @classmethod
    def from_file(cls, file_name: str, invalid_states: List[int] = None, dynprop_states: List[int] = None,
                  ambig_states: List[int] = None) -> 'RadarPC':
        """A pointcloud is loaded from a pcd file, no further adjustments are made (so no transformation to map frame of
        reference"""
        radar_pc = RadarPointCloud.from_file(file_name, invalid_states, dynprop_states, ambig_states)
        return cls.from_radar_point_cloud(radar_pc)

    @classmethod
    def from_sample_data(cls, sample_data_token: str, nusc: NuScenes, nsweeps: int = 1) -> 'RadarPC':
        """
        Load radar from sample data token and transform it to the nusc map frame of reference
        :param sample_data_token: token of sample data this RadarPC is made from
        :param nusc: nuScenes database
        :param nsweeps: number of radar sweeps to use. With 1 only the current sweep is used, otherwise previous sweeps
        will be included, if they are available
        :return: a RadarPC point cloud
        """
        points = np.zeros((cls.nbr_dims(), 0))
        actual_sweeps = 0
        iterate_sd_token = sample_data_token
        # iterate over nsweep previous sweeps (if available), transform them to map frame of reference and add together
        for sweep in range(nsweeps):
            sample_data = nusc.get('sample_data', iterate_sd_token)
            filename = osp.join(nusc.dataroot, sample_data['filename'])
            pc = cls.from_file(filename)
            pc.transform_to_nusc_map_frame(nusc, iterate_sd_token)
            pc.sweep_nums()[:] = np.ones(pc.nbr_points())*sweep

            # add points together
            points = np.hstack((points, pc.points))
            actual_sweeps += 1
            # select next sample_data
            iterate_sd_token = sample_data['prev']
            if iterate_sd_token == '':
                break
        pc_out = cls(points, map_resolution=0.1, nsweeps=actual_sweeps, sample_data_token=sample_data_token, nusc=nusc)
        return pc_out

    @staticmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        return 26

    @classmethod
    def extended_filters(cls) -> None:
        """
        Set the extended settings for all radar filter settings. For invalid states all valid states in one way or
        another are included. And for ambig states stationary candidates are taken into account too
        Note that this method affects the global settings.
        """
        RadarPointCloud.invalid_states = [0, 4, 8, 9, 10, 11, 12, 15, 16, 17]
        RadarPointCloud.dynprop_states = list(range(7))
        RadarPointCloud.ambig_states = [3, 4]

    # Get functions
    def x(self, in_pixels: bool = False) -> np.ndarray:
        if in_pixels:
            return self.points[0] / self.map_resolution
        else:
            return self.points[0]

    def y(self, in_pixels: bool = False) -> np.ndarray:
        if in_pixels:
            return self.points[1] / self.map_resolution
        else:
            return self.points[1]

    def z(self, in_pixels: bool = False) -> np.ndarray:
        if in_pixels:
            return self.points[2] / self.map_resolution
        else:
            return self.points[2]

    def xy(self, in_pixels: bool = False, normalized: bool = False) -> np.ndarray:
        assert not (in_pixels and normalized), 'the points cannot be both in pixels and normalized'
        if in_pixels:
            return self.points[:2] / self.map_resolution
        elif normalized:
            if np.array_equal(self.points[21], np.zeros(self.nbr_points())):
                self.points[21:23] = self.__normalize_features(self.points[:2])
            return self.points[21:23]
        else:
            return self.points[:2]

    def xyz(self, in_pixels: bool = False) -> np.ndarray:
        if in_pixels:
            return self.points[:3] / self.map_resolution
        else:
            return self.points[:3]

    def xyv(self, normalized: bool = False) -> np.ndarray:
        return np.vstack((self.xy(normalized=normalized), self.v_rad(normalized=normalized)))

    def dynprop(self) -> np.ndarray:
        """
        dynProp: Dynamic property of cluster to indicate if is moving or not.
        0: moving
        1: stationary
        2: oncoming
        3: stationary candidate
        4: unknown
        5: crossing stationary
        6: crossing moving
        7: stopped
        """
        return self.points[3]

    def invalid_state(self) -> np.ndarray:
        """
        invalid_state: state of Cluster validity state.
        (Invalid states)
        0x01	invalid due to low RCS
        0x02	invalid due to near-field artefact
        0x03	invalid far range cluster because not confirmed in near range
        0x05	reserved
        0x06	invalid cluster due to high mirror probability
        0x07	Invalid cluster because outside sensor field of view
        0x0d	reserved
        0x0e	invalid cluster because it is a harmonics
        (Valid states)
        0x00	valid
        0x04	valid cluster with low RCS
        0x08	valid cluster with azimuth correction due to elevation
        0x09	valid cluster with high child probability
        0x0a	valid cluster with high probability of being a 50 deg artefact
        0x0b	valid cluster but no local maximum
        0x0c	valid cluster with high artefact probability
        0x0f	valid cluster with above 95m in near range
        0x10	valid cluster with high multi-target probability
        0x11	valid cluster with suspicious angle
        """
        return self.points[14]

    def ambig_state(self) -> np.ndarray:
        """
        ambig_state: State of Doppler (radial velocity) ambiguity solution.
        0: invalid
        1: ambiguous
        2: staggered ramp
        3: unambiguous
        4: stationary candidates
        """
        return self.points[11]

    def sweep_nums(self) -> np.ndarray:
        """
        the number of sweep each point comes from. 0 means the current one, 1 means one sweep in the past etc
        """
        return self.points[19]

    def vx_comp(self) -> np.ndarray:
        return self.points[8]

    def vy_comp(self) -> np.ndarray:
        return self.points[9]

    def v_comps(self) -> np.ndarray:
        return self.points[8:10]

    def v_rad(self, normalized: bool = False) -> np.ndarray:
        if normalized:
            if np.array_equal(self.points[24], np.zeros(self.nbr_points())):
                self.points[24] = self.__normalize_features((self.v_comps()**2).sum(0))
            return self.points[24]
        else:
            if np.array_equal(self.points[20], np.zeros(self.nbr_points())):
                self.points[20] = (self.v_comps()**2).sum(0)
            return self.points[20]

    def distance_to_ego(self) -> np.ndarray:
        if np.array_equal(self.points[25], np.zeros(self.nbr_points())):
            if self.ego_position is None:
                self.points[25] = np.zeros(self.nbr_points())
            else:
                self.points[25] = np.sqrt(sum((self.xy() - self.ego_position[:2].reshape((2, 1))) ** 2))
        return self.points[25]

    def labels(self) -> np.ndarray:
        return self.points[18]

    def pixel_ego_position(self):
        if self.ego_position is None:
            return None
        else:
            return self.ego_position / self.map_resolution

    def get_boxes(self, nusc: NuScenes, in_pixels: bool = False) -> List[Box]:
        return self.get_boxes_base(nusc, self.sample_data_token, in_pixels)

    # Helper functions
    @staticmethod
    def __normalize_features(features):
        if len(features.shape) == 1:
            return (features - features.mean()) / features.std()
        else:
            dim = features.shape[0]
            return (features - features.mean(1).reshape(dim, 1))/features.std(1).reshape(dim, 1)

    def get_boxes_base(self, nusc: NuScenes, sample_data_token, in_pixels: bool = False) -> List[Box]:
        """
        Return list of annotated bounding boxes
        :param nusc: nuscenes dataset
        :param sample_data_token: sample data token
        :param in_pixels: whether boxes have to be given in pixel coordinates (needed for rendering)
        :return: list of bounding boxes
        """
        boxes = nusc.get_boxes(sample_data_token)
        if in_pixels:
            for box in boxes:
                box.center /= self.map_resolution
                box.wlh /= self.map_resolution
        return boxes

    def __set_fields_from_sd_token(self, sample_data_token: str, nusc: NuScenes):
        assert self.sample_data_token is None or self.sample_data_token == sample_data_token, \
            'sample_data_token parameter should be the same as token stored in the RadarPC'
        sample_data = nusc.get('sample_data', sample_data_token)
        ego_pose = nusc.get('ego_pose', nusc.get('sample_data', sample_data_token)['ego_pose_token'])
        if self.sample_data_token is None:
            self.sample_data_token = sample_data_token
        if self.filename is None:
            self.filename = sample_data['filename']
        if self.ego_position is None:
            # todo: find more elegant way to do this, since ego_position is not always transformed while the points are
            self.ego_position = np.array(ego_pose['translation'].copy())
        if self.ego_direction is None:
            rot_mat = Quaternion(ego_pose['rotation']).rotation_matrix
            self.ego_direction = np.dot(rot_mat, np.array((1, 0, 0)))
        self.set_map_fields_from_sd_token(sample_data_token, nusc)

    def set_map_fields_from_sd_token(self, sample_data_token: str, nusc: NuScenes):
        sample_data = nusc.get('sample_data', sample_data_token)
        log_token = nusc.get('scene', nusc.get('sample', sample_data['sample_token'])['scene_token'])['log_token']
        map_token = nusc.get('log', log_token)['map_token']
        basic_mask: MapMask = nusc.get('map', map_token)['mask']
        if self.map_resolution is None:
            self.map_resolution = basic_mask.resolution
        if self.map_token is None:
            self.map_token = map_token

    def transform(self, transf_matrix: np.ndarray, transform_coords: bool = True, transform_vel: bool = True,
                  transform_ego: bool = False) -> None:
        # todo: check if the direction of rotated velocities is correct
        rot_matrix = (transf_matrix / np.sqrt(np.sum(transf_matrix ** 2, 0)))[:3, :3]
        if transform_coords:
            super().transform(transf_matrix)
        if transform_vel:
            self.points[8:10] = np.dot(rot_matrix[:2, :2], self.points[8:10])
        if transform_ego:
            ego_position = np.ones(4)
            ego_position[:3] = self.ego_position
            self.ego_position = np.dot(transf_matrix, ego_position)[:3]
            self.ego_direction = np.dot(rot_matrix, self.ego_direction)

    def transform_to_nusc_map_frame(self, nusc: NuScenes, sample_data_token: str = None) -> None:
        # Store fields that can be acquired from the sample data token
        sample_data_token = self.sample_data_token if sample_data_token is None else sample_data_token
        self.__set_fields_from_sd_token(sample_data_token, nusc)

        # get nuscenes database entries
        sample_data = nusc.get('sample_data', sample_data_token)
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        cal_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])

        # todo: maybe I have to do something with "flat coordinates", like in render radar
        car_from_senor = transform_matrix(cal_sensor['translation'], Quaternion(cal_sensor['rotation']), inverse=False)
        global_from_car = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)

        # Combine tranformation matrices
        global_from_sensor = np.dot(global_from_car, car_from_senor)

        # Transform coordinates, ego position and velocities
        self.transform(global_from_sensor, transform_vel=True)
        # todo: check if it is correct to rotate velocities with map_from_global

        # remove height information, since this is not included in the rotation matrices, so incorrect
        self.z()[:] = np.zeros(self.nbr_points())

    def extend_with(self, radar_pc: RadarPointCloud, check_sd_token: str = None):
        """
        add the point clouds of self and radar_pc together, but keep all the other attributes of self
        :param radar_pc: the points from this radar_pc are added to the points of self
        :param check_sd_token: it is checked if this sample data token is the same the one from object self. When using
        multi sweep radar point clouds this can be the token of the main keyframe.
        """
        if check_sd_token is not None:
            assert self.sample_data_token == check_sd_token, "sample data token is not the same as the sample data " \
                                                             "token that is inputted as a check. They are " \
                                                             "respectively {} and {}"\
                .format(self.sample_data_token, check_sd_token)
        self.points = np.hstack((self.points, radar_pc.points))

    def save(self, filename: str = None, base_directory: str = '', scaling_factor: float = 1e-4) -> None:
        """
        Save this point cloud as a .las file. A .las file of version 1.4 with point format 4 is used. For information
        about the file type see https://laspy.readthedocs.io/en/latest/tut_background.html
        :param scaling_factor: .las file scaling factor. Effectively dictates how many decimals are stored
        :param base_directory: base directory to store files. Directory needs to have a subdirectory samples which has a
        subdirectory for each radar channel
        :param filename: name to store file under, has to include .las
        """
        assert base_directory == '' or osp.exists(base_directory), 'Base directory: \"{}\" does not exist' \
            .format(base_directory)
        if filename is not None:
            osp.join(base_directory, filename).replace('\\', '/')
        elif self.filename is not None:
            filename = osp.join(base_directory, self.filename.replace('.pcd', '.las')).replace('\\', '/')
        else:
            raise AssertionError('RadarPointCloud object has no filename assigned, give filename as function input')

        assert filename[-4:] == '.las', 'Filename should end in .las'

        # delete file if it already exists
        if osp.exists(filename):
            os.remove(filename)
        # make sure sub folders exist and create them if necessary except when directory name is ''
        # (current folder is used in that case)
        elif not osp.dirname(filename) == '' and not osp.exists(osp.dirname(filename)):
            folders: list = osp.dirname(filename).split('/')
            if len(folders) > 0:
                path = folders[0]
                if not osp.exists(path):
                    os.mkdir(path)
                for f in folders[1:]:
                    path += '/' + f
                    if not osp.exists(path):
                        os.mkdir(path)

        # initialize the las file
        header = las.header.Header(file_version=1.4, point_format=4)
        out = las.file.File(filename, mode='w', header=header)
        out.header.scale = np.ones(3) * scaling_factor
        # save sample data token
        out.header.software_id = self.sample_data_token
        out.header.file_source_id = self.nsweeps
        # set coordinates
        out.x = self.x()
        out.y = self.y()
        out.z = self.z()
        # set classification
        out.classification = self.labels().astype(int)
        # set velocities, these are already compensated for the ego vehicle velocity
        out.x_t = self.vx_comp()
        out.y_t = self.vy_comp()
        # save dynamic property states (indicating if target is moving or not) to user_data field
        out.user_data = self.dynprop()
        # save ambiguity state
        out.wave_packet_desc_index = self.ambig_state()
        # save invalid states
        out.waveform_packet_size = self.invalid_state()
        # save the sweep numbers, indicating which radar sweep a point belongs to
        out.pt_src_id = self.sweep_nums()
        out.close()


class RadarPCCombined(RadarPC):
    """A combination of multiple standard Radar point clouds"""
    __all_radar_modalities = ('RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT',
                              'RADAR_BACK_RIGHT')
    __allowed_parent_levels = ('sample', 'scene', 'log')
    __ref_channel = 'RADAR_FRONT'

    def __init__(self, points: np.ndarray, parent_level: str = None, parent_token: str = None,
                 sd_token_list: list = None, ego_positions: np.ndarray = None, ego_directions: np.ndarray = None):
        assert parent_level is None or parent_level in self.__allowed_parent_levels,\
            'Parent level was {}, but should be one of: {}'.format(parent_level, self.__allowed_parent_levels)
        super().__init__(points)
        self.sd_token_list = [] if sd_token_list is None else sd_token_list
        self.parent_level = parent_level
        self.parent_token = parent_token
        self.ego_positions = ego_positions
        self.ego_directions = ego_directions
        del self.filename, self.sample_data_token, self.ego_position, self.ego_direction

    def __getattr__(self, item):
        if item == 'sample_data_token':
            return self.sd_token_list[0]
        if item == 'ego_position':
            return self.ego_positions[:, 0]
        if item == 'ego_direction':
            return self.ego_directions[:, 0]

    @classmethod
    def from_sample(cls, sample_token: str, nusc: NuScenes, from_las: bool = True, base_directory: str = '',
                    nsweeps: int = 1) -> 'RadarPCCombined':
        """Makes a combined point cloud from all radar sensors in this sample"""
        points, sd_token_list, ego_position, ego_direction = \
            cls.__combine_sample_points(sample_token, nusc, from_las, base_directory, nsweeps)
        # Setup RadarPCCombined
        # todo: change how I handle ego_direction and position. First of all it needs to be transformed to ego vehicle
        #  position, not sensor position. Second of all this transposing stuff doesn't work in apply_baseline
        pc_comb = cls(points, 'sample', sample_token, sd_token_list,
                      ego_position.reshape((3, 1)), ego_direction.reshape((3, 1)))
        pc_comb.set_map_fields_from_sd_token(sd_token_list[0], nusc)
        pc_comb.nsweeps = int(np.max(pc_comb.sweep_nums())+1)
        return pc_comb

    @classmethod
    def from_scene(cls, scene_token: str, nusc: NuScenes) -> 'RadarPCCombined':
        points, sd_token_list, ego_positions, ego_directions = cls.__combine_scene_points(scene_token, nusc)
        pc_comb = cls(points, 'scene', scene_token, sd_token_list, ego_positions, ego_directions)
        pc_comb.set_map_fields_from_sd_token(sd_token_list[0], nusc)
        return pc_comb

    # helper functions
    @classmethod
    def __combine_sample_points(cls, sample_token: str, nusc: NuScenes, from_las: bool = True,
                                base_directory: str = '', nsweeps: int = 1):
        """
        Combines the point fields from all radar sample data entries of this sample
        :param sample_token: sample token
        :param nusc: nuscenes data set
        :param nsweeps: number of radar sweeps. Only used when not loaded from las
        :return: tuple of (points, sd_token_list) or (points, sd_token_list, ego_position, ego_direction)
        """
        sample = nusc.get('sample', sample_token)
        points = np.zeros((cls.nbr_dims(), 0))
        sd_token_list = []
        ego_position, ego_direction = None, None
        for chan in cls.__all_radar_modalities:
            sample_data_token = sample['data'][chan]
            if from_las:
                filename = nusc.get('sample_data', sample_data_token)['filename'].replace('.pcd', '.las')
                radar_pc = RadarPC.from_las(osp.join(base_directory, filename), nusc, nsweeps=nsweeps, do_logging=False)
            else:
                radar_pc = RadarPC.from_sample_data(sample_data_token, nusc, nsweeps)
            sd_token_list.append(sample_data_token)
            points = np.hstack((points, radar_pc.points))
            if chan == cls.__ref_channel:
                ego_position = radar_pc.ego_position
                ego_direction = radar_pc.ego_direction
        return points, sd_token_list, ego_position, ego_direction

    @classmethod
    def __combine_scene_points(cls, scene_token: str, nusc: NuScenes):
        """
        Combines all radar point clouds in this scene
        :param scene_token: token of the scene to combine
        :param nusc: nuscenes data set
        :return: tuple of (points, sd_token_list)
        """
        scene = nusc.get('scene', scene_token)
        points_combined = np.zeros((cls.nbr_dims(), 0))
        sd_token_list_combined = []
        ego_position_table = np.zeros((3, scene['nbr_samples']))
        ego_direction_table = np.zeros((3, scene['nbr_samples']))
        i, sample_token = 0, scene['first_sample_token']
        while sample_token != '':
            points, sd_token_list, ego_position, ego_direction = cls.__combine_sample_points(sample_token, nusc)
            points_combined = np.hstack((points_combined, points))
            sd_token_list_combined += sd_token_list
            ego_position_table[:, i] = ego_position
            ego_direction_table[:, i] = ego_direction
            i, sample_token = i + 1, nusc.get('sample', sample_token)['next']
        return points_combined, sd_token_list_combined, ego_position_table, ego_direction_table

    def get_boxes(self, nusc: NuScenes, in_pixels: bool = False) -> List[Box]:
        return self.get_boxes_base(nusc, self.sd_token_list[0], in_pixels)

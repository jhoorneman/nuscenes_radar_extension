# General imports
import sys
from collections import OrderedDict
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from colour import Color
from matplotlib.axes import Axes
from pyquaternion import Quaternion
import laspy as las
from functools import reduce
from typing import List, Tuple
import logging
import os.path as osp
import cv2 as cv
# Nuscenes imports
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils.data_classes import RadarPointCloud, Box
from nuscenes.utils.map_mask import MapMask
from nuscenes.utils.geometry_utils import view_points, BoxVisibility, transform_matrix
# Own code imports
from radar_pointcloud import RadarPC, RadarPCCombined
from road_estimation import RoadEstimation


class NuScenesRadarExtension(NuScenesExplorer):
    """
    Custom extension of NuScenesExplorer class that focuses on handling radar data
    """
    __ann_categories = {(255, 61, 99): 'bike/motor', (255, 158, 0): 'vehicle', (0, 0, 230): 'pedestrian',
                        (0, 0, 0): 'cone/barrier', (255, 0, 255): 'others'}
    __all_radar_modalities = ('RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT',
                              'RADAR_BACK_RIGHT')

    def __init__(self, nusc: NuScenes, modified_base_directory: str = '', preload_contour_maps: bool = False):
        """
        Initializes the Radar extension class
        :param nusc: the NuScenes data set that the functions of this class will use
        :param modified_base_directory: the base directory where modified files (contour map, las point cloud files)
        will be stored. It's functions as a replacement for nusc.dataroot
        :param preload_contour_maps: if true the contour maps will get preloaded. If false these maps will get stored
        once they have been loaded once by one of the functions
        """
        super().__init__(nusc)
        self.base_directory = modified_base_directory
        self.contour_maps = {}
        if preload_contour_maps:
            # Load all contour map versions of the nuscenes maps into the contour_maps dictionary
            for map_ in nusc.map:
                self.__contour_mask_from_token(map_['token'])

    #####################
    # Rendering methods #
    #####################

    def render_radar(self,
                     sample_token: str,
                     with_anns: bool = True,
                     box_vis_level: BoxVisibility = BoxVisibility.ANY,
                     axes_limit: float = 80,
                     ax: Axes = None,
                     nsweeps: int = 1,
                     out_path: str = None,
                     underlay_map: bool = True,
                     use_flat_vehicle_coordinates: bool = True,
                     radar_modalities: tuple = None) -> None:
        """
        Render sample data onto axis.
        :param sample_token: Sample token.
        :param with_anns: Whether to draw annotations.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param axes_limit: Axes limit for radar (measured in meters).
        :param ax: Axes onto which to render.
        :param nsweeps: Number of sweeps for radar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param underlay_map: When set to true, radar data is plotted onto the map. This can be slow.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
            aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
            can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
            setting is more correct and rotates the plot by ~90 degrees.
        :param radar_modalities: tuple of radar modalities to be shown, if empty all radar modalities are shown
        """
        # if radar modalities are not specified use all
        if radar_modalities is None:
            radar_modalities = self.__all_radar_modalities

        for modality in radar_modalities:
            if self.__all_radar_modalities.count(modality) != 1:
                raise ValueError("{} is not a valid radar sensor modality!".format(modality))

        # get
        sample_rec = self.nusc.get('sample', sample_token)
        ref_chan = 'LIDAR_TOP'
        ref_sd_token = sample_rec['data'][ref_chan]
        ref_sd_record = self.nusc.get('sample_data', ref_sd_token)

        points_total = np.empty([18, 1])
        velocities_total = np.empty([3, 1])
        # repeat the pointcloud gathering and velocity calculation for all radar types in radar_modalities
        for chan in radar_modalities:
            sd_token = sample_rec['data'][chan]
            sd_record = self.nusc.get('sample_data', sd_token)

            # Get aggregated radar point cloud in reference frame.
            # The point cloud is transformed to the reference frame for visualization purposes.
            pc, times = RadarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)

            # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
            # point cloud.
            radar_cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            ref_cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
            velocities = pc.points[8:10, :]  # Compensated velocity
            velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
            velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
            velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
            velocities[2, :] = np.zeros(pc.points.shape[1])

            if chan == radar_modalities[0]:
                points_total = pc.points
                velocities_total = velocities
            else:
                points_total = np.hstack((points_total, pc.points))
                velocities_total = np.hstack((velocities_total, velocities))

        # By default we render the sample_data top down in the sensor frame.
        # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
        # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
        if use_flat_vehicle_coordinates:
            # Retrieve transformation matrices for reference point cloud.
            cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
            pose_record = self.nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
            ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                          rotation=Quaternion(cs_record["rotation"]))

            # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
            ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            rotation_vehicle_flat_from_vehicle = np.dot(
                Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                Quaternion(pose_record['rotation']).inverse.rotation_matrix)
            vehicle_flat_from_vehicle = np.eye(4)
            vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
            viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
        else:
            viewpoint = np.eye(4)

        # Init axes.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 9))

        # Render map if requested.
        if underlay_map:
            assert use_flat_vehicle_coordinates, 'Error: underlay_map requires use_flat_vehicle_coordinates, as ' \
                                                 'otherwise the location does not correspond to the map!'
            self.render_ego_centric_map(sample_data_token=ref_sd_token, axes_limit=axes_limit, ax=ax)
            # This is fine for plotting, but slightly inaccurate when labeling: each radar sensor should have it's own
            # transformed map

        # Show point cloud.
        points = view_points(points_total[:3, :], viewpoint, normalize=False)
        dists = np.sqrt(np.sum(points_total[:2, :] ** 2, axis=0))
        # get color range and convert it to rgb
        red = Color("red")
        color_range = list(red.range_to(Color("green"), 101))
        rgba_range = np.ones((len(color_range), 4))
        for i in range(len(color_range)):
            rgba_range[i, 0:3] = color_range[i].get_rgb()
        color_idx = np.round(np.minimum(1, dists / axes_limit / np.sqrt(2)), 2) * 100
        colors = rgba_range[color_idx.astype(int), :]
        point_scale = 3.0
        ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)

        # Show velocities.
        points_vel = view_points(points_total[:3, :] + velocities_total, viewpoint, normalize=False)
        deltas_vel = points_vel - points
        deltas_vel = 6 * deltas_vel  # Arbitrary scaling
        max_delta = 20
        deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
        # colors_rgba = scatter.to_rgba(colors)
        for i in range(points.shape[1]):
            # ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i], color=colors_rgba[i])
            ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i], color=colors[i])

        # Show ego vehicle.
        ax.plot(0, 0, 'x', color='red')

        # Get boxes in lidar frame.
        _, boxes, _ = self.nusc.get_sample_data(ref_sd_token, box_vis_level=box_vis_level,
                                                use_flat_vehicle_coordinates=use_flat_vehicle_coordinates)

        # Show boxes.
        if with_anns:
            for box in boxes:
                c = np.array(self.get_color(box.name)) / 255.0
                box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=1)

        # Limit visible range.
        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)

        ax.axis('off')
        ax.set_title('Radar channels: {}'.format(radar_modalities))

        if out_path is not None:
            plt.savefig(out_path)

        print('Displayed {} radar points'.format(points_total.shape[1]))

    # noinspection PyArgumentList
    @classmethod
    def render_labels(cls, pointcloud: np.ndarray, background_map: np.ndarray, labels: np.ndarray, ax: Axes = None,
                      **kwargs) -> Axes:

        # Handle kwargs
        # plot settings
        plot_legend: bool = kwargs['plot_legend'] if 'plot_legend' in kwargs else False
        plot_ego_position: bool = kwargs['plot_ego_position'] if 'plot_ego_position' in kwargs else True
        plot_ego_direction: bool = kwargs['plot_ego_direction'] if 'plot_ego_direction' in kwargs else True
        plot_axes: bool = kwargs['plot_axes'] if 'plot_axes' in kwargs else True
        crop_map: bool = kwargs['crop_map'] if 'crop_map' in kwargs else True
        small_outliers: bool = kwargs['small_outliers'] if 'small_outliers' in kwargs else True
        scale_tick_labels: bool = kwargs['scale_tick_labels'] if 'scale_tick_labels' in kwargs else True
        plot_background: bool = kwargs['plot_background'] if 'plot_background' in kwargs else True
        figsize: tuple = kwargs['figsize'] if 'figsize' in kwargs else (16, 9)

        # parameters
        title: str = kwargs['title'] if 'title' in kwargs else None
        row_header: str = kwargs['row_header'] if 'row_header' in kwargs else None
        radius: float = kwargs['radius'] if 'radius' in kwargs else None
        point_scale: float = kwargs['point_scale'] if 'point_scale' in kwargs else 4
        map_resolution: float = kwargs['map_resolution'] if 'map_resolution' in kwargs else 1
        boxes: List[Box] = kwargs['boxes'] if 'boxes' in kwargs else None
        map_border: int = kwargs['map_border'] if 'map_border' in kwargs else 10
        label_names: dict = kwargs['label_names'] if 'label_names' in kwargs else {}

        # map colors
        color_neutral: float = kwargs['color_neutral'] if 'color_neutral' in kwargs else 255
        color_road: float = kwargs['color_road'] if 'color_road' in kwargs else 0
        color_search_area: float = kwargs['color_search_area'] if 'color_search_area' in kwargs else 150

        if 'ego_position' in kwargs and kwargs['ego_position'] is not None:
            ego_positions: np.ndarray = kwargs['ego_position'].reshape((3, 1))[:2]
        elif 'ego_positions' in kwargs:
            ego_positions: np.ndarray = kwargs['ego_positions'][:2, :]
        else:
            ego_positions: np.ndarray = np.zeros((2, 0))
        if 'ego_direction' in kwargs and kwargs['ego_direction'] is not None:
            ego_directions: np.ndarray = kwargs['ego_direction'].reshape((3, 1))[:2]
        elif 'ego_directions' in kwargs:
            ego_directions: np.ndarray = kwargs['ego_directions'][:2, :]
        else:
            ego_directions: np.ndarray = np.zeros((2, 0))

        # Assertions
        assert ego_directions.shape[1] == 0 or ego_directions.shape[1] == ego_positions.shape[1], \
            'Ego position and ego direction tables should have similar shape or the ego direction table should be empty'
        assert pointcloud.shape[0] == 2, 'Point cloud should have 3 dimensions (3D coordinates), ' \
                                         'but it had {}'.format(pointcloud.shape[0])
        assert pointcloud.shape[1] == len(labels), 'pointcloud and labels should have same length, it is respectively' \
                                                   ' length {} and {}'.format(pointcloud.shape[1], len(labels))

        # values that boundary pixels and neutral pixels have in the contour map
        neutral_value = background_map.min()
        boundary_value = background_map.max()

        # Initialize fig
        # noinspection PyTypeChecker
        fig: plt.Figure = None

        # modify map and point cloud for plotting
        coords = pointcloud
        if crop_map:
            if boxes is not None:
                box_centers = np.array([box.center for box in boxes]).transpose()[:2]
            else:
                box_centers = np.zeros((2, 0))
            background_map, translate_coords = NuScenesRadarExtension. \
                crop_map_around_pc(background_map, np.hstack((coords, ego_positions, box_centers)), map_border)
            coords = coords + translate_coords
        else:
            translate_coords = (0, 0)
        modified_map = background_map.copy()

        # color boundary points and background
        modified_map[background_map == neutral_value] = color_neutral
        modified_map[background_map == boundary_value] = color_road
        # color search area
        if radius is not None:
            x_range = background_map.shape[0]
            y_range = background_map.shape[1]
            for p in coords.transpose():
                indices = NuScenesRadarExtension.radius_to_indices(p, radius, 1, (0, y_range, 0, x_range))
                for idx in indices:
                    # if it is not a boundary point, color it gray
                    if background_map[idx] == neutral_value:
                        modified_map[idx] = color_search_area

        # Plotting
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        # plot contour map, ego vehicle
        if plot_background:
            ax.imshow(modified_map, cmap='gray', vmin=0, vmax=255, aspect='auto')
        if plot_ego_position:
            for ego_position in ego_positions.transpose():
                ego_position[0] = ego_position[0] + translate_coords[0]
                ego_position[1] = ego_position[1] + translate_coords[1]
                ax.plot(ego_position[0], ego_position[1], 'x', color='red', label='ego position')
        # plot annotations
        if boxes is not None:
            for box in boxes:
                box = BoxExtended(box)
                box.center[0] += translate_coords[0]
                box.center[1] += translate_coords[1]
                c = cls.get_color(box.name)
                label = cls.__ann_categories[c]
                c = np.array(c) / 255.0
                box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=1, label=label)
        # plot all point clouds with a certain label separately
        num_points = 0
        for label in np.unique(labels):
            point_scale_ = point_scale / 4 if small_outliers and label == -1 else point_scale
            filtered_coords = coords[:, labels == label]
            label_name = label_names[label] if label in label_names else 'cluster {}'.format(int(label))
            ax.scatter(filtered_coords[0], filtered_coords[1], s=point_scale_, label=label_name)
            num_points += filtered_coords.shape[1]
        assert num_points == coords.shape[1], 'All points of point cloud should get plotted, only {} of {} where' \
            .format(num_points, coords.shape[1])
        # plot ego direction if the table exists
        if plot_ego_direction and plot_ego_position and ego_directions.shape[1] > 0:
            scaling_factor = 50
            end_points = ego_positions + ego_directions * scaling_factor
            for i in range(ego_directions.shape[1]):
                ax.plot((ego_positions[0, i], end_points[0, i]), (ego_positions[1, i], end_points[1, i]), color='red')

        # More settings
        if plot_legend:
            handles, labels = ax.get_legend_handles_labels()
            unique_labels_only = OrderedDict(zip(labels, handles))
            ax.legend(unique_labels_only.values(), unique_labels_only.keys())
        if title is not None:
            ax.set_title(title)
        if row_header is not None:
            # decide what the labelpad value should be based on the longest line of the string
            pad = 0
            for line in row_header.split('\n'):
                pad = len(line) * 4 if len(line) * 4 > pad else pad
            ax.set_ylabel(row_header, rotation=0, size='large', labelpad=pad)
        if plot_axes and scale_tick_labels:
            ax.set_xticklabels((ax.get_xticks() * map_resolution).astype(int))
            ax.set_yticklabels((ax.get_yticks() * map_resolution).astype(int))
        elif not plot_axes:
            ax.set_xticks([])
            ax.set_yticks([])
        if fig is not None:
            fig.tight_layout()

        return ax

    def render_radar_pc(self, radar_pc: 'RadarPC', background_map: np.ndarray = None, ax: Axes = None,
                        **kwargs) -> Axes:
        """
        Render a point cloud from the RadarPC class
        :param radar_pc: the point cloud to render
        :param background_map: a specified background map, this map gets flipped in up/down direction, since that is
        how the point clouds are oriented
        :param ax: a specific axes to plot this on
        :param kwargs: settings for the render_radar_pc or render_labels function
        :return: the axes the point cloud was rendered on
        """
        # Handle kwargs
        labels = kwargs.pop('labels') if 'labels' in kwargs else radar_pc.labels()
        plot_normal_map: bool = kwargs['plot_normal_map'] if 'plot_normal_map' in kwargs else False
        with_anns: bool = kwargs['with_anns'] if 'with_anns' in kwargs else False
        with_fit_lines: bool = kwargs['with_fit_lines'] if 'with_fit_lines' in kwargs else False
        do_log: bool = kwargs['do_log'] if 'do_log' in kwargs else True
        if 'radius' in kwargs:
            kwargs['radius'] = kwargs['radius'] / radar_pc.map_resolution
        if type(radar_pc) is RadarPC:
            kwargs['ego_position'] = radar_pc.pixel_ego_position()
            kwargs['ego_direction'] = radar_pc.ego_direction
        elif type(radar_pc) is RadarPCCombined:
            # noinspection PyUnresolvedReferences
            kwargs['ego_positions'] = radar_pc.ego_positions / radar_pc.map_resolution
            # noinspection PyUnresolvedReferences
            kwargs['ego_directions'] = radar_pc.ego_directions
        if with_anns:
            kwargs['boxes'] = radar_pc.get_boxes(self.nusc, in_pixels=True)
        kwargs['map_resolution'] = radar_pc.map_resolution
        # Select map
        if background_map is None:
            if plot_normal_map:
                background_map = np.flipud(self.nusc.get('map', radar_pc.map_token)['mask'].mask())
            else:
                background_map = self.__contour_mask_from_token(radar_pc.map_token)
        else:
            background_map = np.flipud(background_map)
        # plotting
        ax = NuScenesRadarExtension.render_labels(radar_pc.xy(True), background_map, labels, ax, **deepcopy(kwargs))
        if with_fit_lines:
            ax = NuScenesRadarExtension.render_fit_lines(radar_pc, labels, ax, **deepcopy(kwargs))
        if do_log:
            print('Diplayed {} radar points'.format(radar_pc.nbr_points()))
        return ax

    @staticmethod
    def render_fit_lines(radar_pc: RadarPC, cluster_labels: np.ndarray, ax: Axes = None, **kwargs) -> Axes:
        # Handle kwargs
        fit_through: np.ndarray = kwargs['fit_through'] if 'fit_through' in kwargs else \
            np.unique(cluster_labels[cluster_labels >= 0])
        line_coeffs: np.ndarray = kwargs['line_coeffs'] if 'line_coeffs' in kwargs else \
            RoadEstimation.fit_lines_through_clusters(radar_pc, cluster_labels, fit_through)
        long_lines: bool = kwargs['long_lines'] if 'long_lines' in kwargs else False
        line_length_multiplier: float = kwargs['line_length_multiplier'] if 'line_length_multiplier' in kwargs else None
        map_border: int = kwargs['map_border'] if 'map_border' in kwargs else 10
        crop_map: bool = kwargs['crop_map'] if 'crop_map' in kwargs else True
        boxes: List[Box] = kwargs['boxes'] if 'boxes' in kwargs else None
        linewidth: float = kwargs['linewidth'] if 'linewidth' in kwargs else None
        line_color: str = kwargs['line_color'] if 'line_color' in kwargs else None
        line_alpha: float = kwargs['line_alpha'] if 'line_alpha' in kwargs else None
        if 'ego_position' in kwargs and kwargs['ego_position'] is not None:
            ego_positions: np.ndarray = kwargs['ego_position'].reshape((3, 1))[:2]
        elif 'ego_positions' in kwargs:
            ego_positions: np.ndarray = kwargs['ego_positions'][:2, :]
        elif radar_pc.ego_position is not None:
            ego_positions: np.ndarray = np.array(radar_pc.pixel_ego_position()).reshape((3, 1))[:2]
        else:
            ego_positions: np.ndarray = np.zeros((2, 0))
        # Assertions
        assert len(line_coeffs) == len(fit_through), 'The number of coefficient sets and clusters should be the same'
        # Other pre-calculations
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 9))
        if crop_map:
            if boxes is not None:
                box_centers = np.array([box.center for box in boxes]).transpose()[:2]
            else:
                box_centers = np.zeros((2, 0))
            _, translation = NuScenesRadarExtension.crop_map_around_pc(
                np.zeros((1, 1)), np.hstack((radar_pc.xy(True), ego_positions, box_centers)), map_border)
        else:
            translation = (0, 0)
        # Plotting lines
        # repeat first line to make cluster and line colors match up (since first cluster color is used for outliers)
        if line_coeffs.size > 0:
            line_coeffs = np.vstack((line_coeffs[0], line_coeffs))
            fit_through = np.hstack((fit_through[0], fit_through))
        for i, coeff in enumerate(line_coeffs):
            assert len(coeff) == 2, 'Only polynomials of the first degree are supported at the moment'
            # compensate for map resolution
            coeff[1] = coeff[1] / radar_pc.map_resolution
            poly = np.poly1d(coeff)
            # set lenght of lines: spanning the complete width of the plot, the width of the cluster or a line_length_multiplier x
            # the cluster width

            if long_lines:
                # max line lenght
                x = RoadEstimation.get_x_in_lim(poly, translation, ax.get_xlim(), ax.get_ylim())
            else:
                # cluster width line length
                x = (radar_pc.x(True)[cluster_labels == fit_through[i]].min(),
                     radar_pc.x(True)[cluster_labels == fit_through[i]].max())
                if line_length_multiplier is not None:
                    x_length = (x[1]-x[0]) * line_length_multiplier
                    x = np.array((x[0] - x_length / 2, x[1] + x_length / 2))
                    x_lim = RoadEstimation.get_x_in_lim(poly, translation, ax.get_xlim(), ax.get_ylim())
                    x = (max(x_lim[0], x[0]), min(x_lim[1], x[1]))

            x = np.array(x)
            x_plot = x + translation[0]
            y_plot = poly(x) + translation[1]
            ax.plot(x_plot, y_plot, linewidth=linewidth, color=line_color, alpha=line_alpha)
        return ax

    def render_radar_grid(self, arg_matrix: 'ArgumentMatrix', title: str = '', figsize: tuple = None) \
            -> Tuple[plt.Figure, np.ndarray]:
        """Renders a grid plot using render_radar_pc with arg_matrix specified in ArgumentMatrix"""
        # Some standard settings for plotting in grids, will be used unless specified otherwise
        arg_matrix.add_univ_kwarg('plot_axes', False)
        arg_matrix.add_univ_kwarg('do_log', False)
        figsize = (16, 9) if figsize is None else figsize
        fig, axes = plt.subplots(arg_matrix.height, arg_matrix.width, figsize=figsize)
        for i in range(arg_matrix.height):
            for j in range(arg_matrix.width):
                arguments = arg_matrix[i, j]
                # add the axes to the kwargs dictionary
                if type(axes) != np.ndarray:
                    arguments.kwargs['ax'] = axes
                elif axes.ndim == 1:
                    arguments.kwargs['ax'] = axes[max(i, j)]
                else:
                    arguments.kwargs['ax'] = axes[i, j]

                self.render_radar_pc(*arguments.args, **arguments.kwargs)

        fig.suptitle(title)
        fig.tight_layout()
        fig.subplots_adjust(left=0.10, top=0.90, wspace=0.01, hspace=0.01)
        return fig, axes

    def render_cluster_grid(self, radar_pc: RadarPC, eps_params: tuple, min_pnts_params: tuple, **kwargs) \
            -> Tuple[plt.Figure, np.ndarray]:
        """
        Render an overview plot of the clustering of a radar_pc with different clustering parameters
        """
        assert len(eps_params) <= 5, 'a maximum of 5 eps variants can get plotted'
        assert len(min_pnts_params) <= 5, 'a maximum of 5 min_pnts variants can get plotted'
        # add universal settings to kwargs dictionary if they were not already specified
        _kwargs = {'plot_axes': False, 'plot_ego_direction': False, 'small_outliers': True}
        for key in _kwargs:
            if key not in kwargs:
                kwargs[key] = _kwargs[key]
        title = 'DBSCAN algorithm on radar point cloud for different parameters'
        # initialize argument matrix
        arg_mat = ArgumentMatrix(len(min_pnts_params), len(eps_params), kwargs)
        # filling argument matrix
        for i, min_pnts in enumerate(min_pnts_params):
            for j, eps in enumerate(eps_params):
                args = arg_mat[i, j]
                args.args.append(radar_pc)
                args.kwargs['labels'] = RoadEstimation.clustering(radar_pc.xyv(normalized=True), eps, min_pnts)
                if j == 0:
                    args.kwargs['row_header'] = 'min\npoints: {}'.format(min_pnts)
                if i == 0:
                    args.kwargs['title'] = 'epsilon: {}'.format(eps)
        return self.render_radar_grid(arg_mat, title)

    ##########################
    # Infrastructure methods #
    ##########################

    def load_las(self, sample_data_token: str = None, filename: str = None):
        """
        Read las file like the ones saved by pcd_to_las and return it as a 6D numpy array with the entries being
        (x, y, z, vx_comp, vy_comp, labels)
        """
        if sample_data_token is not None:
            sample_data = self.nusc.get('sample_data', sample_data_token)
            filename = osp.join(self.base_directory, sample_data['filename'].replace('.pcd', '.las'))
        elif filename is None:
            raise ValueError('Specify sample_data_token or  a file name')

        file = las.file.File(filename, mode='r')
        output = np.vstack((file.x, file.y, file.z, file.x_t, file.y_t, file.classification))
        file.close()
        return output

    def __transform_pc(self, sample_data_token: str):
        """
       Transforms a radar point cloud and the corresponding part of the map the same way as happens in
       render_sample_data and RadarPointCloud.from_file_multisweep to match the point cloud and map coordinates.
       This is a work around solution
       :return: transformed point cloud (coordinates and velocities)
        """
        # nuscenes stuff
        current_sample_token, _, current_log_token = self.tokens_from_sample_data(sample_data_token)
        current_sample_record = self.nusc.get('sample', current_sample_token)
        current_sd_record = self.nusc.get('sample_data', sample_data_token)
        current_pose_record = self.nusc.get('ego_pose', current_sd_record['ego_pose_token'])
        current_cs_record = self.nusc.get('calibrated_sensor', current_sd_record['calibrated_sensor_token'])

        ref_channel = 'LIDAR_TOP'
        ref_sd_token = current_sample_record['data'][ref_channel]
        ref_sd_record = self.nusc.get('sample_data', ref_sd_token)
        ref_pose_record = self.nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
        ref_cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])

        # get radar point cloud
        pc = RadarPointCloud.from_file(osp.join(self.nusc.dataroot, current_sd_record['filename']))
        points = np.vstack((pc.points[:3, :], np.ones(pc.points.shape[1])))

        # Transformations done in from_file_multisweep
        # Homogeneous transform from ego car frame to reference frame.
        ref_from_car = transform_matrix(ref_cs_record['translation'], Quaternion(ref_cs_record['rotation']),
                                        inverse=True)
        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(ref_pose_record['translation'], Quaternion(ref_pose_record['rotation']),
                                           inverse=True)
        global_from_car = transform_matrix(current_pose_record['translation'],
                                           Quaternion(current_pose_record['rotation']), inverse=False)
        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        car_from_current = transform_matrix(current_cs_record['translation'], Quaternion(current_cs_record['rotation']),
                                            inverse=False)
        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
        points = trans_matrix.dot(points)

        # Transformations done in render_sample_data
        # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
        ref_to_ego = transform_matrix(translation=ref_cs_record['translation'],
                                      rotation=Quaternion(ref_cs_record["rotation"]))
        ego_yaw = Quaternion(ref_pose_record['rotation']).yaw_pitch_roll[0] + \
                  Quaternion(current_pose_record['rotation']).yaw_pitch_roll[0]
        rotation_vehicle_flat_from_vehicle = np.dot(
            Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
            Quaternion(ref_pose_record['rotation']).inverse.rotation_matrix)
        vehicle_flat_from_vehicle = np.eye(4)
        vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
        viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
        points = viewpoint.dot(points)

        # Transform radar velocities (x is front, y is left)
        velocities = pc.points[8:10, :]  # Compensated velocity
        velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
        velocities = np.dot(Quaternion(current_cs_record['rotation']).rotation_matrix, velocities)
        velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
        points = np.vstack((points[:3, :], velocities[:2, :]))
        return points

    def __transform_contour_map(self, sample_data_token: str, axes_limit: int = 150, base_contour_map:
    np.ndarray = None) -> tuple:

        def crop_image(image: np.array,
                       x_px: int,
                       y_px: int,
                       axes_limit_px: int) -> np.array:
            x_min = int(x_px - axes_limit_px)
            x_max = int(x_px + axes_limit_px)
            y_min = int(y_px - axes_limit_px)
            y_max = int(y_px + axes_limit_px)

            cropped_image = image[y_min:y_max, x_min:x_max]

            return cropped_image

        # nuscenes stuff
        current_sample_token, _, current_log_token = self.tokens_from_sample_data(sample_data_token)
        current_sd_record = self.nusc.get('sample_data', sample_data_token)
        current_pose_record = self.nusc.get('ego_pose', current_sd_record['ego_pose_token'])

        # get map information
        basic_mask: MapMask = self.nusc.get('map', self.nusc.get('log', current_log_token)['map_token'])['mask']
        if base_contour_map is None:
            mask_raster = self.__contour_mask_from_sd(sample_data_token)
        else:
            mask_raster = base_contour_map

        # Transformations done in render_egocentric_map
        # Retrieve and crop mask.
        pixel_coords = basic_mask.to_pixel_coords(current_pose_record['translation'][0],
                                                  current_pose_record['translation'][1])
        scaled_limit_px = int(axes_limit * (1.0 / basic_mask.resolution))
        # noinspection PyTypeChecker
        ego_centric_map = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], scaled_limit_px)
        return ego_centric_map, basic_mask.resolution

    def transform_pc_and_contour(self, sample_data_token: str, axes_limit: int = 150, base_contour_map:
    np.ndarray = None, do_plot: bool = False) -> tuple:
        """
        Transforms a radar point cloud and the corresponding part of the map the same way as happens in
        render_sample_data, RadarPointCloud.from_file_multisweep and render_egocentric_map to match the point cloud and
        map coordinates.
        This is a work around solution
        :return: tuple of transformed point cloud (coordinates and velocities), transformed contour map and map
        resolution
        """
        points = self.__transform_pc(sample_data_token)
        ego_centric_map, resolution = self.__transform_contour_map(sample_data_token, axes_limit, base_contour_map)

        # Transformation done to compensate for not using the matplotlib.imshow extent parameter
        points[:2, :] += np.array((axes_limit, axes_limit)).reshape((2, 1))
        ego_centric_map = np.flipud(ego_centric_map)

        # Plotting
        if do_plot:
            plt.figure()
            plt.imshow(ego_centric_map, cmap='gray', vmin=0, vmax=255, aspect='auto')
            plt.scatter(points[0, :] / resolution, points[1, :] / resolution, s=3)
            plt.plot(axes_limit / resolution, axes_limit / resolution, 'x', color='red')

        return points, ego_centric_map, resolution

    def save_map_contour(self, map_token: str) -> bool:
        """
        Converts a binary map to a binary map with only contours
        :param map_token: map to convert
        """
        map_ = self.nusc.get('map', map_token)
        mask = map_['mask']
        mask_raster = mask.mask()
        contours, hierarchy = cv.findContours(mask_raster, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        cont_image = np.zeros(mask_raster.shape).astype(np.uint8)
        for cont in contours:
            for point in cont:
                cont_image[point[0][1], point[0][0]] = 255

        return np.save(osp.join(self.base_directory, map_['filename'].replace('.png', '.npy')), cont_image)

    ######################
    # Annotation methods #
    ######################

    def annotate_base(self, radar_pc: 'RadarPC', radius: float, dynprop_states: list, boundary_categories: tuple,
                      print_status: bool = False) -> list:
        """
        Annotate a point cloud for boundary points using the global contour map

        :param dynprop_states: these dynprop states can be boundary points. Notable states: 1 stationary, 3 stationary
        candidate, 5 (crossing stationary)
        :param radar_pc: numpy array of 3D radar points and velocities
        :param radius: if a point is within radius of a contour point is gets labeled as a boundary point
        :param boundary_categories: points in an annotation bounding box of these types can still be boundary points
        :param print_status: print status updates when this is True
        :return: return a list of labels corresponding to the point cloud
        """

        coords = radar_pc.xyz().transpose()
        contour_map = self.__contour_mask_from_token(radar_pc.map_token)
        is_stationary = [p in dynprop_states for p in radar_pc.dynprop()]

        # points inside an annotated bounding box should not be labeled as boundary point
        if type(radar_pc) is RadarPCCombined:
            sd_token = radar_pc.sd_token_list[0]
        else:
            sd_token = radar_pc.sample_data_token
        in_bb = self.points_in_boxes(self.nusc.get_boxes(sd_token), coords.transpose(),
                                     exclude_names=boundary_categories)

        labels = []
        # label a point as boundary when a contour point is in range
        for i, c in enumerate(coords):
            # if the point is not stationary or in a bounding box, label it as non-boundary and go to next point
            if not is_stationary[i] or in_bb[i]:
                labels.append(0)
                continue

            # optimizations: first check the biggest square that is contained by the cirlce and the smallest square that
            # is containing the circle for road boundaries
            side_length = radius*2
            mask_x = [np.ceil((c[0] - side_length / 2) / radar_pc.map_resolution).astype(int),
                      np.floor((c[0] + side_length / 2) / radar_pc.map_resolution).astype(int) + 1]
            mask_y = [np.ceil((c[1] - side_length / 2) / radar_pc.map_resolution).astype(int),
                      np.floor((c[1] + side_length / 2) / radar_pc.map_resolution).astype(int) + 1]
            # make sure the mask don't cross the map boundaries
            mask_x[0] = max(mask_x[0], 0)
            mask_x[1] = min(mask_x[1], contour_map.shape[1])
            mask_y[0] = max(mask_y[0], 0)
            mask_y[1] = min(mask_y[1], contour_map.shape[0])
            # switch x and y in mask, since that is how the coordinates on the contour map work
            mask = tuple(np.ogrid[mask_y[0]:mask_y[1], mask_x[0]:mask_x[1]])
            in_radius = contour_map[mask]
            if in_radius.sum() == 0:
                labels.append(0)
                continue

            # square contained by circle:
            side_length = np.sqrt(2)*radius
            mask_x = [np.ceil((c[0] - side_length / 2) / radar_pc.map_resolution).astype(int),
                      np.floor((c[0] + side_length / 2) / radar_pc.map_resolution).astype(int) + 1]
            mask_y = [np.ceil((c[1] - side_length / 2) / radar_pc.map_resolution).astype(int),
                      np.floor((c[1] + side_length / 2) / radar_pc.map_resolution).astype(int) + 1]
            # make sure the mask don't cross the map boundaries
            mask_x[0] = max(mask_x[0], 0)
            mask_x[1] = min(mask_x[1], contour_map.shape[1])
            mask_y[0] = max(mask_y[0], 0)
            mask_y[1] = min(mask_y[1], contour_map.shape[0])
            # switch x and y in mask, since that is how the coordinates on the map work
            mask = tuple(np.ogrid[mask_y[0]:mask_y[1], mask_x[0]:mask_x[1]])
            in_radius = contour_map[mask]
            if in_radius.sum() > 0:
                labels.append(1)
                continue

            # if the two square approximations don't give an answer, check each point in a circle (more expensive)
            labeled = False
            indices = self.radius_to_indices(c, radius, radar_pc.map_resolution,
                                             (0, contour_map.shape[1], 0, contour_map.shape[0]))
            indices_x = [ind[0] for ind in indices]
            indices_y = [ind[1] for ind in indices]
            for j in indices:
                if contour_map[j] > 0:
                    labels.append(1)
                    labeled = True
                    break
            if not labeled:
                labels.append(0)
        assert len(labels) == len(coords), 'each radar point should have a label. There were {} and {} labels and ' \
                                           'radar points respectively'.format(len(labels), len(coords))

        if print_status:
            logging.info('Annotated {} points. {} were boundary points'.format(len(coords), np.sum(labels)))

        radar_pc.labels()[:] = labels
        return labels

    def annotate_sample_data(self, sample_data_token: str, radius: float, dynprop_states: list, boundary_categories: tuple,
                             nsweeps: int, do_logging: bool = False) -> tuple:
        """
        Annotate the point cloud of this sample_data entry and save the result in a .las file.
        :param sample_data_token: token of sample_data to annotate
        :param radius: if a point is within radius of a contour point is gets labeled as a boundary point
        :param dynprop_states: these dynprop states can be boundary points. Notable states: 1 stationary, 3 stationary
        candidate, 5 (crossing stationary)
        :param boundary_categories: points in an annotation bounding box of these types can still be boundary points
        :param nsweeps: number radar sweeps to include. With nsweeps = 1, only the current radar pointcloud is used
        :param do_logging: print status updates when this is True
        :return: return tuple of (annotated points, total points)
        """
        assert type(nsweeps) == int and nsweeps > 0, "nsweeps should be an integer bigger than 0. It was {} of type {}" \
            .format(nsweeps, type(nsweeps))

        # initial point cloud
        radar_pc = RadarPC.from_sample_data(sample_data_token, self.nusc)
        self.annotate_base(radar_pc, radius, dynprop_states, boundary_categories, print_status=False)
        # for each sweep label the boundary points in this point cloud and add them together
        sd_token_loop = sample_data_token
        actual_sweeps = 1
        for sweep in range(1, nsweeps):
            sd_token_loop = self.nusc.get('sample_data', sd_token_loop)['prev']
            if sd_token_loop == '':
                break
            radar_pc_loop = RadarPC.from_sample_data(sd_token_loop, self.nusc)
            radar_pc_loop.sweep_nums()[:] = np.ones(radar_pc_loop.nbr_points())*sweep
            self.annotate_base(radar_pc_loop, radius, dynprop_states, boundary_categories, print_status=False)
            radar_pc.extend_with(radar_pc_loop)
            actual_sweeps += 1
        radar_pc.nsweeps = actual_sweeps
        labels = radar_pc.labels()

        channel = self.nusc.get('sample_data', sample_data_token)['channel']
        channel = channel + ' ' * (17 - len(channel))
        if radar_pc.nbr_points() > 0:
            if do_logging:
                logging.info('\t\t\tAnnotated {} from {} sweeps. Found {}/{} boundary points.'
                             .format(channel, actual_sweeps, int(np.sum(labels)), len(labels)))
            radar_pc.save(base_directory=self.base_directory)
            return np.sum(labels), len(labels)
        else:
            if do_logging:
                logging.warning('\t\t\tWarning: point cloud from channel {} was empty so no .las file was saved'
                                .format(channel))
            return 0, 0

    def annotate_sample(self, sample_token: str, radius: float, dynprop_states: list, boundary_categories: tuple,
                        nsweeps: int, do_logging: bool = False, sample_number: int = None) -> tuple:
        """
        annotates all radar data point cloud of a sample using the annotate_sample_data method
        :param sample_token: token the to annotate sample
        :param radius: if a point is within radius of a contour point is gets labeled as a boundary point
        :param dynprop_states: these dynprop states can be boundary points. Notable states: 1 stationary, 3 stationary
        candidate, 5 (crossing stationary)
        :param boundary_categories: points in an annotation bounding box of these types can still be boundary points
        :param nsweeps: number radar sweeps to include. With nsweeps = 1, only the current radar pointcloud is used
        :param do_logging: print status updates when this is True
        :param sample_number: to be printed in debug message
        :return: return tuple of (annotated points, total points) tuples from each sample data
        """
        sample = self.nusc.get('sample', sample_token)
        sample_data_tokens = []
        for data in sample['data']:
            # only append radar data
            token = sample['data'][data]
            if self.nusc.get('sample_data', token)['sensor_modality'] == 'radar':
                sample_data_tokens.append(token)
        if do_logging:
            if sample_number is None:
                sample_number = ''
            else:
                sample_number = str(sample_number)
            logging.info('\t Annotating data from sample ' + sample_number + ', token: ' + sample_token)

        output = []
        for token in sample_data_tokens:
            out = self.annotate_sample_data(token, radius, dynprop_states, boundary_categories, nsweeps, do_logging)
            output.append(out)

        return tuple(output)

    def annotate_scene(self, scene_token: str, radius: float, dynprop_states: list, boundary_categories: tuple,
                       nsweeps: int, do_logging: bool = False) -> tuple:
        """
        annotate a complete scene by looping over all samples
        :return: return tuple of tuples with annotation data from each sample
        """
        scene = self.nusc.get('scene', scene_token)
        sample_token = scene['first_sample_token']
        if do_logging:
            logging.info('Annotating data from ' + scene['name'])

        sample_number = 0
        output = []
        while sample_token != '':
            sample = self.nusc.get('sample', sample_token)
            out = self.annotate_sample(sample_token, radius, dynprop_states, boundary_categories, nsweeps,
                                       do_logging, sample_number)
            output.append(out)
            sample_token = sample['next']
            sample_number += 1
        return tuple(output)

    def annotation_table(self, scene_list: list, ann_data: list) -> str:
        """ Lists all scenes with some meta data. """

        def scene_ann_count(scene_ann_data) -> tuple:
            ann_points = 0
            total_points = 0
            for sample_data in scene_ann_data:
                for pc_data in sample_data:
                    ann_points += int(pc_data[0])
                    total_points += int(pc_data[1])
            return ann_points, total_points

        def bound(text: str, length: int) -> str:
            return text.ljust(length)[:length]

        csize = (10, 32, 18, 12, 42)
        out_string = '\n' + bound('Name', csize[0]) + ' ' + \
                     bound('Token', csize[1]) + ' ' + \
                     bound('Location', csize[2]) + ' ' + \
                     bound('Annotated', csize[3]) + ' ' + \
                     bound('Description', csize[4])
        for i, scene in enumerate(scene_list):
            location = self.nusc.get('log', scene['log_token'])['location']
            ann_count = scene_ann_count(ann_data[i])
            out_string += '\n' + bound(scene['name'], csize[0]) + ' ' + \
                          bound(scene['token'], csize[1]) + ' ' + \
                          bound(location, csize[2]) + ' ' + \
                          bound('{}/{}'.format(ann_count[0], ann_count[1]), csize[3]) + ' ' + \
                          bound(scene['description'], csize[4]) + '...'

        return out_string

    ######################
    # Evaluation methods #
    ######################

    def evaluate_baseline_from_sd(self, sample_data_token: str, params: dict) -> Tuple[np.ndarray, list]:
        """
        Evaluate the baseline performance on a sample data radar pointcloud
        :param sample_data_token: sample data token
        :param params: dict of evaluation parameters. At the moment this consist of: (eps, min_pnts, assign_radius)
        """
        # load radar_pc file and set its relevant field from the sample data token
        sample_data = self.nusc.get('sample_data', sample_data_token)
        filename = osp.join(self.base_directory, sample_data['filename'].replace('.pcd', '.las'))
        radar_pc = RadarPC.from_las(filename, self.nusc, do_logging=False)
        return RoadEstimation.evaluate_baseline(radar_pc, params)

    def evaluate_baseline_from_sample(self, sample_token: str, params: dict, do_per_sd: bool = False,
                                      do_logging: bool = False) -> dict:
        """Evaluates baseline on a sample by evaluating each radar sensor individually
        :param sample_token: sample token to run the baseline method on
        :param params: baseline paramaters to use dictionary with ie eps, min_pnts and assign_radius
        :param do_per_sd: when true the baseline will run on each radar sample_data instance seperately
        :param do_logging: when true log messages will be printed
        """
        sample = self.nusc.get('sample', sample_token)
        if do_per_sd:
            raise NotImplementedError('Evaluating baseline per sample data no longer supported')
            # sample_data_tokens = []
            # confusion_matrix = np.zeros((2, 2))
            # for data in sample['data']:
            #     # only append radar data
            #     token = sample['data'][data]
            #     if self.nusc.get('sample_data', token)['sensor_modality'] == 'radar':
            #         sample_data_tokens.append(token)
            #
            # for token in sample_data_tokens:
            #     confusion_matrix += self.evaluate_baseline_from_sd(token, params)
        else:
            radar_pc = RadarPCCombined.from_sample(sample_token, self.nusc, True, self.base_directory,
                                                   params['nsweeps'])
            confusion_matrix, boundary_distance = RoadEstimation.evaluate_baseline(radar_pc, params)
        if do_logging:
            logging.info('\t Evaluating baseline on sample: ' + sample_token)
        point_data = np.vstack((radar_pc.labels(), boundary_distance, radar_pc.distance_to_ego()))
        return {'conf_matrix': confusion_matrix,
                'point_data': point_data}

    def evaluate_baseline_from_scene(self, scene_token: str, params: dict, do_logging: bool = False) \
            -> dict:
        """Evaluate baseline on a scene by evaluating each sample individually and adding results together
        :param scene_token: scene to evaluate on
        :param params: dictionary holding all relevant parameters
        :param do_logging: log intermediate results
        :return: Dict[confusion matrix of scene, list of f1-scores for each of the samples, ground truth labels,
        boundary distance and ego distance for each point in the scene]"""

        scene = self.nusc.get('scene', scene_token)
        sample_token = scene['first_sample_token']
        confusion_matrix = np.zeros((2, 2))
        sample_f1_list = []
        point_data = np.zeros((3, 0))
        while sample_token != '':
            sample = self.nusc.get('sample', sample_token)
            sample_out = self.evaluate_baseline_from_sample(sample_token, params, do_logging=False)
            confusion_matrix += sample_out['conf_matrix']
            point_data = np.hstack((point_data, sample_out['point_data']))
            _, _, f1 = RoadEstimation.get_precision_recall_f1(confusion_matrix)
            sample_f1_list.append(f1)
            sample_token = sample['next']
        if do_logging:
            stats = RoadEstimation.get_precision_recall_f1(confusion_matrix)
            logging.info('Evaluated baseline on ' + scene['name'] + '. Precision: {:.3f}, Recall: {:.3f}, F1 Score '
                                                                    '{:.3f}'
                         .format(stats[0], stats[1], stats[2]))
        return {'conf_matrix': confusion_matrix,
                'sample_f1': sample_f1_list,
                'point_data': point_data}

    ###################
    # Support methods #
    ###################

    def sample_tokens_from_scene(self, scene_token: str) -> list:
        sample_token = self.nusc.get('scene', scene_token)['first_sample_token']
        token_list = []
        while sample_token != '':
            token_list.append(sample_token)
            sample_token = self.nusc.get('sample', sample_token)['next']
        return token_list

    def las_filename_from_sd(self, sample_data_token: str):
        sample_data = self.nusc.get('sample_data', sample_data_token)
        return osp.join(self.base_directory, sample_data['filename'].replace('.pcd', '.las'))

    @staticmethod
    def radius_to_indices(center: tuple, radius: float, map_resolution: float = 1, boundaries: tuple = None) -> list:
        """
        Returns indices that are in radius range of the center point. x and y values of pointcloud get swapped
        :param center: center point as 2D array index, so (x, y)
        :param radius: radius in which the indices have to lie
        :param map_resolution: how much [measuring unit] one pixel is
        :param boundaries: the indices have to stay within these boundaries tuple of (xlower, xupper, ylower, yupper)
        :return: list of indices each with coordinates (y, x)
        """

        def is_out_of_bounds(value, lower_bound, upper_bound):
            return value < lower_bound or value >= upper_bound

        index_list = []
        assert boundaries is None or len(boundaries) == 4, 'Boundaries has to be a tuple of length 4, now it is {}' \
            .format(len(boundaries))

        if boundaries is not None:
            (xlower, xupper, ylower, yupper) = boundaries
            assert xupper >= xlower and yupper >= ylower, 'upper boundary can\'t be lower than lower boundary, ' \
                                                          'boundaries: {}'.format(boundaries)
        else:
            (xlower, xupper, ylower, yupper) = (-sys.maxsize, sys.maxsize, -sys.maxsize, sys.maxsize)
        # get radius and center coordinates in pixels instead of meters
        radius = radius / map_resolution
        cx = center[0] / map_resolution
        cy = center[1] / map_resolution
        max_y = int(np.floor(cy + radius))
        min_y = int(np.ceil(cy - radius))
        for y in range(min_y, max_y + 1):
            if is_out_of_bounds(y, ylower, yupper):
                continue
            max_x = int(np.floor(cx + np.sqrt(radius ** 2 - (y - cy) ** 2)))
            min_x = int(np.ceil(cx - np.sqrt(radius ** 2 - (y - cy) ** 2)))
            for x in range(min_x, max_x + 1):
                if is_out_of_bounds(x, xlower, xupper):
                    continue
                # x and y are switched since that is how the map coordinates work
                index_list.append((y, x))
        return index_list

    @staticmethod
    def crop_map_around_pc(map_: np.ndarray, pc: np.ndarray, border: int = 1) -> tuple:
        """
        Crop map around the point cloud
        :return: return tuple of cropped map and transformation for point cloud to stay in the same space
        """
        assert pc.shape[0] == 2, "point cloud should have 2 dimensions"
        assert border >= 1, "border parameter should be 1 or higher"
        min_x, max_x = int(np.round(np.min(pc[0])) - border), int(np.round(np.max(pc[0])) + border)
        min_y, max_y = int(np.round(np.min(pc[1])) - border), int(np.round(np.max(pc[1])) + border)
        # switch x and y around because matrix coordinates are (y, x)
        return map_[min_y:max_y, min_x:max_x], np.array((-min_x, -min_y)).reshape((2, 1))

    def __contour_mask_from_sd(self, sample_data_token: str) -> np.ndarray:
        """
        get map mask from sample data token
        """
        sample_data = self.nusc.get('sample_data', sample_data_token)
        return self.__contour_mask_from_sample(sample_data['sample_token'])

    def __contour_mask_from_sample(self, sample_token: str) -> np.ndarray:
        """
        get map mask from sample token
        """
        sample = self.nusc.get('sample', sample_token)
        return self.__contour_mask_from_scene(sample['scene_token'])

    def __contour_mask_from_scene(self, scene_token: str) -> np.ndarray:
        """
        get map mask from scene token
        """
        scene = self.nusc.get('scene', scene_token)
        map_token = self.nusc.get('log', scene['log_token'])['map_token']
        return self.__contour_mask_from_token(map_token)

    def __contour_mask_from_token(self, map_token: str) -> np.ndarray:
        """
        get map from map_token, also stores the map in a class wide dictionary so it doesn't have to get loaded again
        """
        if map_token in self.contour_maps:
            return self.contour_maps[map_token]
        else:
            # When the base directory contains no maps folder check if the parent or parent-parent directory contains
            # a maps folder
            if osp.isdir(osp.join(self.base_directory, 'maps')):
                map_base_directory = self.base_directory
            elif osp.isdir(osp.join(self.base_directory, '../maps')):
                map_base_directory = osp.join(self.base_directory, '..')
            elif osp.isdir(osp.join(self.base_directory, '../../maps')):
                map_base_directory = osp.join(self.base_directory, '../..')
            else:
                raise FileNotFoundError('No maps folder found in {} and its parent or parent-parent directories'
                                        .format(self.base_directory))

            filename = osp.join(map_base_directory, self.nusc.get('map', map_token)['filename'].replace('.png', '.npy'))
            im = np.flipud(np.load(filename))
            assert im is not None, 'No map was found at location {}'.format(filename)
            assert len(im.shape) == 2, 'Contour map should be 2D, this one is {}D'.format(len(im.shape))
            self.contour_maps[map_token] = im
            return im

    def tokens_from_sample_data(self, sample_data_token) -> tuple:
        """
        Gives tokens of higher data base categories
        :param sample_data_token:
        :return: tuple of sample, scene and log tokens
        """
        sample_token = self.nusc.get('sample_data', sample_data_token)['sample_token']
        tokens = self.tokens_from_sample(sample_token)
        return sample_token, tokens[0], tokens[1]

    def tokens_from_sample(self, sample_token) -> tuple:
        """
        Gives tokens of higher data base categories
        :param sample_token:
        :return: tuple of scene and log tokens
        """
        sample = self.nusc.get('sample', sample_token)
        scene = self.nusc.get('scene', sample['scene_token'])
        return scene['token'], scene['log_token']

    @staticmethod
    def __points_in_box(box, points, wlh_factor: float = 1.0, exclude_names: tuple = ()):
        """
        Checks whether points are inside the box.
        Z coordinates are ignored, code that test for z coordinates is commented out

        Picks one corner as reference (p1) and computes the vector to a target point (v).
        Then for each of the 3 axes, project v onto the axis and compare the length.
        Inspired by: https://math.stackexchange.com/a/1552579
        :param box: <Box>.
        :param points: <np.float: 3, n>.
        :param wlh_factor: Inflates or deflates the box.
        :param exclude_names: boxes whose name includes one of these terms are excluded
        :return: <np.bool: n, >.
        """
        assert type(exclude_names) == tuple, "exclude names should be a tuple, but was {}. Exclude names content: {}" \
            .format(type(exclude_names), exclude_names)
        for name in exclude_names:
            if name in box.name:
                # logging.info('name: {}, was in the exclude names list so box {} is ignored'.format(name, box.name))
                return np.zeros(points.shape[1], dtype=bool)

        corners = box.corners(wlh_factor=wlh_factor)

        p1 = corners[:, 0]
        p_x = corners[:, 4]
        p_y = corners[:, 1]
        # p_z = corners[:, 3]

        i = p_x - p1
        j = p_y - p1
        # k = p_z - p1

        v = points - p1.reshape((-1, 1))

        iv = np.dot(i, v)
        jv = np.dot(j, v)
        # kv = np.dot(k, v)

        mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
        mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
        # mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))
        # mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)
        mask = np.logical_and(mask_x, mask_y)

        return mask

    @staticmethod
    def points_in_boxes(boxes, points, exclude_names: tuple = ()):
        """
        finds all points that lie within one of the bounding boxes
        :param boxes: list of bounding boxes
        :param points: np.array containing the points <np.float: 3, n>
        :param exclude_names: boxes whose name includes one of these terms are excluded
        :return: <np.bool: n, >
        """
        mask = np.zeros(points.shape[1], dtype=bool)
        for box in boxes:
            mask = np.logical_or(mask, NuScenesRadarExtension.__points_in_box(box, points, exclude_names=exclude_names))
        return mask


class BoxExtended(Box):
    """
    Slightly adjusted version of the nuScenes Box class, mainly for plotting purposes
    """

    def __init__(self, box: Box):
        super().__init__(box.center, list(box.wlh), box.orientation, box.label, box.score, box.velocity, box.name,
                         box.token)

    def render(self,
               axis: Axes,
               view: np.ndarray = np.eye(3),
               normalize: bool = False,
               colors: Tuple = ('b', 'r', 'k'),
               linewidth: float = 2,
               label: str = None) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        :param label: Label of the box, so it can be shown in the legend
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth, label=label)
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=colors[2], linewidth=linewidth, label=label)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=colors[0], linewidth=linewidth, label=label)


class Arguments:
    """Class that holds arg_matrix for that can be inputted into a function"""

    def __init__(self, args: list = None, kwargs: dict = None):
        """
        Save arg_matrix list for function in Arguments object
        :param args: list of standard arg_matrix for function
        :param kwargs: **kwargs for function
        """
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __repr__(self):
        return 'Argument(args: {}, kwargs: {})'.format(self.args, self.kwargs)


class ArgumentMatrix:
    """2D matrix of function arg_matrix"""

    def __init__(self, height=1, width=1, universal_kwargs=None):
        """
        matrix of arguments used as function input
        :param height: height of the matrix, number of rows
        :param width: width of the matrix, number of columns
        :param universal_kwargs: sort of global kwargs dictionary entries, will add these to the Arguments kwargs if they
        are not already in there
        """
        self.data = [[Arguments(kwargs=deepcopy(universal_kwargs)) for _ in range(width)] for _ in range(height)]
        self.__universal_kwargs = {} if universal_kwargs is None else universal_kwargs

    def __getitem__(self, index: tuple) -> Arguments:
        return self.data[index[0]][index[1]]

    def __setitem__(self, index: tuple, arg: Arguments):
        self.data[index[0]][index[1]] = arg
        for key in self.__universal_kwargs:
            arg.kwargs[key] = self.__universal_kwargs[key] if key not in arg.kwargs else arg.kwargs[key]

    def __getattr__(self, item):
        if item == 'width':
            return len(self.data[0])
        if item == 'height':
            return len(self.data)

    def __repr__(self):
        return 'ArgumentMatrix (width: {}, height: {}, universal_kwargs: {})'.format(len(self.data[0]), len(self.data),
                                                                                     self.__universal_kwargs)

    def add_univ_kwarg(self, key: str, item):
        """Add a universal kwarg and to the ArgumentMatrix and update all entries with it"""
        self.__universal_kwargs[key] = item
        for row in self.data:
            for arg in row:
                arg.kwargs[key] = item if key not in arg.kwargs else arg.kwargs[key]

    def remove_kwarg(self, key):
        """Remove a kwarg from the universal kwargs dictionary and/or all Arguments this object contains"""
        if key in self.__universal_kwargs:
            self.__universal_kwargs.pop(key)
        for row in self.data:
            for arg in row:
                if key in arg.kwargs:
                    arg.kwargs.pop(key)

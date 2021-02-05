"""Example of uses of code so you don't have to think them up later"""

# General imports
import os.path as osp
import os
import matplotlib.pyplot as plt
import numpy as np
# Nuscenes imports
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
# Own code imports
from nuscenes_radar_extension import NuScenesRadarExtension
from radar_pointcloud import RadarPC, RadarPCCombined
from road_estimation import RoadEstimation


# select disk where data set is stored
if os.environ['COMPUTERNAME'] == 'DESKTOP-BR3NISI':
    disk = 'F:'
else:
    disk = 'D:'

# Base directory where all files have to be stored
base_directory = disk + '/Databases/nuscenes_modified'

# Setup nuScenes basics
nusc = NuScenes(version='v1.0-mini', dataroot=disk+'/Databases/nuscenes_mini', verbose=False)
ext = NuScenesRadarExtension(nusc, modified_base_directory='')
# nusc_map = NuScenesMap(dataroot=disk+'/Databases/nuScenes', map_name='singapore-onenorth')

# Basic nuScenes data
sample_token = 'ed5fc18c31904f96a8f0dbb99ff069c0'
sample = nusc.get('sample', sample_token)
sample_data_token = sample['data']['RADAR_FRONT']
sample_data = nusc.get('sample_data', sample_data_token)
cal_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
scene = nusc.get('scene', sample['scene_token'])
ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
log = nusc.get('log', scene['log_token'])
map_ = nusc.get('map', log['map_token'])

# Aannotation
radius = 1
dynprop_states = [1, 3, 5]
boundary_categories = ('barrier',)
nsweeps = 1
# Sample annotation
sample_ann_out = ext.annotate_sample(sample_token, radius, dynprop_states, boundary_categories, nsweeps, do_logging=True)
# Scene annotation
base_directory = disk + '/Databases/nuscenes_modified'
scene_ann_data = ext.annotate_scene(scene['token'], radius, dynprop_states, boundary_categories, nsweeps, do_logging=True)
print(ext.annotation_table([scene], [scene_ann_data]))

# Load map
filename_map = map_['filename']
# from file
contour_map = plt.imread(filename_map)
# from nuscenes
contour_map = map_['mask'].mask()

# RadarPointCloud
filename_pc = osp.join(nusc.get('sample_data', sample_data_token)['filename'].replace('.pcd', '.las'))
pc = RadarPointCloud.from_file(osp.join(nusc.dataroot, sample_data['filename']))
radar_pc_from_pc = RadarPC.from_radar_point_cloud(pc)
radar_pc_from_sd = RadarPC.from_sample_data(sample_data_token, nusc, True)
radar_pc_from_sd.save(filename_pc)
radar_pc_from_las = RadarPC.from_las(filename_pc)
# Point cloud with settings like in paper Road Boundaries Detection based on Modified Occupancy Grid Map Using
# Millimeter-wave Radar
radar_pc_from_pcd = RadarPC.from_file(osp.join(nusc.dataroot, nusc.get('sample_data', sample_data_token)['filename']),
                                      ambig_states=[3, 4], invalid_states=[0, 4, 11, 16])
# test radar_pc extention
radar_pc_extended = RadarPC.from_sample_data(sample_data_token, nusc, True)
radar_pc_extra = RadarPC.from_sample_data(sample_data['prev'], nusc)
radar_pc_extended.extend_with(radar_pc_extra)

# RadarPCCombined
pc_comb_sample = RadarPCCombined.from_sample(sample_token, nusc)
pc_comb_scene = RadarPCCombined.from_scene(scene['token'], nusc)


# Clustering
params = {'nsweeps': 3,
          'eps': 5/30,
          'min_pnts': 3,
          'delta_phi': np.deg2rad(10),
          'dist_lat': 10.0,
          'r_assign': 1.0,
          'dist_max': 75.0,
          'normalize_points': True}

clustering_labels = RoadEstimation.clustering(radar_pc_from_sd.xyv(normalized=True), params['eps'], params['min_pnts'])


# Evaluate baseline
output_dict = ext.evaluate_baseline_from_scene('bebf5f5b2a674631ab5c88fd1aa9e87a',
                                                                    params, do_logging=True)
confusion_matrix, sample_f1_list = output_dict['conf_matrix'], output_dict['sample_f1']


# Road side estimation
road_side_labels, curve_coeffs = RoadEstimation.apply_baseline(radar_pc_from_sd, params)


# Rendering
ax = ext.render_radar_pc(radar_pc_from_sd)
# render selected fitted lines
ax = ext.render_radar_pc(radar_pc_from_sd, with_fit_lines=True)
# show clusters and fit lines through them
labels = RoadEstimation.clustering(radar_pc_from_sd.xyv(normalized=True), params['eps'], params['min_pnts'])
ext.render_radar_pc(radar_pc_from_sd, with_fit_lines=True, labels=labels, crop_map=True, plot_normal_map=True,
                    long_lines=True)
# Plotting clustering variants
eps_params = (3, 5, 7)
min_pnts_params = (3, 4, 5)
fig, axes = ext.render_cluster_grid(pc_comb_sample, eps_params, min_pnts_params,
                                    with_fit_lines=False, plot_normal_map=True)

# information storing templates
scene_data_template = {'name': '', 'param_key': '000' * 8, 'nsweeps': 0,
                       'eps': 0, 'min_pnts': 0, 'delta_phi': 0, 'dist_lat': 0, 'r_assign': 0, 'dist_max': 0,
                       'recall': 0, 'precision': 0, 'f1-score': 0,
                       'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0,
                       'scene_token': ''}

data_template = {'param_key': [], 'nsweeps': 0,
                 'eps': 0, 'min_pnts': 0, 'delta_phi': 0, 'dist_lat': 0, 'r_assign': 0, 'dist_max': 0,
                 'recall': 0, 'precision': 0, 'f1-score': 0,
                 'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, }

runtime_template = {'param_key': [],
                    'conf_matrix': np.zeros((2, 2)).astype(np.int64)}
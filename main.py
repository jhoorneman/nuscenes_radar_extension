"""
Main modifications of nuScenes data set
"""
# noinspection PyPep8
# General imports
import csv
import os
import os.path as osp
import sys
import time
import warnings
from datetime import date
import logging
from typing import Tuple, List

import numpy as np
import tabulate
from prettytable import prettytable
from sklearn import metrics
# import matplotlib.pyplot as plt
# Nuscenes imports
from nuscenes.nuscenes import NuScenes, RadarPointCloud
# Own code imports
from nuscenes_radar_extension import NuScenesRadarExtension
from radar_pointcloud import RadarPC, RadarPCCombined
from road_estimation import RoadEstimation
from experiment_support import ExperimentSupport
from diagnostics import NuScenesDiagnostic

""""---------STORE DATA FILE LIST COMBINATIONS------------"""


"""---------GENERAL SETTINGS---------"""

# run database from harddisk (has complete data set) or ssd (faster, but has only mini data set)
use_harddisk = True
# Decide which data set to use: mini (only 10 scenes) or trainval (1000 scenes)
use_trainval = True

# Decide which parts to run
convert_to_contour_maps = False
annotate_dataset = False
train_parameters_on_dataset = False
test_parameter_on_dataset = False
combine_experiment_data = False
save_qualitative_plots = False
number_of_experiment_points = True

"""---------ANNOTATION SETTINGS---------"""

# which radar points to include
annotate_filters = 'extended'
# annotate only the scenes in the following subsets. If list is empty, annotate all scenes
annotate_subsets = ['random20', 'barrier30']
# place the annotated radar pointcloud files in a new submap of the main base_directory
new_annotation_submap = True
# annotation parameters
annotation_radii = [0.75, 1, 1.25]
dynprop_states = [1, 3, 5]  # Notable states: 1 stationary, 3 stationary candidate, 5 (crossing stationary)
boundary_categories = ('barrier',)
nsweeps = 5

"""---------EXPERIMENT SETUP BASELINE SETTINGS---------"""
# experiment setup
# data_set_split options:
# 'holdout' for a 50/50 split,
# 'test' for 1 scene in each
# 'random30' for 30 random scenes, or any other number
# 'barrier40' for 40 scenes with the highest barrier count
data_set_split = 'barrier30'
experiment_filter = 'static'
normalize_points = False
# Specify this when you want to use a non-standard annotated data set
experiment_sub_directories = ['r_annotate=1']

list_nsweeps = np.array([1, 3, 5])

"""---------TRAINING PARAMETERS SETTINGS---------"""
# Don't repeat parameter combinations already done in this experiment
extend_from_files = None
# Grid-search parameters
grid_search_params = {
    # Clustering
    # 'list_eps': np.array([0.025, 0.0375, 0.05, 0.0625, 0.075]),
    'list_eps': np.array([0.25, 0.5, 1, 1.5, 2, 2.5, 3]),
    'list_min_pnts': np.arange(2, 5),
    # Boundary cluster selection
    'list_delta_phi': np.deg2rad(np.arange(0, 30, 10) + 10),
    'list_dist_lat': np.arange(4, 10, 2) + 2,
    # Estimation label assignment
    'list_r_assign': np.array([0.5, 1, 2, 3, 4]),
    'list_dist_max': np.array([50, 75, 100, 125]),
    # Number of sweeps
    'list_nsweeps': list_nsweeps}

"""---------TEST PARAMETER SET SETTINGS---------"""
training_data_keys = ['random20-static-False-0.75',
                      'random20-static-False-1',
                      'random20-static-False-1.25',
                      'barrier30-static-False-1']
number_of_param_combs = 6

"""---------SAVE QUALITATIVE EVALUATION PLOTS---------"""
# fill in the keys for experiments as defined in _index.txt
eval_test_keys = ['barrier30-static-False-std',
                  'random20-static-False-1',
                  'barrier30-static-False-1',
                  'random20-default-True-1']
eval_num_param_combs = 1
eval_num_of_each = 3
"""------------Count number of points per experiment-------------"""
count_sub_directory = 'r_annotate=1'
count_experiments = [{'name': 'main-random', 'split': 'random20', 'filter': 'default', 'sweeps': [1, 3, 5]},
                     {'name': 'main-barrier', 'split': 'barrier30', 'filter': 'default', 'sweeps': [1, 3, 5]},
                     {'name': 'static-random', 'split': 'random20', 'filter': 'static', 'sweeps': [1, 3, 5]}]
"""-----------------------------------"""


# Helper functions
def file_name(name, keyword='_v', extension='.log') -> str:
    """Checks if a file with name exists. If it does make the number one higher. Version number should appear after
    the keyword"""
    if not osp.isfile(name):
        return name
    else:
        separated = name.split(keyword)
        last = separated.pop(-1)
        try:
            separated.append(keyword + str(int(last.replace(extension, '')) + 1) + extension)
        except ValueError:
            raise Exception('Version number couldn\'t be increased by one in {}. Filename should end in '
                            '{}#version number#{}'.format(name, keyword, extension))
        return file_name(''.join(separated), keyword, extension)


def initialize_logger(output_dir, name):
    today = date.today().strftime("%y-%m-%d")
    fmt = '%(levelname)s: %(asctime)s.%(msecs)03d: %(message)s'
    datefmt = '%H:%M:%S'

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt, datefmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create error file handler and set level to error
    handler = logging.FileHandler(file_name(osp.join(output_dir, name + "_errors_{}_v0.log".format(today))),
                                  encoding=None, delay=True)
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter(fmt, datefmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create general file handler and set level to info
    handler = logging.FileHandler(file_name(osp.join(output_dir, name + "_general_{}_v0.log".format(today))))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt, datefmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def split_experiment_data(split_type: str, scene_list: list) -> Tuple[list, list]:
    if split_type == 'holdout':
        _train_set = scene_list[::2]
        _test_set = scene_list[1::2]
    elif split_type == 'test':
        _train_set = [scene_list[0]]
        _test_set = [scene_list[1]]
    elif 'random' in split_type:
        split_size = int(split_type[6:])
        train_idx, test_idx = ExperimentSupport.random_train_test_split(split_size, len(scene_list), split_size)
        _train_set = [scene_list[i] for i in train_idx]
        _test_set = [scene_list[i] for i in test_idx]
    elif 'barrier' in split_type:
        split_size = int(split_type[7:])
        with open(osp.join(base_directory, 'data', 'counting_barrier_trafficcone_in_v1.0-trainval.data'), newline='') \
                as csvFile:
            barrier_counts = list(csv.DictReader(csvFile))
        scene_token_list = [data['scene_token'] for data in barrier_counts[-2 * split_size:]]
        scene_list = ExperimentSupport.scene_list_from_tokens(scene_token_list, nusc)
        _train_set = scene_list[::2]
        _test_set = scene_list[1::2]
        del barrier_counts, scene_token_list, scene_list
    else:
        raise NotImplementedError('data split method ' + split_type + ' not implemented yet')

    return _train_set, _test_set


def set_radarpc_filter(filter_type: str):
    if filter_type == 'extended':
        RadarPC.extended_filters()
    elif filter_type == 'default':
        RadarPointCloud.default_filters()
    elif filter_type == 'disable':
        RadarPointCloud.disable_filters()
    elif filter_type == 'static':
        RadarPointCloud.default_filters()
        RadarPointCloud.dynprop_states = [1, 3, 5]
    else:
        raise NotImplementedError(experiment_filter + 'is not a valid RadarPointCloud filter')


start_time = time.time()

# select disk where mini data set is stored
if os.environ['COMPUTERNAME'] == 'DESKTOP-BR3NISI':
    disk_ssd = 'G:'
else:
    disk_ssd = 'F:'
# select disk where mini data set is stored
if os.environ['COMPUTERNAME'] == 'DESKTOP-BR3NISI':
    disk_harddisk = 'F:'
else:
    disk_harddisk = 'E:'

# Select base directory where all modified files will be stored and the mini or main data set
nusc_needed = convert_to_contour_maps or \
              annotate_dataset or \
              train_parameters_on_dataset or \
              test_parameter_on_dataset or \
              save_qualitative_plots or \
              number_of_experiment_points
if use_harddisk:
    base_directory = disk_harddisk + '/Databases/nuscenes_modified'
    if use_trainval and nusc_needed:
        nusc = NuScenes(version='v1.0-trainval', dataroot=disk_harddisk + '/Databases/nuscenes_main', verbose=True)
    elif nusc_needed:
        nusc = NuScenes(version='v1.0-mini', dataroot=disk_harddisk + '/Databases/nuscenes_main', verbose=True)
    else:
        nusc = None
else:
    base_directory = disk_ssd + '/Databases/nuscenes_modified'
    if nusc_needed:
        nusc = NuScenes(version='v1.0-mini', dataroot=disk_ssd + '/Databases/nuscenes_mini', verbose=True)
    else:
        nusc = None

if nusc_needed:
    # Load nuScenes extension class
    ext = NuScenesRadarExtension(nusc, base_directory)
    # Load nuScenes diagnostics class
    diag = NuScenesDiagnostic(nusc, base_directory)
else:
    ext, diag = None, None

# Convert the four binary maps to contour maps
if convert_to_contour_maps:
    map_tokens = ['36092f0b03a857c6a3403e25b4b7aab3', '37819e65e09e5547b8a3ceaefba56bb2',
                  '53992ee3023e5494b90c316c183be829', '93406b464a165eaba6d9de76ca09f5da']
    for map_token in map_tokens:
        ext.save_map_contour(map_token)

# Annotate all scenes in mini data set
if annotate_dataset:
    # Set up logging
    initialize_logger(base_directory + '/logs', 'annotation')
    start_time = time.time()

    # Decide what scenes to annotate
    if len(annotate_subsets) == 0:
        scene_list = nusc.scene
    else:
        scene_list = []
        for subset in annotate_subsets:
            out = split_experiment_data(subset, nusc.scene)
            scene_list += out[0]
            scene_list += out[1]

        # remove duplicate scenes
        scene_names = [s['name'] for s in scene_list]
        _, idx = np.unique(scene_names, return_index=True)
        scene_list = [scene_list[i] for i in idx]

    for i, annotation_radius in enumerate(annotation_radii):
        # Define in what folder the pointclouds have to be saved
        assert not (len(annotation_radii) > 1 and not new_annotation_submap), 'with multiple annotation radii they ' \
                                                                              'have to be stored in submaps'
        if new_annotation_submap:
            annotation_base_directory = base_directory + '/annotation_alternatives/r_annotate={}'\
                .format(annotation_radius)
            if not osp.exists(annotation_base_directory):
                os.mkdir(annotation_base_directory)
            ext.base_directory = annotation_base_directory
            diag.base_directory = annotation_base_directory
        else:
            annotation_base_directory = base_directory

        logging.info('Starting annotation of {} scenes in the {} data set. Will be saved in {}.'
                     ' \n\t\t\t\t Parameters: search radius = {} m, ''dynprop states = {}, boundary annotation categories '
                     '= {}, number of sweeps = {}'
                     .format(len(scene_list), nusc.version, annotation_base_directory,
                             annotation_radius, dynprop_states, boundary_categories, nsweeps))
        # Setup radar pc filter
        set_radarpc_filter(annotate_filters)

        failed_scenes = []
        ann_data = []
        for j, scene in enumerate(scene_list):
            # noinspection PyBroadException
            try:
                current_runtime = time.time() - start_time
                if i==0 and j == 0:
                    remaining_time = 0
                else:
                    remaining_time = current_runtime / (i*len(annotation_radii)+j) * \
                                     (len(annotation_radii)*len(scene_list)-(i*len(annotation_radii)+j))
                logging.info('Scene number {} of {}. Annotation radius: {} of {}. Ran for: {}:{:02d}:{:02d}. '
                             'Estimated remaining time: {}:{:02d}:{:02d}'
                             .format(j + 1, len(scene_list), annotation_radius, annotation_radii,
                                     int(current_runtime / 3600), int((current_runtime % 3600) / 60),
                                     int(current_runtime % 60), int(remaining_time / 3600),
                                     int((remaining_time % 3600) / 60), int(remaining_time % 60)))
                out = ext.annotate_scene(scene['token'], annotation_radius, dynprop_states, boundary_categories, nsweeps,
                                         do_logging=True)
                ann_data.append(out)
            except Exception:
                error = sys.exc_info()[0]
                logging.error('Error occurred while annotating {}. Annotation of this scene will be skipped. Error message:'
                              ' {}'.format(scene['name'], error))
                failed_scenes.append(scene['token'])
                ann_data.append(())

        current_runtime = time.time() - start_time
        logging.info('Annotating done. {} out of {} scenes were annotated. The code ran for {}:{:02d}:{:02d}'
                     .format(len(scene_list) - len(failed_scenes), len(scene_list), int(current_runtime / 3600),
                             int((current_runtime % 3600) / 60), int(current_runtime % 60)) +
                     '\n\nParameters: search radius = {} m, dynprop states = {}, boundary annotation categories = '
                     '{}, number of sweeps = {}'.format(annotation_radius, dynprop_states, boundary_categories, nsweeps) +
                     '\nDyn prop: {}, ambig states: {}, invalid states: {}'.format(list(RadarPointCloud.dynprop_states),
                                                                                   list(RadarPointCloud.ambig_states),
                                                                                   list(RadarPointCloud.invalid_states)) +
                     '\nSummary:' +
                     ext.annotation_table(scene_list, ann_data))
        if len(failed_scenes) > 0:
            logging.warning('{} scene(s) failed annotating. The following scene token(s) weren\'nt fully annotated: \n {}'
                            .format(len(failed_scenes), failed_scenes))


if train_parameters_on_dataset:
    # Set up logging
    initialize_logger(base_directory + '/logs', 'baseline_parameter_training')
    # Warning filters (gets reset to default at the end of this section)
    warnings.filterwarnings('ignore', category=np.RankWarning)

    # Select training and testing set
    train_set, _ = split_experiment_data(data_set_split, nusc.scene)

    # Setup radar pc filter
    set_radarpc_filter(experiment_filter)

    # each sub directory needs to have a separate extend from file
    assert extend_from_files is None or len(extend_from_files) == len(experiment_sub_directories), \
        'the length of extend_from_files and experiment_sub_directories should be the same'

    # Implementation of multiple sub directories is basic at the moment: it just loops the complete experiment code for
    # each sub directory. Thing like the estimated remaining time do not keep other iterations of this loop in mind. It
    # just shows as if it was the only one.
    for i_sub, experiment_sub_directory in enumerate(experiment_sub_directories):

        # Select base directory
        if experiment_sub_directory == '':
            experiment_base_directory = base_directory
        else:
            experiment_base_directory = base_directory + '/annotation_alternatives/' + experiment_sub_directory

        if extend_from_files is None:
            extend_from_file = None
        else:
            extend_from_file = extend_from_files[i_sub]
        if extend_from_file:
            if type(extend_from_file) == list:
                # noinspection PyTypeChecker
                train_data = ExperimentSupport.load_data_combined(extend_from_file, base_directory,
                                                                  goal_param_dict=grid_search_params)
            else:
                # noinspection PyTypeChecker
                train_data = ExperimentSupport.load_data(extend_from_file, base_directory, do_sort=True,
                                                         goal_param_dict=grid_search_params)

        logging.info('Starting evaluation of baseline on data set with the following settings: data set splitting setting: '
                     '{}, normalized data: {}, sweeps: {}, filter setting: {}, ambig states: {}, invalid states: {}, '
                     'dynprop states: {}\nPoint clouds base directory: {}'
                     .format(data_set_split,
                             normalize_points,
                             list_nsweeps,
                             experiment_filter,
                             list(RadarPointCloud.ambig_states),
                             list(RadarPointCloud.invalid_states),
                             list(RadarPointCloud.dynprop_states),
                             experiment_base_directory))
        logging.info('Parameter ranges (length, min, max): eps: {}, min_pnts: {}, lat_dist: {}, delta_phi: {}, '
                     'assign_radius: {}, max_dist: {}'
            .format(
                    (len(grid_search_params['list_eps']), grid_search_params['list_eps'][0], grid_search_params['list_eps'][-1]),
                    (len(grid_search_params['list_min_pnts']), grid_search_params['list_min_pnts'][0],
                        grid_search_params['list_min_pnts'][-1]),
                    (len(grid_search_params['list_dist_lat']), grid_search_params['list_dist_lat'][0],
                        grid_search_params['list_dist_lat'][-1]),
                    (len(grid_search_params['list_delta_phi']), grid_search_params['list_delta_phi'][0],
                        grid_search_params['list_delta_phi'][-1]),
                    (len(grid_search_params['list_r_assign']), grid_search_params['list_r_assign'][0],
                        grid_search_params['list_r_assign'][-1]),
                    (len(grid_search_params['list_dist_max']), grid_search_params['list_dist_max'][0],
                        grid_search_params['list_dist_max'][-1])))
        if extend_from_file:
            logging.info('Experiment will extend the one described in {}'.format(extend_from_file))
            # noinspection PyUnboundLocalVariable
            param_comb_counts = ExperimentSupport.check_param_comb_counts(train_data, grid_search_params, False)
            pt_comb_counts = prettytable.PrettyTable(('type', 'value', 'already calculated #', 'total #', 'ratio [%]'))
            pt_comb_counts.add_rows(row.values() for row in param_comb_counts)
            pt_comb_counts.align = 'r'
            logging.info('Parameter combinations already calculated\n' + str(pt_comb_counts))

        # Estimate runtime
        # noinspection PyUnboundLocalVariable
        runtime = diag.estimate_baseline_evaluation_time(grid_search_params, train_set, repeats=1)

        # information storing templates
        runtime_template = {'param_key': [],
                            'conf_matrix': np.zeros((2, 2)).astype(np.int64)}

        # handle data gathering related tasks
        train_data_list = []
        start_time = time.time()
        nbr_train_samples = 0
        for scene in train_set:
            nbr_train_samples += scene['nbr_samples']
        nbr_train_samples *= len(list_nsweeps)
        try:
            # loop over all samples in training set with different sweep numbers
            for i0, scene in enumerate(train_set):
                # initialize list that holds all results for this scene
                for i1, sample_token in enumerate(ext.sample_tokens_from_scene(scene['token'])):
                    for i2, sweep_num in enumerate(list_nsweeps):
                        # if this parameter has already been computed and all following substeps are already computed,
                        # skip this substep
                        # noinspection PyUnboundLocalVariable
                        if extend_from_file and ExperimentSupport.data_completion_ratio(train_data,
                                                                                        {'nsweeps': sweep_num},
                                                                                        grid_search_params,
                                                                                        do_print=False) >= 1:
                            continue

                        try:
                            radar_pc = RadarPCCombined.from_sample(sample_token, nusc, True, experiment_base_directory, sweep_num)
                            absolute_sample_nbr = (i0 * 40 * len(list_nsweeps) + i1 * len(list_nsweeps) + i2)
                            progress_percent = absolute_sample_nbr / nbr_train_samples * 100
                            # todo: right now this does not take into account that different sweeps can have different
                            #  durations. Especially with extending experiments this is the case. Update this
                            remaining_time = (time.time() - start_time) / absolute_sample_nbr * \
                                             (nbr_train_samples - absolute_sample_nbr) if absolute_sample_nbr != 0 else 0
                            logging.info('Progress: {:5.2f}%. Estimated remaining time: {}:{:02d}:{:02d}. Sweep number {}. '
                                         'Sample {:02d} from scene {:03d} of {:03d}. {}'
                                         .format(progress_percent, int(remaining_time / 3600),
                                                 int((remaining_time % 3600) / 60),
                                                 int(remaining_time % 60), sweep_num, i1 + 1, i0 + 1, len(train_set), radar_pc))
                            # loop over all combinations of clustering parameters eps and min_points
                            for i3, eps in enumerate(grid_search_params['list_eps']):
                                for i4, min_pnts in enumerate(grid_search_params['list_min_pnts']):
                                    # if this parameter has already been computed and all following substeps are already computed,
                                    # skip this substep
                                    # noinspection PyUnboundLocalVariable
                                    if extend_from_file and ExperimentSupport.data_completion_ratio(train_data,
                                                                                                    {'nsweeps': sweep_num,
                                                                                                     'eps': eps,
                                                                                                     'min_pnts': min_pnts},
                                                                                                    grid_search_params,
                                                                                                    do_print=False) >= 1:
                                        continue
                                    # todo: look into enabling mutlithreading for suitably large point clouds
                                    if experiment_filter == 'static':
                                        # when the filter is static only static points are include, so velocity doesn't have
                                        # to be used
                                        cluster_labels = RoadEstimation.clustering(radar_pc.xy(normalize_points), eps, min_pnts)
                                    else:
                                        cluster_labels = RoadEstimation.clustering(radar_pc.xyv(normalize_points), eps, min_pnts)
                                    line_coefficients = RoadEstimation.fit_lines_through_clusters(radar_pc, cluster_labels)

                                    # loop over all combinations of boundary cluster selection parameters
                                    for i5, delta_phi in enumerate(grid_search_params['list_delta_phi']):
                                        for i6, dist_lat in enumerate(grid_search_params['list_dist_lat']):
                                            # if this parameter has already been computed and all following substeps are already computed,
                                            # skip this substep
                                            # noinspection PyUnboundLocalVariable
                                            if extend_from_file and ExperimentSupport.data_completion_ratio(train_data,{'nsweeps': sweep_num,
                                                                                                                        'eps': eps,
                                                                                                                        'min_pnts': min_pnts,
                                                                                                                        'delta_phi': delta_phi,
                                                                                                                        'dist_lat': dist_lat},
                                                                                                            grid_search_params,
                                                                                                            do_print=False) >= 1:
                                                continue
                                            boundary_clusters = RoadEstimation.select_boundary_cluster(radar_pc, cluster_labels,
                                                                                                       delta_phi, dist_lat,
                                                                                                       line_coefficients)
                                            # loop over all combinations of estimation label assignment parameters
                                            for i7, r_assign in enumerate(grid_search_params['list_r_assign']):
                                                for i8, dist_max in enumerate(grid_search_params['list_dist_max']):
                                                    # if this parameter has already been computed and all following substeps are already computed,
                                                    # skip this substep
                                                    # noinspection PyUnboundLocalVariable
                                                    if extend_from_file and ExperimentSupport.data_completion_ratio(
                                                            train_data, {'nsweeps': sweep_num,
                                                                         'eps': eps,
                                                                         'min_pnts': min_pnts,
                                                                         'delta_phi': delta_phi,
                                                                         'dist_lat': dist_lat,
                                                                         'r_assign': r_assign,
                                                                         'dist_max': dist_max},
                                                            grid_search_params,
                                                            do_print=False) >= 1:
                                                        continue
                                                    point_est_labels = RoadEstimation.label_points_with_curves(radar_pc,line_coefficients[boundary_clusters], r_assign, dist_max)
                                                    conf_matrix = metrics.confusion_matrix(radar_pc.labels(), point_est_labels)
                                                    param_key = [i2, i3, i4, i5, i6, i7, i8]
                                                    data_runtime = next(
                                                        (item for item in train_data_list if item['param_key'] == param_key),
                                                        None)
                                                    if data_runtime is not None:
                                                        data_runtime['conf_matrix'] += conf_matrix
                                                    else:
                                                        data_runtime = runtime_template.copy()
                                                        data_runtime['param_key'] = param_key
                                                        data_runtime['conf_matrix'] = conf_matrix
                                                        train_data_list.append(data_runtime)

                        except Exception as e:
                            error = sys.exc_info()[1]
                            logging.error('Error occurred while evaluating sample-{} of {}. Further evaluation of this sample '
                                          'will be skipped. Error message: {}. Info (scene_token, sample_token, sweep_num): {}'
                                          .format(i1, scene['name'], error, (scene['token'], sample_token, sweep_num)))
        except Exception as e:
            error = sys.exc_info()[1]
            logging.error('Error occurred while evaluating baseline, {} parameter combinations were evaluated and will be'
                          ' stored. Error message: {}'.format(len(train_data_list), error))
        finally:
            # Save data to file
            train_data_list_extended = []
            for data_runtime in train_data_list:
                data_extended = ExperimentSupport.get_data_template(data_runtime, grid_search_params)
                train_data_list_extended.append(data_extended)

            today = date.today().strftime("%y-%m-%d")
            if experiment_sub_directory == '':
                r_annotate_key = 'std'
            else:
                r_annotate_key = experiment_sub_directory.split('=')[-1]
            filename = file_name(osp.join(base_directory + '/data',
                                          'experiment-' + data_set_split + '-' + experiment_filter + '-' +
                                          str(normalize_points) + '-' + r_annotate_key +
                                          "-{}_v0.data".format(today)),
                                 extension='.data')

            ExperimentSupport.write_dictlist_to_csv(train_data_list_extended, filename)

            # Prepare summary
            if len(train_data_list_extended) > 0:
                pt_summary = prettytable.PrettyTable(train_data_list_extended[0].keys())
                pt_summary.add_rows(ExperimentSupport.change_data_for_print(row).values() for row in train_data_list_extended)
                pt_summary.align = 'r'
            else:
                pt_summary = '\n'
            # Prepare parameter table
            param_table = ExperimentSupport.param_dict_to_table(grid_search_params)
            pt_params = prettytable.PrettyTable(param_table[0].keys())
            pt_params.add_rows([row.values() for row in param_table])
            # Print summary
            comp_time = time.time() - start_time
            scene_names = [scene['name'] for scene in train_set]
            logging.info('Finished running baseline on {} scenes. It took {}:{}:{:.2f} hours.\n'
                         .format(len(train_set), int(comp_time/3600), int(comp_time%3600/60), comp_time%3600%60)
                         + 'Scenes used: {}\nData stored in: {}\nExtended from: {}\n'.format(scene_names, filename,
                                                                                             extend_from_file)
                         + 'Parameters used:\n' + str(pt_params))

    # reset warning filter
    warnings.filterwarnings('default')

if test_parameter_on_dataset:
    # Set up logging
    initialize_logger(base_directory + '/logs', 'baseline_parameter_testing')
    # Warning filters (gets reset to default at the end of this section)
    warnings.filterwarnings('ignore', category=np.RankWarning)

    for training_data_key in training_data_keys:
        # Set data selection parameters based on filename
        key_split = training_data_key.split('-')
        data_set_split = key_split[0]
        experiment_filter = key_split[1]
        normalize_points = 'True' in key_split[2]
        r_annotate_key = key_split[3]

        # noinspection PyUnboundLocalVariable
        if r_annotate_key == 'std':
            experiment_base_directory = base_directory
        else:
            experiment_base_directory = base_directory + '/annotation_alternatives/r_annotate=' + r_annotate_key

        ext.base_directory = experiment_base_directory

        # Select training and testing set
        _, test_set = split_experiment_data(data_set_split, nusc.scene)
        # Setup radar pc filter
        set_radarpc_filter(experiment_filter)

        logging.info(
            'Starting testing of {} best performing parameter sets on data set with the following settings:\ndata set split'
            ': {}, normalized data: {}, filter setting: {}, ambig states: {}, invalid states: {}, dynprop states: {}\n'
            'Point clouds base directory: {}'
            .format(number_of_param_combs,
                    data_set_split,
                    normalize_points,
                    experiment_filter,
                    list(RadarPointCloud.ambig_states),
                    list(RadarPointCloud.invalid_states),
                    list(RadarPointCloud.dynprop_states),
                    experiment_base_directory))

        # select parameter combinations
        train_data_file = ExperimentSupport.parse_index()[training_data_key]['train']
        data_file = ExperimentSupport.load_data(train_data_file, base_directory, do_sort=True)
        param_set_list = data_file[:number_of_param_combs]

        test_data_list = []
        point_data_total = np.zeros((4, 0))
        start_time = time.time()
        for i0, param_set in enumerate(param_set_list):
            # add parameters not stored in file
            param_set['normalize_points'] = normalize_points
            # if only static points are included in data, don't cluster with velocity
            if experiment_filter == 'static':
                param_set['cluster_v'] = False
            # store parameters in string for plotting
            parameter_string = ''
            for param in ['nsweeps', 'eps', 'min_pnts', 'delta_phi', 'dist_lat', 'r_assign', 'dist_max']:
                parameter_string += '{}: {}, '.format(param, param_set[param])

            point_data = np.zeros((3, 0))
            # noinspection PyUnboundLocalVariable
            for i1, scene in enumerate(test_set):
                try:
                    result_dict = ext.evaluate_baseline_from_scene(scene['token'], param_set, do_logging=False)
                    # process data and put in list
                    scene_data = param_set.copy()

                    # remove information that is not wanted in save file
                    scene_data.pop('normalize_points')
                    if 'cluster_v' in scene_data:
                        scene_data.pop('cluster_v')

                    # add new information
                    scene_data = ExperimentSupport.store_results_in_data_template(scene_data, result_dict['conf_matrix'])
                    scene_data['scene_name'] = scene['name']
                    scene_data['scene_token'] = scene['token']
                    scene_data['sample_f1'] = result_dict['sample_f1']
                    test_data_list.append(scene_data)

                    # append point data
                    point_data = np.hstack((point_data, result_dict['point_data']))

                    # logging
                    logging.info('Parameter set {:02d} of {:02d}. Scene: {:03d} of {:03d}. '
                                 'Mean training F1: {:.04f}. Scene F1: {:.04f}. '
                                 'Name: {}. Token: {}, parameters: {}'
                                 .format(i0 + 1, len(param_set_list), i1 + 1, len(test_set),
                                         param_set['f1-score'], scene_data['f1-score'],
                                         scene['name'], scene['token'], parameter_string))
                except Exception as e:
                    error = sys.exc_info()[1]
                    logging.error('An error occurred while running on {}, token: {}, error message: {}, parameters: {}'
                                  .format(scene['name'], scene['token'], error, parameter_string))
            point_data = np.vstack((np.ones(point_data.shape[1])*i0, point_data))
            point_data_total = np.hstack((point_data_total, point_data))
        # Save files
        fieldnames = ['scene_name',
                      'nsweeps', 'eps', 'min_pnts', 'delta_phi', 'dist_lat', 'r_assign', 'dist_max',
                      'recall', 'precision', 'f1-score', 'TP', 'FP', 'TN', 'FN',
                      'param_key', 'scene_token', 'sample_f1']
        today = date.today().strftime("%y-%m-%d")

        filename_base = 'testing-' + data_set_split + '-' + experiment_filter + '-' + str(normalize_points) + \
                        '-' + r_annotate_key + '-' + str(number_of_param_combs) + "-{}_v0".format(today)
        filename_scene_data = file_name(osp.join(base_directory + '/data', filename_base + ".data"), extension='.data')
        filename_point_data = file_name(osp.join(base_directory + '/data', filename_base + '.npy'), extension='.npy')

        ExperimentSupport.write_dictlist_to_csv(test_data_list, filename_scene_data, fieldnames)
        np.save(filename_point_data, point_data_total)

        # Print summary
        pt = prettytable.PrettyTable()
        for i, row in enumerate(test_data_list):
            test_data_list[i] = ExperimentSupport.change_data_for_print(row)
        for field in fieldnames[:-1]:
            pt.add_column(field, [row[field] for row in test_data_list])
        pt.align = 'r'
        # Print summary
        scene_names = [scene['name'] for scene in test_set]
        logging.info('Finished running baseline on {} scenes. It took {:.2f} seconds.\n'
                     .format(len(test_set), time.time() - start_time)
                     + 'Scenes used: {}\nBased on parameter training stored in: {}\nScene data stored in: {}\n'
                       'Point data stored in: {}\n'
                     .format(scene_names, train_data_file, filename_scene_data, filename_point_data)
                     + str(pt))

    # reset warning filter
    warnings.filterwarnings('default')

if combine_experiment_data:
    # information
    barrier30_filenames = ['experiment-barrier30-default-True-20-12-13_v0.data',
                           'experiment-barrier30-default-True-20-12-26_v0.data',
                           'experiment-barrier30-default-True-21-01-02_v0.data',
                           'experiment-barrier30-default-True-21-01-03_v0.data']
    random20_filenames = ['experiment-random20-default-True-20-12-12_v0.data',
                          'experiment-random20-default-True-20-12-25_v0.data',
                          'experiment-random20-default-True-20-12-25_v1.data',
                          'experiment-random20-default-True-21-01-02_v0.data',
                          'experiment-random20-default-True-21-01-03_v0.data']
    static20_filenames = ['experiment-random20-static-False-21-01-04_v0.data',
                          'experiment-random20-static-False-21-01-04_v1.data']

    random20_params = {
        # Clustering
        'list_eps': (np.arange(0.5, 3, 0.5) + 0.5) / 40,
        'list_min_pnts': np.arange(2, 5),
        # Boundary cluster selection
        'list_delta_phi': np.deg2rad(np.arange(0, 30, 10) + 10),
        'list_dist_lat': np.arange(4, 10, 2) + 2,
        # Estimation label assignment
        'list_r_assign': np.array([0.5, 1, 2, 3, 4]),
        'list_dist_max': np.array([50, 75, 100, 125]),
        # Number of sweeps
        'list_nsweeps': [1, 3, 5]}
    barrier30_params = random20_params
    static20_params = {
        # Clustering
        'list_eps': np.array([0.25, 0.5, 1, 1.5, 2, 2.5, 3]),
        'list_min_pnts': np.arange(2, 5),
        # Boundary cluster selection
        'list_delta_phi': np.deg2rad(np.arange(0, 30, 10) + 10),
        'list_dist_lat': np.arange(4, 10, 2) + 2,
        # Estimation label assignment
        'list_r_assign': np.array([0.5, 1, 2, 3, 4]),
        'list_dist_max': np.array([50, 75, 100, 125]),
        # Number of sweeps
        'list_nsweeps': [1, 3, 5]}

    # combines all grid-search data files into experiment A, B or C files
    new_filenames_base = ('experimentA', 'experimentB', 'experimentC')
    limiting_params = (random20_params, barrier30_params, static20_params)
    for i, filename_list in enumerate((random20_filenames, barrier30_filenames, static20_filenames)):
        train_data = ExperimentSupport.load_data_combined(filename_list, base_directory,
                                                          goal_param_dict=limiting_params[i])
        filename_elements = [new_filenames_base[i]] + filename_list[0].split('-')[1:4]
        filename = '-'.join(filename_elements) + '_v0.data'
        filename = file_name(osp.join(base_directory + '/data', filename), extension='.data')
        ExperimentSupport.write_dictlist_to_csv(train_data, filename)
        print('New data file is stored in {} that combines the data from following files: {}'
              .format(filename, filename_list))


if save_qualitative_plots:
    for test_key in eval_test_keys:
        test_data, = ExperimentSupport.load_from_key(test_key, include_train=False,
                                                     include_test=True, include_point=False)
        normalize_points = 'True' in test_key.split('-')[2]
        diag.save_qualitative_evaluation_plots(test_data, test_key, eval_num_of_each, eval_num_param_combs,
                                               normalize_points)

if number_of_experiment_points:
    count_results = []
    count_base_directory = osp.join(base_directory, 'annotation_alternatives', count_sub_directory)
    for experiment in count_experiments:
        set_radarpc_filter(experiment['filter'])
        train_scenes, test_scenes = split_experiment_data(experiment['split'], nusc.scene)
        for t, scene_list in enumerate((train_scenes, test_scenes)):
            for sweeps in experiment['sweeps']:

                sub_results = {'experiment': experiment['name'], 'type': ['optimise', 'test'][t],
                               '$n_{sweeps}$': sweeps, 'total points': 0, 'boundary points': 0}
                for scene in scene_list:
                    for sample_token in ext.sample_tokens_from_scene(scene['token']):
                        radar_pc = RadarPCCombined.from_sample(sample_token, nusc, True, count_base_directory, sweeps)
                        sub_results['total points'] += radar_pc.nbr_points()
                        sub_results['boundary points'] += sum(radar_pc.labels())
                sub_results['ratio'] = round(sub_results['boundary points']/sub_results['total points'], 3)
                count_results.append(sub_results)
    header = {key: key for key in count_results[0]}
    latex_table = tabulate.tabulate(count_results, header, tablefmt='latex_raw')
    print(latex_table)

# Own code imports
import csv
import logging
import os
from typing import Any, Union, List, Tuple

from nuscenes.utils.data_classes import RadarPointCloud
from prettytable import prettytable
from sklearn.cluster import DBSCAN

from nuscenes_radar_extension import NuScenesRadarExtension, Arguments, ArgumentMatrix
from radar_pointcloud import RadarPC, RadarPCCombined
from road_estimation import RoadEstimation
from experiment_support import ExperimentSupport
# Other imports
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import time as _time
from copy import deepcopy


class NuScenesDiagnostic(NuScenesRadarExtension):
    """Class for small bits of code that finds stuff out about the nuscenes dataset"""

    def __init__(self, nusc, modified_base_directory: str = ''):
        super().__init__(nusc, modified_base_directory)

    # qualitative evaluation of results
    def evaluate_experiment_results(self, sample_and_param_key: tuple, test_data: List[dict],
                                    titles: Tuple[str, str, str, str] = None, normalize_points: bool = None) -> plt.Figure:
        sample_token, param_key = sample_and_param_key
        param_data = next((item for item in test_data if item['param_key'] == param_key))
        if normalize_points is not None:
            param_data['normalize_points'] = normalize_points
        return self.show_baseline_in_progress(param_data, sample_token, one_plot=True, titles=titles)

    def save_qualitative_evaluation_plots(self, test_data: List[dict], name: str,
                                          num_samples_of_each: int = 5, num_param_combs:int = -1,
                                          normalize_points: bool = None):
        plot_directory = osp.join(self.base_directory, 'qualitative_eval', name)
        if not osp.exists(plot_directory):
            os.mkdir(plot_directory)
        out = self.best_average_worst_samples_in_test_data(test_data, num_samples_of_each, num_param_combs)
        plt.ioff()
        for _type in ['best', 'average', 'worst']:
            for i, item in enumerate(out[_type]):
                # get sample information
                scene_token = self.nusc.get('sample', item[0])['scene_token']
                sample_token_list = self.sample_tokens_from_scene(scene_token)
                idx = sample_token_list.index(item[0])
                data_item = next(_item for _item in test_data if _item['param_key'] == item[1]
                                 and _item['scene_token'] == scene_token)
                f1 = data_item['sample_f1'][idx]
                # construct titles
                titles = ('sample {} of {}, nsweeps: {}, F1-score: {:.3f}'.format(idx, data_item['scene_name'],
                                                                                 data_item['nsweeps'], f1),
                          'clustering. eps: {:.3f}, min_pnts: {}'.format(data_item['eps'], data_item['min_pnts']),
                          'boundary selection. delta_phi: {:.3f}, dist_lat: {:.1f}'.format(data_item['delta_phi'],
                                                                                           data_item['dist_lat']),
                          'results. r_assign: {:.2f}, dist_max: {}'.format(data_item['r_assign'], data_item['dist_max'])
                          )
                # get and save figure
                fig = self.evaluate_experiment_results(item, test_data, titles, normalize_points)
                plt.close(fig)
                fig.savefig(osp.join(plot_directory, _type + '_{}_f1_{}.png'.format(i, round(f1, 3))))
                self.nusc.render_sample(item[0], out_path=osp.join(plot_directory, _type + '_{}_f1_{}_sensors.png'
                                                                   .format(i, round(f1, 3))))
                plt.close(plt.gcf())
        print('Results saved in {}'.format(plot_directory))

    # Diagnostics for the code
    def show_baseline_in_progress(self, params: dict = None, sample_token: str = None, one_plot: bool = False,
                                  titles: Tuple[str, str, str, str] = None):
        """
        shows all the substeps of the baseline method on a random sample data
        :param params: dictionary holding all needed parameters (see example below)
        :param sample_token: sample token to show baseline on, if not specified random sample token is chosen
        :param one_plot: plot all progress plots as subplots in one. Only works when eps and min_pnts have one value
        :param titles: titles of each of the four substeps shown. If not specified, standard values will be used
        :return: when one_plot is True it returns the figure. Otherwise it returns the RadarPC
        """

        if params is None:
            params = {'nsweeps': 3,
                      'eps': [3 / 30, 5 / 30],
                      'min_pnts': [2, 3],
                      'delta_phi': np.deg2rad(10),
                      'dist_lat': 10.0,
                      'r_assign': 1.0,
                      'dist_max': 75.0,
                      'normalize_points': True}
            print('The params variable was not specified so a standard set is used: {}'.format(params))

        # if eps and min points are only single values, make them a list. Make a copy to not edit the original version
        params = deepcopy(params)
        for param in ['eps', 'min_pnts']:
            if not (type(params[param]) == np.ndarray or type(params[param]) == list):
                params[param] = [params[param]]

        # place in normalize points in dict, since it is not standard in
        if 'normalize_points' not in params:
            params['normalize_points'] = True
            print('The params dict had no field for normalize_points so the field has been added as True')

        # Select random non-empty radar pc
        nbr_points = 0
        radar_pc = None
        if sample_token is not None:
            radar_pc = RadarPCCombined.from_sample(sample_token, self.nusc, True, self.base_directory, params['nsweeps'])
            nbr_points = radar_pc.nbr_points()
            if nbr_points == 0:
                "Specified sample results in empty radar pointcloud, random sample is used instead"

        attempts = 0
        while nbr_points == 0:
            sample_token = self.rand_sample_token(False)
            radar_pc = RadarPCCombined.from_sample(sample_token, self.nusc, base_directory=self.base_directory)
            nbr_points = radar_pc.nbr_points()
            attempts += 1
            assert attempts < 100, 'while loop looped more than 100 times'

        # Titles of plots
        if titles is None:
            step0 = 'Radar pointcloud from sample ' + radar_pc.parent_token
            step1 = 'Step 1: clustering'
            step2 = 'Step 2: selected boundary clusters with line'
            step3 = 'Step 3: results of boundary estimation'
        else:
            assert len(titles) == 4, 'the titles field should have four values, instead it had {}'.format(len(titles))
            step0, step1, step2, step3 = titles

        # set size of figures (since maximizing doesn't work
        figsize = (20, 11.25)

        # Handle plotting all results in one plot if one_plot = True
        assert one_plot and len(params['eps']) == 1 and len(params['min_pnts']) == 1 or not one_plot, \
            'substeps can only be plotted when there is one eps and one min_pnts value'

        if one_plot:
            master_fig, master_axes = plt.subplots(2, 2, figsize=figsize)
        else:
            master_fig, master_axes = None, None

        # Render radar pc on normal map to see how it looks
        moving_labels = [0, 2, 6, 7]
        is_moving = [p in moving_labels for p in radar_pc.dynprop()]
        labels = radar_pc.labels()
        labels[is_moving] = 2
        self.render_radar_pc(radar_pc,
                             ax=master_axes[0, 0] if master_axes is not None else None,
                             title=step0,
                             labels=labels,
                             plot_legend=True,
                             plot_axes=False,
                             label_names={0: 'non-boundary points', 1: 'boundary points', 2: 'dynamic points'},
                             figsize=figsize)
        if not one_plot:
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.showMaximized()
        print('Showing baseline substeps for sample data: ' + radar_pc.sample_data_token)
        # print('Showing baseline substeps for sample: ' + radar_pc.parent_token)

        # Setup ArgumentMatrix for clustering
        # universal settings
        kwargs = {'plot_axes': False, 'plot_ego_direction': True, 'small_outliers': True, 'with_fit_lines': True,
                  'line_color': 'gray', 'line_alpha': 0.9}
        # initialize argument matrix
        arg_mat = ArgumentMatrix(len(params['min_pnts']), len(params['eps']), kwargs)
        # filling argument matrix
        for i, min_pnts in enumerate(params['min_pnts']):
            for j, eps in enumerate(params['eps']):
                args = arg_mat[i, j]
                args.args.append(radar_pc)
                args.kwargs['labels'] = RoadEstimation.clustering(radar_pc.xyv(normalized=params['normalize_points']), eps, min_pnts)
                if not(len(params['min_pnts']) == 1 and len(params['eps']) == 1):
                    if j == 0:
                        args.kwargs['row_header'] = 'min\npoints: {}'.format(min_pnts)
                    if i == 0:
                        args.kwargs['title'] = 'epsilon: {}'.format(round(eps, 4))

        # Render clustering step
        if one_plot:
            arg_mat[0, 0].kwargs['ax'] = master_axes[0, 1]
            arg_mat[0, 0].kwargs['title'] = step1
            self.render_radar_pc(*arg_mat[0, 0].args, **arg_mat[0, 0].kwargs)
        else:
            self.render_radar_grid(arg_mat, step1, figsize)
            # maximizing doesn't work for some reason
            # fig_manager = plt.get_current_fig_manager()
            # fig_manager.window.showMaximized()

        # Boundary selection step
        arg_mat.add_univ_kwarg('line_length_multiplier', 20)
        arg_mat.add_univ_kwarg('linewidth', 0.7)
        arg_mat.remove_kwarg('line_color')
        arg_mat.add_univ_kwarg('line_alpha', 0.8)
        for i, min_pnts in enumerate(params['min_pnts']):
            for j, eps in enumerate(params['eps']):
                args = arg_mat[i, j]
                params_for_baseline = params.copy()
                params_for_baseline['eps'] = eps
                params_for_baseline['min_pnts'] = min_pnts
                valid_clusters, fit_coeffs = RoadEstimation.apply_baseline(radar_pc, params_for_baseline)
                # make the labels of all non valid (so non boundary) clusters -1
                for k, label in enumerate(args.kwargs['labels']):
                    if label not in valid_clusters:
                        args.kwargs['labels'][k] = -1
                args.kwargs['line_coeffs'] = fit_coeffs
                args.kwargs['fit_through'] = valid_clusters

        # Render chosen boundary clusters and assign radius
        if one_plot:
            arg_mat[0, 0].kwargs['ax'] = master_axes[1, 0]
            arg_mat[0, 0].kwargs['title'] = step2
            self.render_radar_pc(*arg_mat[0, 0].args, **arg_mat[0, 0].kwargs)
        else:
            self.render_radar_grid(arg_mat, step2, figsize)
            # maximizing doesn't work for some reason
            # fig_manager = plt.get_current_fig_manager()
            # fig_manager.window.showMaximized()

        # Label assignment
        arg_mat.remove_kwarg('radius')
        arg_mat.remove_kwarg('with_fit_lines')
        arg_mat.add_univ_kwarg('plot_legend', True)
        arg_mat.add_univ_kwarg('label_names', {-1: 'TN', 0: 'FN', 1: 'TP', 2: 'FP'})

        summaries = []
        for i, min_pnts in enumerate(params['min_pnts']):
            for j, eps in enumerate(params['eps']):
                args = arg_mat[i, j]
                baseline_labels = RoadEstimation.label_points_with_curves(radar_pc, args.kwargs['line_coeffs'],
                                                                          params['r_assign'], params['dist_max'])
                # label -1 for TN, 1 for TP, 0 for FN, 2 for FP. radar.pc.labels() is the ground truth
                for k, label in enumerate(radar_pc.labels()):
                    flag = -1
                    if label == 1 and baseline_labels[k] == 1:
                        # TP
                        flag = 1
                    elif label == 1 and baseline_labels[k] == 0:
                        # FN
                        flag = 0
                    elif label == 0 and baseline_labels[k] == 1:
                        # FP
                        flag = 2
                    args.kwargs['labels'][k] = flag
                # Get information for qualitative summary at the end
                params_for_baseline = params.copy()
                params_for_baseline['eps'] = eps
                params_for_baseline['min_pnts'] = min_pnts
                conf_matrix, _ = RoadEstimation.evaluate_baseline(radar_pc, params_for_baseline)
                summary = {'parameters': (eps, min_pnts, params['r_assign']),
                           'confusion_matrix': conf_matrix}
                summaries.append(summary)

        # Render final result
        if one_plot:
            arg_mat[0, 0].kwargs['ax'] = master_axes[1, 1]
            arg_mat[0, 0].kwargs['title'] = step3
            self.render_radar_pc(*arg_mat[0, 0].args, **arg_mat[0, 0].kwargs)

            # fix overal look of the plot
            master_fig.subplots_adjust(left=.01, bottom=.01, right=.99, top=.95, wspace=.02, hspace=.1)
        else:
            self.render_radar_grid(arg_mat, step3, figsize)

        # maximizing doesn't work for some reason
        # fig_manager = plt.get_current_fig_manager()
        # fig_manager.window.showMaximized()
        # fig_manager.show()
        # Print precision results
        summary_string = 'Summary\nParameters: eps, min_pnts, assign_radius. Precision, Recall, F1 score'
        for summary in summaries:
            params = summary['parameters']
            stats = RoadEstimation.get_precision_recall_f1(summary['confusion_matrix'])
            summary_string += '\nParameters: {:.1f}, {}, {:.1f}. Precision: {:.3f}, Recall: {:.3f}, F1 score: {:.3f}' \
                .format(params[0], params[1], params[2], stats[0], stats[1], stats[2])
        print(summary_string)
        if one_plot:
            return master_fig
        else:
            return radar_pc

    # Tests to find out how stuff works in nuScenes
    def scrap_annotation_log(self, filename, save_to_filename: str = None, relative_path: bool = True):
        """Retrieves information of boundary/non-boundary points per scene from the log file. Very susceptible to
        changes in log file layout"""
        if relative_path:
            filename = osp.join(self.base_directory, 'logs', filename)
            if save_to_filename is not None:
                save_to_filename = osp.join(self.base_directory, 'logs', save_to_filename)
        info = {}
        most_recent_scene = ''
        f = open(filename, "r")
        for line in f.readlines():
            if 'Parameters' in line and 'params' not in info:
                info['params'] = line.replace('\t', '').replace('\n', '')
                if info['params'][0] == ' ':
                    info['params'] = info['params'][1:]
            elif 'data from scene' in line:
                most_recent_scene = line.split()[-1]
                info[most_recent_scene] = []
            elif 'sample' in line:
                info[most_recent_scene].append([0, 0])
            elif 'Annotated RADAR' in line:
                data = line.split()[-3].split('/')
                info[most_recent_scene][-1][0] += int(data[0])
                info[most_recent_scene][-1][1] += int(data[1])
        f.close()
        matrix = []
        # convert information from dictionary to matrix (list in list) so it can get printed
        total = {'boundary_pnts': 0, 'total_pnts': 0, 'max_ratio': 0, 'max_pnts': 0, 'min_ratio': 1, 'min_pnts': 0}
        for key in info:
            boundary_pnts = 0  # number of boundary points in scene
            total_pnts = 0  # total number of points in scene
            max_ratio = 0  # maximum annotation ratio in a single sample
            max_pnts = 0  # total number of points for that sample
            min_ratio = 1  # minimum annotation ratio in a single sample
            min_pnts = 0  # total number of points for that sample
            if key == 'params':
                continue
            else:
                sample: list
                for sample in info[key]:  # each sample is a list of [#boundary points, total # points]
                    if sample[1] == 0:  # if sample has no points, continue to next sample
                        continue
                    else:
                        boundary_pnts += sample[0]
                        total_pnts += sample[1]
                        ratio = sample[0] / sample[1]
                        if ratio > max_ratio:
                            max_ratio = ratio
                            max_pnts = sample[1]
                        if ratio < min_ratio:
                            min_ratio = ratio
                            min_pnts = sample[1]

                # process info for complete data
                if total_pnts == 0:
                    total_ratio = 0
                else:
                    total_ratio = boundary_pnts / total_pnts
                    total['boundary_pnts'] += boundary_pnts
                    total['total_pnts'] += total_pnts
                    if total_ratio > total['max_ratio']:
                        total['max_ratio'] = total_ratio
                        total['max_pnts'] = total_pnts
                    if total_ratio < total['min_ratio']:
                        total['min_ratio'] = total_ratio
                        total['min_pnts'] = total_pnts
                # edit data to print well
                scene_info = [key,
                              boundary_pnts,
                              total_pnts,
                              str(round(total_ratio * 100, 1)) + '%',
                              '{}% ({})'.format(round(max_ratio * 100, 1), max_pnts),
                              '{}% ({})'.format(round(min_ratio * 100, 1), min_pnts)]

                matrix.append(scene_info)
        # edit info for summarizing line
        if total['total_pnts'] == 0:
            total_ratio = 0
        else:
            total_ratio = total['boundary_pnts'] / total['total_pnts']
        matrix.append(['total',
                       '{}k'.format(round(total['boundary_pnts'] / 1000, 1)),
                       '{}k'.format(round(total['total_pnts'] / 1000, 1)),
                       str(round(total_ratio * 100, 1)) + '%',
                       '{}% ({}k)'.format(round(total['max_ratio'] * 100, 1), round(total['max_pnts'] / 1000, 1)),
                       '{}% ({}k)'.format(round(total['min_ratio'] * 100, 1), round(total['min_pnts'] / 1000, 1))])

        output = info['params'] + '\n' + self.__matrix_to_string(matrix, ['name',
                                                                          'boundary',
                                                                          'total',
                                                                          'ann%',
                                                                          'max% (+#pnts)',
                                                                          'min% (-#pnts)'])
        print(output)
        if save_to_filename is not None:
            f = open(save_to_filename, 'w')
            f.write(output)
            f.close()
        return info

    def point_num_sample(self, sample_token: str, do_print: bool = True) -> int:
        """find number of radar points in a sample"""
        radar_pc = RadarPCCombined.from_sample(sample_token, self.nusc, False)
        if do_print:
            print('Sample: {}. Points: {}'.format(sample_token, radar_pc.nbr_points()))
        return radar_pc.nbr_points()

    def point_num_scene(self, scene_token: str, print_samples: bool = True):
        """returns number of radar points in a scene"""
        scene = self.nusc.get('scene', scene_token)
        sample_token = scene['first_sample_token']
        points = 0
        print('Scene {}, ambig states {}, invalid states {}, dynprop {}'
              .format(scene_token, RadarPointCloud.ambig_states,
                      RadarPointCloud.invalid_states, RadarPointCloud.dynprop_states))
        while sample_token != '':
            points += self.point_num_sample(sample_token, print_samples)
            sample_token = self.nusc.get('sample', sample_token)['next']
        print('Points total: {}. Average: {:.1f} per sample'.format(points, points / scene['nbr_samples']))

    def ego_pos_in_sample(self, sample_token: str):
        """finds the timestamp, and ego position for all sample data's in a sample"""
        sample = self.nusc.get('sample', sample_token)
        matrix = []
        sample_time = sample['timestamp']
        for i, chan in enumerate(sample['data']):
            sd = self.nusc.get('sample_data', sample['data'][chan])
            ep = self.nusc.get('ego_pose', sd['ego_pose_token'])
            matrix.append([chan, (sd['timestamp'] - sample_time) / 1e6, np.round(ep['translation'], 3)])
        print(self.__matrix_to_string(matrix, ['channel', 'time [s]', 'ego location [m]']))
        return matrix

    def num_static_dynamic_sample(self, sample_token: str = None, nsweeps: int = 1):
        """prints number of static and dynamic points in a sample"""
        if sample_token is None:
            sample_token = self.rand_sample_token()
        radar_pc = RadarPCCombined.from_sample(sample_token, self.nusc,
                                               from_las=False, base_directory=self.base_directory, nsweeps=nsweeps)
        self.num_static_dynamic_radarpc(radar_pc)

    def num_static_dynamic_sd(self, sd_token: str = None, nsweeps: int = 1):
        """prints number of static and dynamic points in a sample data"""
        if sd_token is None:
            sd_token = self.rand_sd_token(True)
        radar_pc = RadarPC.from_sample_data(sd_token, self.nusc, nsweeps)
        self.num_static_dynamic_radarpc(radar_pc)

    def count_anns_in_sample(self, sample_token: str, categories: tuple, do_print: bool = True):
        """Counts for each word in categories how often a category name of each annotation is containing it"""
        sample = self.nusc.get('sample', sample_token)
        assert type(categories) is tuple, \
            'categories variable should be a tuple, right now it is a {}'.format(type(categories))
        anns = sample['anns']
        counter = [0] * len(categories)
        for ann in anns:
            sample_ann = self.nusc.get('sample_annotation', ann)
            for i, category in enumerate(categories):
                if category in sample_ann['category_name']:
                    counter[i] += 1
        if do_print:
            print('Sample {}. Categories: {}. Occurrence: {}'.format(sample_token, categories, counter))
        return counter

    def count_anns_in_scene(self, scene_token: str, categories: tuple, do_print: bool = True) -> dict:
        """Counts for each word in categories how often a category name of each annotation is containing it"""
        scene = self.nusc.get('scene', scene_token)
        out = {'name': scene['name'], 'scene_token': scene_token}
        sample_token = scene['first_sample_token']
        counter = [0] * len(categories)
        while sample_token != '':
            sample_counter = self.count_anns_in_sample(sample_token, categories, False)
            for i in range(len(counter)):
                counter[i] += sample_counter[i]
            sample_token = self.nusc.get('sample', sample_token)['next']
        for i, category in enumerate(categories):
            out[category] = counter[i]
        if do_print:
            print(out)

        return out

    def count_anns_in_dataset(self, scene_list: list, categories: tuple, do_print: bool = True) -> List[dict]:
        """Counts how much annotations of a certain list of categories are in the dataset. Sort the output list
        on the number of the first category specified from low to high"""
        count_list = []
        for scene in scene_list:
            count_list.append(self.count_anns_in_scene(scene['token'], categories, False))

        # sort the count_list
        count_list = sorted(count_list, key=lambda k: k[categories[0]])

        if do_print:
            pt = prettytable.PrettyTable(count_list[0].keys())
            pt.add_rows(row.values() for row in count_list)
            print(pt)

        # save data to csv file
        filename = 'counting'
        if len(categories) < 5:
            for category in categories:
                filename += '_' + category
        else:
            filename += str(len(categories)) + '_annotations'
        filename += '_in_' + self.nusc.version
        filename = osp.join(self.base_directory, 'data', filename)
        ExperimentSupport.write_dictlist_to_csv(count_list, filename)

        return count_list

    ####################
    # TIMING FUNCTIONS #
    ####################

    def time_in_between_sweeps(self, scene_token: str = None, modality: str = 'RADAR_FRONT') -> list:
        """Returns and print the time in between all sweeps for a scene"""
        if scene_token is None:
            scene_token = self.rand_scene_token(False)
        scene = self.nusc.get('scene', scene_token)
        sd = self.sd_from_scene(scene_token, modality, False)
        last_time = sd['timestamp'] / 1e6
        time_diffs = []
        _next = sd['next']
        while _next != '':
            sd = self.nusc.get('sample_data', _next)
            time_diffs.append(sd['timestamp'] / 1e6 - last_time)
            last_time = sd['timestamp'] / 1e6
            _next = sd['next']
        print('Timestamps in {} for {}:\n'.format(scene['name'], modality),
              'A total of {} frames of which {} were annotated\n'.format(len(time_diffs) + 1, scene['nbr_samples']),
              'Average, min and max difference [s]: {:.4f}, {:.4f}, {:.4f}\n'
              .format(np.mean(time_diffs), np.min(time_diffs), np.max(time_diffs)),
              'Frequency [Hz]: {:.2f}'.format(1 / np.mean(time_diffs)))
        return time_diffs

    def time_dbscan(self, radar_pc=None, eps: float = 3, min_pnts: int = 3, repeats=10, dbargs: dict = None):
        """Timing dbscan algorithms"""
        if radar_pc is None:
            radar_pc = RadarPCCombined.from_sample(self.rand_sample_token(), self.nusc,
                                                   base_directory=self.base_directory)

        time_list = [_time.time()]
        for _ in range(repeats):
            DBSCAN(eps=eps, min_samples=min_pnts, **dbargs).fit_predict(radar_pc.xyz().transpose())
            time_list.append(_time.time())
        time_list = np.diff(time_list)
        print('Running clustering {} times, with settings: {} on radar pc: {}'.format(repeats, dbargs, str(radar_pc)))
        print('Average: {:.3f} ms, min: {:.3f} ms, max: {:.3f} ms'
              .format(np.mean(time_list) * 1000, np.min(time_list) * 1000, np.max(time_list) * 1000))

    @staticmethod
    def time_function(func, args: tuple, repeats: int = 100, do_print: bool = True) -> float:
        """Times execution of certain function func. Returns the average time in mili seconds"""
        time_list = [_time.time()]
        for _ in range(repeats):
            func(*args)
            time_list.append(_time.time())
        time_list = np.diff(time_list)
        if do_print:
            print('Timing function {} with {} repetitions. Function arguments: {}'
                  .format(func.__qualname__, repeats, args))
            print('Average: {:.3f} ms, min: {:.3f} ms, max: {:.3f} ms'
                  .format(np.mean(time_list) * 1000, np.min(time_list) * 1000, np.max(time_list) * 1000))
        return float(np.mean(time_list) * 1000)

    def estimate_baseline_evaluation_time(self, params: dict, train_set: list, repeats: int = 100) -> float:
        """Estimate the runtime of baseline evaluation with grid search based on performing each substep on a number of
         random samples. Return the total runtime in seconds"""
        nbr_estimation_samples = 5
        nbr_training_samples = 0
        for scene in train_set:
            nbr_training_samples += scene['nbr_samples']
        # initialize sub processes
        sub_processes = [{'name': 'radar pc loading', 'exec_nbrs': 0, 'avg_time [ms]': 0, 'total_time [s]': 0},
                         {'name': 'clustering', 'exec_nbrs': 0, 'avg_time [ms]': 0, 'total_time [s]': 0},
                         {'name': 'cluster line fitting', 'exec_nbrs': 0, 'avg_time [ms]': 0, 'total_time [s]': 0},
                         {'name': 'boundary cluster selection', 'exec_nbrs': 0, 'avg_time [ms]': 0,
                          'total_time [s]': 0},
                         {'name': 'estimation label assignment', 'exec_nbrs': 0, 'avg_time [ms]': 0,
                          'total_time [s]': 0}]

        # set number of executions
        sub_processes[0]['exec_nbrs'] = nbr_training_samples * len(params['list_nsweeps'])
        sub_processes[1]['exec_nbrs'] = sub_processes[0]['exec_nbrs'] * len(params['list_eps']) * len(
            params['list_min_pnts'])
        sub_processes[2]['exec_nbrs'] = sub_processes[1]['exec_nbrs']
        sub_processes[3]['exec_nbrs'] = sub_processes[2]['exec_nbrs'] * len(params['list_delta_phi']) * len(
            params['list_dist_lat'])
        sub_processes[4]['exec_nbrs'] = sub_processes[3]['exec_nbrs'] * len(params['list_r_assign']) * len(
            params['list_dist_max'])

        # estimate computation time for each sub process
        for i in range(nbr_estimation_samples):
            sample_token = self.rand_sample_token(False)
            # loading data
            nsweeps = np.round(np.mean(params['list_nsweeps']))
            # loading
            func_params = (sample_token, self.nusc, True, self.base_directory, nsweeps)
            radar_pc = RadarPCCombined.from_sample(*func_params)
            time = self.time_function(RadarPCCombined.from_sample, func_params, repeats=repeats, do_print=False)
            sub_processes[0]['avg_time [ms]'] += time / nbr_estimation_samples

            # clustering
            func_params = (radar_pc.xyv(True), params['list_eps'][-1], params['list_min_pnts'][-1])
            cluster_labels = RoadEstimation.clustering(*func_params)
            time = self.time_function(RoadEstimation.clustering, func_params, repeats=repeats, do_print=False)
            sub_processes[1]['avg_time [ms]'] += time / nbr_estimation_samples

            # cluster line fitting
            func_params = (radar_pc, cluster_labels)
            line_coeffs = RoadEstimation.fit_lines_through_clusters(*func_params)
            time = self.time_function(RoadEstimation.fit_lines_through_clusters, func_params, repeats=repeats, do_print=False)
            sub_processes[2]['avg_time [ms]'] += time / nbr_estimation_samples

            # boundary cluster selection
            func_params = (
            radar_pc, cluster_labels, params['list_delta_phi'][-1], params['list_dist_lat'][-1], line_coeffs)
            boundary_clusters = RoadEstimation.select_boundary_cluster(*func_params)
            time = self.time_function(RoadEstimation.select_boundary_cluster, func_params, repeats=repeats, do_print=False)
            sub_processes[3]['avg_time [ms]'] += time / nbr_estimation_samples

            # estimation label assignment
            func_params = (
            radar_pc, line_coeffs[boundary_clusters], params['list_r_assign'][-1], params['list_dist_max'][-1])
            RoadEstimation.label_points_with_curves(*func_params)
            time = self.time_function(RoadEstimation.label_points_with_curves, func_params, repeats=repeats, do_print=False)
            sub_processes[4]['avg_time [ms]'] += time / nbr_estimation_samples

        # calculate total times
        total_time = 0
        for sub_process in sub_processes:
            sub_process['total_time [s]'] = (sub_process['avg_time [ms]'] * sub_process['exec_nbrs']) / 1000
            total_time += sub_process['total_time [s]']
            for field in ['avg_time [ms]', 'total_time [s]']:
                sub_process[field] = round(sub_process[field], 3)

        # print table with substeps
        pt = prettytable.PrettyTable(sub_processes[0].keys())
        pt.add_rows(row.values() for row in sub_processes)
        logging.info('Number of scenes: {}. Estimated runtime: {:.2f} s or {:.3f} hr. Warning: inaccurate estimation\n'
                     .format(len(train_set), total_time, total_time / 3600) + str(pt))

        return total_time / 1000

    ##############
    # VALIDATION #
    ##############

    def compare_sample_data_tokens(self):
        """
        I found a situation where the data from sample_data token '62666e6073244d61ad97394b41c4057a' was stored in the
        las file with the token '95397af858e34021893bab6ecdfa073f'. This code checks if there are more mismatches like
        that
        """
        num_files = 0
        num_mismatches = 0
        update_points = self.nusc.sample_data[::round(len(self.nusc.sample_data) / 10)][1:]
        for sample_data in self.nusc.sample_data:
            if len(update_points) > 0 and sample_data == update_points[0]:
                print('Status update: processed {} files, found {} mismatches'.format(num_files, num_mismatches))
                update_points = update_points[1:]
            if sample_data['sensor_modality'] != 'radar':
                continue

            filename = osp.join(self.base_directory, sample_data['filename'].replace('.pcd', '.las'))
            radar_pc = RadarPC.from_las(filename, do_logging=False)

            if radar_pc.nbr_points() > 0:
                num_files += 1
            if radar_pc.nbr_points() > 0 and radar_pc.sample_data_token != sample_data['token']:
                num_mismatches += 1
                tokens = self.tokens_from_sample_data(sample_data['token'])
                print('Sd tokens for {} not equal, token in file: {}, channel: {:17s}, sample: {}, scene: {}, '
                      'filename: {}'
                      .format(sample_data['token'], radar_pc.sample_data_token, sample_data['channel'],
                              tokens[0], tokens[1], filename))
        print('Done searching. Found {} mismatches in {} files'.format(num_mismatches, num_files))

    def best_average_worst_samples_in_test_data(self, test_data: List[dict], number_of_each: int = 3,
                                                num_param_combs: int = -1) -> dict:
        """Return the tokens of a the best, average and worst performing samples from a test"""
        param_keys = []
        for row in test_data:
            if not row['param_key'] in param_keys:
                param_keys.append(row['param_key'])

        test_data_subset = []
        for i, param_key in enumerate(param_keys):
            # only include the specified number of param comps, break loop when it is reached
            if num_param_combs != -1 and i == num_param_combs:
                break
            test_data_subset += [row for row in test_data if row['param_key'] == param_key]

        sample_token_list = []
        sample_f1_list = []
        param_token_list = []
        for i, scene_test in enumerate(test_data_subset):
            sample_token_list += self.sample_tokens_from_scene(scene_test['scene_token'])
            sample_f1_list += scene_test['sample_f1']
            param_token_list += [scene_test['param_key']]*len(scene_test['sample_f1'])
        sorted_idx = np.argsort(sample_f1_list)
        sample_token_list = np.array(sample_token_list)[sorted_idx]
        param_token_list = np.array(param_token_list)[sorted_idx]
        mean_idx = int(len(sample_token_list) / 2)
        best_mean_worst = tuple(np.array(sample_f1_list)[sorted_idx][[-1, mean_idx, 0]])
        print('Best sample: {:.04f}, average sample: {:.04f}, worst sample: {:.04f}'.format(*best_mean_worst))
        out = {'best': [(sample_token_list[i], list(param_token_list[i]))
                        for i in range(len(sample_token_list) - number_of_each, len(sample_token_list))[::-1]],
               'average': [(sample_token_list[i], list(param_token_list[i])) for i in
                           range(mean_idx, mean_idx + number_of_each)],
               'worst': [(sample_token_list[i], list(param_token_list[i])) for i in range(number_of_each)]}
        return out

    @staticmethod
    def num_static_dynamic_radarpc(radar_pc: RadarPC):
        """
        Checks the dynprop field (the fourth field of the points table) to determine which points are static and dynamic
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
        dynprop = radar_pc.points[3, :]
        static = np.sum(np.logical_or(dynprop == 1, dynprop == 3, dynprop == 7))
        unknown = np.sum(dynprop == 4)
        dynamic = len(dynprop) - static - unknown
        print('Radar pc {} \t Static: {} \t Dynamic {} \t Unknown {}'
              .format(radar_pc.sample_data_token, static, dynamic, unknown))

    def render_dynprop(self, radar_pc: RadarPC = None, nsweeps: int = 1):
        """Render a radar pc with the dynprop values as its label"""
        if radar_pc is None:
            radar_pc = RadarPCCombined.from_sample(self.rand_sample_token(False), self.nusc,
                                                   from_las=False, nsweeps=nsweeps)
        dynprop_dict = {0: 'moving',
                        1: 'stationary',
                        2: 'oncoming',
                        3: 'stationary candidate',
                        4: 'unknown',
                        5: 'crossing stationary',
                        6: 'crossing moving',
                        7: 'stopped'}
        self.render_radar_pc(radar_pc, labels=radar_pc.dynprop(), plot_legend=True, label_names=dynprop_dict)

    # Additional nusc get methods
    def sd_from_sample(self, sample_token: str, modality: str = 'RADAR_FRONT', do_print: bool = True):
        """For testing purposes, gives the sample data of radar front"""
        if do_print:
            print('Sample data of sensor:', modality, ' from sample: ' + sample_token)
        return self.nusc.get('sample_data', self.nusc.get('sample', sample_token)['data'][modality])

    def sd_from_scene(self, scene_token: str, modality: str = 'RADAR_FRONT', do_print: bool = True):
        """For testing purposes, gives the sample data of radar front of the first sample"""
        if do_print:
            print('Sample data of sensor:', modality, ' from scene: ' + scene_token)
        return self.sd_from_sample(self.nusc.get('scene', scene_token)['first_sample_token'], modality, do_print=False)

    def rand_sample_token(self, do_print: bool = True) -> str:
        """returns a random sample token"""
        sample_token = self.nusc.sample[np.random.randint(len(self.nusc.sample))]['token']
        if do_print:
            print('Generated random sample token: ' + sample_token)
        return sample_token

    def rand_scene_token(self, do_print: bool = True) -> str:
        """returns a random scene token"""
        scene_token = self.nusc.scene[np.random.randint(len(self.nusc.scene))]['token']
        if do_print:
            print('Generated random scene token: ' + scene_token)
        return scene_token

    def rand_sd_token(self, radar_token: bool = True, do_print: bool = True):
        """returns a random sample data token, it is of a radar sensor if radar_token is True"""
        if radar_token:
            sample = self.nusc.sample[np.random.randint(len(self.nusc.sample))]
            radar_channels = []
            for chan in sample['data']:
                if 'RADAR' in chan:
                    radar_channels.append(chan)
            sd_token = sample['data'][radar_channels[np.random.randint(len(radar_channels))]]
            if do_print:
                print('Generated random radar sample data token: ' + sd_token)
        else:
            sd_token = self.nusc.sample_data[np.random.randint(len(self.nusc.sample_data))]['token']
            if do_print:
                print('Generated random sample data token: ' + sd_token)

        return sd_token

    def las_name_from_sd(self, sd_token: str):
        """returns las filename from sample data"""
        sd = self.nusc.get('sample_data', sd_token)
        return sd['filename'].replace('.pcd', '.las')

    def rand_radar_pc(self) -> RadarPC:
        """returns a random RadarPC object"""
        attempts = 0
        radar_pc = None
        nbr_points = 0
        while nbr_points == 0:
            sample_token = self.rand_sample_token(False)
            radar_pc = RadarPCCombined.from_sample(sample_token, self.nusc, base_directory=self.base_directory)
            nbr_points = radar_pc.nbr_points()
            attempts += 1
        return radar_pc

    # Helper function
    @staticmethod
    def __matrix_to_string(matrix, header=None):
        """returns a string of a list in a list as a nice looking matrix"""

        def bound(text: str, length: int) -> str:
            return text.ljust(length)[:length]

        if header is not None:
            matrix.insert(0, header)

        csize = [0] * len(matrix[0])
        # get max size for each column
        for line in matrix:
            for i, col in enumerate(line):
                if len(str(col)) > csize[i]:
                    csize[i] = len(str(col))

        out_string = ''
        for line in matrix:
            if line != matrix[0]:
                out_string += '\n'
            for i, col in enumerate(line):
                out_string += bound(str(col), csize[i]) + ' '
        return out_string

import csv
import os
import warnings
from copy import deepcopy, copy
from os import path as osp
from typing import List, Tuple, Dict, Union

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nuscenes import NuScenes
from prettytable import prettytable
from sklearn.metrics import precision_recall_curve
from tabulate import tabulate
from sklearn import metrics

from road_estimation import RoadEstimation


class ExperimentSupport:
    # Hard coded, can differ depending on computer
    data_base_directory = 'E:\\Databases\\nuscenes_modified\\data'
    # Class values
    hyper_params = ['eps', 'min_pnts', 'delta_phi', 'dist_lat', 'r_assign', 'dist_max']
    hyper_params_str = ['$\epsilon$', '$n_{min}$', '$\Delta\phi_{thres}$',
                        '$d_{lat, thresh}$', '$r_{assign}$', '$d_{max}$']
    hyper_params_plus = hyper_params + ['nsweeps']
    hyper_params_plus_str = hyper_params_str + ['$n_{sweeps}$']
    report_headers = hyper_params_plus + ['train f1', 'test f1', 'combination number']
    report_headers_str = hyper_params_plus_str + ['optimisation $F_1$-score', 'test $F_1$-score', '\#']
    metrics_str = ['$F_1$-$score$', '$recall$', '$precision$']
    param_dict_keys = ['list_eps', 'list_min_pnts', 'list_delta_phi', 'list_dist_lat', 'list_r_assign',
                       'list_dist_max', 'list_nsweeps']
    # in the order used in param_keys
    hyper_params_plus_sweepfirst = ['nsweeps'] + hyper_params
    param_dict_keys_sweepfirst = [param_dict_keys[-1]] + param_dict_keys[:-1]

    @classmethod
    def check_data_base_directory(cls) -> bool:
        """Check whether the hardcoded base directory holds data, by checking if it has a _index.txt file"""
        return osp.exists(osp.join(cls.data_base_directory, '_index.txt'))

    @classmethod
    def assert_base_directory(cls):
        """asserts base_directory exists"""
        assert cls.check_data_base_directory(), \
            'base directory has no file _index.txt. Path: ' + cls.data_base_directory

    @classmethod
    def validate_index(cls):
        """Validates that all words in experiments keys and filenames line up"""
        cls.assert_base_directory()
        index = cls.parse_index()
        everything_is_correct = True
        for key in index:
            exp = index[key]
            if '-std' in key:
                _key = key.replace('-std', '')
            else:
                _key = key
            for field in ['train', 'test', 'point']:
                if exp[field] != '-' and _key not in exp[field]:
                    everything_is_correct = False
                    print('Filename of {} from experiment {} does not line up. Filename: {}'
                          .format(field, key, exp[field]))

        if everything_is_correct:
            print('Index file is validated. All filenames match up with experiment keys.')
        else:
            raise AssertionError('Some experiment keys do not match up with the filenames. '
                                 'See prints above this message.')

    @staticmethod
    def write_dictlist_to_csv(dict_list: list, filename: str, fieldnames: list = None):
        if len(dict_list) == 0:
            with open(filename, 'w', newline=''):
                pass
            return

        if fieldnames is None:
            fieldnames = [key for key in dict_list[0]]
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dict_list)

    @staticmethod
    def change_data_for_print(data_dict: dict) -> dict:
        data_dict = data_dict.copy()
        # Round the float fields to
        for field in ['eps', 'delta_phi', 'recall', 'precision', 'f1-score', 'train f1', 'test f1']:
            if field in data_dict:
                data_dict[field] = round(data_dict[field], 4)
        # Special cases
        if 'sample_f1' in data_dict:
            data_dict['sample_f1'] = [round(item, 3) for item in data_dict['sample_f1']]
        return data_dict

    @staticmethod
    def get_data_template(runtime_template: dict, params: dict) -> dict:
        """"converts a runtime information template to a more extensive data template that can be stored to file"""
        data_template = {'param_key': runtime_template['param_key'], 'nsweeps': 0, 'eps': 0, 'min_pnts': 0,
                         'delta_phi': 0, 'dist_lat': 0, 'r_assign': 0, 'dist_max': 0, 'recall': 0, 'precision': 0,
                         'f1-score': 0, 'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

        data_template = ExperimentSupport.store_results_in_data_template(data_template, runtime_template['conf_matrix'])
        for i, param_name in enumerate(['nsweeps'] + ExperimentSupport.hyper_params):
            data_template[param_name] = params['list_' + param_name][data_template['param_key'][i]]
        return data_template

    @staticmethod
    def param_dict_duplicates(param_dict1: dict, param_dict2: dict) -> dict:
        param_dict_out = {}
        for key in ExperimentSupport.param_dict_keys:
            param_list = []
            for param in param_dict1[key]:
                if param in param_dict2[key]:
                    param_list.append(param)
            param_dict_out[key] = np.array(param_list)
        return param_dict_out

    @staticmethod
    def param_dict_to_table(param_dict: dict, round_to: int = 4) -> List[dict]:
        max_param_len = max([len(params[1]) for params in param_dict.items()])
        param_table = []
        for i in range(max_param_len):
            param_row = {}
            for j, key in enumerate(ExperimentSupport.hyper_params_plus):
                if i < len(param_dict[ExperimentSupport.param_dict_keys[j]]):
                    value = param_dict[ExperimentSupport.param_dict_keys[j]][i]
                    if round_to >= 0:
                        param_row[key] = round(value, round_to)
                    else:
                        param_row[key] = value
                else:
                    param_row[key] = ''
            param_table.append(param_row)
        return param_table

    @classmethod
    def table_to_latex(cls, table: List[dict], headers: dict = None, do_print: bool = True) -> str:

        if headers is not None:
            # When headers are specified: change order of fields so it matches the header
            new_table = []
            for row in table:
                new_table.append({key: row[key] for key in headers})
            table = new_table
        else:
            headers = {param: param for param in ExperimentSupport.hyper_params_plus}

        # Change headers to their string representation
        for i, param in enumerate(ExperimentSupport.report_headers):
            if param in headers:
                headers[param] = cls.report_headers_str[i]
        table_begin = '\\begin{table}[h]\n\\caption{}\n\\label{}\n'
        table_end = '\n\\end{table}'
        latex_table = tabulate(table, headers=headers, tablefmt='latex_raw', numalign='left')
        latex_table = table_begin + latex_table + table_end
        if do_print:
            print(latex_table)

        return latex_table

    @classmethod
    def update_param_keys(cls, data_list: List[dict], goal_param_dict: dict, rebuild_param_keys: bool = False) \
            -> List[dict]:
        """Update the param_keys of a data list to reflect the situation of the goal param dict.
        :param data_list: the data to update
        :param goal_param_dict: the new param dict the param keys should show the index of
        :param rebuild_param_keys: when part of the data in data_list was deleted, the param keys are incorrect and the
        standard (faster) update method cannot be used so the param keys are recalculated from the ground up.
        """
        data_param_dict = cls.train_data_to_param_dict(data_list)
        # compute the new indices for each of the param keys
        idx_switch = []
        # make sure list_nsweeps is first in the list so that the parameter order matches up with the param_key
        _param_dict_keys = [cls.param_dict_keys[-1]] + list(cls.param_dict_keys[:-1])
        for i, key in enumerate(_param_dict_keys):
            idx_switch.append([np.where(goal_param_dict[key] == param_value)[0][0]
                               for param_value in data_param_dict[key]])
        # update the param keys
        for row in data_list:
            if not rebuild_param_keys:
                param_key = row['param_key']
                new_key = []
                for i, param_num in enumerate(param_key):
                    new_value = idx_switch[i][param_num]
                    new_key.append(new_value)
                row['param_key'] = new_key
            else:
                new_key = []
                for i, key in enumerate(cls.hyper_params_plus):
                    new_key.append(list(goal_param_dict[cls.param_dict_keys[i]]).index(row[key]))
                # put last number (nsweeps) first in param key, since that is the order they are from the main file
                row['param_key'] = [new_key[-1]] + new_key[:-1]
        return data_list

    @classmethod
    def combine_data(cls, data_base: List[dict], data_ext: List[dict], goal_param_dict: dict = None,
                     always_rebuild_param_keys: bool = False) -> List[dict]:
        """Combines two data lists into one and update the param keys to reflect the new situation.
        Standard, both complete lists are combined, but with goal_param_dict specified only the parameters in there
         will be included. This is to make sure the """
        data_base = deepcopy(data_base)
        data_ext = deepcopy(data_ext)

        params_base = cls.train_data_to_param_dict(data_base)
        params_ext = cls.train_data_to_param_dict(data_ext)
        if not goal_param_dict:
            params_new = {key: np.unique(np.append(params_base[key], params_ext[key])) for key in params_base.keys()}
            data_ext = cls.update_param_keys(data_ext, params_new, always_rebuild_param_keys)
        else:
            params_new = goal_param_dict
            # remove parameters not included in params_new
            for i, key in enumerate(cls.param_dict_keys):
                for data in (data_base, data_ext):
                    for j, row in reversed(list(enumerate(data))):
                        if row[cls.hyper_params_plus[i]] not in params_new[key]:
                            del data[j]
        data_base = cls.update_param_keys(data_base, params_new, bool(goal_param_dict) or always_rebuild_param_keys)

        return data_base + data_ext

    @staticmethod
    def train_data_to_param_dict(train_data: List[dict]) -> Dict[str, np.ndarray]:
        param_dict = {}
        for key in ExperimentSupport.param_dict_keys:
            param_dict[key] = []

        for row in train_data:
            for i, key in enumerate(ExperimentSupport.param_dict_keys):
                if row[ExperimentSupport.hyper_params_plus[i]] not in param_dict[key]:
                    param_dict[key].append(row[ExperimentSupport.hyper_params_plus[i]])

        for key in ExperimentSupport.param_dict_keys:
            param_dict[key].sort()
            param_dict[key] = np.array(param_dict[key])

        return param_dict

    @staticmethod
    def store_results_in_data_template(data_template: dict, conf_matrix: np.ndarray) -> dict:
        precision, recall, f1 = RoadEstimation.get_precision_recall_f1(conf_matrix)
        data_template['recall'] = recall
        data_template['precision'] = precision
        data_template['f1-score'] = f1
        data_template['TP'] = int(conf_matrix[1, 1])
        data_template['FP'] = int(conf_matrix[0, 1])
        data_template['TN'] = int(conf_matrix[0, 0])
        data_template['FN'] = int(conf_matrix[1, 0])
        return data_template

    @staticmethod
    def random_train_test_split(split_size: int, total_size: int, seed: int = None) -> Tuple[list, list]:
        """Returns random indices for a training and test split of a data size"""
        assert not 2 * split_size > total_size, 'split size {} cannot be more than twice the total size {}' \
            .format(split_size, total_size)
        rand = np.random.default_rng(seed)
        train_indices = np.sort(rand.choice(total_size, size=split_size, replace=False))
        # get random test split and make sure it doesn't have duplicate entries
        remaining_indices = list(range(total_size))
        for i in train_indices:
            remaining_indices.remove(i)
        substep_indices = np.sort(rand.choice(len(remaining_indices), size=split_size, replace=False))
        test_indices = [remaining_indices[idx] for idx in substep_indices]
        return list(train_indices), list(test_indices)

    @classmethod
    def load_data(cls, filename: str, base_directory: str = None, label: str = None, do_sort: bool = False,
                  goal_param_dict: dict = None) \
            -> List[dict]:

        if base_directory is not None:
            if '/data' in base_directory or '\\data' in base_directory:
                filename = osp.join(base_directory, filename)
            else:
                filename = osp.join(base_directory, 'data', filename)
        with open(filename, newline='') as csvFile:
            data_list = list(csv.DictReader(csvFile))

        # cast to the correct type
        int_keys = ['nsweeps', 'min_pnts', 'TP', 'FP', 'TN', 'FN']
        float_keys = ['eps', 'delta_phi', 'dist_lat', 'r_assign', 'dist_max', 'recall', 'precision', 'f1-score']
        list_keys = ['param_key', 'sample_f1']
        list_item_type = ['int', 'float']
        for data in data_list:
            for int_key in int_keys:
                if int_key not in data:
                    continue
                data[int_key] = int(data[int_key])
            for float_key in float_keys:
                if float_key not in data:
                    continue
                data[float_key] = float(data[float_key])
            # special case: param_keys is a list of ints
            for i, list_key in enumerate(list_keys):
                if list_key not in data:
                    continue
                temp = data[list_key].replace('[', '').replace(']', '').split(', ')
                if list_item_type[i] == 'int':
                    data[list_key] = [int(item) for item in temp]
                elif list_item_type[i] == 'float':
                    data[list_key] = [float(item) for item in temp]

        # filter out all param combinations not in goal_param_dict
        if goal_param_dict:
            # remove parameters not included in params_new
            for i, key in enumerate(cls.param_dict_keys):
                for j, row in reversed(list(enumerate(data_list))):
                    if row[cls.hyper_params_plus[i]] not in goal_param_dict[key]:
                        del data_list[j]

        if do_sort:
            data_list.sort(key=lambda k: k['f1-score'], reverse=True)

        if label is not None:
            # add a fake entry in the data list that hold the label of this data
            # get a template dict and delete all data from it
            label_entry = deepcopy(data_list[0])
            for key in label_entry.keys():
                label_entry[key] = None
            # add label and append to data list
            label_entry['label'] = label
            data_list.append(label_entry)
        return data_list

    @staticmethod
    def load_point_data(filename: str, base_directory: str = '') -> np.ndarray:
        if '/data' in base_directory or '\\data' in base_directory:
            return np.load(osp.join(base_directory, filename.replace('.data', '.npy')))
        else:
            return np.load(osp.join(base_directory + '/data', filename.replace('.data', '.npy')))

    @classmethod
    def load_data_combined(cls, filename_list: list, base_directory: str = '', do_sort: bool = False,
                           goal_param_idx: int = None, goal_param_dict: dict = None) -> List[dict]:
        """
        Load data from a list of files and combine them into a single data_list
        :param do_sort: sort the data list based on f1 score
        :param filename_list: all the data of the filenames in this list will be combined
        :param base_directory: absolute path to the base_directory where the data files are stored
        :param goal_param_idx: only the parameters present in data of the filename at this index will be represented in
        the returned data list
        :param goal_param_dict: only the parameters present in this data dict will be represented in the returned data
        list
        """
        assert goal_param_idx is None or goal_param_idx < len(filename_list), \
            'goal_param_idx cannot be larger than length of filename_list'

        assert goal_param_idx is None or goal_param_dict is None, \
            'one of goal_param_idx and goal_param_dict must be None'

        filename_list = filename_list.copy()

        if goal_param_idx is not None:
            goal_param_dict = cls.train_data_to_param_dict(cls.load_data(filename_list[goal_param_idx], base_directory))

        data_list = cls.load_data(filename_list.pop(0), base_directory, goal_param_dict=goal_param_dict)
        for filename in filename_list:
            new_data = cls.load_data(filename, base_directory)
            data_list = cls.combine_data(data_list, new_data, goal_param_dict,
                                         always_rebuild_param_keys=True)
        if do_sort:
            data_list.sort(key=lambda k: k['f1-score'], reverse=True)

        return data_list

    @classmethod
    def parse_index(cls) -> Dict[str, Dict[str, str]]:
        """Parses the _index.txt file to get filepaths of all experiments"""
        cls.assert_base_directory()
        index = {}
        with open(osp.join(cls.data_base_directory, '_index.txt'), 'r') as file:
            lines = file.readlines()
        current_experiment = ''
        for line in lines:
            if line[0] == '#' or line[0] == '\n':
                continue
            else:
                line_split = line.split()
                if line_split[0] == 'experiment':
                    current_experiment = line_split[1]
                    index[current_experiment] = {'train': '-', 'test': '-', 'point': '-'}
                elif line_split[0] == 'train':
                    index[current_experiment]['train'] = line_split[1]
                elif line_split[0] == 'test':
                    index[current_experiment]['test'] = line_split[1]
                elif line_split[0] == 'point':
                    index[current_experiment]['point'] = line_split[1]
                else:
                    raise IOError('Keyword ' + line_split[0] + ' is not defined for _index.txt')
        return index

    @classmethod
    def load_from_key(cls, key: str, include_train: bool = True, include_test: bool = True, include_point: bool = False) \
            -> Tuple[Union[List[dict], np.ndarray], ...]:
        """Load data files using the key and _index.txt
        :param key: Experiment key that will be used to lookup filepath in _index.txt
        :param include_train: include training data in output
        :param include_test: include testing data in output
        :param include_point: include point data in output
        """
        assert len(key.split('-')) == 4, 'The key should consist of 4 parts'
        assert include_train or include_test or include_point, 'one of the three options should be included at least'
        index = cls.parse_index()
        out = []
        if include_train:
            train = cls.load_data(index[key]['train'], cls.data_base_directory, key)
            out.append(train)
        if include_test:
            test = cls.load_data(index[key]['test'], cls.data_base_directory, key)
            out.append(test)
        if include_point:
            point = cls.load_point_data(index[key]['point'], cls.data_base_directory)
            out.append(point)
        return tuple(out)

    @staticmethod
    def get_data_field(data_list: List[dict], key: str) -> list:
        """Returns a list of all values of one field of a data list"""
        return [data[key] for data in data_list]

    @staticmethod
    def metrics_from_point_data(point_data: np.array, param_comb_num: int, r_assign: float, dist_max: float = None)\
            -> Tuple[float, float, float]:
        assert point_data.shape[0] == 4, 'point_data shoud have 4 rows, this had {} instead'.format(point_data.shape[0])
        point_data = point_data[:, point_data[0] == param_comb_num]
        groundtruth_labels = point_data[1]
        if dist_max is None:
            predicted_labels = (point_data[2] <= r_assign)
        else:
            predicted_labels = (point_data[2] <= r_assign) & (point_data[3] <= dist_max)
        conf_matrix = metrics.confusion_matrix(groundtruth_labels, predicted_labels)
        return RoadEstimation.get_precision_recall_f1(conf_matrix)

    @classmethod
    def f_from_point_data(cls, point_data: np.array, param_comb_num: int, r_assign: float, dist_max: float = None) \
            -> float:
        _, _, f1 = cls.metrics_from_point_data(point_data, param_comb_num, r_assign, dist_max)
        return f1

    @classmethod
    def show_all_results(cls, experiment_key: str, test_param_combs: int = 3) -> Tuple[list, list, np.ndarray]:
        """Show all results from training and testing fase for one experiment"""
        train, test, point = cls.load_from_key(experiment_key,
                                               include_train=True,
                                               include_test=True,
                                               include_point=True)
        cls.process_training_results(train)
        cls.process_testing_results(test, train, point, test_param_combs)

        return train, test, point

    @classmethod
    def process_training_results(cls, data_list: List[dict]):
        """Make plots and stuff from experiment results"""
        warnings.filterwarnings('ignore', message='invalid value encountered', category=RuntimeWarning)
        # if the data list has a label, get it
        label_entry = next((row for row in data_list if 'label' in row.keys()), None)
        if label_entry is None:
            experiment_label = '-'
        else:
            data_list.remove(label_entry)
            experiment_label = label_entry['label']

        figsize = (16, 9)
        f1 = [data['f1-score'] for data in data_list]
        boundary_ratio = (data_list[0]['TP'] + data_list[0]['FN']) / (
                data_list[0]['TP'] + data_list[0]['FN'] + data_list[0]['FP'] + data_list[0]['TN'])
        f1_baseline = 2 * boundary_ratio / (1 + boundary_ratio)
        print('Baseline f1-score: {}. Max f1-score: {}'.format(round(f1_baseline, 4), np.round(np.nanmax(f1), 4)))

        # metric histograms
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig_label = experiment_label + '/metric_histogram'
        fig.set_label(fig_label)
        fig.canvas.set_window_title(fig_label)
        # fig.subplots_adjust(bottom=.08, left=.04)
        # fig.suptitle('histogram of each of the three metrics')
        nbins = min(max(round(len(data_list) / 250), 10), 50)
        for i, metric in enumerate(['f1-score', 'recall', 'precision']):
            ax = axes[i]
            ax.hist([data[metric] for data in data_list], nbins)
            if metric == 'f1-score':
                ax.axvline(f1_baseline, linestyle='--', color='r')
            elif metric == 'precision':
                ax.axvline(boundary_ratio, linestyle='--', color='r')
            if metric == 'precision':
                ax.legend(('baseline', 'experiment data'))
            ax.set_xlabel(ExperimentSupport.metrics_str[i])
        fig.tight_layout()

        # influence of each parameter on f1-score
        parameters = cls.hyper_params_plus
        fig, axes = plt.subplots(2, int(len(parameters) / 2) + 1, figsize=figsize)
        fig_label = experiment_label + '/param_f1_score'
        fig.set_label(fig_label)
        fig.canvas.set_window_title(fig_label)
        # fig.subplots_adjust(wspace=.25, hspace=.25, left=.06, bottom=.08)
        # fig.suptitle('Scatter plot of f1 score in relation to each parameter')
        for i, param in enumerate(parameters):
            ax: plt.axes = axes[i % 2, int(i / 2)]
            param_data = np.array(([data[param] for data in data_list], f1))
            # sort and find starting index of each new parameter value
            param_data = param_data[:, param_data[0].argsort()]
            group_indices = np.unique(param_data[0].round(10), return_index=True)[1]
            # holds an array of all f1 scores of the grid search where one param in fixed at a certain value
            f1_subsets = []
            for j in range(len(group_indices)):
                if j + 1 != len(group_indices):
                    f1_subset = param_data[1, group_indices[j]:group_indices[j + 1]]
                else:
                    f1_subset = param_data[1, group_indices[j]:]
                f1_subsets.append(f1_subset)
            box_labels = np.unique(param_data[0]).round(3)
            if param == 'nsweeps' or param == 'min_pnts':
                box_labels = box_labels.astype(int)
            ax.boxplot(f1_subsets, whis=[0, 100], labels=box_labels)
            ax.axhline(f1_baseline, linestyle='--', color='r', label='baseline')
            ax.set_xlabel(cls.hyper_params_plus_str[i])
            ax.set_ylabel(cls.metrics_str[0])

            # add legend to last plot only
            if ax == axes[0, int(len(parameters) / 2 - 0.5)]:
                ax.legend()

        # look for empty plots and remove axes from them
        for ax in axes.reshape(axes.size):
            if not ax.lines and not ax.collections:
                ax.axis('off')
        fig.tight_layout()

        # If there was a label entry in the data list, add it back in
        data_list.append(label_entry)

        # influence of parameter combinations on max f1
        fig, ax = cls.plot_f1_table(data_list)

        warnings.filterwarnings('default')


    @classmethod
    def process_testing_results(cls, test_list: List[dict], train_list: List[dict] = None,
                                point_data: np.ndarray = None, param_combs: int = -1):
        """
        Plot results from running RBE algorithm on test set
        :param test_list: test data gained from .data file
        :param train_list: train data gained from .data file. Train f1 score is included in table if this not None
        :param point_data: data on a per point level. Precision-recall curve will get plotted if this is not None
        :param param_combs: number of parameter combinations to show. At -1 all available combinations are shown
        :return: the test results table containing hyperparameter for all combinations and test and train f1 score
        """
        # if the data list has a label, get it
        test_label_entry = next((row for row in test_list if 'label' in row.keys()), None)
        if test_label_entry is None:
            experiment_label = '-'
        else:
            test_list.remove(test_label_entry)
            experiment_label = test_label_entry['label']

        train_label_entry = None
        if train_list is not None:
            train_label_entry = next((row for row in train_list if 'label' in row.keys()), None)
            if train_label_entry is None:
                train_label = '-'
            else:
                train_list.remove(train_label_entry)
                train_label = train_label_entry['label']
            assert experiment_label == train_label, 'label from test and train data should be the same. Test label: ' \
                                                    '{}, train label: {}'.format(experiment_label, train_label)

        param_keys = []
        for row in test_list:
            if not row['param_key'] in param_keys:
                param_keys.append(row['param_key'])

        test_results = []
        test_list_per_param_key = []
        scene_order = []
        for i, param_key in enumerate(param_keys):
            # only include the specified number of param comps, break loop when it is reached
            if param_combs != -1 and i == param_combs:
                break
            single_param_key_list = [row for row in test_list if row['param_key'] == param_key]
            if i == 0:
                # sort from highest to lowest f1-score
                single_param_key_list.sort(key=lambda k: k['f1-score'], reverse=True)
                scene_order = [scene['scene_name'] for scene in single_param_key_list]
            else:
                # order scenes in the same way as first param key combination
                single_param_key_list = [next((item for item in single_param_key_list if item['scene_name']
                                               == scene_name)) for scene_name in scene_order]
            test_list_per_param_key.append(single_param_key_list)
            conf_matrix = np.zeros((2, 2))
            for row in single_param_key_list:
                conf_matrix += np.array([[row['TN'], row['FP']], [row['FN'], row['TP']]])
            _, _, f1 = RoadEstimation.get_precision_recall_f1(conf_matrix)
            test_result = deepcopy(single_param_key_list[0])
            for key in ['TP', 'TN', 'FP', 'FN', 'recall', 'precision',
                        'scene_name', 'scene_token', 'sample_f1', 'f1-score']:
                test_result.pop(key)
            if train_list is not None:
                train_data = next((item for item in train_list if item['param_key'] == param_key))
                test_result['train f1'] = train_data['f1-score']
            else:
                test_result['train f1'] = 0
            test_result['test f1'] = f1
            test_results.append(test_result)

        # set subplots stats for all plots
        if 4 < len(test_list_per_param_key) <= 8:
            ncols = 3
        elif len(test_list_per_param_key) > 8:
            ncols = 5
        else:
            ncols = len(test_list_per_param_key)
        nrows = np.ceil(len(test_list_per_param_key) / ncols).astype(int)

        if nrows == 1 and ncols == 1:
            figsize = (8, 4.5)
        elif nrows == 1:
            figsize = (16, 4.5)
        else:
            figsize = (16, 9)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        fig_label = experiment_label + '/scene_f1_score'
        fig.set_label(fig_label)
        fig.canvas.set_window_title(fig_label)
        # fig.subplots_adjust(wspace=0.22, hspace=0.20, left=.06)
        # fig.suptitle('f1 score per scene for best performing parameter combinations')
        for i, single_param_key_test in enumerate(test_list_per_param_key):
            f1_baseline = next((item for item in test_results
                                if item['param_key'] == single_param_key_test[0]['param_key']))['test f1']

            if nrows == 1 and ncols == 1:
                ax = axes
            elif nrows == 1:
                ax = axes[i]
            else:
                ax = axes[int(i / ncols), i % ncols]
            ax.scatter(range(len(single_param_key_test)), [data['f1-score'] for data in single_param_key_test])
            # ax.scatter(range(len(single_param_key_test)), [np.nanmean(np.array(data_dict['sample_f1'])) for data_dict in single_param_key_test])
            ax.axhline(f1_baseline, linestyle='--', color='m')
            if i == ncols - 1:
                ax.legend(('complete test set', 'per scene'))
            if not (nrows == 1 and ncols == 1):
                ax.set_title('parameter combination ' + str(i + 1))
            ax.set_ylabel(ExperimentSupport.metrics_str[0])
            ax.set_xlabel('scene')
            ax.set_xticks([])
        fig.tight_layout()

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        fig_label = experiment_label + '/sample_f1_score'
        fig.set_label(fig_label)
        fig.canvas.set_window_title(fig_label)

        # fig.subplots_adjust(hspace=0.31, bottom=.08)
        # fig.suptitle('f1 distribution per sample for best performing parameter combinations')
        for i, single_param_key_test in enumerate(test_list_per_param_key):
            # plot histogram of all samples for a single parameter combination
            sample_f1_list = []
            for scene_test in single_param_key_test:
                sample_f1_list += scene_test['sample_f1']
            f1_baseline = next((item for item in test_results if item['param_key'] ==
                                single_param_key_test[0]['param_key']))['test f1']

            if nrows == 1 and ncols == 1:
                ax = axes
            elif nrows == 1:
                ax = axes[i]
            else:
                ax = axes[int(i / ncols), i % ncols]

            nbins = min(max(round(len(sample_f1_list) / 60), 10), 50)
            ax.hist(sample_f1_list, nbins)
            ax.axvline(f1_baseline, linestyle='--', color='m')
            if not (nrows == 1 and ncols == 1):
                ax.set_title('parameter combination ' + str(i + 1))
            ax.set_xlabel(ExperimentSupport.metrics_str[0])

            if i == ncols - 1:
                ax.legend(('complete test set', 'per sample'))
        fig.tight_layout()

        # precision-recall curve
        if point_data is not None:
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            fig_label = experiment_label + '/prec_rec_curve'
            fig.set_label(fig_label)
            fig.canvas.set_window_title(fig_label)
            for i, single_param_key_test in enumerate(test_list_per_param_key):

                if nrows == 1 and ncols == 1:
                    ax = axes
                elif nrows == 1:
                    ax = axes[i]
                else:
                    ax = axes[int(i / ncols), i % ncols]

                point_data_subset = point_data[:, point_data[0] == i]
                point_data_subset[2, point_data_subset[3] > single_param_key_test[0]['dist_max']] = 1e6
                point_data_subset[2] = -point_data_subset[2]

                precision, recall, thresholds = precision_recall_curve(point_data_subset[1], point_data_subset[2])

                # plot precision-recall curve
                # the first value of recall is 1 because all points are labeled as boundary points (including the ones
                # that have assign radius 1e6). However this is not wanted behaviour so these are ignored.
                # The last value of precision is set to 1 by sklearn, this is also distorting the curve so ignored
                ax.plot(recall[1:-1], precision[1:-1])
                ax.set_xlabel('recall')
                ax.set_ylabel('precision')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                if not (nrows == 1 and ncols == 1):
                    ax.set_title('parameter combination ' + str(i + 1))

            fig.tight_layout()

        # print latex table
        # remove param keys and add parameter combination number
        for i, row in enumerate(test_results):
            row['param_key'].pop()
            row['combination number'] = i
            test_results[i] = ExperimentSupport.change_data_for_print(row)
        headers = ['combination number',
                   'nsweeps', 'eps', 'min_pnts', 'delta_phi', 'dist_lat', 'r_assign', 'dist_max',
                   'train f1', 'test f1']
        headers = {header: header for header in headers}
        cls.table_to_latex(test_results, headers, True)
        # pt = prettytable.PrettyTable()
        # for i, row in enumerate(test_results):
        #     test_results[i] = ExperimentSupport.change_data_for_print(row)
        # for field in headers:
        #     pt.add_column(field, [row[field] for row in test_results])
        # pt.align = 'r'
        # print(pt)

        # if labels were specified add them back
        if test_label_entry is not None:
            test_list.append(test_label_entry)
        if train_label_entry is not None:
            train_list.append(train_label_entry)

        return test_results

    @classmethod
    def max_f1_table(cls, data_list: List[dict], include_params: list = 'all', half_table: bool = False) -> np.ndarray:

        horizontal_params = cls.hyper_params_plus
        if include_params == 'all':
            vertical_params = cls.hyper_params_plus
        else:
            raise NotImplementedError('only showing all parameter combinations is supported yet')
            # vertical_params = include_params

        param_dict = cls.train_data_to_param_dict(data_list)
        param_len_cumsum = np.append(0, np.cumsum([len(param_dict[key]) for key in cls.param_dict_keys]))

        f1 = [data['f1-score'] for data in data_list]
        lookup_table = np.zeros((0, len(f1)))
        for param in cls.hyper_params_plus:
            lookup_table = np.vstack((lookup_table, [data[param] for data in data_list]))
        lookup_table = np.vstack((lookup_table, f1))

        results_table = np.ones((param_len_cumsum[-1], param_len_cumsum[-1]))*np.nan
        for v, v_param in enumerate(vertical_params):
            # sort and find starting index of each new parameter value
            lookup_table = lookup_table[:, lookup_table[v].argsort()]
            param_value_indices = np.unique(lookup_table[v].round(10), return_index=True)[1]

            # loop over all individual values of v_param
            for i, vi in enumerate(range(len(param_value_indices))):
                if vi + 1 != len(param_value_indices):
                    sub_lookup_table = lookup_table[:, param_value_indices[vi]:param_value_indices[vi + 1]]
                else:
                    sub_lookup_table = lookup_table[:, param_value_indices[vi]:]

                for h, h_param in enumerate(horizontal_params):
                    if half_table and v >= h or v_param == h_param:
                        continue
                    # sort and find starting index of each new parameter value
                    sub_lookup_table = sub_lookup_table[:, sub_lookup_table[h].argsort()]
                    sub_param_value_indices = np.unique(sub_lookup_table[h].round(10), return_index=True)[1]

                    for j, hj in enumerate(range(len(sub_param_value_indices))):
                        if hj + 1 != len(sub_param_value_indices):
                            f1s_for_comb = sub_lookup_table[-1, sub_param_value_indices[hj]:sub_param_value_indices[hj + 1]]
                        else:
                            f1s_for_comb = sub_lookup_table[-1, sub_param_value_indices[hj]:]
                        results_table[param_len_cumsum[v]+i, param_len_cumsum[h]+j] = np.nanmax(f1s_for_comb)

        return results_table

    @classmethod
    def plot_f1_table(cls, data_list: List[dict], ax=None, limit_color_map: bool = False):
        # if the data list has a label, get it
        label_entry = next((row for row in data_list if 'label' in row.keys()), None)
        if label_entry is None:
            experiment_label = '-'
        else:
            data_list.remove(label_entry)
            experiment_label = label_entry['label']

        f1_table = cls.max_f1_table(data_list)
        param_dict = cls.train_data_to_param_dict(data_list)
        param_len_cumsum = np.append(0, np.cumsum([len(param_dict[key]) for key in cls.param_dict_keys]))
        param_values = []
        for dict_key in cls.param_dict_keys:
            param_values += list(np.round(param_dict[dict_key], 3))

        # plotting
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        else:
            fig = plt.gcf()
        fig_label = experiment_label + '/f1_max_table'
        fig.set_label(fig_label)
        fig.canvas.set_window_title(fig_label)
        cmap = copy(plt.cm.get_cmap('inferno'))
        cmap.set_bad(color='white')
        if limit_color_map:
            if np.nanmin(f1_table) < 0.15 or np.nanmax(f1_table) > 0.26:
                print('Warning: the min and max values are hardcoded at 0.15 and 0.26, and the min or max value of '
                      'this f1 table falls outside this range')
            mat = ax.matshow(f1_table, cmap=cmap, vmin=0.15, vmax=0.26)
        else:
            mat = ax.matshow(f1_table, cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(mat, cax=cax)

        # major ticks (hyperparameter names)
        ax.set_xticks(param_len_cumsum[:-1]-0.5)
        ax.set_yticks(param_len_cumsum[:-1]-0.5)
        ax.set_xticklabels(cls.hyper_params_plus_str)
        ax.set_yticklabels(cls.hyper_params_plus_str)
        ax.tick_params('x', which='major', pad=15, length=18)
        ax.tick_params('y', which='major', pad=20, length=15)

        # minor ticks (hyperparameter values)
        ax.set_xticks(range(len(param_values)), minor=True)
        ax.set_yticks(range(len(param_values)), minor=True)
        ax.set_xticklabels(param_values, minor=True)
        ax.set_yticklabels(param_values, minor=True)
        ax.tick_params('x', which='minor', labelrotation=270, pad=1)

        # draw lines in between parameters
        for xory in param_len_cumsum[1:-1]-0.5:
            ax.axhline(xory, color='k', linewidth=0.5)
            ax.axvline(xory, color='k', linewidth=0.5)

        fig.tight_layout()

        # add label entry back
        if label_entry is not None:
            data_list.append(label_entry)

        return fig, ax

    @classmethod
    def save_current_experiment_results(cls, base_directory: str = None):
        """Saves all figures currently opened in base_directory/experiment_results/folder"""
        if base_directory is None:
            cls.assert_base_directory()
            base_directory = cls.data_base_directory.replace('data', 'experiment_results')

        save_directories = []
        for fignums in plt.get_fignums():
            fig = plt.figure(fignums)

            label = fig.get_label() if fig.get_label() != '' else 'plot'
            label_split = label.split('/')
            assert len(label_split) == 2, 'the figure label should have a subdirectory specified'
            save_directory = osp.join(base_directory, label_split[0])
            if not osp.exists(save_directory):
                os.mkdir(save_directory)
            if save_directory not in save_directories:
                save_directories.append(save_directory)

            label = label_split[1]
            label += '_v'
            v_num = 0
            while osp.exists(osp.join(save_directory, label + str(v_num) + '.png')):
                v_num += 1
            fig.savefig(osp.join(save_directory, label + str(v_num) + '.png'))
        print('{} currently opened figures are saved in {}'.format(len(plt.get_fignums()), save_directories))

    @staticmethod
    def scene_list_from_tokens(token_list: list, nusc: NuScenes):
        scene_list = []
        for token in token_list:
            scene_list.append(nusc.get('scene', token))
        return scene_list

    @staticmethod
    def get_parameters_from_data(data: dict, decimals: int = -1) -> list:
        out = []
        for param in ['eps', 'min_pnts', 'delta_phi', 'dist_lat', 'r_assign', 'dist_max', 'nsweeps']:
            if decimals >= 0:
                out.append(round(data[param], decimals))
            else:
                out.append(data[param])
        return out

    @classmethod
    def check_param_comb_counts(cls, data_list: List[dict], param_dict: dict = None, do_print: bool = True) -> List[
        dict]:
        """checks if all param combinations in the param_dict are present. When param_dict is not specified,
        it will get generated from the data_list"""
        results = []
        param_dict = cls.train_data_to_param_dict(data_list) if param_dict is None else param_dict
        total_count = int(np.prod([len(params) for params in param_dict.values()]))
        for i, param in enumerate(cls.hyper_params_plus_sweepfirst):
            param_values = param_dict[cls.param_dict_keys_sweepfirst[i]]
            for j, param_value in enumerate(param_values):
                data_count = len([True for data in data_list if data[param] == param_value])
                results.append({'type': param,
                                'value': round(param_value, 3),
                                'actual_count': data_count,
                                'goal_count': int(total_count / len(param_values)),
                                'ratio': round(data_count / (int(total_count / len(param_values))) * 100, 1)
                                })
        results.append({'type': 'total',
                        'value': '-',
                        'actual_count': len(data_list),
                        'goal_count': total_count,
                        'ratio': round(len(data_list) / total_count * 100, 1)})

        if do_print:
            pt = prettytable.PrettyTable(('type', 'value', 'is #', 'should be #', 'ratio [%]'))
            pt.add_rows(row.values() for row in results)
            pt.align = 'r'
            print(pt)

        return results

    @classmethod
    def data_completion_ratio(cls, data_list: List[dict], params: dict, param_dict: dict = None,
                              do_print: bool = False) -> float:
        """Returns the ratio of completion of data parameter combinations if the specified parameters are set"""
        for key in params:
            assert key in cls.hyper_params_plus_sweepfirst, 'the params dict can only contain the class specified ' \
                                                            'parameters. {} is not one of them'.format(key)
        param_dict = cls.train_data_to_param_dict(data_list) if param_dict is None else param_dict
        count_per_param = {}
        for i, key in enumerate(cls.param_dict_keys_sweepfirst):
            count_per_param[cls.hyper_params_plus_sweepfirst[i]] = len(param_dict[key])
        total_count = int(np.prod(list(count_per_param.values())))

        # calculate number of data entries there should be
        goal_count = total_count
        for key in params:
            goal_count /= count_per_param[key]
        assert goal_count % 1 == 0, 'goal_count should be an integer but was {}'.format(goal_count)
        goal_count = int(goal_count)

        # count number of data entries there actually are
        actual_count = 0
        for row in data_list:
            do_continue = False
            for key in params:
                if row[key] != params[key]:
                    do_continue = True
                    break
            if do_continue:
                continue
            actual_count += 1
        ratio = actual_count / goal_count
        assert 0 <= ratio <= 1, 'The ratio should be between 0 and 1, now it was {}'.format(ratio)
        if do_print:
            print('{} of {} entries are already done, this is {}%'.
                  format(actual_count, goal_count, round(ratio * 100, 1)))
        return ratio

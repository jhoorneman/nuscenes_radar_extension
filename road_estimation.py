# General imports
from typing import Tuple

import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
# Own code imports
from radar_pointcloud import RadarPC


class RoadEstimation:
    @staticmethod
    def clustering(features: np.ndarray, eps: float, min_pnts: int) -> np.ndarray:
        """
        Apply dbscan clustering algorithm to a set of features of a radar pointcloud radar_pc
        :param features: features to cluster. Typical application can be xyz values or xy and velocity
        :param eps: epsilon parameter of dbscan algorithm
        :param min_pnts: minimum points parameter of dbscan algorithm
        :return: an np.ndarray of labels describing for each point what cluster they are in
        """
        nbr_points = features.shape[1] > 0
        if nbr_points > 0:
            # when there are more then 1000 points in the feature space, using multithreading is worth it
            n_jobs = -1 if nbr_points > 1000 else None
            labels = DBSCAN(eps=eps, min_samples=min_pnts, n_jobs=n_jobs).fit_predict(features.transpose())
        else:
            labels = np.zeros(0)
        return labels

    @staticmethod
    def select_boundary_cluster(radar_pc: RadarPC, point_cluster_labels: np.ndarray,
                                delta_angle: float, lat_distance: float, line_coefficients: np.ndarray = None) -> list:
        """Select which of the clusters is a boundary cluster
        :param radar_pc: radar pointcloud containing the points
        :param point_cluster_labels: cluster label for each point
        :param delta_angle: parameter that determines maximum angle difference between cluster and ego vehicle
        :param lat_distance: parameter max lateral distance from cluster to ego vehicle
        :param line_coefficients: precomputed coefficients of fit lines. If not specified they are calculated locally
        :return: list of boundary clusters
        """
        clusters = np.unique(point_cluster_labels[point_cluster_labels >= 0])
        if line_coefficients is None:
            line_coefficients = RoadEstimation.fit_lines_through_clusters(radar_pc, point_cluster_labels)

        # curve selection
        line_angles = np.arctan(line_coefficients[:, 0])
        ego_coeff = np.polyfit((radar_pc.ego_position[0], radar_pc.ego_position[0] + radar_pc.ego_direction[0]),
                               (radar_pc.ego_position[1], radar_pc.ego_position[1] + radar_pc.ego_direction[1]), 1)
        ego_angle = np.arctan(ego_coeff[0])
        valid_clusters = []
        for i, cluster in enumerate(clusters):
            center = np.mean(radar_pc.xy()[:, point_cluster_labels == cluster], 1)
            distance = np.abs(-ego_coeff[0] * center[0] + center[1] - ego_coeff[1]) / np.sqrt(ego_coeff[0] ** 2 + 1)
            # todo: does this take into account a line that can be rotated 180 degrees?
            if ego_angle - delta_angle < line_angles[i] < ego_angle + delta_angle and distance < lat_distance:
                valid_clusters.append(cluster)
        return valid_clusters

    @staticmethod
    def fit_lines_through_clusters(radar_pc: RadarPC, cluster_labels: np.ndarray, to_be_fit_clusters: np.ndarray = None,
                                   degree: int = 1) -> np.ndarray:
        """
        Fit lines through a set of cluster. Lines are fitted with basic polynomial fitting using the np.polyfit function
        :param radar_pc: radar point cloud
        :param cluster_labels: cluster labels for each point in the radar_pc
        :param to_be_fit_clusters: label of clusters that need to get a line fitted. If not specified all clusters (not
        outliers) get a line fitted
        :param degree: polynomial degree of the fit line. Standard at 1
        :return: a np.ndarray line coefficients that can get used to plot the fitted lines
        """
        if to_be_fit_clusters is None:
            to_be_fit_clusters = np.unique(cluster_labels[cluster_labels >= 0])
        curve_coefficients = np.zeros((len(to_be_fit_clusters), degree + 1))
        for i, fit_label in enumerate(to_be_fit_clusters):
            curve_coefficients[i] = np.polyfit(radar_pc.x()[cluster_labels == fit_label],
                                               radar_pc.y()[cluster_labels == fit_label], degree)
        return curve_coefficients

    @staticmethod
    def label_points_with_curves(radar_pc: RadarPC, curve_coefficients, assign_radius, max_distance,
                                 return_distances: bool = False):
        """
        Label points that are within radius of a boundary curves and that are not further than max_distance away
        from the ego_position
        :param radar_pc: point cloud
        :param curve_coefficients: coefficient of the estimated road boundaries
        :param assign_radius: when a point is this distance from a estimated boundary is will be labeled as boundary
        :param max_distance: point has to be within max_distance of the ego position to be labeled as boundary point
        :param return_distances: the shortest distance to a boundary line for each point is returned as well
        :return: array of labels 1 for estimated boundary, 0 for non boundary, additionally the boundary distances can
        get returned, depending on return_distances
        """
        distances = radar_pc.distance_to_ego()
        curve_labels = np.zeros(radar_pc.nbr_points())
        boundary_distances = np.zeros((0, radar_pc.nbr_points()))
        for coeff in curve_coefficients:
            lat_distances = np.abs(-coeff[0] * radar_pc.x() + radar_pc.y() - coeff[1]) / np.sqrt(coeff[0] ** 2 + 1)
            if return_distances:
                boundary_distances = np.vstack((boundary_distances, lat_distances))
            curve_labels[(lat_distances < assign_radius) & (distances < max_distance)] = 1

        if return_distances:
            if boundary_distances.size > 0:
                boundary_distances = boundary_distances.min(0)
            else:
                boundary_distances = np.ones(radar_pc.nbr_points())*1e6
            return curve_labels, boundary_distances
        else:
            return curve_labels

    @staticmethod
    def apply_baseline(radar_pc: RadarPC, params: dict) -> Tuple[list, np.ndarray]:
        """Apply baseline road coarse estimation method as used in the paper: Lim, Sohee, Seongwook Lee, and Seong-
        Cheol Kim. "Clustering of Detected Targets Using DBSCAN in Automotive Radar Systems." 2018
        :param params: dictonary holding all relevant parameters
        :return: Tuple(list of indices of estimated boundary clusters, np.ndarry of line coefficient for fit lines
        through the boundary clusters"""

        for param in ('eps', 'min_pnts', 'r_assign', 'dist_max', 'normalize_points'):
            assert param in params, \
                param + ' was not in params dictionary, dict keys: ' + str([key for key in params.keys()])

        # clustering and curve fitting
        if 'cluster_v' in params and not params['cluster_v']:
            point_cluster_labels = RoadEstimation.clustering(radar_pc.xy(normalized=params['normalize_points']),
                                                             params['eps'], params['min_pnts'])
        else:
            point_cluster_labels = RoadEstimation.clustering(radar_pc.xyv(normalized=params['normalize_points']),
                                                             params['eps'], params['min_pnts'])
        clusters = np.unique(point_cluster_labels[point_cluster_labels >= 0])

        # cluster selection
        line_coefficients = RoadEstimation.fit_lines_through_clusters(radar_pc, point_cluster_labels, clusters)
        valid_clusters = RoadEstimation.select_boundary_cluster(radar_pc, point_cluster_labels,
                                                                params['delta_phi'], params['dist_lat'],
                                                                line_coefficients)

        return valid_clusters, line_coefficients[valid_clusters]

    @staticmethod
    def evaluate_baseline(radar_pc: RadarPC, params: dict) -> Tuple[np.ndarray, list]:
        """
        Compare labels of each point in the pointcloud with the labels the baseline generates and return the resulting
        confusion matrix
        :param params: dictionary of parameters, should contain eps, min_pnts and assign radius atm
        :param radar_pc: point cloud
        :return: Tuple[confusion matrix, distance to closest boundary]
        """
        for param in ('eps', 'min_pnts', 'delta_phi', 'dist_lat', 'r_assign', 'dist_max', 'normalize_points'):
            assert param in params, \
                param + ' was not in params dictionary, dict keys: ' + str([key for key in params.keys()])

        if radar_pc.nbr_points() > 0:
            valid_clusters, valid_coeffs = RoadEstimation.apply_baseline(radar_pc, params)
            baseline_labels, boundary_distances = RoadEstimation.label_points_with_curves(radar_pc,
                                                                                          valid_coeffs,
                                                                                          params['r_assign'],
                                                                                          params['dist_max'],
                                                                                          return_distances=True)
            groundtruth_labels = radar_pc.labels()
            confusion_matrix = metrics.confusion_matrix(groundtruth_labels, baseline_labels)
        else:
            confusion_matrix = np.zeros((2, 2))
            boundary_distances = []
        return confusion_matrix, boundary_distances

    ##################
    # Helper methods #
    ##################

    @staticmethod
    def get_x_in_lim(poly: np.poly1d, offset: tuple, xlim: tuple, ylim: tuple) -> tuple:
        def bound_x(x, ylim_, poly_):
            y = poly(x)
            if y < ylim_[0]:
                x = (ylim_[0] - poly_.c[1]) / poly_.c[0]
            elif y > ylim_[1]:
                x = (ylim_[1] - poly_.c[1]) / poly_.c[0]
            return x

        xlim = xlim[0] - offset[0], xlim[1] - offset[0]
        ylim = ylim[0] - offset[1], ylim[1] - offset[1]
        x0 = bound_x(xlim[0], ylim, poly)
        x1 = bound_x(xlim[1], ylim, poly)
        x0 = x0 if xlim[0] < x0 < xlim[1] else xlim[0]
        x1 = x1 if xlim[0] < x1 < xlim[1] else xlim[1]
        return x0, x1

    @staticmethod
    def get_precision_recall_f1(conf_matrix: np.ndarray) -> Tuple[float, float, float]:
        if np.sum(conf_matrix) > 0 and (conf_matrix[1, 1] + conf_matrix[0, 1]) > 0:
            precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
            recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
            if recall+precision != 0:
                f1 = 2 * recall * precision / (recall + precision)
            else:
                f1 = 0
            return precision, recall, f1
        else:
            return 0, 0, 0




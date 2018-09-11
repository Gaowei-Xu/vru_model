#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time    : 18-7-30 下午3:01
# @Author  : Gaowei Xu
# @Email   : gaowxu@hotmail.com
# @TEL     : +86-15800836035
# @File    : vru_infer.py
#
import os
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from vru_model import VRUModel
from vru_config import set_deploy_args
from data_loader import DataLoader
from image_tools import ImageTools
import pandas as pd
import numpy as np
import shutil
import random


class AgentSimulator(object):
    """
    Simulation class (simulate vehicle predicting trajectory in real time)
    """

    def __init__(self):
        """
        initialize simulator: model and dataset
        """
        self._args = args = set_deploy_args()

        self._model = None
        self._saver = None
        self._optimal_model_path = args.optimal_model_path

        print "Loading RNN model weights..."

        self._agent_id = None
        self._batch_size = args.batch_size
        self._frequency = args.frequency
        self._data_root_path = args.data_root_path
        self._seq_length = args.seq_length
        self._discrete_timestamps = args.discrete_timestamps
        self._features = args.features
        self._feed, self._ground_truth = self.prepare_data()

        print "Infer agent {} in dataset:\n".format(self._agent_id)

        self._predictions = list()
        self._image_tools = ImageTools(default_fps=8)

    def prepare_data(self):
        """
        prepare data for infer
        :return: feed: (?, feature_size)
                 ground_truth: (?, discrete_steps, feature_size)
        """
        cols_names = ['Agent_ID'].append(self._features)
        df_lc = pd.read_csv(self._data_root_path, usecols=cols_names)

        agent_ids = np.unique(df_lc[['Agent_ID']].values)

        self._agent_id = 471 #agent_ids[random.randint(0, len(agent_ids))]
        agent_features_abs = np.array(df_lc.loc[df_lc.loc[:, 'Agent_ID'] == self._agent_id, self._features].values)

        # transform agent_features from absolute coordinate to relative coordinates
        agent_features_rel = DataLoader.agent_coord_trans(agent_features_abs)
        sample_length = self._seq_length + max(self._discrete_timestamps) * self._frequency
        samples_amount = len(agent_features_rel) - sample_length

        feed_samples_rel = agent_features_rel[:samples_amount]

        return feed_samples_rel, agent_features_abs

    def visualize(self, dump_root_dir):
        xmin = np.min(np.array(self._ground_truth[:, 0]))
        xmax = np.max(np.array(self._ground_truth[:, 0]))

        ymin = np.min(np.array(self._ground_truth[:, 1]))
        ymax = np.max(np.array(self._ground_truth[:, 1]))

        for i, prediction in enumerate(self._predictions):
            if i == 0:
                continue
            fig = plt.figure(figsize=(15, 8))
            ax = fig.gca()
            # ax.set_aspect('equal')

            # plot whole trajectory
            plt.scatter(self._ground_truth[:, 1], self._ground_truth[:, 0], marker='.', c='b')

            # plot current position
            plt.scatter(self._ground_truth[i, 1], self._ground_truth[i, 0], marker='o', c='r')

            # plot targets shape(6,4)
            for time in self._discrete_timestamps:
                plt.scatter(
                    self._ground_truth[i + int(time * self._frequency), 1],
                    self._ground_truth[i + int(time * self._frequency), 0],
                    marker='o', c='g')

            # plot predictions with RNN
            l1 = plt.scatter(prediction[:, 1], prediction[:, 0], marker='^', c='m')

            # plot predictions with Linear Regression
            buffer_len = 10
            fx, fy, valid = self.traj_fit(self._ground_truth, i, buffer_len)
            if valid:
                l2 = plt.scatter(fy, fx, marker='^', c='k')

            plt.axis([ymin-0.5, ymax+0.5, xmin-0.5, xmax+0.5])
            plt.legend([l1, l2], ['RNN', 'Linear Regression'], loc='upper right')
            plt.savefig(os.path.join(dump_root_dir, '{}.jpg'.format(str(i))), dpi=300)
            plt.close()

    def traj_fit(self, traj, curr_index, buffer_len):
        """
        using linear regression to fit the trajecotry

        :param traj:
        :param curr_index:
        :param len:
        :return:
        """
        len = buffer_len

        if curr_index - buffer_len + 1 < 0 and curr_index < 1:
            return None, None, False

        if curr_index - buffer_len + 1 < 0 and curr_index >= 1:
            len = curr_index + 1

        timestamp_list = np.arange(0, len, 1.0) / self._frequency
        x_list = traj[curr_index-len+1:curr_index + 1, 0]
        y_list = traj[curr_index-len+1:curr_index + 1, 1]

        dt = 1.0
        time_horizon = 6.5

        _, f_x = self.vru_trajectory_fit(
            timestamp_list=timestamp_list,
            observation_list=x_list,
            dt=dt,
            time_horizon=time_horizon)

        _, f_y = self.vru_trajectory_fit(
            timestamp_list=timestamp_list,
            observation_list=y_list,
            dt=dt,
            time_horizon=time_horizon)

        return f_x, f_y, True

    def vru_trajectory_fit(self, timestamp_list, observation_list, dt, time_horizon):
        """
        curve fit (1-order linear regression) of timestamp vs variable, this function is used
        for linear regression between two variables

        :param timestamp_list: list of timestamp
        :param observation_list: list of observations (such as x, y, velocity, acc, etc)
        :param dt: timestamp interval
        :param time_horizon: time horizon to predict
        :return:
        """
        assert (len(timestamp_list) == len(observation_list))
        assert (len(timestamp_list) >= 2)

        timestamp_list = np.array(timestamp_list)
        observation_list = np.array(observation_list)
        num_samples = len(timestamp_list)

        # perform normalization in that the values of input list are usually very large
        timestamp_max = max(timestamp_list)
        timestamp_min = min(timestamp_list)
        timestamp_list_norm = (timestamp_list - timestamp_min) / (timestamp_max - timestamp_min)

        temp = np.vstack((np.ones(num_samples), timestamp_list_norm))
        lineqn_a = np.dot(temp, temp.transpose())
        lineqn_b = np.dot(temp, observation_list.reshape(num_samples, 1))
        coeffs = np.linalg.solve(lineqn_a, lineqn_b)

        # extra-sampling of the fitted line
        future_timestamps = np.arange(timestamp_list[-1]+dt, timestamp_list[-1] + time_horizon, dt)
        future_timestamps_norm = (future_timestamps - timestamp_min) / (timestamp_max - timestamp_min)
        future_observations = coeffs[1, 0] * future_timestamps_norm + coeffs[0, 0]
        return future_timestamps, future_observations

    def infer(self):
        """
        Predict the future trajectory

        :return:
        """
        assert self._batch_size == 1
        assert self._seq_length == 1

        tf.reset_default_graph()
        self._model = VRUModel(self._args)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            self._saver = tf.train.Saver()
            self._saver.restore(sess, self._optimal_model_path)

            # initialize initial state
            state = sess.run(fetches=self._model.initial_state)

            for i in range(len(self._feed)):

                # feed 3D shape=(1, 1, feature_size)
                feed = [[self._feed[i]]]
                # target shape=(1, 1, discrete_steps, feature_size)
                dummy_target = np.zeros([self._batch_size,
                                         self._seq_length,
                                         len(self._discrete_timestamps),
                                         len(self._features)])

                feeds = {self._model.input_data: feed,
                         self._model.initial_state: state,
                         self._model.target_data: dummy_target}

                fetches = [self._model.predicted_outputs, self._model.final_state]

                # forward the prediction
                results, state = sess.run(feed_dict=feeds, fetches=fetches)

                # result shape is [self._batch_size, self._seq_length, self._discrete_step_size, self._feature_size]
                prediction_each_frame = self.coord_trans_rel2abs(results, i)

                self._predictions.append(prediction_each_frame)

        dump_root_dir = './frames'
        if not os.path.exists(dump_root_dir):
            os.makedirs(dump_root_dir)

        # visualize predictions with target
        self.visualize(dump_root_dir)

        if not os.path.exists(dump_root_dir):
            os.makedirs(dump_root_dir)

        # make frames into videos
        self._image_tools.frames2video(
            load_frames_root_dir="./frames/",
            video_reconstruct_full_path="./outputs/Agent_{}.avi".format(self._agent_id))

        shutil.rmtree(dump_root_dir)

        return

    def coord_trans_rel2abs(self, results, index):
        """
        calculate the coordinates transformations

        :param results: output of RNN, relative results, shape is
                        [self._batch_size, self._seq_length, self._discrete_step_size, self._feature_size]
        :param index: current index
        :return: absolute coordinates
        """
        assert results.shape[0] == 1
        assert results.shape[1] == 1

        if index >= 5:
            tr_matrix = DataLoader.get_inverse_trans_matrix(
                self._ground_truth[index],
                self._ground_truth[index - 5])
        else:
            tr_matrix = DataLoader.get_inverse_trans_matrix(
                self._ground_truth[index + 1],
                self._ground_truth[index])

        for i in range(results.shape[2]):
            coord_rel = results[0][0][i][0:2]
            coord_rel = np.matrix(np.append(coord_rel, 1.0)).transpose()
            coord_abs = tr_matrix * coord_rel
            coord_abs = np.array(coord_abs[:-1]).reshape([2, ])
            results[0][0][i][0:2] = coord_abs

        return results[0][0]


if __name__ == '__main__':
    agent = AgentSimulator()
    agent.infer()



#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time    : 18-7-30 下午3:01
# @Author  : Gaowei Xu
# @Email   : gaowxu@hotmail.com
# @TEL     : +86-15800836035
# @File    : data_loader.py

import os
import numpy as np
import random
import pandas as pd
import pickle


class DataLoader(object):
    def __init__(self,
                 cache_root_path,
                 data_root_path,
                 batch_size,
                 seq_length,
                 frequency,
                 discrete_timestamps,
                 features):
        self._cache_root_path = cache_root_path
        if not os.path.exists(self._cache_root_path):
            os.makedirs(self._cache_root_path)
        self._data_root_path = data_root_path
        self._batch_size = batch_size
        self._seq_length = seq_length
        self._frequency = frequency
        self._discrete_timestamps = discrete_timestamps
        self._features = features
        self._train_pointer = 0
        self._val_pointer = 0
        self._train_batch_num = 0
        self._val_batch_num = 0
        self._cached_file_name = 'all_feed_samples.pkl'
        self._cached = os.path.exists(os.path.join(self._cache_root_path, self._cached_file_name))

        # generate dataset for training phase and validation phase
        self._all_feed_samples_train, self._all_feed_samples_val = self.data_generator(train_val_ratio=4.5)

    @staticmethod
    def agent_coord_trans(agent_features_abs):
        """
        transform features from absolute coordinate system to relative coordinates

        :param agent_features_abs: the whole trajectory (may contain velocity, etc) needs to be transformed
        :return: agent_features_rel (relative)
        """
        agent_features_rel = list()

        for index in range(len(agent_features_abs)):
            features_abs = agent_features_abs[index]

            if index >= 4:
                tr_matrix = DataLoader.get_inverse_trans_matrix(
                    agent_features_abs[index - 1],
                    agent_features_abs[index - 4])

                xy_abs = np.array([features_abs[0], features_abs[1]])
                xy_abs = np.matrix(np.append(xy_abs, 1.0)).transpose()
                xy_rel = tr_matrix.I * xy_abs
                xy_rel = np.array(xy_rel[:-1]).reshape([2, ])
                xy_rel = list(xy_rel)
                xy_rel.extend(features_abs[2:])
                agent_features_rel.append(xy_rel)
            else:
                xy_rel = [0.0, 0.0]
                xy_rel.extend(features_abs[2:])
                agent_features_rel.append(xy_rel)

        return np.array(agent_features_rel)

    @staticmethod
    def target_coord_trans(gt_abs, index, agent_features_abs):
        """
        transform ground truth from absolute coordinate system to relative coordinates

        :param gt_abs: shape=[len(self._discrete_timestamps), len(self._features)]
        :param index: current index
        :param agent_features_abs:
        :return:
        """
        gt_rel = np.zeros(shape=gt_abs.shape)
        if index >= 3:
            tr_matrix = DataLoader.get_inverse_trans_matrix(
                agent_features_abs[index],
                agent_features_abs[index - 3])

            for k in range(gt_abs.shape[0]):
                gt_rel[k] = gt_abs[k]
                xy_abs = np.array([gt_abs[k][0], gt_abs[k][1]])
                xy_abs = np.matrix(np.append(xy_abs, 1.0)).transpose()
                xy_rel = tr_matrix.I * xy_abs
                xy_rel = np.array(xy_rel[:-1]).reshape([2, ])
                gt_rel[k][0:2] = [xy_rel[0], xy_rel[1]]
        else:
            for k in range(gt_abs.shape[0]):
                gt_rel[k] = gt_abs[k]
                gt_rel[k][0:2] = [0.0, 0.0]

        return gt_rel

    def data_generator(self, train_val_ratio=5.0):
        """
        data generator for training and validation dataset

        :param train_val_ratio: ratio of training and validation samples amount
        :return:
            all_feed_samples_train: shape = [train_samples_amount, seq_length, feature_size]
            all_feed_samples_val: shape = [validation_samples_amount, seq_length, discrete_timestamps, feature_size]
        """
        if not self._cached:
            cols_names = ['Agent_ID', 'FrameIndex'].append(self._features)
            df_lc = pd.read_csv(self._data_root_path, usecols=cols_names)

            columns = df_lc.columns
            print 'Loading train/val file {} with columns = {}\n'.format(self._data_root_path, columns)
            print 'Generating dataset for training and validation phase...'

            all_feed_samples = list()
            agent_ids = np.unique(df_lc[['Agent_ID']].values)
            agent_amount = len(agent_ids)
            sample_length = self._seq_length + max(self._discrete_timestamps) * self._frequency

            for agent_index, agent_id in enumerate(agent_ids):
                agent_features_abs = np.array(df_lc.loc[df_lc.loc[:, 'Agent_ID'] == agent_id, self._features].values)

                # transform agent_features from absolute coordinate to relative coordinates
                agent_features_rel = DataLoader.agent_coord_trans(agent_features_abs)
                samples_amount = len(agent_features_rel) - sample_length

                for curr_index in range(samples_amount):
                    # input_data shape = (self._seq_length, feature_size)
                    input_data_rel = agent_features_rel[curr_index:curr_index+self._seq_length]

                    # target_data shape = (self._seq_length, discrete_timestamps, feature_size)
                    gt_data_rel = np.zeros(shape=[self._seq_length, len(self._discrete_timestamps), len(self._features)])

                    for offset in range(self._seq_length):

                        gt_each_frame_abs = np.zeros(shape=[len(self._discrete_timestamps), len(self._features)])
                        for k, t in enumerate(self._discrete_timestamps):
                            gt_each_frame_abs[k] = agent_features_abs[curr_index + offset + t * self._frequency]

                        gt_each_frame_rel = DataLoader.target_coord_trans(gt_each_frame_abs,
                                                                          curr_index + offset,
                                                                          agent_features_abs)

                        # assign ground truth (with relative coordinates) to gt_data_rel
                        gt_data_rel[offset] = gt_each_frame_rel

                    all_feed_samples.append([input_data_rel, gt_data_rel])

                print 'Complete generating dataset for agent {}, totally {} samples, ({}/{})...'.format(agent_id,
                                                                                                        samples_amount,
                                                                                                        agent_index,
                                                                                                        agent_amount)

            pkl_file = open(os.path.join(self._cache_root_path, self._cached_file_name), 'wb')
            pickle.dump(all_feed_samples, pkl_file)

        else:       # cached
            print 'Loading cached trainval dataset...'
            pkl_file = open(os.path.join(self._cache_root_path, self._cached_file_name), 'rb')
            all_feed_samples = pickle.load(pkl_file)

        all_feed_samples = list(all_feed_samples)
        random.shuffle(all_feed_samples)

        # split samples into train and val
        train_dataset_amount = int(train_val_ratio * len(all_feed_samples) / (train_val_ratio + 1))
        all_feed_samples_train = all_feed_samples[:train_dataset_amount]
        all_feed_samples_val = all_feed_samples[train_dataset_amount:]

        self._train_batch_num = int(len(all_feed_samples_train) / self._batch_size)
        self._val_batch_num = int(len(all_feed_samples_val) / self._batch_size)

        print 'Totally generate {} training batches and {} validation batches. \n'.format(self._train_batch_num,
                                                                                          self._val_batch_num)

        return all_feed_samples_train, all_feed_samples_val

    def next_batch(self, mode):
        """
        get the next batch data for training

        :param mode: train or val
        :return: input_batch or input_batch_delta: (batch_size, seq_length, features)
                 target_batch or target_batch_delta: (batch_size, seq_length, discrete_steps, features)
        """
        if mode == 'train':
            pointer = self._train_pointer

            # batch_bundle shape = [self._batch_size, 2]
            batch_bundle = self._all_feed_samples_train[pointer:pointer + self._batch_size]

            input_batch_rel = np.array([x[0] for x in batch_bundle])
            target_batch_rel = np.array([x[1] for x in batch_bundle])

            self._train_pointer += self._batch_size
            return input_batch_rel, target_batch_rel

        elif mode == 'val':
            pointer = self._val_pointer

            # batch_bundle shape = [self._batch_size, 2]
            batch_bundle = self._all_feed_samples_train[pointer:pointer + self._batch_size]

            input_batch_rel = np.array([x[0] for x in batch_bundle])
            target_batch_rel = np.array([x[1] for x in batch_bundle])

            self._val_pointer += self._batch_size
            return input_batch_rel, target_batch_rel

        else:
            raise RuntimeError('Mode can only be train or val.')

    @staticmethod
    def get_inverse_trans_matrix(p1, p2):
        """
        calculate inverse transformation matrix, p1 is later observed than p2
        Attention: the transformed y-axis direction is defined by vector P2P1 (NOT P1P2)

        [transform from rel to abs: coord_abs = inverse_trans_matrix * coord_rel]
        coord_rel = predict[0:2]
        coord_rel = np.matrix(np.append(coord_rel, 1.0)).transpose()
        coord_abs = inverse_trans_matrix * coord_rel
        coord_abs = np.array(coord_abs[:-1]).reshape([2, ])
        abs_x = coord_abs[0]
        abs_y = coord_abs[1]

        [transform from abs to rel: coord_rel = inverse_trans_matrix.I * coord_abs]
        coord_abs = np.array([feed[0], feed[1]])
        coord_abs = np.matrix(np.append(coord_abs, 1.0)).transpose()
        coord_rel = inverse_trans_matrix.I * coord_abs
        coord_rel = np.array(coord_rel[:-1]).reshape([2, ])
        rel_x = coord_rel[0]
        rel_y = coord_rel[1]

        :param p1: the first point
        :param p2: the second point
        :return:
        """
        y_axis_vec = np.array([p1[0], p1[1]]) - np.array([p2[0], p2[1]])
        theta = np.arctan2(y_axis_vec[1], y_axis_vec[0])
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        inverse_trans_matrix = np.matrix([[cos_theta, -sin_theta, p1[0]],
                                          [sin_theta, cos_theta, p1[1]],
                                          [0, 0, 1]])
        return inverse_trans_matrix

    def reset_train_pointer(self):
        self._train_pointer = 0

    def reset_val_pointer(self):
        self._val_pointer = 0

    @property
    def train_batch_num(self):
        return self._train_batch_num

    @property
    def val_batch_num(self):
        return self._val_batch_num

    @property
    def train_pointer(self):
        return self._train_pointer

    @property
    def val_pointer(self):
        return self._val_pointer






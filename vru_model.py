#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time    : 18-7-30 下午3:01
# @Author  : Gaowei Xu
# @Email   : gaowxu@hotmail.com
# @TEL     : +86-15800836035
# @File    : vru_model.py
#
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, layers


class VRUModel(object):
    def __init__(self, args):
        """
        configure model parameters

        :param args: model configuration parameters
        """
        self._mode = args.mode
        self._batch_size = args.batch_size

        self._feature_size = len(args.features)
        self._discrete_step_size = len(args.discrete_timestamps)
        self._output_size = len(args.discrete_timestamps) * len(args.features)
        self._rnn_size = args.rnn_size
        self._seq_length = args.seq_length
        self._n_layers = args.n_layers

        self._lr = args.lr
        self._keep_prob = args.keep_prob

        self._input_data = tf.placeholder(
            tf.float32, shape=[self._batch_size, self._seq_length, self._feature_size], name='input_data')
        self._target_data = tf.placeholder(
            tf.float32, shape=[self._batch_size, self._seq_length, self._discrete_step_size, self._feature_size], name='target_data')

        # global step variable used for decaying optimizer
        self._global_step = tf.Variable(0, trainable=False)
        self._predicted_outputs = None
        self._loss = None
        self._summary_op = None
        self._train_op = None
        self._initial_state = None
        self._final_state = None
        self._loss_weights = np.ones(shape=[self._batch_size, self._seq_length,
                                            self._discrete_step_size, self._feature_size]) * args.loss_weights

        self.build_model()

    def build_model(self):

        def get_a_cell():
            # initialize a recurrent unit
            single_cell = rnn.GRUCell(num_units=self._rnn_size)

            # wrap a dropout layer if applicable
            if self._mode == 'train' and self._keep_prob < 1.0:
                single_cell = rnn.DropoutWrapper(cell=single_cell, output_keep_prob=self._keep_prob)

            return single_cell

        with tf.variable_scope('rnn'):
            cell = rnn.MultiRNNCell([get_a_cell() for _ in range(self._n_layers)])

            # initial cell state, shape=(N, rnn_size)
            self._initial_state = cell.zero_state(batch_size=self._batch_size, dtype=tf.float32)

            # sequence length argument in rnn
            _sequence_length = tf.convert_to_tensor([self._seq_length for _ in range(self._batch_size)])

            # dynamic rnn, output shape (batch_size, seq_length, rnn_size)
            rnn_outputs, self._final_state = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=self._input_data,
                sequence_length=_sequence_length,
                initial_state=self._initial_state,
                dtype=None,
                parallel_iterations=None,
                swap_memory=False,
                time_major=False,
                scope=None)

        with tf.variable_scope('fc'):
            rnn_outputs = tf.reshape(rnn_outputs, shape=[-1, self._rnn_size])
            outputs = layers.fully_connected(rnn_outputs, self._output_size)
            outputs = tf.reshape(
                outputs,
                shape=[self._batch_size, self._seq_length, self._discrete_step_size, self._feature_size])
            self._predicted_outputs = outputs

        with tf.variable_scope('loss'):
            #  shape is [self._batch_size, self._seq_length, self._discrete_step_size, self._feature_size]
            residual_abs_square = tf.square(outputs - self._target_data)
            self._loss = loss = tf.reduce_mean(residual_abs_square * self._loss_weights)

            # add summary operation
            tf.summary.scalar('training_loss', self._loss)
            self._summary_op = tf.summary.merge_all()

        if self._mode == 'Infer':
            return

        with tf.variable_scope('optimizer'):
            self._train_op = tf.train.AdamOptimizer(learning_rate=self._lr).minimize(loss, global_step=self._global_step)

    @property
    def input_data(self):
        return self._input_data

    @property
    def target_data(self):
        return self._target_data

    @property
    def predicted_outputs(self):
        return self._predicted_outputs

    @property
    def loss(self):
        return self._loss

    @property
    def summary_op(self):
        return self._summary_op

    @ property
    def train_op(self):
        return self._train_op

    @ property
    def final_state(self):
        return self._final_state

    @ property
    def initial_state(self):
        return self._initial_state

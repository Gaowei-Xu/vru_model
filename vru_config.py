#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time    : 18-7-30 下午3:01
# @Author  : Gaowei Xu
# @Email   : gaowxu@hotmail.com
# @TEL     : +86-15800836035
# @File    : vru_config.py
#

import argparse


def set_train_args():
    """
    configure the training arguments

    :return: args
    """
    parser = argparse.ArgumentParser()

    # mode
    parser.add_argument('--mode', type=str, default='train',
                        help='mode of training, train or infer')

    # size of each batch for training dataset parameter
    parser.add_argument('--batch_size', type=int, default=96,
                        help='training set minibatch size')

    # list of features parameters to feed in rnn
    parser.add_argument('--features', type=list, default=[
        'X',
        'Y',
        'YawAbs',
        'VelocityAbs'
    ], help='List of feature names selected to learn by RNN')

    # weights of loss of each dimension
    parser.add_argument('--loss_weights', type=list, default=[5.0, 5.0, 1.0, 1.0],
                        help='loss_weights')

    # length of time sequence parameter
    parser.add_argument('--seq_length', type=int, default=48,
                        help='length of the time sequence')

    # predict frequency
    parser.add_argument('--frequency', type=int, default=10,
                        help='predict frequency, should match with the dataset observations.')

    # number of discrete steps
    parser.add_argument('--discrete_timestamps', type=list, default=[1, 2, 3, 4, 5, 6],
                        help='List of the specific timestamp to be predict')

    # size of the rnn outputs parameter
    parser.add_argument('--rnn_size', type=int, default=64,
                        help='the output size of rnn')

    # number of layers of each rnn cell parameter
    parser.add_argument('--n_layers', type=int, default=1,
                        help='the output size of rnn')

    # learning rate for unet parameter
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate for RNN')

    # Dropout probability parameter
    parser.add_argument('--keep_prob', type=float, default=0.9,
                        help='dropout keep probability')

    # -- training
    # Number of epoch parameter
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs')

    # each number of epoch parameter to save model parameters
    parser.add_argument('--save_every_epoch', type=int, default=1,
                        help='save model variables for every how many epoch(s)')

    # path of root data for dataloader
    parser.add_argument('--cache_root_path', type=str, default='cache/',
                        help='pickle file path for accelerate dataset loading')

    # path of root data for dataloader
    parser.add_argument('--data_root_path', type=str, default='dataset/trainval.csv',
                        help='csv data path for dataloader')

    # directory for train summary FileWriter
    parser.add_argument('--train_summary', type=str, default='train_log/train/',
                        help='train summary FileWriter')

    # directory for val summary FileWriter
    parser.add_argument('--val_summary', type=str, default='train_log/val/',
                        help='val summary FileWriter')

    # target path to dump model parameters parameter
    parser.add_argument('--dump_model_para_root_dir', type=str, default='model_params/',
                        help='directory path to dump model parameters while training')

    args = parser.parse_args()
    return args


def set_deploy_args():
    """
    configure the deploy arguments

    :return: args
    """
    parser = argparse.ArgumentParser()

    # mode
    parser.add_argument('--mode', type=str, default='infer',
                        help='mode of training, train or infer')

    # agent id to get dataset from for simulation
    parser.add_argument('--agent_id', type=int, default=471,
                        help='the agent id to get dataset from')

    # path of optimal model /home/yuhuang/Desktop/VRUPrediction/model_params/epoch1_0.114497_0.121583.ckpt.index
    parser.add_argument('--optimal_model_path', type=str,
                        default='model_params/epoch12_3.979129_4.051616.ckpt',
                        help='path of optimal model parameters')

    # weights of loss of each dimension
    parser.add_argument('--loss_weights', type=list, default=[5.0, 5.0, 1.0, 1.0],
                        help='loss_weights')

    # path of root data for dataloader
    parser.add_argument('--data_root_path', type=str, default='dataset/trainval.csv',
                        help='csv data path for dataloader')

    # predict frequency
    parser.add_argument('--frequency', type=int, default=10,
                        help='predict frequency, should match with the dataset observations.')

    # infer batch size
    parser.add_argument('--batch_size', type=int, default=1,
                        help='infer batch size')

    # length of time sequence parameter for infer
    parser.add_argument('--seq_length', type=int, default=1,
                        help='length of the sequence when infer')

    # list of features parameters to feed in rnn
    parser.add_argument('--features', type=list, default=['X', 'Y', 'YawAbs', 'VelocityAbs'],
                        help='List of feature names selected to learn by RNN')

    # number of discrete steps
    parser.add_argument('--discrete_timestamps', type=list, default=[1, 2, 3, 4, 5, 6],
                        help='List of the specific timestamp to be predict')

    # size of the rnn outputs parameter
    parser.add_argument('--rnn_size', type=int, default=64,
                        help='the output size of rnn')

    # number of layers of each rnn cell parameter
    parser.add_argument('--n_layers', type=int, default=1,
                        help='the output size of rnn')

    # learning rate for unet parameter
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate for RNN')

    # Dropout probability parameter
    parser.add_argument('--keep_prob', type=float, default=0.9,
                        help='dropout keep probability')

    args = parser.parse_args()
    return args

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time    : 18-7-30 下午3:01
# @Author  : Gaowei Xu
# @Email   : gaowxu@hotmail.com
# @TEL     : +86-15800836035
# @File    : vru_train.py
#
from data_loader import DataLoader
from vru_model import VRUModel
from vru_config import *

import os
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')


def train():
    """
    Train phase main process

    :return:
    """
    args = set_train_args()
    dataloader = DataLoader(
                            cache_root_path=args.cache_root_path,
                            data_root_path=args.data_root_path,
                            batch_size=args.batch_size,
                            seq_length=args.seq_length,
                            frequency=args.frequency,
                            discrete_timestamps=args.discrete_timestamps,
                            features=args.features
                            )
    print '\nDatasets loaded.'

    model = VRUModel(args)
    print '\nModel initialized.'

    train_error = np.zeros(args.num_epochs)
    valid_error = np.zeros(args.num_epochs)

    step = 0

    # configure GPU training
    gpuConfig = tf.ConfigProto(allow_soft_placement=True)
    gpuConfig.gpu_options.allow_growth = True

    with tf.Session(config=gpuConfig) as sess:
        train_writer = tf.summary.FileWriter('./train/', sess.graph)
        val_writer = tf.summary.FileWriter('./val/', sess.graph)

        # initialize all variables in the graph
        tf.global_variables_initializer().run()

        # initialize a saver that saves all the variables in the graph
        saver = tf.train.Saver(max_to_keep=None)

        print '\n\nStart Training... \n'
        for e in range(args.num_epochs):

            dataloader.reset_train_pointer()
            dataloader.reset_val_pointer()

            # --- TRAIN ---
            for batch in range(dataloader.train_batch_num):

                # input_batch: (batch_size, seq_length , feature)
                # target_batch: (batch_size, seq_length, discrete_steps, feature)
                input_batch_rel, target_batch_rel = dataloader.next_batch('train')

                loss, predicted_outputs, summary, __ = sess.run(
                    fetches=[
                        model.loss,
                        model.predicted_outputs,
                        model.summary_op,
                        model.train_op
                    ],
                    feed_dict={
                        model.input_data: input_batch_rel,
                        model.target_data: target_batch_rel
                    })

                # add summary and accumulate stats
                train_writer.add_summary(summary, step)
                train_error[e] += loss
                step += 1
                if batch % 100 == 0:
                    print "Predict results = {}".format(predicted_outputs[0][10])
                    print 'Ground truth = {}\n'.format(target_batch_rel[0][10])

            # normalise running means by number of batches
            train_error[e] /= dataloader.train_batch_num

            # --- VALIDATION ---
            for batch in range(dataloader.val_batch_num):
                input_batch_rel, target_batch_rel = dataloader.next_batch('val')

                loss, __ = sess.run(
                    fetches=[
                        model.loss,
                        model.summary_op
                    ],
                    feed_dict={
                        model.input_data: input_batch_rel,
                        model.target_data: target_batch_rel
                    })
                val_writer.add_summary(summary, step)
                valid_error[e] += loss
            valid_error[e] /= dataloader.val_batch_num

            # checkpoint model variable
            if (e + 1) % args.save_every_epoch == 0:
                model_name = 'epoch{}_{:2f}' \
                             '_{:2f}.ckpt'.format(e + 1, train_error[e], valid_error[e])

                dump_model_full_path = os.path.join(args.dump_model_para_root_dir, model_name)
                saver.save(sess=sess, save_path=dump_model_full_path)
                tf.add_to_collection("predict", model.predicted_outputs)

            print('Epoch {0:02d}: err(train)={1:.2f}, err(valid)={2:.2f}'.format(e + 1, train_error[e], valid_error[e]))

    train_writer.close()
    val_writer.close()
    sess.close()

    return train_error, valid_error


if __name__ == '__main__':
    train()


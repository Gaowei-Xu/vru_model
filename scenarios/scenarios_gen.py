#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time    : 18-7-30 下午3:01
# @Author  : Gaowei Xu
# @Email   : gaowxu@hotmail.com
# @TEL     : +86-15800836035
# @File    : scenarios_gen.py
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import random


class ScenarioGen(object):
    """
    Scenarios generator
    """
    def __init__(self, scenario_type, amount, dump_root_dir):
        """
        initialization of generator

        :param scenario_type:
        :param amount:
        """
        self._scenario_type = scenario_type
        self._amount = amount
        self._dump_root_dir = os.path.join(dump_root_dir, scenario_type)
        if not os.path.exists(self._dump_root_dir):
            os.makedirs(self._dump_root_dir)

    def generate(self):
        if self._scenario_type == 'TurnLeft':
            self.generate_scenario_turn_left()

    def generate_scenario_turn_left(self):
        """

        :return:
        """
        csv_file = open(os.path.join(self._dump_root_dir, 'trainval.csv'), 'w')
        csv_file.write('Agent_ID,FrameID,X,Y,YawAbs,VelocityAbs\n')

        for k in range(self._amount):
            agent = list()

            center_x = float(random.randint(0, 10000))
            center_y = float(random.randint(0, 10000))

            raduis = float(random.randint(30, 60))

            theta_start = random.randint(0, 628) * 2 * 3.141592 / 628.0
            theta_end = theta_start + 3.141592 / 2.0

            for theta in np.arange(theta_start, theta_end, 0.01):
                x = center_x + raduis * np.cos(theta)
                y = center_y + raduis * np.sin(theta)

                x = x + random.uniform(-0.5, 0.5)
                y = y + random.uniform(-0.5, 0.5)
                theta = theta + random.uniform(-0.01, 0.01)

                agent.append([x, y, theta, 0.0])

            delta_t = 0.08
            for i in range(1, len(agent)):
                v_abs = np.sqrt((agent[i][0] - agent[i-1][0]) * (agent[i][0] - agent[i-1][0]) + \
                        (agent[i][1] - agent[i-1][1]) * (agent[i][1] - agent[i-1][1])) / delta_t
                agent[i][-1] = v_abs

            # write into csv file
            for i in range(0, len(agent)):
                csv_file.write('{},{},{},{},{},{}\n'.format(k, i, agent[i][0], agent[i][1], agent[i][2], agent[i][3]))

            agent = np.array(agent)
            fig = plt.figure(figsize=(8, 8))
            ax = fig.gca()
            ax.set_aspect('equal')
            plt.scatter(agent[:, 0], agent[:, 1], marker='o', c='k')
            plt.savefig(os.path.join(self._dump_root_dir, '{}.jpg'.format(k+1)), dpi=300)
            plt.close()

        csv_file.close()


if __name__ == '__main__':
    gen = ScenarioGen(
        scenario_type='TurnLeft',
        amount=400,
        dump_root_dir='./scenarios'
    )

    gen.generate()













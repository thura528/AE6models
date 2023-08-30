# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 22:57:18 2023

@author: thura
"""

import os
import pylab
import matplotlib.pyplot as plt
import argparse


class ParseGRU():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', default='', help='dataset directory')
        parser.add_argument('--log_folder', default='./logs', help='log directory')
        parser.add_argument('--batch_size', type=int,default=16)
        parser.add_argument('--video_batch', type=int,default=16)
        parser.add_argument('--image_size', default=64)
        parser.add_argument('--T', type=int, default=4)
        parser.add_argument('--check_point', type=int, default=200)
        parser.add_argument('--n_channels', type=int, default=1)
        parser.add_argument('--n_test', type=int, default=1,help='number of test image which saved')
        parser.add_argument('--n_itrs', type=int, default=10000)
        parser.add_argument('--z_dim', type=int, default=64)
        parser.add_argument('--gru_dim', type=int, default=100)#512,128,32
        parser.add_argument('--learning_rate', type=int, default=1e-4)#1e-2
        parser.add_argument('--cuda', type=bool, default=True)

        self.args = parser.parse_args()

class Visualizer():
    def __init__(self,opt):
        self.opt = opt

    def plot_loss(self):
        pylab.xlim(0, self.opt.n_itrs)  # *self.len
        pylab.ylim(0, max(self.losses)/100)
        plt.plot(self.losses, label='loss')#'+','.join(self.opt.dis_loss))#if wanna print type of loss
        plt.legend()
        plt.savefig(os.path.join(self.opt.log_folder, 'loss.pdf'))
        plt.close()

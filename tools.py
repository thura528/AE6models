# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 11:51:46 2022

@author: thura
"""

import numpy as np


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):  # 除以每列的和
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward, normalize=True):
    I = edge2mat(self_link, num_node)
    In = edge2mat(inward, num_node)
    Out = edge2mat(outward, num_node)
    if normalize:
        In = normalize_digraph(In)
        Out = normalize_digraph(Out)
    A = np.stack((I, In, Out))
    return A
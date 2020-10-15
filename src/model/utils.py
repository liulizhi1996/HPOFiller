#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Walk with Restart (RWR) is used to extract features from
similarity graph for protein nodes or HPO term nodes.
"""
import numpy as np


def rwr(A, restart_prob):
    """
    Random Walk with Restart (RWR) on similarity network.
    :param A: n x n, similarity matrix
    :param restart_prob: probability of restart
    :return: n x n, steady-state probability
    """
    n = A.shape[0]
    A = (A + A.T) / 2
    A = A - np.diag(np.diag(A))
    A = A + np.diag(sum(A) == 0)
    P = A / sum(A)
    Q = np.linalg.inv(np.eye(n) - (1 - restart_prob) * P) @ (restart_prob * np.eye(n))
    return Q

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import dense, norm

# set random generator seed to allow reproducibility
t.manual_seed(12345)


class GCMC(nn.Module):
    """
    The core model of HPOFiller goes here. GCMC means using graph
    convolutional networks to finish matrix completion.
    """
    def __init__(self, n_prot, n_term):
        """
        :param n_prot: the number of proteins
        :param n_term: the number of HPO terms
        """
        super(GCMC, self).__init__()

        self.bm_input_x = norm.BatchNorm(n_prot)
        self.bm_input_y = norm.BatchNorm(n_term)

        self.dense_x = nn.Linear(n_prot, 800)
        self.dense_y = nn.Linear(n_term, 800)

        self.bm_x0 = norm.BatchNorm(800)
        self.bm_y0 = norm.BatchNorm(800)

        self.gcn_xy1 = dense.DenseGraphConv(800, 800)
        self.gcn_x1 = dense.DenseGCNConv(800, 800)
        self.gcn_y1 = dense.DenseGCNConv(800, 800)
        self.bm_xy1 = norm.BatchNorm(800)
        self.bm_x1 = norm.BatchNorm(800)
        self.bm_y1 = norm.BatchNorm(800)

        self.gcn_xy2 = dense.DenseGraphConv(800, 800)
        self.gcn_x2 = dense.DenseGCNConv(800, 800)
        self.gcn_y2 = dense.DenseGCNConv(800, 800)
        self.bm_xy2 = norm.BatchNorm(800)
        self.bm_x2 = norm.BatchNorm(800)
        self.bm_y2 = norm.BatchNorm(800)

        self.linear_x1 = nn.Linear(800, 400)
        self.linear_y1 = nn.Linear(800, 400)
        self.linear_x2 = nn.Linear(400, 200)
        self.linear_y2 = nn.Linear(400, 200)
        self.linear_x3 = nn.Linear(200, 100)
        self.linear_y3 = nn.Linear(200, 100)
        self.bm_lx1 = norm.BatchNorm(400)
        self.bm_ly1 = norm.BatchNorm(400)
        self.bm_lx2 = norm.BatchNorm(200)
        self.bm_ly2 = norm.BatchNorm(200)

    def forward(self, X_prot, X_term, A_prot, A_term, A_rel):
        """
        :param X_prot: n_prot x dim, feature matrix of proteins
        :param X_term: n_term x dim, feature matrix of HPO terms
        :param A_prot: n_prot x n_prot, similarity of proteins
        :param A_term: n_term x n_term, similarity of HPO terms
        :param A_rel: (n_prot + n_term) x (n_prot + n_term), protein-HPO term
                      association matrix, i.e. HPO annotations
        :return: predicted protein-HPO term matrix
        """
        m, n = X_prot.shape[0], X_term.shape[0]

        X = self.bm_input_x(X_prot)
        Y = self.bm_input_y(X_term)
        X = F.leaky_relu(self.dense_x(X))
        Y = F.leaky_relu(self.dense_y(Y))
        X = self.bm_x0(X)
        Y = self.bm_y0(Y)

        XY = t.squeeze(F.leaky_relu(self.gcn_xy1(t.cat((X, Y)), A_rel)))
        XY = self.bm_xy1(XY)
        X, Y = t.split(XY, (m, n))
        X = t.squeeze(F.leaky_relu(self.gcn_x1(X, A_prot)))
        Y = t.squeeze(F.leaky_relu(self.gcn_y1(Y, A_term)))
        X = self.bm_x1(X)
        Y = self.bm_y1(Y)

        XY = t.squeeze(F.leaky_relu(self.gcn_xy2(t.cat((X, Y)), A_rel)))
        XY = self.bm_xy2(XY)
        X, Y = t.split(XY, (m, n))
        X = t.squeeze(F.leaky_relu(self.gcn_x2(X, A_prot)))
        Y = t.squeeze(F.leaky_relu(self.gcn_y2(Y, A_term)))
        X = self.bm_x2(X)
        Y = self.bm_y2(Y)

        X = F.relu(self.linear_x1(X))
        Y = F.relu(self.linear_y1(Y))
        X = self.bm_lx1(X)
        Y = self.bm_ly1(Y)
        X = F.relu(self.linear_x2(X))
        Y = F.relu(self.linear_y2(Y))
        X = self.bm_lx2(X)
        Y = self.bm_ly2(Y)
        X = F.relu(self.linear_x3(X))
        Y = F.relu(self.linear_y3(Y))

        return X.mm(Y.t())

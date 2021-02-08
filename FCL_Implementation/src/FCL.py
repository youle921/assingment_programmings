# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 21:26:50 2020

@author: youle
"""
import numpy as np

np.random.seed(0)

class FCL:

    def __init__(self, data_num, dim, cluster_num, theta = 2):

        # setting hyper parameter

        # number of clusters
        self.c_num = cluster_num
        # theta
        self.theta = theta

        # initialize membership and prototype randomly
        self.membership_u = np.random.rand(data_num, cluster_num)
        self.prototype_b = np.random.rand(cluster_num, dim)
        self.prototype_a = np.random.rand(cluster_num, dim)

        self.membership_u_normalize()
        self.prototype_a_normalize()

    def membership_u_normalize(self):

        self.membership_u /= np.sum(self.membership_u, axis = 1)[:, None]

    def prototype_a_normalize(self):

        self.prototype_a /= np.sum(self.prototype_a**2, axis = 1)[:, None]

    def calc_loss(self, data):

        loss = 0
        for c in range(self.c_num):
            for n in range(data.shape[0]):
                loss += self.membership_u[n, c]**self.theta*(np.sum((data[n] - self.prototype_b[c])**2)\
                    -np.dot(self.prototype_a[c], data[n] - self.prototype_b[c])**2)

        return loss

    def update_membership(self, data):

        for i in range(data.shape[0]):

            for c in range(self.c_num):

                numer = np.sum((data[i] - self.prototype_b[c, :])**2) \
                    - np.dot(self.prototype_a[c], data[i] - self.prototype_b[c, :])**2

                tmp = 0
                for cl in range(self.c_num):
                    denom = np.sum((data[i] - self.prototype_b[cl, :])**2) \
                        - np.dot(self.prototype_a[cl], data[i] - self.prototype_b[cl, :])**2
                    tmp += (numer/denom)**(1/(self.theta - 1))

                self.membership_u[i, c] = 1 / tmp

    def update_b(self, data):

        for c in range(self.c_num):

            self.prototype_b[c] = np.sum(self.membership_u[:, c][:, None]*data, axis = 0)/np.sum(self.membership_u[:, c])**self.theta

    def update_a(self, data):

        for c in range(self.c_num):
            mat = np.zeros([data.shape[1], data.shape[1]])
            for i in range(data.shape[0]):
                mat += self.membership_u[i, c]**self.theta * np.dot(data[i][:, None], data[i][None, :])

            eig_val, eig_vec = np.linalg.eig(mat)
            idx = np.where(eig_val == np.max(eig_val))[0][0]
            self.prototype_a[c] = eig_vec[:, idx]

    def execute(self, data, itr = 100, eps = 10e-8):

        loss_old = self.calc_loss(data)

        for i in range(itr):

            self.update_membership(data)
            self.update_b(data)
            self.update_a(data)

            self.membership_u_normalize()
            self.prototype_a_normalize()

            loss = self.calc_loss(data)

            if abs(loss - loss_old) < eps:
                return

            loss_old = loss

data = np.loadtxt("dataset/waking_data.csv", delimiter = ",")

data, label = np.hsplit(data, [data.shape[1] - 1])

FCL_algorithm = FCL(*data.shape, np.unique(label).shape[0])
FCL_algorithm.execute(data, itr = 1000)

pred_label = np.argmax(FCL_algorithm.membership_u, axis = 1)

pred1 = np.bincount(pred_label[:30])
pred2 = np.bincount(pred_label[30:60])
pred3 = np.bincount(pred_label[60:])
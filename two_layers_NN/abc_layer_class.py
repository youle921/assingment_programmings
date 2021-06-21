# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 23:09:32 2021

@author: youle
"""
from abc import ABCMeta, abstractmethod

class abc_layer(metaclass=ABCMeta):

    @abstractmethod
    def forward(self, data):
        pass

    @abstractmethod
    def backward(self, data):
        pass

    @abstractmethod
    def update_params(self, lr):
        pass
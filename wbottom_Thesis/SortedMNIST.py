# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:19:48 2024

@author: wbott
"""
from bindsnet.datasets import MNIST
import torch

class SortedMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data, self.targets = self.sort_by_label(self.data, self.targets)

    @staticmethod
    def sort_by_label(data, targets):
        sorted_indices = torch.argsort(targets)
        return data[sorted_indices], targets[sorted_indices]
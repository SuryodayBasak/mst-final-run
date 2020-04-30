import pandas as pd
import numpy as np
import os

"""
Dataset 1: MPG.
"""
class Mpg:
  def __init__(self):
    self.dataset = np.genfromtxt('data/mpg/data.csv', delimiter = ',')

  def Data(self):
    return self.dataset[:, 1:-1], self.dataset[:, 0]

"""
Dataset 2: Airfoil Self Noise.
"""
class Airfoil:
  def __init__(self):
    self.dataset = np.genfromtxt('data/airfoil/data.csv', delimiter = ',')

  def Data(self):
    return self.dataset[:, 0:-1], self.dataset[:, -1]


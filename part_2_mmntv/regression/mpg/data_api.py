import pandas as pd
import numpy as np
import os

"""
Dataset 23: MPG.
"""
class Mpg:
  def __init__(self):
    self.dataset = np.genfromtxt('data/mpg/data.csv', delimiter = ',')

  def Data(self):
    return self.dataset[:, 1:], self.dataset[:, 0]


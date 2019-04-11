#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 10:12:28 2018

@author: zqwu
"""

import numpy as np
import pandas as pd
import pickle
import os
from data_analysis import background_subtraction
from glob import glob

def generate_train_labels(X, profiles, dist_factor=20., itgr_factor=2.):
  """ Generate labels for training

  Args:
  -----
  X: np.array
      N*2 array of rt and intensities
  profiles: np.array
      peak profile
  dist_factor: float
      importance factor for predicted boundary differences
      smaller value => larger importance
  itgr_factor: float
      importance factor for predicted abundance differences

  Returns:
  --------
  y: np.array
      N*2 array of p_start and p_end

  """
  y = np.zeros((X.shape[0], 2))
  rt_start = profiles[1]
  rt_end = profiles[2]
  # Ground truths for start and end positions
  left = np.argmin(np.abs(X[:, 0] - rt_start))
  right = np.argmin(np.abs(X[:, 0] - rt_end))

  y[left, 0] = 1
  y[right, 1] = 1
  signals = background_subtraction(X, profiles)
  abd = np.sum(signals[left:(right+1), 1]) + 1e-9
  for i in range(10):
    #Left of left end
    if left - i >= 0:
      diff_abd = 100 * float(np.sum(signals[left-i:left, 1]))/abd
      y[left - i, 0] = np.exp(-i**2/dist_factor) * np.exp(-diff_abd**2/itgr_factor)
    #Right of left end
    if left + i < (left + right)/2:
      diff_abd = 100 * float(np.sum(signals[left:left+i, 1]))/abd
      y[left + i, 0] = np.exp(-i**2/dist_factor) * np.exp(-diff_abd**2/itgr_factor)
    #Left of right end
    if right - i > (left + right)/2:
      diff_abd = 100 * float(np.sum(signals[right-i+1:right+1, 1]))/abd
      y[right - i, 1] = np.exp(-i**2/dist_factor) * np.exp(-diff_abd**2/itgr_factor)
    #Right of right end
    if right + i < y.shape[0]:
      diff_abd = 100 * float(np.sum(signals[right+1:right+i+1, 1]))/abd
      y[right + i, 1] = np.exp(-i**2/dist_factor) * np.exp(-diff_abd**2/itgr_factor)

  return y

def weights_by_intensity(data, max_val=120., min_val=50.):
  """ Generate sample weights according to their max signal intensity """
  weights = []
  for d in data:
    ind_start, ind_end = list(np.argmax(d[1], 0))
    max_peak_value = max(d[0][ind_start:(ind_end+1), 1])
    weight = max((min(max_peak_value, max_val) - min_val)/(max_val - min_val), 0.)
    weights.append(weight)
  return weights


def load_references_from_table(path=None):
  """ In the structure of:
    {peak_name(str): [(0)precursor_mz(float), 
                      (1)product_mz(float), 
                      (2)rt(float),
                      (3)rt_start(float),
                      (4)rt_end(float),
                      (5)rt_width(float),
                      (6)valid(float, 0 or 1)]
    }
  """
  data = np.array(pd.read_csv(path))
  n_peaks = data.shape[0]
  all_peaks = {}
  for i in range(n_peaks):
    peak_name = data[i, 0]
    all_peaks[peak_name] = [data[i, 1], # precursor_mz
                            data[i, 2], # product_mz
                            data[i, 4], # rt
                            data[i, 5], # rt_start
                            data[i, 6], # rt_end
                            data[i, 7], # rt_width
                            1]
  return all_peaks
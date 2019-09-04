#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 15:37:21 2018

@author: zqwu
"""

import pandas as pd
import numpy as np
import os
import pickle
from data_utils import generate_train_labels, load_references_from_table
from data_analysis import get_start_end
from data_analysis import calculate_abundance
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

def load_sample_labels(path='./utils/multiple_annotators.csv'):
  """ In the structure of:
    {(sample_name(str), peak_name(str)): 
         [(0)response(float), 
          (1)rt_start(float), 
          (2)rt_end(float), 
          (3)S/N(float), 
          (4)width(float),
          (5)rt(float), 
          (6)rt_difference(float)]
    }
  """
  peaks = {'PP%02d' % i: {} for i in range(1, 13)}
  labels = np.array(pd.read_csv(path, header=None, dtype=str))
  references = load_references_from_table('./utils/ref_rt.csv')
  labels[np.where(labels=="?")] = "-1"
  sample_names = labels[2:, 0].astype(str)
  peak_names = labels[0, 1:].reshape((-1, 7))[:, 0]
  peak_profiles = labels[2:, 1:].reshape((sample_names.size, peak_names.size, 7))
  for i, sample_name in enumerate(sample_names):
    for j, peak_name in enumerate(peak_names):
      if peak_name.startswith('PP'):
        peak = peak_name.split(' ')[0][5:]
        picker_id = peak_name[:4]
        p_p = peak_profiles[i, j].astype(float)
        if references[peak][3] - p_p[1] > 0.2:
          continue
        if - references[peak][4] + p_p[2] > 0.2:
          continue
        peaks[picker_id][(sample_name, peak)] = p_p
  return peaks


parallel_peaks = load_sample_labels()
peaks = [list(val.keys()) for val in parallel_peaks.values()]
shared_peaks = set(peaks[0])
for p in peaks:
  shared_peaks = shared_peaks | set(p)


test_data = pickle.load(open('./test_input.pkl', 'rb'))
pred = pickle.load(open('./test_preds.pkl', 'rb'))
pred_ref = pickle.load(open('./test_preds_ref.pkl', 'rb'))
test_data_keys = [t[3] for t in test_data]

shared_peaks = list(shared_peaks & set(test_data_keys))

# 0 - prediction
# 1 - prediction(ref)
# 2 - original label
# 3~14 - PP01~PP12
abds = [[] for _ in range(15)] 
for key in shared_peaks:
  t = test_data[test_data_keys.index(key)]
  X = t[0]
  
  prediction = pred[test_data_keys.index(key)]
  abds[0].append(calculate_abundance(prediction, t))

  prediction_ref = pred_ref[test_data_keys.index(key)]
  abds[1].append(calculate_abundance(prediction_ref, t))
  
  label = generate_train_labels(X, t[2])
  abds[2].append(calculate_abundance(label, t))
  
  for i in range(1, 13):
    if key in parallel_peaks['PP%02d' % i]:
      label_ = generate_train_labels(X, parallel_peaks['PP%02d' % i][key])
      abds[i + 2].append(calculate_abundance(label_, t))
    else:
      abds[i + 2].append(None)

abds = np.stack(abds, 0).astype(float)
mean_abds = np.nanmean(abds[2:], 0)
std_abds = np.nanstd(abds[2:], 0)

step = 0.001
hs1, bins1 = np.histogram(np.abs(abds[0] - mean_abds)/mean_abds, bins=np.arange(0.0, 0.02, step))
hs2, bins2 = np.histogram(np.abs(abds[1] - mean_abds)/mean_abds, bins=np.arange(0.0, 0.02, step))
hs3, bins3 = np.histogram(std_abds/mean_abds, bins=np.arange(0.0, 0.02, step))

font = {'weight': 'normal',
        'size': 16}

# Figure 4
plt.clf()
bins = np.arange(0.0, 0.02, step)[:-1]
plt.bar(bins+step/4, hs3, width=step/4, color=(0.1, 0.1, 0.1, 0.3), label='Human Annotators')
plt.bar(bins-step/4, hs1, width=step/4, color=(44./256, 83./256, 169./256, 1.), label='Sequential PB-Net')
plt.bar(bins, hs2, width=step/4, color=(69./256, 209./256, 163./256, 1), label='Ref-based PB-Net')
plt.legend(fontsize=14)
plt.xlabel("Relative Error/Standard Deviation", fontdict=font)
plt.ylabel("Counts", fontdict=font)
plt.tight_layout()
plt.savefig('./figs/fig4_comp_annotators.eps', format='eps')

print(np.mean(std_abds/mean_abds))
print(np.mean(np.abs(abds[1] - mean_abds)/mean_abds)) # ref
print(np.mean(np.abs(abds[0] - mean_abds)/mean_abds)) # seq

#########################################################################

diffs = np.zeros((15, 15))
for i in range(0, 15):
  for j in range(i+1, 15):
    inds = np.array(list(set(list(np.where(abds[i] == abds[i])[0])) & set(list(np.where(abds[j] == abds[j])[0]))))
    diffs[i, j] = np.mean(np.abs(abds[i][inds] - abds[j][inds])/mean_abds[inds])
diffs = diffs + np.transpose(diffs)
diffs_within = diffs[2:, 2:].flatten()
diffs_within = diffs_within[np.nonzero(diffs_within)]

print(str(np.mean(diffs_within)) + ' plus/minus ' + str(np.std(diffs_within)))
print(str(np.mean(diffs[1, 2:])) + ' plus/minus ' + str(np.std(diffs[1, 2:]))) # ref
print(str(np.mean(diffs[0, 2:])) + ' plus/minus ' + str(np.std(diffs[0, 2:]))) # seq


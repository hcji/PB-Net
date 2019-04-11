#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 00:29:21 2018

@author: zqwu
"""

import numpy as np
import pandas as pd
from data_analysis import eval_accuracy, eval_abundance_accuracy, eval_dists, eval_abundance_correlation, eval_confidences
from scipy.stats import spearmanr, pearsonr
import pickle
import os
from data_analysis import calculate_abundance, background_subtraction, get_start_end

test_data = pickle.load(open('./test_input.pkl', 'rb'))

df = pd.read_csv('./utils/skyline_results.csv')
all_peaks = set([t[3][1] for t in test_data])
sample_names = list(df['Sample.Name'])
peak_names = list(df.columns[1:])
results = np.array(df)[:, 1:]

labels =  -np.ones_like(results)
for i, t in enumerate(test_data):
  if t[3][0] in sample_names and t[3][1] in peak_names:
    labels[sample_names.index(t[3][0])][peak_names.index(t[3][1])] = calculate_abundance(t[1], t)

r_scores = []
sr_scores = []
total_ct = 0
correct_ct = 0
for i in range(results.shape[1]):
  pred = results[:, i].astype(float)
  label = labels[:, i].astype(float)
  valid_inds = np.array(sorted(list(set(np.where(pred == pred)[0]) & 
                                    set(np.where(label > 0)[0]))))
  pred = pred[valid_inds]
  label = label[valid_inds]
  if len(valid_inds) < 60:
    print(all_peaks[i])
    continue
  r_scores.append(pearsonr(np.log(label + 1e-9), np.log(pred + 1e-9))[0])
  sr_scores.append(spearmanr(label, pred).correlation)
  total_ct += len(label)
  r = pred/label
  correct_ct += len(set(np.where(r <= 1.05)[0]) & set(np.where(r >= 0.95)[0]))

print(np.mean(r_scores))
print(np.mean(sr_scores))
print(float(correct_ct)/total_ct)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 10:24:01 2018

@author: zqwu
"""
import numpy as np
import matplotlib
matplotlib.use('AGG') 
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

def background_subtraction(rt_intensity, profile):
  """ Subtract signal background
      background = lowest signal in 
         [rt_start(human label) - 0.1, rt_end(human label) + 0.1]

  rt_intensity: np.array
      N*2 array of retention time/intensity pairs
  profile: tuple/list
      peak profiles
  """
  window_start = profile[1] - 0.1
  window_end = profile[2] + 0.1
  ind_start = np.argmin(np.abs(rt_intensity[:, 0] - window_start))
  ind_end = np.argmin(np.abs(rt_intensity[:, 0] - window_end))
  background = min(rt_intensity[ind_start:(ind_end+1), 1])
  signal = rt_intensity[:, 1] - background
  signal = np.max(np.stack([signal, np.zeros_like(signal)], 0), 0)
  return np.stack([rt_intensity[:, 0], signal], 1)

def get_start_end(pred):
  """ Get discrete start/end position from pred

  pred: np.array
      N*2 array of predictions on peak start/end
  """
  start = np.argmax(pred[:, 0])
  # Prevent end position appears prior to start position
  end = np.argmax(pred[start:, 1]) + start
  return start, end

def calculate_abundance(pred, sample, intg=False, sub=True, sum_of_signals=False):
  """ Calculate abundance of a XIC curve

  pred: np.array
      N*2 array of predictions on peak start/end
  sample: list
      standard input structure: [X, y, profiles, names]
  intg: bool, optional
      if to use weighted peak start/end
  sub: bool, optional
      if to substract baseline
  sum_of_signals: bool, optional
      if to use the simple sum of signals/integration of signals over retention time
  """
  if sub:
    out = background_subtraction(sample[0], sample[2])
  else:
    out = sample[0]
  if not intg:
    se_pred = get_start_end(pred)
    if sum_of_signals:
      return np.sum(out[se_pred[0]:(se_pred[1]+1), 1])
    else:
      step = out[(se_pred[0]+1):(se_pred[1]+1), 0] - \
          out[se_pred[0]:se_pred[1], 0]
      intensity = (out[(se_pred[0]+1):(se_pred[1]+1), 1] + \
          out[se_pred[0]:se_pred[1], 1])/2
      return np.sum(step*intensity) * 60
  else:
    if sum_of_signals:
      intg_mat = np.triu(np.repeat(np.transpose(out[:, 1:2]), 
                                   out.shape[0], 0))
    else:
      step = out[1:, 0] - out[:-1, 0]   
      intensity = (out[1:, 1] + out[:-1, 1])/2
      areas = np.concatenate([np.zeros((1,)), step*intensity])
      intg_mat = np.triu(np.repeat(areas.reshape((1, -1)), out.shape[0], 0) * (1-np.eye(out.shape[0]))) * 60
    intg_mat = np.cumsum(intg_mat, axis=1)
    s_thr = np.max(pred[:, 0]) * 0.05
    e_thr = np.max(pred[:, 1]) * 0.05
    s_inds = pred[:, 0] < s_thr
    pred[s_inds, 0] = 0
    e_inds = pred[:, 1] < e_thr
    pred[e_inds, 1] = 0
    p_mat = np.triu(pred[:, 0:1] * np.transpose(pred[:, 1:2]))
    if np.sum(p_mat) == 0:
      return 0
    else:
      return np.sum(intg_mat * p_mat) / np.sum(p_mat)

def eval_dists(test_data, preds):
  """ Evaluate the distances(in points) between 
      boundary predictions and labels 

  test_data: list
      list of standard input structure
  preds: list
      list of np.array(predictions of shape N*2)
  """
  dists = []
  for i, sample in enumerate(test_data):
    label = sample[1]
    output = preds[i]
    def calculate_dist(pred, true):
      start, end = get_start_end(pred)
      left_dist = np.abs(start - np.mean(np.argmax(true[:, 0])))
      right_dist = np.abs(end - np.mean(np.argmax(true[:, 1])))
      return (left_dist + right_dist)/2

    dists.append(calculate_dist(output, label))
  print("Mean error %f" % np.nanmean(dists))
  return dists

def eval_accuracy(test_data, preds, thr=2):
  """ Evaluate the accuracy of boundary predictions
      Accurate is defined as average boundaries prediction error within 'thr'

  test_data: list
      list of standard input structure
  preds: list
      list of np.array(predictions of shape N*2)
  thr: int, optional
      threshold of error on boundary prediction(in number of points) for "accurate" predictions
  """
  n_samples = len(preds)
  correct_ct = 0
  for i, sample in enumerate(test_data):
    label = sample[1]
    output = preds[i]
    start, end = get_start_end(output)
    if np.abs(start - np.argmax(label[:, 0])) + np.abs(end - np.argmax(label[:, 1])) <= 2*thr:
      correct_ct += 1
  accuracy = float(correct_ct)/n_samples
  print("Boundary accuracy at threshold %d: %f" % (thr, accuracy))
  return accuracy

def eval_accuracy2(test_data, preds, thr=0.05):
  """ Evaluate the accuracy of abundance predictions
      Accurate is defined as sum of absolute abundance differences
      between prediction and label less than `thr` * abundance(label)

      Evaluate the accuracy of apex predictions:
      Accurate is defined as correct apex being picked in the prediction

  test_data: list
      list of standard input structure
  preds: list
      list of np.array(predictions of shape N*2)
  thr: float, optional
      threshold of error on abundance prediction for "accurate" predictions
  """
  n_samples = len(preds)
  correct_ct = 0
  wrong_ct = 0
  for i, sample in enumerate(test_data):
    signals = background_subtraction(sample[0], sample[2])
    label = sample[1]
    output = preds[i]
    start, end = get_start_end(output)
    start_ref = np.argmax(label[:, 0])
    end_ref = np.argmax(label[:, 1])
    if np.max(signals[start_ref:(end_ref+1), 1]) != np.max(signals[start:(end+1), 1]):
      wrong_ct += 1
      continue
    
    abundance = np.sum(signals[start_ref:(end_ref + 1), 1])
    start_diff_abd = sum(signals[min(start_ref, start):max(start_ref, start), 1])
    end_diff_abd = sum(signals[(min(end_ref, end)+1):(max(end_ref, end)+1), 1])
    if (start_diff_abd + end_diff_abd)/abundance  < thr:
      correct_ct += 1
  accuracy = float(correct_ct)/n_samples
  wrong_rate = float(wrong_ct)/n_samples
  print("(Abs)Abundance accuracy at threshold %f: %f" % (thr, accuracy))
  print("Apex accuracy: %f" % (1 - wrong_rate))
  return accuracy


def eval_abundance_accuracy(test_data, preds, thr=0.05, intg=False):
  """ Evaluate the accuracy of abundance predictions
      Accurate is defined as (total) abundance differences
      between prediction and label less than `thr` * abundance(label)

      Note this is different from `eval_accuracy2`.

  test_data: list
      list of standard input structure
  preds: list
      list of np.array(predictions of shape N*2)
  thr: float, optional
      threshold of error on abundance prediction for "accurate" predictions
  intg: bool, optional
      if to use weighted peak start/end
  """
  y_trues = []
  y_preds = []
  for i, sample in enumerate(test_data):
    output = preds[i]
    y_trues.append(calculate_abundance(sample[1], sample))
    y_preds.append(calculate_abundance(output, sample, intg=intg))
  y_trues = np.array(y_trues)
  y_preds = np.array(y_preds)

  total_ct = len(test_data)
  ratio = y_preds/y_trues
  correct_ct = len(set(np.where(ratio <= 1+thr)[0]) & set(np.where(ratio >= 1-thr)[0]))
  accuracy = float(correct_ct)/total_ct
  print("Abundance accuracy at threshold %f: %f" % (thr, accuracy))
  return accuracy


def eval_abundance_correlation(test_data, preds, mode='r2', intg=False):
  """ Evaluate correlation of abundance(integration) of peak

  test_data: list
      list of standard input structure
  preds: list
      list of np.array(predictions of shape N*2)
  mode: str, optional
      correlation type to be reported
  intg: bool, optional
      if to use weighted peak start/end
  """
  y_trues = []
  y_preds = []
  for i, sample in enumerate(test_data):
    output = preds[i]
    y_trues.append(calculate_abundance(sample[1], sample))
    y_preds.append(calculate_abundance(output, sample, intg=intg))
  y_trues = np.array(y_trues)
  y_preds = np.array(y_preds)
  if mode == 'r2':
    corr1 = r2_score(y_trues, y_preds)
    corr2 = r2_score(np.log(y_trues + 1e-9), np.log(y_preds + 1e-9))
    corr3 = spearmanr(y_trues, y_preds)
    print("Abundance correlation: %f\t Log: %f" % (corr1, corr2))
    print(corr3)
    return corr1, corr2, corr3.correlation
  elif mode == 'l1':
    error = np.abs(y_trues - y_preds)
    print("Abundance mae: %f" % np.mean(error))
    return error
  elif mode == 'l2':
    squared_error = np.square(y_trues - y_preds)
    print("Abundance rmse: %f" % np.sqrt(np.mean(error)))
    return squared_error
  elif mode == 'ratio':
    ratio = y_preds/y_trues
    print("Mean abundance ratio: %f" % np.mean(ratio))
    return ratio
  else:
    raise ValueError("mode should be in: r2, l1, l2")

def plot_peak(pair, pred=None):
  """ Plot a peak

  pair: list
      standard input structure: [X, y, profiles, names]
  pred: None or np.array, optional
      if provided, predicted probabilities of peak start/end will also be plotted
  """
  X = pair[0]
  profile = pair[2]
  plt.plot(X[:, 0], X[:, 1], 'b', label='X')
  start = profile[1]
  end = profile[2]
  plt.vlines(start, min(X[:, 1]), max(X[:, 1]), colors='r')
  plt.vlines(end, min(X[:, 1]), max(X[:, 1]), colors='r')
  if pred is not None:
    plt.plot(X[:, 0], min(X[:, 1]) + pred[:, 0]*(max(X[:, 1]) - min(X[:, 1])), 'g', label='y_pred_start')
    plt.plot(X[:, 0], min(X[:, 1]) + pred[:, 1]*(max(X[:, 1]) - min(X[:, 1])), 'c', label='y_pred_end')
  plt.xlabel('Retention Time')
  plt.ylabel('Intensity')
  plt.legend(loc=4)
  
def calculate_confidence(pred, thr=0.2):
  """ Calculate confidence of a peak start/end prediction
  
  pred: np.array
      N*1 array of predictions on peak start or end
  thr: float, optional
      threshold of defining a local maximum (fixed to 0.2 in most cases)
  """
  apex = np.max(pred)
  base = np.sqrt(apex)
  over_threshold = []
  for i, intensity in enumerate(pred):
    if intensity > apex * thr:
      if (i > 0) and (intensity < pred[i-1]):
        continue
      if (i < len(pred) - 1) and (intensity < pred[i+1]):
        continue
      over_threshold.append((i, intensity))
  selected_points = over_threshold
  
  selected_points.sort(key=lambda x: x[0])
  if len(selected_points) == 1:
    return base
  elif base < 0.01:
    return base
  else:
    intensities = [p[1] for p in selected_points]
    total_score = np.max(intensities) * (len(pred) - 1)
    apex_position = np.argmax(intensities)
    score = 0
    if apex_position > 0:
      for i in range(0, apex_position):
        intensities[i] = np.max(intensities[:(i+1)])
        score += intensities[i] * (selected_points[i+1][0] - selected_points[i][0])
    if apex_position < len(selected_points) - 1:
      for i in range(apex_position+1, len(selected_points)):
        intensities[i] = np.max(intensities[i:])
        score += intensities[i] * (selected_points[i][0] - selected_points[i-1][0])
    return (1. - score/total_score) * base

def eval_confidences(preds, thr=0.2):
  """ Calculate confidences for a list of peak predictions
  
  preds: list
      list of np.array(predictions of shape N*2)
  thr: float, optional
      threshold of defining a local maximum (fixed to 0.2 in most cases)
  """
  conf = [calculate_confidence(p[:, 0], thr=thr) * \
          calculate_confidence(p[:, 1], thr=thr) for p in preds]
  return conf

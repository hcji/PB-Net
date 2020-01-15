#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 10:12:28 2018

@author: zqwu
"""

from pyteomics import mzxml
from pyopenms import MzMLFile, MSExperiment
import numpy as np
import pandas as pd
import pickle
import os
from data_analysis import background_subtraction
from glob import glob
from copy import deepcopy
from scipy.stats import truncnorm

# Default paths (data for train/valid set)
UTILS_PATH = os.path.join(os.environ['HOME'], 'ms', 'utils')
REFERENCE_PATH = os.path.join(UTILS_PATH, 'GPX-002_SS reference with Peak Start-End time.csv')
TIDY_TRANSITION_LIST = os.path.join(UTILS_PATH, 'Tidy_transitonlist.csv')
LABEL_PATH = os.path.join(UTILS_PATH, 'GPX002 Processed Data with additional features_180727_refined.csv')
DATA_PATH = os.path.join(os.environ['HOME'], 'hd', 'GPX002_RAW')

# Helper functions for mzXML preprocessing, deprecated
def product_mz(s):
  return float(s['m/z array'][0])
  
def precursor_mz(s):
  return float(s['precursorMz'][0]['precursorMz'])
  
def rt(s):
  return float(s['retentionTime'])
  
def intensity(s):
  return float(s['intensity array'][0])

def simplify(s):
  new_s = {}
  new_s['m/z array'] = s['m/z array']
  new_s['precursorMz'] = s['precursorMz']
  new_s['retentionTime'] = s['retentionTime']
  new_s['intensity array'] = s['intensity array']
  return new_s

def load_references_from_table(path=REFERENCE_PATH, clean=False):
  """ Read reference transition list
      Output will be a dict In the structure of:
      {peak_name(str): [(0)precursor_mz(float), 
                        (1)product_mz(float), 
                        (2)rt(float),
                        (3)rt_start(float),
                        (4)rt_end(float),
                        (5)rt_width(float),
                        (6)valid(float, 0 or 1)]}

  path: str, optional
      path to the transition list
  clean: bool, optional
      if to remove peaks not appearing in `TIDY_TRANSITION_LIST`
  """
  data = np.array(pd.read_csv(path))
  n_peaks = data.shape[0]
  all_peaks = {}
  if clean:
    # Excluding all peaks mentioned in the `tidy_list`, the path is now fixed
    tidy_list = np.unique(np.array(pd.read_csv(TIDY_TRANSITION_LIST, header=None)))
  for i in range(n_peaks):
    peak_name = data[i, 0]
    if (not clean) or (clean and peak_name not in tidy_list):
      all_peaks[peak_name] = [data[i, 1], # precursor_mz
                              data[i, 2], # product_mz
                              data[i, 4], # rt
                              data[i, 5], # rt_start
                              data[i, 6], # rt_end
                              data[i, 7], # rt_width
                              # validity, invalid if zero response or no peak selected
                              ((data[i, 4] == data[i, 4]) & (data[i, 7]>0)) * 1]
      if not clean:
        all_peaks[peak_name][6] = 1
  return all_peaks

def load_sample_labels_from_table(path=LABEL_PATH, 
                                  reference_path=REFERENCE_PATH, 
                                  check_reference=True,
                                  clean=False,
                                  log=False):
  """ Read labels
      Output will be a dict in the structure of:
      {(sample_name(str), peak_name(str)): 
           [(0)response(float), 
            (1)rt_start(float), 
            (2)rt_end(float), 
            (3)S/N(float), 
            (4)width(float),
            (5)rt(float), 
            (6)rt_difference(float)]
      Note the value structure is the standard structure for `profile`

  path: str, optional
      path to the label file
  reference_path: str, optional
      path to the transition list
  check_reference: bool, optional
      if to only choose peaks appearing in the transition list
  clean: bool, optional
      if to remove peaks not appearing in `TIDY_TRANSITION_LIST`
  log: bool, optional 
      if to print error information
  """
  if check_reference:
    # All selected transitions should be in the reference transition list
    references = load_references_from_table(reference_path, clean=clean)

  # TODO: move epsilon to arg
  epsilon = 1e-5
  peaks = {}
  labels = np.array(pd.read_csv(path, header=None, dtype=str))
  sample_names = labels[2:, 0].astype(str)
  peak_names = labels[0, 1:].reshape((-1, 7))[:, 0]
  peak_profiles = labels[2:, 1:].reshape((sample_names.size, peak_names.size, 7))
  for i, sample_name in enumerate(sample_names):
    for j, peak_name in enumerate(peak_names):
      peak_name = peak_name
      if check_reference:
        if not peak_name in references or references[peak_name][6] != 1:
          continue
      try:
        # Excluding peaks with low response/abundance
        if float(peak_profiles[i, j, 0]) > epsilon:
          p_p = peak_profiles[i, j].astype(float)
          # Excluding peaks whose rt_start/rt_end annotations 
          # are deviated from reference for more than 0.2 min
          assert references[peak_name][3] - p_p[1] < 0.2
          assert - references[peak_name][4] + p_p[2] < 0.2
          peaks[(sample_name, peak_name)] = p_p
      except Exception as e:
        if log:
          print("Error in loading peak %s %s" % (sample_name, peak_name))
          print(e)
  print("Effective peaks: %d" % len(peaks))
  return peaks

def load_signals(path):
  """ Load raw signals from mzXML/mzML:
      For mzXML, return a dict of:
        {retention_time(float): [signals(dict)]}
      For mzML, return a dict of:
        {(precursor_mz, product_mz): [signals(N*2 array of retention time/intensity pairs)]}

  path: str
      path to the raw signal file (.mzXML or .mzML)
  """
  if os.path.splitext(path)[1] == '.mzXML':
    reader = mzxml.read(path)
    all_signals = list(reader)
    signals_sorted = {}
    for s in all_signals:
      if rt(s) not in signals_sorted:
        signals_sorted[rt(s)] = []
      signals_sorted[rt(s)].append(simplify(s))
    return signals_sorted, 'mzXML'
  elif os.path.splitext(path)[1] == '.mzML':
    signals = {}
    file = MzMLFile()
    exp = MSExperiment()
    file.load(path, exp)
    Chromatograms = exp.getChromatograms()
    #assert len(list(Chromatograms[0])) == sum([len(list(ch)) for ch in Chromatograms[1:]])
    for ch in Chromatograms[1:]:
      precursor = ch.getPrecursor().getMZ()
      product = ch.getProduct().getMZ()
      if (precursor, product) not in signals:
        signals[(precursor, product)] = []
      signal = np.stack(ch.get_peaks(), 1)
      if pyopenms.__version__ >= '2.4.0': # Since 2.4.0, pyopenms generate peak signals in second (change to minute)
        signal[:, 0] = signal[:, 0]/60.
      signals[(precursor, product)].append(signal)
    return signals, 'mzML'

def select_peak(sorted_signals, 
                precursor, 
                product, 
                rt_start, 
                rt_end, 
                mz_window_size=0.15,
                rt_window_size=0.2,
                mode='mzXML'):
  """ Select the context of a specified transition(peak):
      return an N*2 array of retention time/intensity pairs

  sorted_signals: dict
      raw signals
  precursor: float
      reference precursor m/z
  product: float
      reference product m/z
  rt_start: float
      reference rt start
  rt_end: float
      reference rt end
  mz_window_size: float, optional
      acceptance threshold for precursor/product m/z
  rt_window_size: float, optional
      context size of the peak
  mode: string, 'mzXML' or 'mzML'
      format of the raw signals
  """
  width = rt_end - rt_start
  # rt_window_size before and after
  window_start = rt_start - min(3*width, rt_window_size)
  window_end = rt_end + min(3*width, rt_window_size)

  selected_signals = []
  if mode == 'mzXML':
    for key in sorted(sorted_signals.keys()):
      if key >= window_start and key <= window_end:
        p = sorted_signals[key]
        for s in p:
          if np.abs(precursor_mz(s) - precursor) < mz_window_size and \
             np.abs(product_mz(s) - product) < mz_window_size:
            selected_signals.append((rt(s), intensity(s)))
  elif mode == 'mzML':
    matching_keys = []
    for key in sorted_signals.keys():
      if np.abs(key[0] - precursor) < mz_window_size and \
         np.abs(key[1] - product) < mz_window_size:
        matching_keys.append(key)
    matching_signals = []
    for key in matching_keys:
      for tr in sorted_signals[key]:
        if rt_start > min(tr[:, 0]) - 0.01 and rt_end < max(tr[:, 0]) + 0.01:
          matching_signals.append((key, tr))
    # Pick the trajectory whose center is closest to the desired window
    # TODO: better way to distinguish multiple trajectories with almost same m/z and rt?
    if len(matching_signals) == 0:
      return None
    matching_signals.sort(key=lambda x: np.abs(np.mean(x[1][:, 0]) - (rt_start+rt_end)/2))
    for pair in matching_signals[0][1]:
      if pair[0] >= window_start and pair[0] <= window_end: 
        selected_signals.append(pair)
    selected_signals.sort(key=lambda x: x[0])
  return np.stack(selected_signals)
  

def check_coelute(peak_signals, peak_prof, rt_window_size=0.2):
  """ Check if there is coeluting peak in the window, cut it off if required.
      

  peak_signals: np.array
      N*2 array of retention time/intensity pairs
  peak_prof: list
      peak profile
  rt_window_size: float, optional
      context size of the peak
  """
  # Cut left/right margin if significant co-eluting peaks exist
  signals = background_subtraction(peak_signals, peak_prof)
  rt_start = peak_prof[1]
  rt_end = peak_prof[2]
  signal_max = np.max((signals[:, 0] >= rt_start) * (signals[:, 0] <= rt_end) * signals[:, 1])

  if signal_max < 200:
    return peak_signals # Omit low intensity signals
  left_end = np.argmax(signals[:, 0] * (signals[:, 0] < rt_start))
  if signals[0, 1] > signal_max:
    for left_start in range(0, left_end):
      if signals[left_start, 1] < 0.5*signal_max:
        break
    print("Coeluting peak on the left: %f, %d" % (signals[0, 1]/signal_max, left_start))
  else:
    left_start = 0
    
  right_end = np.argmin(signals[:, 0] * (signals[:, 0] > rt_end) + \
                        (signals[:, 0] + signals[-1, 0]) * (signals[:, 0] <= rt_end))
  if signals[-1, 1] > signal_max:
    for right_start in range(len(signals)-1, right_end-1, -1):
      if signals[right_start, 1] < 0.5*signal_max:
        break
    print("Coeluting peak on the right: %f, %d" % (signals[-1, 1]/signal_max, len(signals) - 1 - right_start))
  else:
    right_start = len(signals) - 1

  signals_curated = signals[left_start:(right_start+1)]
  return signals_curated

def select_all_peaks(sorted_signals, 
                     peaks, 
                     references,
                     mz_window_size=0.15,
                     rt_window_size=0.2,
                     remove_coelute=False,
                     mode='mzXML'):
  """ Select all peaks in a file:
      return a list of (X, profile)
        X: array of retention time/intensity pairs
        profile: peak profile in standard structure

  sorted_signals: dict
      raw signals
  peaks: list
      list of (sample_name, peak_name)
  references: dict
      reference transition profiles
  mz_window_size: float, optional
      acceptance threshold for precursor/product m/z
  rt_window_size: float, optional
      context size of the peak
  remove_coelute: bool, optional
      if to remove coelution peaks
  mode: string, 'mzXML' or 'mzML'
      format of the raw signals
  """
  outs = []
  for p in peaks:
    peak_name = p[1]
    peak_prof = references[peak_name]
    try:
      peak_signals = select_peak(sorted_signals,
                                 peak_prof[0], # precursor_mz
                                 peak_prof[1], # product_mz
                                 peak_prof[3], # rt_start
                                 peak_prof[4], # rt_end
                                 mz_window_size=mz_window_size,
                                 rt_window_size=rt_window_size,
                                 mode=mode)
      assert not peak_signals is None
      if remove_coelute:
        peak_signals = check_coelute(peak_signals, [None, peak_prof[3], peak_prof[4]])
      outs.append((peak_signals, p))
    except:
      pass
  return outs

def generate_train_labels(X, profiles, dist_factor=20., itgr_factor=2.):
  """ Generate labels for training with label smoothing

  X: np.array
      N*2 array of retention time/intensity pairs
  profiles: list
      peak profile
  dist_factor: float
      importance factor for predicted boundary differences
      smaller value => larger importance
  itgr_factor: float
      importance factor for predicted abundance differences

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

def rt_shift_augmentation(data, shift_sampling='uniform', threshold=15, seed=None):
  """ Generate augmented data pair by shifting retention time

  data: list
      list of standard input pairs
  shift_sampling: str, optional
      distribution from which rt shift is sampled
  threshold: int, optional
      upper/lower threshold for rt shift in the augmentation
  seed: int, optional
      random seed
  """
  out_data = []
  if seed is not None:
    np.random.seed(seed)
  for d in data:
    if shift_sampling == 'uniform':
      shift = np.random.randint(-threshold, threshold+1)
    elif shift_sampling == 'gaussian':
      shift = int(truncnorm.rvs(-2, 2) * threshold/2)
    else:
      shift = shift_sampling()
    # data augmentation: shift input intensities
    X = d[0]
    new_X = deepcopy(X)
    new_X[:, 1] = np.concatenate([X[shift:, 1], X[:shift, 1]])
    step_size = np.median(X[1:, 0] - X[:-1, 0])
    new_prof = deepcopy(d[2])
    new_prof[1] -= shift * step_size # rt_start
    new_prof[2] -= shift * step_size # rt_end
    new_prof[5] -= shift * step_size # rt
    new_prof[6] -= shift * step_size # rt_difference
    
    new_y = generate_train_labels(new_X, new_prof)
    out_data.append((new_X, new_y, new_prof, d[3]))
  return out_data

def weights_by_intensity(data, max_val=120., min_val=50.):
  """ Generate sample weights according to their max signal intensity 
  
  data: list
      list of input pairs: (X, y, profile)
  max_val: float, optional
      threshold above which sample will be given weight of 1
  min_val: float, optional
      threshold under which sample will be given weight of 0
  """
  weights = []
  for d in data:
    ind_start, ind_end = list(np.argmax(d[1], 0))
    max_peak_value = max(d[0][ind_start:(ind_end+1), 1])
    weight = max((min(max_peak_value, max_val) - min_val)/(max_val - min_val), 0.)
    weights.append(weight)
  return weights

def baseline_adjust(X, baseline=0.):
  """ Adjust baseline of trajectory to designated value
  
  X: np.array
      N*2 array of retention time/intensity pairs
  baseline: float
      Standard baseline intensity
  """
  baseline_orig = np.min(X[:, 1])
  if baseline_orig < baseline - 2.:
    # Allowing for fluctuation of 2
    adjust_value = baseline - baseline_orig
    X[:, 1] = X[:, 1] + adjust_value
  return X
  
def load_sample(path=DATA_PATH,
                label_path=LABEL_PATH,
                reference_path=REFERENCE_PATH,
                train_labels=True,
                clean=True,
                remove_coelute=False,
                mz_window_size=0.15,
                rt_window_size=0.2,
                mode='mzML',
                reload=True):
  """ Load samples from signals and transition list
      return a list of standard input pairs: (X, y, profile, name)
        X: array of retention time/intensity pairs
        y: array of smoothed peak start/end labels
        profile: peak profile in standard structure
        name: tuple of sample name and peak name

  path: str (folder)
      path to the folder of signal files(.mzXML/.mzML)
  label_path: str (.csv file)
      path to the label file, None if no label is available
  reference_path: str (.csv file)
      path to the reference transition list file
      if no label is provided, all transitions in this list will be extracted
  train_labels: bool
      if to generate labels for training
      only available when label_path is not None
  clean: bool
      if to exclude peaks from TIDY_TRANSITION_LIST
  remove_coelute: bool, optional
      if to remove coelution peaks
  mz_window_size: float, optional
      acceptance threshold for precursor/product m/z
  rt_window_size: float, optional
      context size of the peak
  mode: string, 'mzXML' or 'mzML'
      format of the raw signals
  reload: bool
      if to reuse processed signals for each sample  
  """
  
  assert reference_path is not None
  assert path is not None
  try:
    print("Loading transition list from %s" % reference_path)
    ref = load_references_from_table(reference_path, clean=clean)
  except Exception:
    raise RuntimeError("Error in loading transition list %s" % reference_path)
  
  sample_paths = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.' + mode))]
  sample_names = [os.path.splitext(os.path.split(p)[1])[0] + '.d' for p in sample_paths]
  assert len(sample_names) > 0, "No input files found"
  
  if label_path is not None:
    try:
      print("Loading labels from %s" % label_path)
      labels = load_sample_labels_from_table(path=label_path, 
                                             reference_path=reference_path,
                                             clean=clean,
                                             check_reference=True)
    except Exception:
      raise RuntimeError("Error in loading label file %s" % label_path)
    # Only load samples appeared in the labels
    valid_sample_names = set([key[0] for key in labels.keys()])
    with_label = True
  else:
    labels = None
    valid_sample_names = set(sample_names)
    with_label = False
  
  
  out_samples = []
  for name in sample_names:
    if name in valid_sample_names:
      # Cached processed data files
      if clean:
        sample_file_name = os.path.join(path, name+'_clean.pkl')
      else:
        sample_file_name = os.path.join(path, name+'.pkl')
      
      # Load and curate signals
      if not os.path.exists(sample_file_name) or not reload:
        print ("Reading sample %s" % name)
        file_path = sample_paths[sample_names.index(name)]
        # Load raw signals
        try:
          sample_signals, mode = load_signals(file_path)
        except Exception:
          raise RuntimeError("Error in loading input file %s" % file_path)
        if with_label:
          # Select transitions in the label list
          sample_peaks = [key for key in labels.keys() if key[0] == name]
        else:
          # Enumerate all peaks in the reference list
          sample_peaks = [(name, peak_name) for peak_name in ref.keys()]
        pairs = select_all_peaks(sample_signals, 
                                 sample_peaks, 
                                 ref,
                                 mz_window_size=mz_window_size,
                                 rt_window_size=rt_window_size,
                                 remove_coelute=remove_coelute,
                                 mode=mode)
        if reload:
          with open(sample_file_name, 'wb') as f:
            pickle.dump(pairs, f)
      else:
        print ("Found sample %s" % name)
        pairs = pickle.load(open(sample_file_name, 'rb'))
        
      # Add labels
      for pair in pairs:
        X = pair[0]
        names = pair[1]
        if with_label:
          profiles = labels[names]
          if profiles[1] < np.min(X[:, 0]) - 0.01 or profiles[2] > np.max(X[:, 0]) + 0.01:
            # Neglecting this pair due to out of scope labels
            continue
          if train_labels:
            y = generate_train_labels(X, profiles)
          else:
            y = profiles[1:2]
        else:
          ref_p = ref[names[1]]
          # Build pseudo-labels, indicated by -1 abundance
          profiles = np.array([-1., ref_p[3], ref_p[4], -1., ref_p[5], -1., 0.])
          y = None
        # Manually adjusting baseline to 50 if not met (models trained with baseline 50)
        X = baseline_adjust(X, baseline=50.)
        out_samples.append((X, y, profiles, names))
  if reload:
    if clean:
      with open(os.path.join(path, 'featurized_dataset_clean.pkl'), 'wb') as f:
        pickle.dump(out_samples, f)
    else:
      with open(os.path.join(path, 'featurized_dataset.pkl'), 'wb') as f:
        pickle.dump(out_samples, f)
  return out_samples


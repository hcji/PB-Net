#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 12:19:34 2018

@author: zqwu
"""

import numpy as np

def baseline(pair, thr=0.04, grad_thr=0.001):
  X = pair[0]
  grad = X[1:, 1] - X[:-1, 1]
  grad = np.concatenate([grad, np.array(grad[-1]).reshape((1,))])
  apex = np.argmax(X[:, 1])
  minimum = np.sort(X[:, 1])[int(X.shape[0]*0.05)]
  threshold = minimum + (X[apex, 1] - minimum) * thr
  left = apex
  for left in range(apex, 0, -1):
    if X[left, 1] < threshold:
      break
  start = left
  for start in range(left, 0, -1):
    if grad[start] < np.max(grad)*grad_thr:
      break
  right = apex
  for right in range(apex, X.shape[0]):
    if X[right, 1] < threshold:
      break
  end = right
  for end in range(right, X.shape[0]):
    if grad[end] > np.min(grad)*grad_thr:
      break
  pred = np.zeros_like(X)
  pred[start, 0] = 1
  pred[end, 1] = 1
  return pred

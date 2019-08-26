#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:48:03 2018

@author: zqwu
"""


import torch as t
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from copy import deepcopy
import numpy as np
from sklearn.metrics import r2_score
from biLSTM import MatchLoss
from torch.utils.data import Dataset, DataLoader
import os
from data_utils import generate_train_labels

class BinFeaturizer(object):
  """ Featurize retention time/intensity pair by discretizing over a range of value """
  def __init__(self, 
               rt_low=-1., 
               rt_high=1.,
               rt_feat_size=128,
               int_low=0.,
               int_high=500.,
               int_feat_size=256):
    """
    rt_low: float, optional
        lower boundary for retention time
    rt_high: float, optional
        higher boundary for retention time
    rt_feat_size: int, optional
        number of gaussian kernels/features for retention time
    int_low: float, optional
        lower boundary for intensity
    int_high: float, optional
        higher boundary for intensity
    int_feat_size: int, optional
        number of gaussian kernels/features for intensity
    """
    self.rt_bins = np.linspace(rt_low, rt_high, rt_feat_size)
    self.int_bins = np.linspace(int_low, int_high, int_feat_size)
    self.width_rt = (self.rt_bins[1] - self.rt_bins[0])/2.
    self.width_int = (self.int_bins[1] - self.int_bins[0])/2.
    
  def __call__(self, inputs):
    """ 
    inputs: np.array
        batch of input features, in the shape of (seq_len, batch_size, 2)
    """
    assert inputs.shape[2] == 2
    rt_feat = np.exp(-(inputs[:, :, 0:1] - self.rt_bins.reshape((1, 1, -1)))**2/(2*self.width_rt**2))
    int_feat = np.exp(-(inputs[:, :, 1:2] - self.int_bins.reshape((1, 1, -1)))**2/(2*self.width_int**2))
    outs = np.concatenate([rt_feat, int_feat], axis=2)    
    return outs
  
class BatchSpectraDataset(Dataset):
  """ Dataset object for featurization of data loading """
  def __init__(self, data_batches, featurizers=[]):
    """
    data_batches: list
        list of minibatches
    featurizers: list, optional
        list of featurizer, should correspond to each element in a batch in the same order
    """
    batch_lengths = []
    for batch in data_batches:
      # All elements in a batch should have same size in batch_size dimension
      assert len(set([item.shape[1] for item in batch])) == 1
      batch_lengths.append(batch[0].shape[1])
    # All batches should have same batch size
    assert len(set(batch_lengths)) == 1
    self.batch_size = batch_lengths[0]
    self.batches = data_batches
    self.length = len(data_batches)
    self.n_items = len(self.batches[0])
    self.featurizers = featurizers
    assert len(self.featurizers) <= self.n_items
    # Padding featurizer list with None
    self.featurizers.extend([None]*(self.n_items - len(self.featurizers)))
  
  def __len__(self):
    return self.length

  def __getitem__(self, index):
    if index >= self.length:
      raise IndexError
    out_batch = []
    for item, featurizer in zip(self.batches[index], self.featurizers):
      if featurizer is not None:
        # Featurize element in a batch
        out_item = featurizer(item)
      else:
        out_item = item
      out_item = Variable(t.from_numpy(out_item).float())
      out_batch.append(out_item)
    return out_batch

class Trainer(object):
  """ Default trainer for the network """
  def __init__(self, 
               net, 
               opt, 
               criterion=MatchLoss(),
               rt_feat_size=128,
               intensity_feat_size=256,
               featurize=True):
    """
    net: callable pytorch function/module
        network to be trained
    opt: Config class
        configuraion of training parameters
    criterion: callable pytorch function/module, optional
        loss function
    rt_feat_size: int, optional
        number of gaussian kernels/features for retention time
    int_feat_size: int, optional
        number of gaussian kernels/features for intensity
    featurize: bool, optional
        if to featurize (discretize) inputs
    """
    self.net = net
    self.opt = opt
    self.criterion = criterion
    self.rt_feat_size = rt_feat_size # n_bins for rt
    self.intensity_feat_size = intensity_feat_size # n_bins for intensity
    if featurize:
      self.featurizer = BinFeaturizer(self.opt.lower_limit,
                                      self.opt.upper_limit,
                                      self.rt_feat_size,
                                      0.,
                                      self.opt.signal_max,
                                      self.intensity_feat_size)
    else:
      self.featurizer = None
    if self.opt.gpu:
      self.net = self.net.cuda()
      self.criterion = self.criterion.cuda()
  
  def assemble_batch(self, data, sample_weights=None, batch_size=None, sort=True):
    """ Assemble data into batches
    
    data: list
        list of standard input structure (X, y, profile, name)
    sample_weights: list, optional
        list of float, assigning training weights to samples
    batch_size: None or int, optional
        specify batch size if given
    sort: bool, optional
        if to reorder samples in the data to ease training
    """
    if batch_size is None:
      batch_size = self.opt.batch_size
    if sample_weights is None:
      sample_weights = [1] * len(data)
    
    # Sort by length
    if sort:
      lengths = [x[0].shape[0] for x in data]
      order = np.argsort(lengths)
      data = [data[i] for i in order]
      sample_weights = [sample_weights[i] for i in order]
    
    # Assemble samples with similar lengths to a batch
    data_batches = []
    for i in range(int(np.ceil(len(data)/float(batch_size)))):
      batch = data[i * batch_size:min((i+1) * batch_size, len(data))]
      batch_weights = sample_weights[i * batch_size:min((i+1) * batch_size, len(data))]
      batch_length = max([sample[0].shape[0] for sample in batch])
      out_batch_X = []
      out_batch_y = []
      for sample in batch:
        sample_length = sample[0].shape[0]
        # center retention time to the reference apex/peak argmax(if reference not given)
        rt = sample[2][5]
        if rt < 0:
          # when reference apex not given
          rt = sample[0][np.argmax(sample[0][:, 1]), 0]
        X = deepcopy(sample[0][:])
        X[:, 0] = X[:, 0] - rt
        # pad inputs in a batch to the same sequence length
        out_batch_X.append(np.pad(X, ((0, batch_length - sample_length), (0, 0)), 'constant'))
        if sample[1] is not None:
          y = deepcopy(sample[1][:])
          out_batch_y.append(np.pad(y, ((0, batch_length - sample_length), (0, 0)), 'constant'))
        else:
          out_batch_y.append(np.zeros_like(out_batch_X[-1]))
        
      if len(batch_weights) < batch_size:
        pad_length = batch_size - len(batch_weights)
        # pad samples to keep batch size consistent
        batch_weights.extend([0.] * pad_length)
        out_batch_X.extend([out_batch_X[0]] * pad_length)
        out_batch_y.extend([out_batch_y[0]] * pad_length)
        
      data_batches.append((np.stack(out_batch_X, axis=1), 
                           np.stack(out_batch_y, axis=1), 
                           np.array(batch_weights).reshape((1, -1, 1))))
    return data_batches
  
  def save(self, path):
    t.save(self.net.state_dict(), path)
  
  def load(self, path):
    s_dict = t.load(path, map_location=lambda storage, loc: storage)
    self.net.load_state_dict(s_dict)
 
  def set_seed(self, seed):
    t.manual_seed(seed)
    if self.opt.gpu:
      t.cuda.manual_seed_all(seed)
 
  def train(self, train_data, sample_weights=None, n_epochs=None, **kwargs):
    self.run_model(train_data, sample_weights=sample_weights, train=True, n_epochs=n_epochs, **kwargs)
    return
  
  def display_loss(self, train_data, **kwargs):
    return self.run_model(train_data, train=False, n_epochs=1, **kwargs)
    
  def run_model(self, data, sample_weights=None, train=False, n_epochs=None, **kwargs):
    """ Train/calculate total loss

    data: list
        list of standard input structure (X, y, profile, name)
    sample_weights: list, optional
        list of float, assigning training weights to samples
    train: bool, optional
        if in train mode
    n_epochs: None or int, optional
        specify number of epochs if given
    """
    if train:
      optimizer = Adam(self.net.parameters(),
                       lr=self.opt.lr,
                       betas=(.9, .999))
      self.net.zero_grad()
      epochs = self.opt.max_epoch
    else:
      epochs = 1
    if n_epochs is not None:
      epochs = n_epochs
    n_points = len(data)
    data_batches = self.assemble_batch(data, sample_weights=sample_weights)
    dataset = BatchSpectraDataset(data_batches, 
                                  featurizers=[self.featurizer, None, None])
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    for epoch in range(epochs):      
      loss = []
      print ('start epoch {epoch}'.format(epoch=epoch))
      for batch in data_loader:
        if self.opt.gpu:
          for i, item in enumerate(batch):
            # Omit the first dimension (created by data_loader)
            batch[i] = item[0].cuda()
        # inp: seq_len * batch_size * 2
        # label: seq_len * batch_size * 2
        # sample_weights: 1 * batch_size * 1
        inp, label, sample_weights = batch
        # loss ratio to account for class imbalance
        ratio = 1/label.mean()
        # output: seq_len * batch_size * 2
        output = self.net(inp, dataset.batch_size)
        error = self.criterion(label, output, weights=sample_weights, ratio=ratio)
        loss.append(error)
        error.backward()
        if train:
          optimizer.step()
          self.net.zero_grad()
      print ('epoch {epoch} loss: {loss}'.format(epoch=epoch, loss=sum(loss).data[0]/n_points))
    return loss
      
  def predict(self, test_data):
    """ Generate predictions

    test_data: list
        list of standard input structure (X, y, profile, name)
    """
    test_batches = self.assemble_batch(test_data, batch_size=1, sort=False)
    dataset = BatchSpectraDataset(test_batches, 
                                  featurizers=[self.featurizer, None, None])
    preds = []
    for sample in dataset:
      if self.opt.gpu:
        for i, item in enumerate(sample):
          sample[i] = item.cuda() # No extra first dimension
      inp, label, sample_weights = sample
      preds.append(self.net.predict(inp, 1, self.opt.gpu))
    return preds
  
class ReferenceTrainer(Trainer):
  def assemble_batch(self, 
                     data_ref,
                     sample_weights=None, 
                     batch_size=None, 
                     sort=True,
                     augment=False):
    """ Assemble data into batches
    
    data_ref: (`data`, `ref`)
      `data` is the same list of input structures
      `ref` is a dict of {peak_name: reference input structure}
    sample_weights: list, optional
        list of float, assigning training weights to samples
    batch_size: None or int, optional
        specify batch size if given
    sort: bool, optional
        if to reorder samples in the data to ease training
    augment: bool, optional
        if to use data augmentation(RT shift) for training
    """
    if augment:
      print("Augmenting")
    data, ref = data_ref
    if batch_size is None:
      batch_size = self.opt.batch_size
    if sample_weights is None:
      sample_weights = [1] * len(data)
    # Sort by length
    if sort:
      lengths = [x[0].shape[0] for x in data]
      order = np.argsort(lengths)
      data = [data[i] for i in order]
      sample_weights = [sample_weights[i] for i in order]
    
    # Assemble samples with similar lengths to a batch
    data_batches = []
    for i in range(int(np.ceil(len(data)/float(batch_size)))):
      batch = data[i * batch_size:min((i+1) * batch_size, len(data))]
      batch_weights = sample_weights[i * batch_size:min((i+1) * batch_size, len(data))]
      batch_ref = [ref[d[3][1]] for d in batch]
      
      # Note that batch_length and batch_ref_length might not be the same
      batch_length = max([sample[0].shape[0] for sample in batch])
      batch_ref_length = max([sample[0].shape[0] for sample in batch_ref])

      out_batch_X = []
      out_batch_y = []
      out_batch_ref_X = []
      out_batch_ref_y = []
      out_batch_att = []
      for sample, sample_ref in zip(batch, batch_ref):
        sample_length = sample[0].shape[0]
        sample_ref_length = sample_ref[0].shape[0]
        # center retention time in query and reference sample
        rt = sample[2][5]
        if rt < 0:
          rt = sample[0][np.argmax(sample[0][:, 1]), 0]
        X = deepcopy(sample[0])
        y = deepcopy(sample[1])
        prof = deepcopy(sample[2])
        
        step_size = np.median(X[1:, 0] - X[:-1, 0])
        if augment:
          offset = np.round(np.random.randint(-10, 11)).astype(int)
        else:
          offset = 0
        if offset != 0:
          # data augmentation: shift input intensities
          X[:, 1] = np.concatenate([X[offset:, 1], X[:offset, 1]])
          prof[1] -= offset * step_size
          prof[2] -= offset * step_size
          rt -= offset * step_size

        # query sample
        if y is not None:
          y = generate_train_labels(X, prof)
        X[:, 0] = X[:, 0] - rt
        out_batch_X.append(np.pad(X, ((0, batch_length - sample_length), (0, 0)), 'constant'))
        if y is not None:
          out_batch_y.append(np.pad(y, ((0, batch_length - sample_length), (0, 0)), 'constant'))
        else:
          out_batch_y.append(np.zeros_like(out_batch_X[-1]))
        
        # reference sample (labels required)
        X_ref = deepcopy(sample_ref[0])
        X_ref[:, 0] = X_ref[:, 0] - rt
        out_batch_ref_X.append(np.pad(X_ref, ((0, batch_ref_length - sample_ref_length), (0, 0)), 'constant'))
        y_ref = deepcopy(sample_ref[1])
        out_batch_ref_y.append(np.pad(y_ref, ((0, batch_ref_length - sample_ref_length), (0, 0)), 'constant'))
        
        # ground truth attention map to regularize training
        att_lab = np.exp(-5e3*np.square(X[:, 0:1] + offset * step_size - np.transpose(X_ref[:, 0:1])))
        att_lab = att_lab/np.sum(att_lab, 1, keepdims=True)
        att_lab = np.pad(att_lab,
                         ((0, batch_length - sample_length), (0, batch_ref_length - sample_ref_length)),
                         'constant')
        out_batch_att.append(att_lab)
        
      if len(batch_weights) < batch_size:
        pad_length = batch_size - len(batch_weights)
        batch_weights.extend([0.] * pad_length)
        out_batch_X.extend([out_batch_X[0]] * pad_length)
        out_batch_y.extend([out_batch_y[0]] * pad_length)
        out_batch_ref_X.extend([out_batch_ref_X[0]] * pad_length)
        out_batch_ref_y.extend([out_batch_ref_y[0]] * pad_length)
        out_batch_att.extend([out_batch_att[0]] * pad_length)
        
      data_batches.append((np.stack(out_batch_X, axis=1), 
                           np.stack(out_batch_y, axis=1), 
                           np.stack(out_batch_ref_X, axis=1), 
                           np.stack(out_batch_ref_y, axis=1), 
                           np.array(batch_weights).reshape((1, -1, 1)),
                           np.stack(out_batch_att, axis=1)))
    return data_batches
    
  def run_model(self, 
                data_ref, 
                sample_weights=None, 
                train=False, 
                n_epochs=None,
                use_att_label=False,
                augment=False,
                **kwargs):
    """ Train/calculate total loss

    data_ref: (`data`, `ref`)
      `data` is the same list of input structures
      `ref` is a dict of {peak_name: reference input structure}
    sample_weights: list, optional
        list of float, assigning training weights to samples
    train: bool, optional
        if in train mode
    n_epochs: None or int, optional
        specify number of epochs if given
    use_att_label: bool, optional
        if to add attention map regularization
    augment: bool, optional
        if to use data augmentation(RT shift) for training
    """
    if train:
      optimizer = Adam(self.net.parameters(),
                       lr=self.opt.lr,
                       betas=(.9, .999))
      self.net.zero_grad()
      epochs = self.opt.max_epoch
    else:
      epochs = 1
    if n_epochs is not None:
      epochs = n_epochs
      
    n_points = len(data_ref[0])
    data_batches = self.assemble_batch(data_ref, sample_weights=sample_weights, augment=augment)
    dataset = BatchSpectraDataset(data_batches, 
                                  featurizers=[self.featurizer, None, self.featurizer, None, None, None])
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for epoch in range(epochs):
      loss = 0
      print ('start epoch {epoch}'.format(epoch=epoch))
      for batch in data_loader:
        if self.opt.gpu:
          for i, item in enumerate(batch):
            # Omit the first dimension (created by data_loader)
            batch[i] = item[0].cuda()
        # inp: seq_len * batch_size * 2
        # label: seq_len * batch_size * 2
        # ref_inp: ref_seq_len * batch_size * 2
        # ref_label: ref_seq_len * batch_size * 2
        # sample_weights: 1 * batch_size * 1
        # sample_att: seq_len * batch_size * ref_seq_len 
        inp, label, ref_inp, ref_label, sample_weights, sample_att = batch
        ratio = 1/label.mean()
        if not use_att_label:
          output = self.net(inp, ref_inp, ref_label, dataset.batch_size)
          error = self.criterion(label, output, weights=sample_weights, ratio=ratio)
        else:
          output, att_kld = self.net(inp, ref_inp, ref_label, dataset.batch_size, att_label=sample_att)
          error = self.criterion(label, output, weights=sample_weights, ratio=ratio) + att_kld
        loss += error
        error.backward()
        if train:
          optimizer.step()
          self.net.zero_grad()
      print ('epoch {epoch} loss: {loss}'.format(epoch=epoch, loss=loss.data[0]/n_points))
    
  def predict(self, test_data_ref):
    """ Generate predictions

    test_data_ref: (`data`, `ref`)
      `data` is the same list of input structures
      `ref` is a dict of {peak_name: reference input structure}
    """
    test_batches = self.assemble_batch(test_data_ref, augment=False, batch_size=1, sort=False)
    dataset = BatchSpectraDataset(test_batches, 
                                  featurizers=[self.featurizer, None, self.featurizer, None, None, None])
    preds = []
    for sample in dataset:
      if self.opt.gpu:
        for i, item in enumerate(sample):
          sample[i] = item.cuda() # No extra first dimension
      inp, label, ref_inp, ref_label, sample_weights, sample_att = sample
      preds.append(self.net.predict(inp, ref_inp, ref_label, 1, self.opt.gpu))
    return preds

  def attention_map(self, test_sample, ref):
    """ Extract attention map between a query sample and its reference
    
    test_sample: list
        standard input structure (X, y, profile, name)
    ref: list
        standard input structure for the reference sample
    """
    test_data_ref = [[test_sample], {test_sample[3][1]: ref}]
    test_batches = self.assemble_batch(test_data_ref, batch_size=1, sort=False)
    dataset = BatchSpectraDataset(test_batches, 
                                  featurizers=[self.featurizer, None, self.featurizer, None, None, None])
    assert dataset.length == 1
    for sample in dataset:
      if self.opt.gpu:
        for i, item in enumerate(sample):
          sample[i] = item.cuda() # No extra first dimension
      inp, label, ref_inp, ref_label, sample_weights, sample_att = sample
      lstm_outs = self.net.lstm_module(inp, batch_size=1)
      attention_outs = self.net.att_module(lstm_outs)
      lstm_outs_ref = self.net.lstm_module_ref(ref_inp, batch_size=1)
      attention_outs_ref = self.net.att_module_ref(lstm_outs_ref)
      pred, att = self.net.att_on_ref(Q_in=attention_outs,
                                      K_in=attention_outs_ref,
                                      V_in=ref_label)
      break
    # att: seq_len * ref_seq_len
    att = att[:, 0, :].detach().cpu().data.numpy()
    return att
      


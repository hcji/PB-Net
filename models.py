#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 15:40:34 2018

@author: zqwu
"""
from torch import nn
import torch as t
import torch.nn.functional as F
import numpy as np
from biLSTM import BiLSTM, MultiheadAttention, MatchLoss, Attention

class TestModel(nn.Module):
  def __init__(self,
               input_dim=256,
               hidden_dim_lstm=128,
               hidden_dim_attention=32,
               n_lstm_layers=2,
               n_attention_heads=8,
               gpu=True,
               random_init=True):
    """ Default reference-free model with LSTM and attention layers 

    input_dim: int, optional
        size of input feature dimension
    hidden_dim_lstm: int, optional
        size of hidden LSTM layer
    hidden_dim_attention: int, optional
        size of each head in the attention layer
    n_lstm_layers: int, optional 
        number of LSTM layers
    n_attention_heads: int, optional
        number of attention layer heads
    gpu: bool, optional
        if the model is run on GPU
    random_init: bool, optional
        if the initialize the LSTM hidden state randomly, for debug use
    """
    super(TestModel, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim_lstm = hidden_dim_lstm
    self.hidden_dim_attention = hidden_dim_attention
    self.n_lstm_layers = n_lstm_layers
    self.n_attention_heads = n_attention_heads
    self.gpu = gpu
    self.random_init = random_init

    self.lstm_module = BiLSTM(input_dim=self.input_dim,
                              hidden_dim=self.hidden_dim_lstm,
                              n_layers=self.n_lstm_layers,
                              gpu=self.gpu,
                              random_init=self.random_init)
    self.att_module = MultiheadAttention(Q_dim=self.hidden_dim_lstm,
                                         V_dim=self.hidden_dim_lstm,
                                         head_dim=self.hidden_dim_attention,
                                         n_heads=self.n_attention_heads)
    self.fc = nn.Sequential(
        nn.Linear(self.hidden_dim_lstm, 32),
        nn.ReLU(True),
        nn.Linear(32, 4))

  def forward(self, sequence, batch_size=1):
    """
    sequence: torch Tensor
        input batch of shape seq_len * batch_size * input_dim
    batch_size: int, optional
        batch size
    """
    lstm_outs = self.lstm_module(sequence, batch_size=batch_size)
    attention_outs = self.att_module(sequence=lstm_outs)
    outs = self.fc(attention_outs)
    outs = outs.view((-1, batch_size, 2, 2))
    return F.log_softmax(outs, 3)

  def predict(self, sequence, batch_size=1, gpu=True):
    """
    sequence: torch Tensor
        input batch of shape seq_len * batch_size * input_dim
    batch_size: int, optional
        batch size
    gpu: bool, optional
        if run on GPU
    """
    #assert batch_size == 1
    output = t.exp(self.forward(sequence, batch_size)[:, :, :, 1])
    if gpu:
      output = output.cpu()
    output = output.data.numpy()
    return output
    

class ReferenceModel(nn.Module):
  def __init__(self,
               input_dim=256,
               hidden_dim_lstm=128,
               hidden_dim_attention=32,
               n_lstm_layers=2,
               n_attention_heads=8,
               gpu=True,
               random_init=True):
    """ Reference-based model 

    input_dim: int, optional
        size of input feature dimension
    hidden_dim_lstm: int, optional
        size of hidden LSTM layer
    hidden_dim_attention: int, optional
        size of each head in the attention layer
    n_lstm_layers: int, optional 
        number of LSTM layers
    n_attention_heads: int, optional
        number of attention layer heads
    gpu: bool, optional
        if the model is run on GPU
    random_init: bool, optional
        if the initialize the LSTM hidden state randomly, for debug use
    """
    super(ReferenceModel, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim_lstm = hidden_dim_lstm
    self.hidden_dim_attention = hidden_dim_attention
    self.n_lstm_layers = n_lstm_layers
    self.n_attention_heads = n_attention_heads
    self.gpu = gpu
    self.random_init = random_init

    # Encoding of query sample
    self.lstm_module = BiLSTM(input_dim=self.input_dim,
                              hidden_dim=self.hidden_dim_lstm,
                              n_layers=self.n_lstm_layers,
                              gpu=self.gpu,
                              random_init=self.random_init)
    self.att_module = MultiheadAttention(Q_dim=self.hidden_dim_lstm,
                                         V_dim=self.hidden_dim_lstm,
                                         head_dim=self.hidden_dim_attention,
                                         n_heads=self.n_attention_heads)

    # Encoding of reference sample, same structure
    self.lstm_module_ref = BiLSTM(input_dim=self.input_dim,
                                  hidden_dim=self.hidden_dim_lstm,
                                  n_layers=self.n_lstm_layers,
                                  gpu=self.gpu,
                                  random_init=self.random_init)
    self.att_module_ref = MultiheadAttention(Q_dim=self.hidden_dim_lstm,
                                             V_dim=self.hidden_dim_lstm,
                                             head_dim=self.hidden_dim_attention,
                                             n_heads=self.n_attention_heads)

    # Attention module using samples as queries, reference(and reference labels) as keys and values
    self.att_on_ref = Attention(Q_dim=self.hidden_dim_lstm,
                                V_dim=2,
                                return_attention=True)

  def forward(self, sequence, ref_seq, ref_label, batch_size=1, att_label=None):
    """
    sequence: torch Tensor
        input batch of shape seq_len * batch_size * input_dim
    ref_seq: torch Tensor
        input reference batch of shape ref_seq_len * batch_size * input_dim
    ref_label: torch Tensor
        input reference label of shape ref_seq_len * batch_size * 2
    batch_size: int, optional
        batch size
    att_label: None or tensor Tensor
        used to calculate attention map regularization if given
        attention map of shape seq_len * batch_size * ref_seq_len
    """
    lstm_outs = self.lstm_module(sequence, batch_size=batch_size)
    attention_outs = self.att_module(lstm_outs)
    lstm_outs_ref = self.lstm_module_ref(ref_seq, batch_size=batch_size)
    attention_outs_ref = self.att_module_ref(lstm_outs_ref)
    
    pred, att = self.att_on_ref(Q_in=attention_outs,
                                K_in=attention_outs_ref,
                                V_in=ref_label)
    outs = t.stack([1-pred, pred], 3)
    if att_label is not None:
      KLD = 0.1 * t.sum(att_label * t.log((att_label + 1e-5)/(att + 1e-5)))
      return outs, KLD
    else:
      return outs

  def predict(self, sequence, ref_seq, ref_label, batch_size=1, gpu=True):
    """
    sequence: torch Tensor
        input batch of shape seq_len * batch_size * input_dim
    ref_seq: torch Tensor
        input reference batch of shape ref_seq_len * batch_size * input_dim
    ref_label: torch Tensor
        input reference label of shape ref_seq_len * batch_size * 2
    batch_size: int, optional
        batch size
    gpu: bool, optional
        if run on GPU
    """
    #assert batch_size == 1
    output = self.forward(sequence, ref_seq, ref_label, batch_size)[:, :, :, 1]
    if gpu:
      output = output.cpu()
    output = output.data.numpy()
    return output

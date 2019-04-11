#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 20:54:51 2018

@author: zqwu
script for train the model
"""
import numpy as np
from biLSTM import BiLSTM, MultiheadAttention, MatchLoss, MatchLossRaw
from models import TestModel, ReferenceModel
from trainer import Trainer, ReferenceTrainer
from data_utils import generate_train_labels, weights_by_intensity
import os
import pickle

class Config:
    lr = 0.001
    batch_size = 8
    max_epoch = 1 # =1 when debug
    workers = 2
    gpu = False # use gpu or not
    lower_limit = -1.0
    upper_limit = 1.0
    signal_max = 500
    
opt=Config()

ref = pickle.load(open('./utils/reference_peaks_dat.pkl', 'rb'))

test_input = pickle.load(open('./test_input.pkl', 'rb'))

# Pre-saved sequential PB-Net predictions on test input
#test_preds = pickle.load(open('./test_preds.pkl', 'rb'))

# Pre-saved reference-based PB-Net predictions on test input
#test_preds_ref = pickle.load(open('./test_preds_ref.pkl', 'rb'))

###
net = TestModel(input_dim=384,
                hidden_dim_lstm=128,
                hidden_dim_attention=32,
                n_lstm_layers=2,
                n_attention_heads=8,
                gpu=opt.gpu)
trainer = Trainer(net, opt, MatchLoss(), featurize=True)
trainer.load('./model-seq.pth')

# sequential PB-Net predictions
preds = trainer.predict(test_input)

###
net = ReferenceModel(input_dim=384,
                     hidden_dim_lstm=128,
                     hidden_dim_attention=32,
                     n_lstm_layers=2,
                     n_attention_heads=8,
                     gpu=opt.gpu)
trainer = ReferenceTrainer(net, opt, MatchLossRaw(), featurize=True)
test_inputs = [test_input, ref]
trainer.load('./model-ref.pth')

# reference-based PB-Net predictions
preds_ref = trainer.predict(test_inputs)
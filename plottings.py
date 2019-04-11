# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from data_analysis import eval_accuracy, eval_abundance_accuracy, eval_dists, calculate_abundance, calculate_confidence
from scipy.stats import spearmanr, pearsonr
import pickle
from baseline import baseline
import matplotlib.pyplot as plt
from models import TestModel, ReferenceModel
from trainer import Trainer, ReferenceTrainer
from biLSTM import MatchLoss, MatchLossRaw

font = {'weight': 'normal',
        'size': 16}
color_seq = (44./256, 83./256, 169./256, 1.)
color_ref = (69./256, 209./256, 163./256, 1.)

test_data = pickle.load(open('./test_input.pkl', 'rb'))
# Pre-saved sequential PB-Net predictions on test input
pred = pickle.load(open('./test_preds.pkl', 'rb'))
# Pre-saved reference-based PB-Net predictions on test input
pred_ref = pickle.load(open('./test_preds_ref.pkl', 'rb'))

###########################################################################
pred_bl = [baseline(p) for p in test_data]

intg = False
y_trues = []
y_preds = []
y_preds_ref = []
y_preds_bl = []
for i, sample in enumerate(test_data):
  output = pred[i]
  output_ref = pred_ref[i]
  output_bl = pred_bl[i]
  y_trues.append(calculate_abundance(sample[1], sample))
  y_preds.append(calculate_abundance(output, sample, intg=intg))
  y_preds_ref.append(calculate_abundance(output_ref, sample, intg=intg))
  y_preds_bl.append(calculate_abundance(output_bl, sample, intg=intg))
y_trues = np.array(y_trues)
y_preds = np.array(y_preds)
y_preds_ref = np.array(y_preds_ref)
y_preds_bl = np.array(y_preds_bl)

test_data_peaks = np.array([p[3][1] for p in test_data])
peaks = set(test_data_peaks)
lengths = []
r_scores = []
sr_scores = []
r_scores_ref = []
sr_scores_ref = []
r_scores_bl = []
sr_scores_bl = []
for peak in peaks:
  inds = np.where(test_data_peaks == peak)[0]
  lengths.append(len(inds))
  r_scores.append(pearsonr(np.log(y_trues[inds] + 1e-9), np.log(y_preds[inds] + 1e-9))[0])
  r_scores_ref.append(pearsonr(np.log(y_trues[inds] + 1e-9), np.log(y_preds_ref[inds] + 1e-9))[0])
  r_scores_bl.append(pearsonr(np.log(y_trues[inds] + 1e-9), np.log(y_preds_bl[inds] + 1e-9))[0])
  sr_scores.append(spearmanr(y_trues[inds], y_preds[inds]).correlation)
  sr_scores_ref.append(spearmanr(y_trues[inds], y_preds_ref[inds]).correlation)
  sr_scores_bl.append(spearmanr(y_trues[inds], y_preds_bl[inds]).correlation)

# Table 1 & 2
dists1 = eval_dists(test_data, pred_bl)
print('\n' + str(np.mean(dists1) * 0.6))
accuracy = eval_accuracy(test_data, pred_bl, 2)
print(np.mean(r_scores_bl))
print(np.mean(sr_scores_bl))
accuracy2 = eval_abundance_accuracy(test_data, pred_bl, 0.05)

dists2 = eval_dists(test_data, pred)
print('\n' + str(np.mean(dists2) * 0.6))
accuracy = eval_accuracy(test_data, pred, 2)
print(np.mean(r_scores))
print(np.mean(sr_scores))
accuracy2 = eval_abundance_accuracy(test_data, pred, 0.05)

dists3 = eval_dists(test_data, pred_ref)
print('\n' + str(np.mean(dists3) * 0.6))
accuracy = eval_accuracy(test_data, pred_ref, 2)
print(np.mean(r_scores_ref))
print(np.mean(sr_scores_ref))
accuracy2 = eval_abundance_accuracy(test_data, pred_ref, 0.05)

# Figure 2A
hs1, bins1 = np.histogram(np.array(dists1)*0.6, bins=np.arange(24)*0.5*0.6)
hs2, bins2 = np.histogram(np.array(dists2)*0.6, bins=np.arange(24)*0.5*0.6)
hs3, bins3 = np.histogram(np.array(dists3)*0.6, bins=np.arange(24)*0.5*0.6)
plt.clf()
bins = bins1[:-1]
plt.bar(bins-0.07, hs1, width=0.07, color=(237./256, 106./256, 90./256, 1.), label='Rule-based')
plt.bar(bins, hs2, width=0.07, color=(44./256, 83./256, 169./256, 1.), label='Sequential PB-Net')
plt.bar(bins+0.07, hs3, width=0.07, color=(69./256, 209./256, 163./256, 1.), label='Ref-based PB-Net')
plt.xlim(-0.2, 6.2)
plt.legend(fontsize=14)
plt.xlabel("Absolute Prediction Error (second)", fontdict=font)
plt.ylabel("Counts", fontdict=font)
plt.tight_layout()
plt.savefig('./figs/fig2_test_MAE_bar.png', dpi=300)

# Figure 2B
plt.clf()
bins = bins1[:-1]
plt.plot(bins, np.cumsum(hs1)/float(len(dists1)), 'o-', color=(237./256, 106./256, 90./256, 1.), label='Rule-based')
plt.plot(bins, np.cumsum(hs2)/float(len(dists1)), 'o-', color=(44./256, 83./256, 169./256, 1.), label='Sequential PB-Net')
plt.plot(bins, np.cumsum(hs3)/float(len(dists1)), 'o-', color=(69./256, 209./256, 163./256, 1.), label='Ref-based PB-Net')
plt.xlim(-0.2, 6.2)
plt.legend(fontsize=14)
plt.xlabel("Absolute Prediction Error (second)", fontdict=font)
plt.ylabel("Fraction of Data", fontdict=font)
plt.tight_layout()
plt.savefig('./figs/fig2_test_MAE_cdf.png', dpi=300)

# Figure 2C
plt.clf()
inds = np.where(y_preds > 0.2)[0]
plt.plot(-np.log(y_trues[0]), -np.log(y_preds_bl[0]), '.', markersize=5, color=(237./256, 106./256, 90./256, 1.0), label='Rule-based')
plt.plot(-np.log(y_trues[0]), -np.log(y_preds[0]), '.', markersize=5, color=(44./256, 83./256, 169./256, 1.0), label='Sequential PB-Net')
plt.plot(-np.log(y_trues[0]), -np.log(y_preds_ref[0]), '.', markersize=5, color=(69./256, 209./256, 163./256, 1.0), label='Ref-based PB-Net')
plt.legend(fontsize=14)
plt.scatter(np.log(y_trues[inds]), np.log(y_preds_bl[inds]), s=0.1, color=(237./256, 106./256, 90./256, 0.8))
plt.scatter(np.log(y_trues[inds]), np.log(y_preds[inds]), s=0.1, color=(44./256, 83./256, 169./256, 0.8))
plt.scatter(np.log(y_trues[inds]), np.log(y_preds_ref[inds]), s=0.1, color=(69./256, 209./256, 163./256, 0.8))
plt.plot(np.arange(-2, 18), np.arange(-2, 18), '-', linewidth=0.5, color=(0., 0., 0., 1.))
plt.xlim(-0.5, 15.5)
plt.ylim(-0.5, 15.5)
plt.xlabel("Log Abundance - Human Annotation", fontdict=font)
plt.ylabel("Log Abundance - Prediction", fontdict=font)
plt.tight_layout()
plt.savefig('./figs/fig2_test_abd_correlation.png', dpi=600)

#############################################################
y_trues = []
y_preds1 = []
y_preds2 = []
y_preds_ref1 = []
y_preds_ref2 = []
for i, sample in enumerate(test_data):
  output = pred[i]
  output_ref = pred_ref[i]
  y_trues.append(calculate_abundance(sample[1], sample))
  y_preds1.append(calculate_abundance(output, sample, intg=False))
  y_preds2.append(calculate_abundance(output, sample, intg=True))
  y_preds_ref1.append(calculate_abundance(output_ref, sample, intg=False))
  y_preds_ref2.append(calculate_abundance(output_ref, sample, intg=True))
y_trues = np.array(y_trues)
y_preds1 = np.array(y_preds1)
y_preds2 = np.array(y_preds2)
y_preds_ref1 = np.array(y_preds_ref1)
y_preds_ref2 = np.array(y_preds_ref2)

test_data_peaks = np.array([p[3][1] for p in test_data])
peaks = set(test_data_peaks)
lengths = []
r_scores1 = []
sr_scores1 = []
r_scores2 = []
sr_scores2 = []
r_scores_ref1 = []
sr_scores_ref1 = []
r_scores_ref2 = []
sr_scores_ref2 = []

for peak in peaks:
  inds = np.where(test_data_peaks == peak)[0]
  lengths.append(len(inds))
  r_scores1.append(pearsonr(np.log(y_trues[inds] + 1e-9), np.log(y_preds1[inds] + 1e-9))[0])
  sr_scores1.append(spearmanr(y_trues[inds], y_preds1[inds]).correlation)
  r_scores2.append(pearsonr(np.log(y_trues[inds] + 1e-9), np.log(y_preds2[inds] + 1e-9))[0])
  sr_scores2.append(spearmanr(y_trues[inds], y_preds2[inds]).correlation)
  r_scores_ref1.append(pearsonr(np.log(y_trues[inds] + 1e-9), np.log(y_preds_ref1[inds] + 1e-9))[0])
  sr_scores_ref1.append(spearmanr(y_trues[inds], y_preds_ref1[inds]).correlation)
  r_scores_ref2.append(pearsonr(np.log(y_trues[inds] + 1e-9), np.log(y_preds_ref2[inds] + 1e-9))[0])
  sr_scores_ref2.append(spearmanr(y_trues[inds], y_preds_ref2[inds]).correlation)

# Table 2(parenthesis values)
print(np.mean(r_scores1))
print(np.mean(r_scores2))
print(np.mean(r_scores_ref1))
print(np.mean(r_scores_ref2))

print(np.mean(sr_scores1))
print(np.mean(sr_scores2))
print(np.mean(sr_scores_ref1))
print(np.mean(sr_scores_ref2))

accuracy2 = eval_abundance_accuracy(test_data, pred, 0.05, intg=False)
accuracy2 = eval_abundance_accuracy(test_data, pred, 0.05, intg=True)
accuracy2 = eval_abundance_accuracy(test_data, pred_ref, 0.05, intg=False)
accuracy2 = eval_abundance_accuracy(test_data, pred_ref, 0.05, intg=True)

# Figure S2A
plt.clf()
inds = np.where(y_preds1 > 0.2)[0]
X_seq = np.concatenate([np.log(y_trues[inds]), np.log(y_trues[inds])])
Y_seq = np.concatenate([np.log(y_preds1[inds]), np.log(y_preds2[inds])])
color = np.array([(44./256, 83./256, 169./256, 0.8)] * len(inds) + [(207./256, 159./256, 54./256, 0.8)] * len(inds))
order = np.arange(len(X_seq))
np.random.shuffle(order)
plt.plot(-np.log(y_trues[0]), -np.log(y_preds1[0]), '.', markersize=5., color=(44./256, 83./256, 169./256, 1.), label='Sequential PB-Net, Argmax Abd')
plt.plot(-np.log(y_trues[0]), -np.log(y_preds2[0]), '.', markersize=5., color=(207./256, 159./256, 54./256, 1.), label='Sequential PB-Net, Weighted Abd')
plt.legend(fontsize=14)
plt.scatter(X_seq[order], Y_seq[order], s=0.1, color=color[order])
plt.plot(np.arange(-2, 18), np.arange(-2, 18), '-', linewidth=0.5, color=(0., 0., 0., 1.))
plt.xlim(-0.5, 15.5)
plt.ylim(-0.5, 15.5)
plt.xlabel("Log Abundance - Human Annotation", fontdict=font)
plt.ylabel("Log Abundance - Prediction", fontdict=font)
plt.tight_layout()
plt.savefig('./figs/figS2_weighted_abd_correlation1.png', dpi=600)

# Figure S2B
plt.clf()
inds = np.where(y_preds_ref1 > 0.2)[0]
X_seq = np.concatenate([np.log(y_trues[inds]), np.log(y_trues[inds])])
Y_seq = np.concatenate([np.log(y_preds_ref1[inds]), np.log(y_preds_ref2[inds])])
color = np.array([(69./256, 209./256, 163./256, 0.8)] * len(inds) + [(246./256, 75./256, 201./256, 0.8)] * len(inds))
order = np.arange(len(X_seq))
np.random.shuffle(order)
plt.plot(-np.log(y_trues[0]), -np.log(y_preds_ref1[0]), '.', markersize=5., color=(69./256, 209./256, 163./256, 1.), label='Ref-based PB-Net, Argmax Abd')
plt.plot(-np.log(y_trues[0]), -np.log(y_preds_ref2[0]), '.', markersize=5., color=(246./256, 75./256, 201./256, 1.), label='Ref-based PB-Net, Weighted Abd')
plt.legend(fontsize=14)
plt.scatter(X_seq[order], Y_seq[order], s=0.1, color=color[order])
plt.plot(np.arange(-2, 18), np.arange(-2, 18), '-', linewidth=0.5, color=(0., 0., 0., 1.))
plt.xlim(-0.5, 15.5)
plt.ylim(-0.5, 15.5)
plt.xlabel("Log Abundance - Human Annotation", fontdict=font)
plt.ylabel("Log Abundance - Prediction", fontdict=font)
plt.tight_layout()
plt.savefig('./figs/figS2_weighted_abd_correlation2.png', dpi=600)

############################################################################
y_trues = []
y_preds = []
y_preds_ref = []
confs = []
confs_ref = []
for i, sample in enumerate(test_data):
  output = pred[i]
  output_ref = pred_ref[i]
  y_trues.append(calculate_abundance(sample[1], sample))
  y_preds.append(calculate_abundance(output, sample, intg=False))
  y_preds_ref.append(calculate_abundance(output_ref, sample, intg=False))
  confs.append(calculate_confidence(output[:, 0], thr=0.2) * \
               calculate_confidence(output[:, 1], thr=0.2))
  confs_ref.append(calculate_confidence(output_ref[:, 0], thr=0.2) * \
                   calculate_confidence(output_ref[:, 1], thr=0.2))

y_trues = np.array(y_trues)
y_preds = np.array(y_preds)
y_preds_ref = np.array(y_preds_ref)
confs = np.array(confs)
confs_ref = np.array(confs_ref)

MAEs = []
MAEs_ref = []
for cutoff in np.arange(0.2, 1.0, 0.02):
  selected_inds = np.array([i for i, c in enumerate(confs) if c > cutoff and c <= cutoff+ 0.02])
  dists = eval_dists(np.array(test_data)[selected_inds], np.array(pred)[selected_inds])
  MAEs.append((np.mean(dists), np.std(dists)))
  
  selected_inds = np.array([i for i, c in enumerate(confs_ref) if c > cutoff and c <= cutoff+ 0.02])
  dists_ref = eval_dists(np.array(test_data)[selected_inds], np.array(pred_ref)[selected_inds])
  MAEs_ref.append((np.mean(dists_ref), np.std(dists_ref)))
  
MAEs = np.array(MAEs) * 0.6
MAEs_ref = np.array(MAEs_ref) * 0.6

# Figure S4A
plt.clf()
hists = plt.hist(confs, width=0.015, bins=np.arange(0, 1.02, 0.02), color=(0.1, 0.1, 0.1, 0.3))
bins = [(0.32, 0.36), (0.54, 0.58), (0.76, 0.80), (0.96, 1.0)]
colors = [(76./256, 232./256, 180./256, 0.8),
          (66./256, 201./256, 156./256, 0.8),
          (52./256, 158./256, 122./256, 0.8),
          (40./256, 122./256, 95./256, 0.8)]
for i in range(len(hists[2])):
  for j in range(len(bins)):
    if hists[1][i] >= bins[j][0] and hists[1][i] < bins[j][1]:
      hists[2][i].set_color(colors[j])
plt.xlim(0.18, 1.02)
plt.xlabel("Confidence Score", fontdict=font)
plt.ylabel("Counts", fontdict=font)
plt.tight_layout()
plt.savefig("./figs/figS4_conf_histogram.png", dpi=300)

# Figure S4B
plt.clf()
plt.plot(np.arange(0.21, 1.0, 0.02), MAEs[:, 0], 'o-', color=(0.1, 0.1, 0.1, 0.3), label='Boundary MAE (Sequential PB-Net)')
plt.xlim(0.18, 1.02)
plt.ylim(0.8, 5.2)
plt.legend(fontsize=14)
plt.xlabel("Confidence Score", fontdict=font)
plt.ylabel("MAE (second)", fontdict=font)
plt.tight_layout()
plt.savefig("./figs/figS4_conf_MAE.png", dpi=300)

# Figure S4C
plt.clf()
hists = plt.hist(confs_ref, width=0.015, bins=np.arange(0, 1.02, 0.02), color=(0.1, 0.1, 0.1, 0.3))
plt.xlim(0.18, 1.02)
plt.xlabel("Confidence Score", fontdict=font)
plt.ylabel("Counts", fontdict=font)
plt.tight_layout()
plt.savefig("./figs/figS4_conf_histogram_ref.png", dpi=300)

# Figure S4D
plt.clf()
plt.plot(np.arange(0.21, 1.0, 0.02), MAEs_ref[:, 0], 'o-', color=(0.1, 0.1, 0.1, 0.3), label='Boundary MAE (Ref-based PB-Net)')
plt.xlim(0.18, 1.02)
plt.ylim(0.8, 3.0)
plt.legend(fontsize=14)
plt.xlabel("Confidence Score", fontdict=font)
plt.ylabel("MAE (second)", fontdict=font)
plt.savefig("./figs/figS4_conf_MAE_ref.png", dpi=300)

# Figure S5
bins = [(0.32, 0.36), (0.54, 0.58), (0.76, 0.80), (0.96, 1.0)]
colors = [(76./256, 232./256, 180./256, 0.8),
          (66./256, 201./256, 156./256, 0.8),
          (52./256, 158./256, 122./256, 0.8),
          (40./256, 122./256, 95./256, 0.8)]

for i_b, b in enumerate(bins):
  plt.clf()
  inds = np.where(y_preds > 0.2)[0]
  selected_inds = np.array([i for i, c in enumerate(confs) if c >= b[0] and c <= b[1]])
  selected_inds = np.array(list(set(selected_inds) & set(inds)))
  r = pearsonr(np.log(y_trues[selected_inds]+1e-9), np.log(y_preds[selected_inds]+1e-9))[0]
  dists = eval_dists(np.array(test_data)[selected_inds], np.array(pred)[selected_inds])
  print(np.mean(dists))
  total_ct = len(selected_inds)
  
  plt.plot(-np.log(y_trues[0]), -np.log(y_preds[0]), '.', markersize=5., color=(0.1, 0.1, 0.1, 0.8))
  plt.plot(-np.log(y_trues[0]), -np.log(y_preds[0]), '.', markersize=5., color=colors[i_b], label='Confidence score\n in [%.2f, %.2f]' % b)
  plt.legend(loc=4, fontsize=14)
  plt.scatter(np.log(y_trues[inds]+1e-9), np.log(y_preds[inds]+1e-9), s=0.1, color=(0.1, 0.1, 0.1, 0.8))
  plt.scatter(np.log(y_trues[selected_inds]+1e-9), np.log(y_preds[selected_inds]+1e-9), s=0.1, color=colors[i_b])
  
  plt.xlim(-0.5, 15.5)
  plt.ylim(-0.5, 15.5)
  plt.text(0.1, 12, "%d samples\nBoundary MAE: %.2fs\nLog-Abd Pearson r: %.4f" % (total_ct, np.mean(dists)*0.6, r), fontsize=14)

  plt.xlabel("Log Abundance - Human Annotation", fontdict=font)
  plt.ylabel("Log Abundance - Prediction", fontdict=font)
  plt.tight_layout()
  plt.savefig('./figs/figS5_conf_change_%d.png' % i_b, dpi=600)

####################################################################
  
# Figure 3A
i = 506
X = test_data[i][0]
bg = np.min(X[:, 1]) - (np.max(X[:, 1]) - np.min(X[:, 1])) * 0.02
profile = test_data[i][2]
start = profile[1]
end = profile[2]
f, ax1 = plt.subplots()
ax1.plot(X[:, 0], X[:, 1], color=(0.1, 0.1, 0.1, 0.3), linewidth=2, label='X')
ax1.plot(start, bg, 'r*', markersize=12, label='start')
ax1.plot(end, bg, 'r^', markersize=9, label='end')
ax1.set_ylabel("Intensity", fontdict=font)
ax1.set_xlabel("Retention Time (min)", fontdict=font)
ax1.legend(fontsize=11, loc=2)

ax2 = ax1.twinx()
ax2.plot(X[:, 0], pred[i][:, 0], '-', color=color_seq, label='y pred start (Seq PB-Net)')
ax2.plot(X[:, 0], pred_ref[i][:, 0], '-', color=color_ref, label='y pred start (Ref PB-Net)')
ax2.plot(X[:, 0], pred[i][:, 1], '--', color=color_seq, label='y pred end (Seq PB-Net)')
ax2.plot(X[:, 0], pred_ref[i][:, 1], '--', color=color_ref, label='y pred end (Ref PB-Net)')
ax2.set_ylim(0.01, 1.02)
ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=8,
           ncol=2, mode="expand", borderaxespad=0., fontsize=10)
ax2.set_ylabel("Boundary Prediction Prob.", fontdict=font)
plt.tight_layout()
f.savefig('./figs/fig3_rs_reference.png', dpi=300)

# Figure 3B
plt.clf()
samples = [6, 1509, 2011, 4510, 6015, 7020]
positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
f, axes = plt.subplots(3, 2)
for i, pos in zip(samples, positions):
  ax1 = axes[pos[0]][pos[1]]
  X = test_data[i][0]
  bg = np.min(X[:, 1]) - (np.max(X[:, 1]) - np.min(X[:, 1])) * 0.02
  profile = test_data[i][2]
  start = profile[1]
  end = profile[2]
  ax1.plot(X[:, 0], X[:, 1], color=(0.1, 0.1, 0.1, 0.5), linewidth=2, label='X')
  ax1.plot(start, bg, 'r*', markersize=12, label='start')
  ax1.plot(end, bg, 'r^', markersize=9, label='end')
  ax1.set_xticklabels([])
  ax2 = ax1.twinx()
  ax2.plot(X[:, 0], pred[i][:, 0], '-', color=color_seq, label='y pred start (Seq PB-Net)')
  ax2.plot(X[:, 0], pred_ref[i][:, 0], '-', color=color_ref, label='y pred start (Ref PB-Net)')
  ax2.plot(X[:, 0], pred[i][:, 1], '--', color=color_seq, label='y pred end (Seq PB-Net)')
  ax2.plot(X[:, 0], pred_ref[i][:, 1], '--', color=color_ref, label='y pred end (Ref PB-Net)')
  ax2.set_ylim(0.01, 1.02)
plt.tight_layout()
f.savefig('./figs/fig3_rs_samples.png', dpi=600)

####################################################################
sample_i = 10
ref_i = 510
sample_dat = test_data[sample_i]
ref_dat = test_data[ref_i]

class Config:
    lr = 0.001
    batch_size = 8
    max_epoch = 1 # =1 when debug
    workers = 2
    gpu = False # use gpu or not
    lower_limit = -1.0
    upper_limit = 1.0
    signal_max = 500  
opt = Config()
net = ReferenceModel(input_dim=384,
                     hidden_dim_lstm=128,
                     hidden_dim_attention=32,
                     n_lstm_layers=2,
                     n_attention_heads=8,
                     gpu=opt.gpu,
                     random_init=False)
trainer = ReferenceTrainer(net, opt, MatchLossRaw(), featurize=True)
trainer.load('./model-ref.pth')

att_map = trainer.attention_map(sample_dat, ref_dat)

# Figure S1
X = test_data[ref_i][0]
bg = np.min(X[:, 1]) - (np.max(X[:, 1]) - np.min(X[:, 1])) * 0.02
profile = test_data[ref_i][2]
start = profile[1]
end = profile[2]
plt.clf()
f, ax1 = plt.subplots()
ax1.plot(X[:, 0], X[:, 1], color=(0.1, 0.1, 0.1, 0.5), linewidth=2, label='X')
ax1.plot(start, bg, 'r*', markersize=12, label='start')
ax1.plot(end, bg, 'r^', markersize=9, label='end')
ax1.set_ylabel("Intensity - Reference", fontdict=font)
ax1.set_xlim(42.06, 42.74)
f.savefig('./figs/figS1_att_reference.png', dpi=300)

plt.clf()
plt.imshow(np.transpose(att_map), cmap='BuGn', extent=[42.06, 42.74, 42.74, 42.06])
plt.ylabel("Retention Time - Reference (min)", fontdict={'size': 12})
plt.xlabel("Retention Time - Sample (min)", fontdict={'size': 12})
plt.savefig('./figs/figS1_att_att.png', dpi=300)

X = test_data[sample_i][0]
bg = np.min(X[:, 1]) - (np.max(X[:, 1]) - np.min(X[:, 1])) * 0.02
profile = test_data[sample_i][2]
start = profile[1]
end = profile[2]
plt.clf()
f, ax1 = plt.subplots()
ax1.plot(X[:, 0], X[:, 1], color=(0.1, 0.1, 0.1, 0.5), linewidth=2, label='X')
ax1.plot(start, bg, 'r*', markersize=12, label='start')
ax1.plot(end, bg, 'r^', markersize=9, label='end')
ax1.legend(loc=2, fontsize=11)
ax1.set_xlim(42.06, 42.74)
ax1.set_ylabel("Intensity - Sample", fontdict=font)

ax2 = ax1.twinx()
ax2.plot(X[:, 0], pred_ref[sample_i][:, 0], '-', color=color_ref, label='y pred start (Ref PB-Net)')
ax2.plot(X[:, 0], pred_ref[sample_i][:, 1], '--', color=color_ref, label='y pred end (Ref PB-Net)')
ax2.set_ylim(0.01, 1.02)
ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=8,
           ncol=2, mode="expand", borderaxespad=0., fontsize=10)
ax2.set_ylabel("Boundary Prediction Prob.", fontdict=font)
plt.tight_layout()
f.savefig('./figs/figS1_att_sample.png', dpi=300)

############################################################
y_trues = []
y_preds = []
y_preds_ref = []
confs = []
confs_ref = []
for i, sample in enumerate(test_data):
  output = pred[i]
  output_ref = pred_ref[i]
  y_trues.append(calculate_abundance(sample[1], sample))
  y_preds.append(calculate_abundance(output, sample, intg=False))
  y_preds_ref.append(calculate_abundance(output_ref, sample, intg=False))
  confs.append(calculate_confidence(output[:, 0], thr=0.2) * \
               calculate_confidence(output[:, 1], thr=0.2))
  confs_ref.append(calculate_confidence(output_ref[:, 0], thr=0.2) * \
                   calculate_confidence(output_ref[:, 1], thr=0.2))
inds = np.where(np.array(confs_ref) < 0.6)[0]

# Figure S3
i = 45
print(calculate_confidence(pred[i][:, 1], thr=0.2))
color_seq = (44./256, 83./256, 169./256, 0.7)
X = test_data[i][0]
bg = np.min(X[:, 1]) - (np.max(X[:, 1]) - np.min(X[:, 1])) * 0.02
profile = test_data[i][2]
start = profile[1]
end = profile[2]
f, ax1 = plt.subplots()
ax1.plot(X[:, 0], X[:, 1], color=(0.1, 0.1, 0.1, 0.3), linewidth=1., label='X')
ax1.set_ylabel("Intensity", fontdict=font)
ax1.set_xlabel("Retention Time (min)", fontdict=font)
ax2 = ax1.twinx()
ax2.plot(X[:, 0], pred[i][:, 1], '--', linewidth=1., color=color_seq, label='y pred end (Seq PB-Net)')
ax2.set_ylim(0.0, 1.0)
ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=8,
           ncol=2, mode="expand", borderaxespad=0., fontsize=10)
ax2.set_ylabel("Boundary Prediction Prob.", fontdict=font)
plt.tight_layout()
plt.savefig('./figs/figS3_conf_illus.png', dpi=300)

#################################################################
# Figure 1
i = 0
X = test_data[i][0][np.linspace(0, len(test_data[i][0])-1, 10).astype(int)]
y_pred = pred[i][np.linspace(0, len(test_data[i][0])-1, 10).astype(int)]
y_pred_ref = pred_ref[i][np.linspace(0, len(test_data[i][0])-1, 10).astype(int)]

bg = np.min(X[:, 1]) - (np.max(X[:, 1]) - np.min(X[:, 1])) * 0.02
profile = test_data[i][2]
f, ax1 = plt.subplots()
ax1.plot(X[:, 0], X[:, 1], 'o-', color=(0.1, 0.1, 0.1, 0.8), linewidth=2., label='X')
plt.savefig('./figs/fig1_model_structure_sample.png', dpi=300)

plt.clf()
plt.subplot(2, 1, 1)
plt.plot(X[:, 0], y_pred[:, 0]/y_pred[:, 0].max(), 'o-', color=color_seq)
plt.plot(X[:, 0], y_pred[:, 1]/y_pred[:, 1].max(), 'o--', color=color_seq)
plt.xticks([], [])
plt.yticks([], [])
plt.ylim(-0.1, 1.1)
plt.savefig('./figs/fig1_model_structure_sample_pred.png', dpi=300)

plt.clf()
plt.subplot(2, 1, 1)
plt.plot(X[:, 0], y_pred_ref[:, 0]/y_pred_ref[:, 0].max(), 'o-', color=color_ref)
plt.plot(X[:, 0], y_pred_ref[:, 1]/y_pred_ref[:, 1].max(), 'o--', color=color_ref)
plt.xticks([], [])
plt.yticks([], [])
plt.ylim(-0.1, 1.1)
plt.savefig('./figs/fig1_model_structure_sample_pred_ref.png', dpi=300)

i = 500
X = test_data[i][0][np.linspace(0, len(test_data[i][0])-1, 10).astype(int)]
y_pred = pred[i][np.linspace(0, len(test_data[i][0])-1, 10).astype(int)]
bg = np.min(X[:, 1]) - (np.max(X[:, 1]) - np.min(X[:, 1])) * 0.02
profile = test_data[i][2]
f, ax1 = plt.subplots()
ax1.plot(X[:, 0], X[:, 1], 'o-', color=(0.1, 0.1, 0.1, 0.5), linewidth=2., label='X')
ax1.plot(profile[1], bg, 'r*', markersize=12, label='start')
ax1.plot(profile[2], bg, 'r^', markersize=9, label='end')
plt.savefig('./figs/fig1_model_structure_ref.png', dpi=300)

class Config:
    lr = 0.001
    batch_size = 8
    max_epoch = 1 # =1 when debug
    workers = 2
    gpu = False # use gpu or not
    lower_limit = -1.0
    upper_limit = 1.0
    signal_max = 500  
opt = Config()
net = ReferenceModel(input_dim=384,
                     hidden_dim_lstm=128,
                     hidden_dim_attention=32,
                     n_lstm_layers=2,
                     n_attention_heads=8,
                     gpu=opt.gpu,
                     random_init=False)
trainer = ReferenceTrainer(net, opt, MatchLossRaw(), featurize=True)
trainer.load('./model-ref.pth')
att_map = trainer.attention_map(test_data[0], test_data[500])
plt.clf()
extent=[29.32295, 30.145, 30.145, 29.32295]
plt.imshow(att_map, cmap='BuGn', extent=extent)
plt.xticks([], [])
plt.yticks([], [])
plt.tight_layout()
plt.savefig('./figs/fig1_att.png', dpi=300)

for ct, sample_i in enumerate([24906, 7548]):
  X = test_data[sample_i][0]
  bg = np.min(X[:, 1]) - (np.max(X[:, 1]) - np.min(X[:, 1])) * 0.02
  profile = test_data[sample_i][2]
  start = profile[1]
  end = profile[2]
  plt.clf()
  f, ax1 = plt.subplots()
  ax1.plot(X[:, 0], X[:, 1], color=(0.1, 0.1, 0.1, 0.5), linewidth=5, label='X')
  ax1.plot(start, bg, 'r*', markersize=20, label='start')
  ax1.plot(end, bg, 'r^', markersize=15, label='end')
  ax1.legend(fontsize=20, loc=2)
  ax1.set_ylabel("Intensity", fontdict={'size': 25})
  ax1.set_xlabel("Retention Time (min)", fontdict={'size': 25})
  plt.tight_layout()
  ax1.tick_params(axis='both', which='major', labelsize=15)
  plt.savefig('./figs/fig1_sample%d_up.png' % ct, dpi=300, bbox_inches = "tight")
  
  f, ax2 = plt.subplots()
  ax2.plot(X[:, 0], pred_ref[sample_i][:, 0], '-', color=color_ref, linewidth=2, label='y pred start (Ref PB-Net)')
  ax2.plot(X[:, 0], pred_ref[sample_i][:, 1], '--', color=color_ref, linewidth=2, label='y pred end (Ref PB-Net)')
  
  ax2.plot(X[:, 0], pred[sample_i][:, 0], '-', color=color_seq, linewidth=2, label='y pred start (Seq PB-Net)')
  ax2.plot(X[:, 0], pred[sample_i][:, 1], '--', color=color_seq, linewidth=2, label='y pred end (Seq PB-Net)')
  
  ax2.set_ylim(0.01, 1.02)
  ax2.tick_params(axis='both', which='major', labelsize=15)
  ax2.legend(bbox_to_anchor=(0., -0.75, 1., .102), loc=8,
             ncol=1, mode="expand", borderaxespad=0., fontsize=20)
  ax2.set_ylabel("Boundary Prediction Prob.", fontdict={'size': 25})
  plt.savefig('./figs/fig1_sample%d_down.png' % ct, dpi=300, bbox_inches = "tight")

############################################################
inds = np.arange(len(test_data))
np.random.seed(123)
np.random.shuffle(inds)

# Figure S6
plt.clf()
ct = 0
for i in inds:
  if ct > 11:
    break
  X = test_data[i][0]
  bg = np.min(X[:, 1]) - (np.max(X[:, 1]) - np.min(X[:, 1])) * 0.02
  profile = test_data[i][2]
  if profile[3] < 1:
    continue
  start = profile[1]
  end = profile[2]
  ct += 1
  ax1 = plt.subplot(3, 4, ct)
  ax1.plot(X[:, 0], X[:, 1], color=(0.1, 0.1, 0.1, 0.3), linewidth=1., label='X')
  ax1.plot(start, bg, 'r*', markersize=12, label='start')
  ax1.plot(end, bg, 'r^', markersize=9, label='end')
  ax1.get_xaxis().set_ticks([])
  ax1.get_yaxis().set_ticks([])
  ax2 = ax1.twinx()
  ax2.plot(X[:, 0], pred[i][:, 0], '-', color=color_seq, label='y pred start (Seq PB-Net)')
  ax2.plot(X[:, 0], pred_ref[i][:, 0], '-', color=color_ref, label='y pred start (Ref PB-Net)')
  ax2.plot(X[:, 0], pred[i][:, 1], '--', linewidth=1., color=color_seq, label='y pred end (Seq PB-Net)')
  ax2.plot(X[:, 0], pred_ref[i][:, 1], '--', color=color_ref, label='y pred end (Ref PB-Net)')
  ax2.set_ylim(0.0, 1.0)
  ax2.get_xaxis().set_ticks([])
  ax2.get_yaxis().set_ticks([])

plt.tight_layout()  
plt.savefig('./figs/figS6_samples.png', dpi=300)
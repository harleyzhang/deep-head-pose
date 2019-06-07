#!/usr/bin/env python

# show training loss graph

import argparse
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
  parser = argparse.ArgumentParser(description="Show training loss graph")
  parser.add_argument('log_file', type=str, help="Log file that contains 'loss: number'")
  parser.add_argument('--alpha', type=float, help="alpha of decay", default=1)
  args = parser.parse_args()
  return args


args = parse_args()
losses = []
yaw_losses = []
pitch_losses = []
roll_losses = []
lr = ''
base_model = ''
with open(args.log_file) as f:
  for line in f:
    if not lr: 
      m = re.search(r'^lr: (.*?)$', line)
      if m:
        lr = m.group(1)
  
    if not base_model:
      m = re.search(r'^base model: (.*?)$', line)
      if m:
         base_model = m.group(1) 

    m = re.search(r'Yaw: (.*?), Pitch: (.*?), Roll: (.*?)$', line)
    if m:
      yaw_losses.append(float(m.group(1)))
      pitch_losses.append(float(m.group(2)))
      roll_losses.append(float(m.group(3)))

print('')
print('######################################')
print('base model: {}'.format(base_model))
print('lr: {}'.format(lr))

if len(yaw_losses) > 0: 
  all_losses  = np.array([yaw_losses, pitch_losses, roll_losses])
  
  print('mean loss:')
  mean_losses = np.mean(all_losses, axis=0)
  print(mean_losses)
  
  # median and variance of the best 5 score
  best5 = sorted(mean_losses)[:5]
  print('Median of best 5: %.4f   Std: %.4f' % (np.median(best5), np.std(best5)))
  print('')
  
  
  #plt.plot(range(1, len(losses)+1), losses)
  #plt.show()
  

#!/usr/bin/env python

# show training loss graph

import argparse
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
  parser = argparse.ArgumentParser(description="Show training loss graph")
  parser.add_argument('log_file', type=str, help="Log file that contains 'loss: number'")
  parser.add_argument('--alpha', type=float, help="alpha of decay(not implemented)", default=1)
  args = parser.parse_args()
  return args


args = parse_args()
losses = []
with open(args.log_file) as f:
  for line in f:
    m = re.search(r'Epoch \[(.*?)/25\].*Loss: (.*)$', line)
    if m:
      epoch = int(m.group(1))
      l = float(m.group(2))
      if len(losses)<epoch:
        losses.append([l])
      else:
        losses[epoch-1].append(l)


#show mean of each epoch
for epoch_loss in losses:
  print '%.4f' % np.mean(epoch_loss)
  #print '%.4f' % np.median(epoch_loss)

#plt.plot(range(1, len(losses)+1), losses)
#plt.show()



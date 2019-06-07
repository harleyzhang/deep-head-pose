#!/usr/bin/env python

import glob
import os
import utils
import numpy as np

data_dir = '/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W/AFLW2000'

def correct_angle(x):
  x = x % 360
  if x >180: x = x - 360
  return x

good_file = []
out_fname = 'AFLW2000_valid.txt'
of = open(out_fname, 'w')
for f1 in sorted(glob.glob(data_dir + '/*.jpg')):
  #print(f1)
  mat_f1 = f1[:-4] + '.mat'
  pose = utils.get_ypr_from_mat(mat_f1)

  yrp = np.array([x*180/np.pi for x in pose])#[yaw, roll, pitch]
  if np.all((yrp>=-99) & (yrp<=99)):
     good_file.append(f1)
     of.write('ALFW200/'+os.path.basename(f1)[:-4]+'\n')
  else:
    print('')
    print(mat_f1)
    print(yrp)
    print(list(map(correct_angle, yrp)))


of.close()



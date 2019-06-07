#!/bin/bash

fname=$@
echo $fname

#k80 are not that crowded
#sbatch -n 1 --mem=32000 --time=3-00:00:00 -p gpu --constraint=k80 --gres=gpu:1 $fname

# P100 seems pretty full
sbatch -n 4 --mem=32000 --time=3-00:00:00 -p gpu --gres=gpu:p100:1 $fname



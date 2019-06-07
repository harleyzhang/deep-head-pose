#!/bin/bash

fname=$@
echo $fname

#k80 are not that crowded

# P100 seems pretty full
sbatch -n 4 --mem=32000 --time=3-00:00:00 -p gpu --gres=gpu:k80:1 $fname



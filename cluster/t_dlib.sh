#!/bin/bash

source /homeappl/home/zhangh/Slurm/setenv3.sh
cd ..

pretrained_model=/homeappl/home/zhangh/Work/DONOTREMOVE/headpose/dhp_pretrained/hopenet_alpha1.pkl
face_model=/homeappl/home/zhangh/Work/DONOTREMOVE/headpose/dlib_models/mmod_human_face_detector.dat
video_file=/homeappl/home/zhangh/Work/DONOTREMOVE/Samples/videos/westworld.mp4

python3 \
  code/test_on_video_dlib.py \
  --snapshot $pretrained_model \
  --face_model $face_model \
  --video $video_file \
  --output_string "westwold" \
  --n_frames 6000 \
  --fps 30


#!/bin/bash

#8G mem, time 12h, one gpu
#salloc -n 1 --mem=8152 --time=720 -p gpu --nodelist=meg[10-12] --gres=gpu:1 ./s_ssh.sh
#salloc -n 4 --nodes 1 --mem=8000 -p gpu --time=5-00:00:00 --exclude=meg3,meg5,meg7 --gres=gpu:2 ./s_ssh.sh

#test
#salloc -n 1 --mem=8152 --time=720 -p gpu --gres=gpu:1 ./s_ssh.sh
#ask for 5 days, so no one bothers me

#salloc -n 1 --mem=8152 --time=10-00:00:00 -p gpu --nodelist=meg[10-12] --gres=gpu:1 ./s_ssh.sh


#salloc -n 8 --mem=32000 --time=10-00:00:00 -p gpu --exclude=meg3,meg5,meg7 --gres=gpu:1 ./s_ssh.sh

#salloc -n 2 --mem=16000 --time=10-00:00:00 -p gpu --exclude=meg3,meg5,meg7 --gres=gpu:2 ./s_ssh.sh


# gpu node has 32G mem, 2 k40t
#salloc -n 4 --mem=32000 --time=3-00:00:00 -p gpu --gres=gpu:1 --constraint=k80 ./s_ssh.sh

#salloc -n 4 --mem=32000 --time=3-00:00:00 -p gpu --gres=gpu:1 --constraint=p100 ./s_ssh.sh

#module load python-env/3.5.3-ml
srun -n 4 --mem=32000 --time=0-03:00:00 -p gpulong --gres=gpu:k80:1 --x11=all --pty $SHELL
#srun -N 1 -n 1 -c 4 --mem=128000 --time=0-03:00:00 -p gpu --gres=gpu:k80:1 --x11=all --pty $SHELL
#srun -n 6 --mem=16000 --time=3-00:00:00 -p gpu --gres=gpu:p100:1 --pty $SHELL



# gpu node has 256G mem, 4 k80t
#salloc -n 16 --mem=64000 --time=3-00:00:00 -p gpu --gres=gpu:4 --constraint=k80 ./s_ssh.sh

#ssh -X $nodename



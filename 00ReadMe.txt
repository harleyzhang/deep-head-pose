removed 31 images from AFLW2000

pretrained model (alpha 1):
  Yaw: 6.9200, Pitch:     6.6373, Roll: 5.6745
  mean of ypr: 6.41

pretrained model(alpha 2):
  Yaw: 6.4700, Pitch:     6.5590, Roll: 5.4358
  mean of ypr: 6.15


Resnet50 No dropout lr 1E-6:
  Median of best 5: 6.5006   Var: 0.0038

Resnet50 Dropout lr 1E-7:
  Median of best 5: 6.6704   Var: 0.0007
  
Resnet50 Dropout lr 5E-7:
  Median of best 5: 6.4400   Var: 0.0049

Resnet50 Dropout lr 1E-6:
  Median of best 5: 6.7973   Var: 0.0055

Hopenet Bs16 lr: 1E-5:
  Median of best 5: 6.8151   Var: 0.1279

Hopenet BS64 lr: 1E-5: 
  Median of best 5: 6.2576   Var: 0.0069

Hopenet B64 learning rate drop: 1E-5, drop 0.1 every 10 epochs
  Median of best 5: 6.0397   Var: 0.0001
  something like:  Yaw: 6.2350, Pitch: 6.5665, Roll: 5.3430

Resnet50 lr drop: 5E07:
Median of best 5: 7.2766   Var: 0.0002
  This is not working at all. 

Resnet18 lr drop, 
  lr: 0.0001
    Median of best 5: 8.5419   Var: 0.0805
  lr: 5e-5
    Median of best 5: 7.6431   Var: 0.0382
  lr: 1E-5
    Median of best 5: 7.3642   Var: 0.0465
  lr: 5E-6
    Median of best 5: 6.8634   Var: 0.0031

Resnet50 lr drop:


############################################################
#different network test
######################################
base model: resnet50
lr: 0.001
Median of best 5: 11.7240   Std: 0.0452


######################################
base model: resnet50
lr: 0.0001
Median of best 5: 8.4209   Std: 0.4101


######################################
base model: resnet50
lr: 0.00001
Median of best 5: 6.2057   Std: 0.1312


######################################
base model: resnet50
lr: 0.000001
Median of best 5: 6.2307   Std: 0.0352


######################################
base model: squeezenet1_1
lr: 0.001
Median of best 5: 17.4281   Std: 0.0050


######################################
base model: squeezenet1_1
lr: 0.0001
Median of best 5: 7.9551   Std: 0.0563


######################################
base model: squeezenet1_1
lr: 0.00001
Median of best 5: 7.7576   Std: 0.0350


######################################
base model: squeezenet1_1
lr: 0.000001
Median of best 5: 8.4911   Std: 0.0041


######################################
base model: resnet101
lr: 0.001
Median of best 5: 14.2763   Std: 0.0268


######################################
base model: resnet101
lr: 0.0001
Median of best 5: 8.2122   Std: 0.0657


######################################
base model: resnet101
lr: 0.00001
Median of best 5: 6.2807   Std: 0.0164


######################################
base model: resnet101
lr: 0.000001
Median of best 5: 5.9920   Std: 0.0296


######################################
base model: se_resnext50_32x4d
lr: 0.001
Median of best 5: 10.3435   Std: 0.0361


######################################
base model: se_resnext50_32x4d
lr: 0.0001
Median of best 5: 7.5250   Std: 0.0433


######################################
base model: se_resnext50_32x4d
lr: 0.00001
Median of best 5: 6.2987   Std: 0.0134


######################################
base model: se_resnext50_32x4d
lr: 0.000001
Median of best 5: 6.3262   Std: 0.0207


######################################
base model: nasnetamobile
lr: 0.001
Median of best 5: nan   Std: nan


######################################
base model: nasnetamobile
lr: 0.0001
Median of best 5: 18.8681   Std: 0.9684


######################################
base model: nasnetamobile
lr: 0.00001
Median of best 5: nan   Std: nan

######################################
base model: squeezenet1_1 
lr: 0.00001
Median of best 5: 9.3522   Std: 0



############################################################
# prune
base model: resnet101 
Yaw: 6.3566, Pitch: 6.3764, Roll: 5.4438
50% pruned: 
 Median of best 5: 6.1513   Std: 0.0246
30% pruned:
Median of best 5: 6.0736   Std: 0.0505

base model: resnet18
Yaw: 7.7031, Pitch:     7.1933, Roll: 6.5727
50% pruned:
 Yaw: 18.2621, Pitch    : 9.9929, Roll: 10.2986
 Median of best 5: 12.8512   Std: 2.2544
30% pruned: 
  Yaw: 9.0981, Pitch:     7.6937, Roll: 7.2699
  8.0166   Std: 0.3492


base model: resnet50
Yaw: 6.5108, Pitch     : 6.7120, Roll: 5.9673
50% pruned:
Yaw: 16.0363, Pitc     h: 10.4576, Roll: 9.6150
Median of best 5: 11.9324   Std: 2.2380
30% pruned:
Median of best 5: 6.7076   Std: 0.1321

base model: se_resnext50_32x4d
Yaw: 6.6743, Pitch:     6.8440, Roll: 6.2009
50% pruned:
Median of best 5: 7.5611   Std: 0.5781
30% pruned: 
Median of best 5: 7.1869   Std: 0.2449

############################################################
# implemented dropout to regnet50, and test
  tested different learning rate with dropout
    5E-6  9.6508   Var: 0.0484
    1E-6  7.6579   Var: 0.0064
    5E-7  7.3028   Var: 0.0052
    1E-7  7.5450   Var: 0.0006
    

# test resnet50 different learning rate
  0.1, 0.01, 0.001: does not converge
  0.0001: about 27,10,9 in the early epochs and diverge in the last few epochs
  0.00001: about 17, 8, 7: same as first time results
    Median of best 5: 11.0230   Var: 0.4494
  0.000001: has value of 7.8, 7.6, 6.7
    Median of best 5: 7.3329   Var: 0.0043

# test hopenet(clipped the values)
batch size 64: 
  best values is about 6.5,7.5,6.5, but it does not get better on later epochs
  Median of best 5: 7.3270   Var: 0.0406

batch size 16:
  seems to be worse than batch size 64, expecially in later epochs
    Median of best 5: 7.6283   Var: 0.1383

a previously traine model
  values are 7.7, 7.3, 6.9, so it's about same as my trained model
    Median of best 5: 7.3536   Var: 0.0000

#########################################################
# run test code

  FPS is about 1.38 when do face detection head pose estimation together

  Further evaluated:
  Face detection takes about 0.69 second per frame
  Head pose estimation takes about 0.059 seconds


  Face detection should use Faster-RCNN, which is very slow. 
    It should use Yolo type of face detection
    Another option is to combine face detection and head pose eistimation together.  In that case, it takes about 0.06 second to process one frame and it's not difficult reach 15 FPS

# train a model

  - download databases
  http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm


# test trained model
  Trained using 300W-LP images
  Tested on AFLW2000 images:
  model: resnet50
  snapshot 25:  
    Test error in degrees of the model on the 2000 test images. Yaw: 23.1398, Pitch: 10.6624, Roll: 10.3506
  snapshot 24: 
    Yaw: 20.9425, Pitch: 10.5844, Roll: 10.3606
  snapshot 23:
    Yaw: 21.8621, Pitch: 9.7099, Roll: 10.0020 
  snapshot 20:
    Yaw: 18.4247, Pitch: 10.1399, Roll: 9.7778
  snapshot 13:(best in all snapshots)
    Yaw: 16.6487, Pitch: 9.6098, Roll: 8.8414


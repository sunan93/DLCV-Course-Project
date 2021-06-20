# DLCV-Course-Project
This repository contains the implementation of the DLCV project : Generalization of Data Poisoning attacks to targeted objects. 
It is a modification to [Bullseye Polytope attack](https://github.com/ucsb-seclab/BullseyePoison).

## Install datasets
Download the datasets from [here](https://drive.google.com/file/d/1wVRobdlwvD9-VL9mYKCu_onq8PbbyP0V/view), which contains the split of CIFAR10 dataset and the Multi-View Car Dataset.

Pretrained models can be downloaded from [here](https://drive.google.com/file/d/1TwxNbJ1arDNQrBJdt5AFeaAbKC65HOko/view).


## Install python dependencies
The project is implemented in Pytorch using the following libraries:
```
kmeans-pytorch       0.3
lpips                0.1.3
matplotlib           3.4.1
scikit-image         0.18.1
scikit-learn         0.24.1
scipy                1.6.2
sklearn              0.0
torch                1.8.1
torchaudio           0.8.0a0+e4e171a
torchvision          0.9.1
```

## End-to-end training 
The following command is used to run the Bullseye attack in a multi target mode:
```
bash launch/attack-end2end-12.sh 0 mean 17 1
```
To run the BP-MT attack for all the cars in the multi-veiw dataset, run the following,
```
python launch/end2end_run_attack_all.py 0 mean 1 20 > log.txt
grep -A 6 "SUMMARY" log.txt
```

## Results
The results displayed by the grep command are aggregated in the [excel sheet]() and averaged to get the accuracy of poisoning.
# DLCV-Course-Project
This repository contains the implementation of the DLCV project : Generalization of Data Poisoning attacks to targeted objects. 
It is a modification to [Bullseye Polytope attack](https://github.com/ucsb-seclab/BullseyePoison). The Bullseye Polytope reporsitory can be cloned and setup with the code changes mentioned for each attack. The authors of Bullseye Polytope released the source code along with the substitute networks that they have used. We have used the same code and dataset for our experiments. The experiments have been done using PyTorch-v1.8.1 over Cuda 11.1. 

## Install datasets
Download the datasets from [here](https://drive.google.com/file/d/1wVRobdlwvD9-VL9mYKCu_onq8PbbyP0V/view), which contains the split of CIFAR10 dataset and the Multi-View Car Dataset.

Pretrained models (substitute networks) can be downloaded from [here](https://drive.google.com/file/d/1TwxNbJ1arDNQrBJdt5AFeaAbKC65HOko/view).


## Install python dependencies
The project is implemented in Pytorch using the following libraries on GPU GeForce GTX 1080.
```
kmeans-pytorch       0.3
lpips                0.1.3
matplotlib           3.4.1
scikit-image         0.18.1
scikit-learn         0.24.1
scipy                1.6.2
sklearn              0.0
torch                1.8.1
torchvision          0.9.1
```

## Instructions to run the code
The following command is used to run the Bullseye attack in a multi target mode for a specific car model 17 from the Multi-Target-Mode directory.
```
bash launch/attack-end2end-12.sh 0 mean 17 1
```
To run the BP-MT attack for all the cars in the multi-veiw dataset, run the following command. The ene2end_run_attack_all.py script contains the flag arguments for all the five loss formulations that we tried.
```
cd Multi-Target-Mode/
python launch/end2end_run_attack_all.py 0 mean 1 20 > log.txt
grep -A 6 "SUMMARY" log.txt
```
The output of grep is aggregated in the [excel sheet](https://github.com/sunan93/DLCV-Course-Project/tree/master/Results) to get aggregate results for each attack.

To run BP-MT attack with LPIPS distance variant, run the same commands from the BP-MT-lpips/Multi-Target-Mode directory. 

To run the "BP-MT+nearest" variant, use the option --nearest in the file launch/attack-end2end-12.sh and run the same command.

To run the antidote optimization code, run the same command from the BP-MT-Antidote-opt folder.

## Results
The results displayed by the grep command are aggregated in the [excel sheet](https://github.com/sunan93/DLCV-Course-Project/tree/master/Results) and averaged over multiple trials to get the accuracy numbers. The numbers might change a little(by around 10%) across runs but the trends remain the same as the same seeds are used across multiple runs.
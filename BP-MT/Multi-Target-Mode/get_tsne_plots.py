import sys

sys.path.append("../")
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import argparse
import os
from models import *
from utils import load_pretrained_net, \
    fetch_nearest_poison_bases, fetch_poison_bases, fetch_all_target_cls
from trainer import make_convex_polytope_poisons, train_network_with_poison
from PIL import Image, ExifTags
import cv2
import os
import torch
from sklearn.manifold import TSNE



class Logger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "a+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def fetch_all_external_targets(target_label, root_path, subset, start_idx, end_idx, num, transforms, device='cuda'):
    target_list = []
    indices = []
    # target_label = [int(i == target_label) for i in range(10)]
    if start_idx == -1 and end_idx == -1:
        print("No specific indices are determined, so try to include whatever we find")
        idx = 1
        while True:
            path = '{}_{}_{}.jpg'.format(root_path, '%.2d' % subset, '%.3d' % idx)
            if os.path.exists(path): 
                img = Image.open(path)
                target_list.append([transforms(img).to(device), torch.tensor([target_label])])
                indices.append(idx)
                idx += 1
            else:
                print("In total, we found {} images of target {}".format(len(indices), subset))
                break
    else:
        assert start_idx != -1
        assert end_idx != -1

        for target_index in range(start_idx, end_idx + 1):
            indices.append(target_index)
            path = '{}_{}_{}.jpg'.format(root_path, '%.2d' % subset, '%.3d' % target_index)
            assert os.path.exists(path), "external target couldn't find"
            img = Image.open(path)
            target_list.append([transforms(img)[None, :, :, :].to(device), torch.tensor([target_label]).to(device)])

    i = math.ceil(len(target_list) / num)
    return [t for j, t in enumerate(target_list) if j % i == 0], [t for j, t in enumerate(indices) if j % i == 0], \
           [t for j, t in enumerate(target_list) if j % i != 0], [t for j, t in enumerate(indices) if j % i != 0]
        




if __name__ == '__main__':
    # ======== arg parser =================================================
    parser = argparse.ArgumentParser(description='PyTorch Poison Attack')
    parser.add_argument('--gpu', default='0', type=str)
    # The substitute models and the victim models
    parser.add_argument('--end2end', default=False, choices=[True, False], type=bool,
                        help="Whether to consider an end-to-end victim")
    parser.add_argument('--substitute-nets', default=['ResNet50', 'ResNet18'], nargs="+", required=False)
    parser.add_argument('--victim-net', default=["DenseNet121"], nargs="+", type=str)
    parser.add_argument('--model-resume-path', default='../models-chks-release', type=str,
                        help="Path to the pre-trained models")
    parser.add_argument('--net-repeat', default=1, type=int)
    parser.add_argument("--subs-chk-name", default=['ckpt-%s-4800.t7'], nargs="+", type=str)
    parser.add_argument("--test-chk-name", default='ckpt-%s-4800.t7', type=str)
    parser.add_argument('--subs-dp', default=[0], nargs="+", type=float,
                        help='Dropout for the substitute nets, will be turned on for both training and testing')

    # Parameters for poisons
    parser.add_argument('--target-path', default='../datasets/epfl-gims08/resized/tripod_seq', type=str,
                        help='path to the external images')
    parser.add_argument('--target-index', default=6, type=int,
                        help='model of the car in epfl-gims08 dataset')
    parser.add_argument('--target-start', default='-1', type=int,
                        help='first index of the car in epfl-gims08 dataset')
    parser.add_argument('--target-end', default='-1', type=int,
                        help='last index of the car in epfl-gims08 dataset')
    parser.add_argument('--target-num', default='5', type=int,
                        help='number of targets')

    parser.add_argument('--target-label', default=1, type=int)
    parser.add_argument('--poison-label', '-plabel', default=6, type=int,
                        help='label of the poisons, or the target label we want to classify into')
    parser.add_argument('--poison-num', default=5, type=int,
                        help='number of poisons')

    parser.add_argument('--poison-lr', '-plr', default=4e-2, type=float,
                        help='learning rate for making poison')
    parser.add_argument('--poison-momentum', '-pm', default=0.9, type=float,
                        help='momentum for making poison')
    parser.add_argument('--poison-ites', default=1000, type=int,
                        help='iterations for making poison')
    parser.add_argument('--poison-decay-ites', type=int, metavar='int', nargs="+", default=[])
    parser.add_argument('--poison-decay-ratio', default=0.1, type=float)
    parser.add_argument('--poison-epsilon', '-peps', default=0.1, type=float,
                        help='maximum deviation for each pixel')
    parser.add_argument('--poison-opt', default='adam', type=str)
    parser.add_argument('--nearest', default=False, action='store_true',
                        help="Whether to use the nearest images for crafting the poison")
    parser.add_argument('--subset-group', default=0, type=int)
    parser.add_argument('--original-grad', default=True, choices=[True, False], type=bool)
    parser.add_argument('--tol', default=1e-6, type=float)

    # Parameters for re-training
    parser.add_argument('--retrain-lr', '-rlr', default=0.1, type=float,
                        help='learning rate for retraining the model on poisoned dataset')
    parser.add_argument('--retrain-opt', default='adam', type=str,
                        help='optimizer for retraining the attacked model')
    parser.add_argument('--retrain-momentum', '-rm', default=0.9, type=float,
                        help='momentum for retraining the attacked model')
    parser.add_argument('--lr-decay-epoch', default=[30, 45], nargs="+",
                        help='lr decay epoch for re-training')
    parser.add_argument('--retrain-epochs', default=60, type=int)
    parser.add_argument('--retrain-bsize', default=64, type=int)
    parser.add_argument('--retrain-wd', default=0, type=float)
    parser.add_argument('--num-per-class', default=50, type=int,
                        help='num of samples per class for re-training, or the poison dataset')

    # Checkpoints and resuming
    parser.add_argument('--chk-path', default='chk-black', type=str)
    parser.add_argument('--chk-subdir', default='poisons', type=str)
    parser.add_argument('--eval-poison-path', default='', type=str,
                        help="Path to the poison checkpoint you want to test")
    parser.add_argument('--resume-poison-ite', default=0, type=int,
                        help="Will automatically match the poison checkpoint corresponding to this iteration "
                             "and resume training")
    parser.add_argument('--train-data-path', default='../datasets/CIFAR10_TRAIN_Split.pth', type=str,
                        help='path to the official datasets')
    parser.add_argument('--dset-path', default='datasets', type=str,
                        help='path to the official datasets')

    parser.add_argument('--mode', default='convex', type=str,
                        help='if convex, run the convexpolytope attack proposed by the paper, '
                             'otherwise run the mean method')
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()

    # Set visible CUDA devices
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True

    # load the pre-trained models
    sub_net_list = []
    for n_chk, chk_name in enumerate(args.subs_chk_name):
        for snet in args.substitute_nets:
            if args.subs_dp[n_chk] > 0.0:
                net = load_pretrained_net(snet, chk_name, model_chk_path=args.model_resume_path,
                                          test_dp=args.subs_dp[n_chk])
            elif args.subs_dp[n_chk] == 0.0:
                net = load_pretrained_net(snet, chk_name, model_chk_path=args.model_resume_path)
            else:
                assert False
            sub_net_list.append(net)

    print("subs nets, effective num: {}".format(len(sub_net_list)))

    print("Loading the victims networks")
    targets_net = []
    for vnet in args.victim_net:
        victim_net = load_pretrained_net(vnet, args.test_chk_name, model_chk_path=args.model_resume_path)
        targets_net.append(victim_net)

    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    # Get the target images
    _targets, targets_indices, eval_targets, val_targets_indices = fetch_all_external_targets(args.target_label,
                                                                                              args.target_path,
                                                                                              args.target_index,
                                                                                              args.target_start,
                                                                                              args.target_end,
                                                                                              args.target_num,
                                                                                              transforms=transform_test)
    
    car_imgs, idxes = fetch_all_target_cls(args.target_label, 100, 'others', args.train_data_path, transform_test)
    frog_imgs, idxes = fetch_all_target_cls(args.poison_label, 100, 'others', args.train_data_path, transform_test) 

    targets = [x for x, _ in _targets]
    print("targets = ", targets[0].shape, len(targets), car_imgs[0].shape, len(car_imgs), frog_imgs[0].shape, len(frog_imgs))


    #targets = torch.Tensor(targets).to(args.device)
    #targets.resize_((19,3,32,32))
    targets = torch.stack(targets)
    car_imgs = torch.Tensor(car_imgs).to(args.device)
    frog_imgs = torch.Tensor(frog_imgs).to(args.device)

    with torch.no_grad():
        for n_net, net in enumerate(sub_net_list):
            target_img_feat = net.module.penultimate(targets)
            car_imgs_feat = net.module.penultimate(car_imgs)
            frog_imgs_feat = net.module.penultimate(frog_imgs)
            break
    print(target_img_feat.shape)
    print(car_imgs_feat.shape, frog_imgs_feat.shape)

    px = torch.cat((frog_imgs_feat, car_imgs_feat), dim=0)
    px = torch.cat((target_img_feat, px), dim=0)
    print(px.shape)

    px = px.to('cpu')
    #tsne_em = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1).fit_transform(px)
    #from bioinfokit.visuz import cluster
    #cluster.tsneplot(score=tsne_em)
    torch.save(px, 'feat_arr.pt') 






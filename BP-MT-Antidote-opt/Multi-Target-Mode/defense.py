import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import argparse
import os
from utils import load_pretrained_net, fetch_all_external_targets
from dataloader import PoisonedDataset, FeatureSet
import pandas as pd
import time
from scipy.special import softmax
from scipy.spatial.distance import cdist
from collections import Counter
from torch.utils.data import Subset

import sys
sys.path.append("../")
from models import *

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_network_with_poison(net, eval_targets, poison_tuple_list, poisoned_dset, args, testset, device='cuda', clean_test_acc=False):
    # requires implementing a get_penultimate_params_list() method to get the parameter identifier of the net's last
    # layer

    params = net.module.get_penultimate_params_list()

    if args.retrain_opt == 'adam':
        print("Using Adam for retraining")
        optimizer = torch.optim.Adam(params, lr=args.retrain_lr, weight_decay=args.retrain_wd)
    else:
        print("Using SGD for retraining")
        optimizer = torch.optim.SGD(params, lr=args.retrain_lr, momentum=args.retrain_momentum,
                                    weight_decay=args.retrain_wd)

    net.eval()

    criterion = nn.CrossEntropyLoss().to(device)

    poisoned_loader = torch.utils.data.DataLoader(poisoned_dset, batch_size=args.retrain_bsize, shuffle=True)
    # The test set of clean CIFAR10
    test_loader = torch.utils.data.DataLoader(testset, batch_size=500)

    # create a dataloader that returns the features
    poisoned_loader = torch.utils.data.DataLoader(FeatureSet(poisoned_loader, net, device=args.device),
                                                  batch_size=64, shuffle=True)

    for epoch in range(args.retrain_epochs):
        net.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        time_meter = AverageMeter()

        if epoch in args.lr_decay_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        end_time = time.time()
        for ite, (input, label) in enumerate(poisoned_loader):
            if args.device == 'cuda':
                input, label = input.to('cuda'), label.to(device)

            output = net.module.linear(input)

            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec1 = accuracy(output, label)[0]

            time_meter.update(time.time() - end_time)
            end_time = time.time()
            loss_meter.update(loss.item(), input.size(0))
            acc_meter.update(prec1.item(), input.size(0))

            if epoch % 50 == 0 and (ite == len(poisoned_loader) - 1):
                print("{2}, Epoch {0}, Iteration {1}, loss {loss.val:.3f} ({loss.avg:.3f}), "
                      "acc {acc.val:.3f} ({acc.avg:.3f})".
                      format(epoch, ite, time.strftime("%Y-%m-%d %H:%M:%S"),
                             loss=loss_meter, acc=acc_meter))
            sys.stdout.flush()


    eval_targets_acc_meter = AverageMeter()
    val_targets_preds = []
    val_targets_scores = []
    with torch.no_grad():
        for ite, (input, target) in enumerate(eval_targets):
            input, target = input.to(device), target.to(device)
            output = net(input)
            _, pred = output.topk(1, 1, True, True)
            val_targets_preds.append(str(pred[0][0].item()))
            target_scores = [float(n) for n in list(softmax(output.view(-1).cpu().detach().numpy()))]
            val_targets_scores.append(target_scores)

            prec1 = accuracy(output, torch.tensor([args.poison_label]).to(device))[0]
            eval_targets_acc_meter.update(prec1.item(), input.size(0))
    print("Predictions for the external targets: {}".format(", ".join(val_targets_preds)))
    print("EVAL ATTACK acc: {}".format(eval_targets_acc_meter.avg))

    poison_pred_list = []
    for poison_img, _ in poison_tuple_list:
        base_scores = net(poison_img[None, :, :, :].to(args.device))
        base_score, base_pred = base_scores.topk(1, 1, True, True)
        poison_pred_list.append(base_pred.item())
    print(
        "Target Label: {}, Poison label: {}, Poisons' Predictions:{}".format(
            args.target_label, args.poison_label, poison_pred_list))

    clean_acc = -1.0
    if clean_test_acc:
        # Evaluate the results on the clean test set
        val_acc_meter = AverageMeter()
        with torch.no_grad():
            for ite, (input, label) in enumerate(test_loader):
                input, label = input.to(args.device), label.to(args.device)

                output = net(input)

                prec1 = accuracy(output, label)[0]
                val_acc_meter.update(prec1.item(), input.size(0))

                if False or ite % 100 == 0 or ite == len(test_loader) - 1:
                    print("{2} Epoch {0}, Val iteration {1}, "
                          "acc {acc.val:.3f} ({acc.avg:.3f})".
                          format(epoch, ite, time.strftime("%Y-%m-%d %H:%M:%S"), acc=val_acc_meter))

        print("* Prec: {}".format(val_acc_meter.avg))
        clean_acc = val_acc_meter.avg

    return {'clean acc': clean_acc, 'prediction': val_targets_preds,
            'poisons predictions': poison_pred_list, 'attack acc': eval_targets_acc_meter.avg,
            'scores': target_scores, 'malicious score': target_scores[args.poison_label]}


def knn_defense(net, poison_list, poison_base_indices, poisoned_dset, k):
    poisoned_loader = torch.utils.data.DataLoader(poisoned_dset, batch_size=len(poisoned_dset), shuffle=False)
    poisoned_loader = torch.utils.data.DataLoader(FeatureSet(poisoned_loader, net, device=args.device),
                                                  batch_size=len(poisoned_dset), shuffle=False)

    features, labels = [batch for batch in poisoned_loader][0]
    features = features.cpu().numpy()

    pairwise_distances = cdist(features, features)

    nearest_neighbors = pairwise_distances.argsort(axis=1)[:, 1:k + 1]

    benign_indices = []  # detected by the defense
    for sample_idx, (sample_label, sample_nearest_neighbors) in enumerate(zip(labels, nearest_neighbors)):
        c = Counter()
        for neighbor_idx in sample_nearest_neighbors:
            c[labels[neighbor_idx].item()] += 1

        c = c.most_common()
        most_cnt = c[0][1]
        mod_labels = []
        for label, label_cnt in c:
            if label_cnt == most_cnt:
                mod_labels.append(label)
            else:
                break

        if sample_label.item() in mod_labels:
            benign_indices.append(sample_idx)

    poisoned_dset_filtered = Subset(poisoned_dset, indices=benign_indices)
    poison_filtered_tuple_list = []
    deleted_poisons_base_indices = []
    for idx, p in enumerate(poison_list):
        if idx in benign_indices:
            poison_filtered_tuple_list.append(p)
        else:
            deleted_poisons_base_indices.append(poison_base_indices[idx])

    num_deleted_samples = len(poisoned_dset) - len(poisoned_dset_filtered)

    return poisoned_dset_filtered, poison_filtered_tuple_list, deleted_poisons_base_indices, num_deleted_samples


def l2outlier_defense(net, poison_list, poison_base_indices, poisoned_dset, fraction):
    poisoned_loader = torch.utils.data.DataLoader(poisoned_dset, batch_size=len(poisoned_dset), shuffle=False)
    poisoned_loader = torch.utils.data.DataLoader(FeatureSet(poisoned_loader, net, device=args.device),
                                                  batch_size=len(poisoned_dset), shuffle=False)

    features, labels = [batch for batch in poisoned_loader][0]
    features = features.cpu().numpy()

    benign_indices = []
    for c in range(10):  # CIFAR10 class labels are zero to nine!
        c_indices = [idx for idx, label in enumerate(labels) if label == c]
        c_features = features[c_indices]
        c_center = c_features.mean(axis=0)
        diff = np.linalg.norm(c_features - c_center, axis=1)

        num = math.ceil(len(c_indices) * (1 - fraction))
        c_filtered_indices = [c_indices[idx] for idx in diff.argsort()[:num]]
        benign_indices.extend(c_filtered_indices)

    poisoned_dset_filtered = Subset(poisoned_dset, indices=benign_indices)
    poison_filtered_tuple_list = []
    deleted_poisons_base_indices = []
    for idx, p in enumerate(poison_list):
        if idx in benign_indices:
            poison_filtered_tuple_list.append(p)
        else:
            deleted_poisons_base_indices.append(poison_base_indices[idx])

    num_deleted_samples = len(poisoned_dset) - len(poisoned_dset_filtered)

    return poisoned_dset_filtered, poison_filtered_tuple_list, deleted_poisons_base_indices, num_deleted_samples


if __name__ == '__main__':
    # ======== arg parser =================================================
    parser = argparse.ArgumentParser(description='PyTorch Poison Attack')
    parser.add_argument('--gpu', default='0', type=str)
    # The substitute models and the victim models
    parser.add_argument('--target-net', default=["DenseNet121"], nargs="+", type=str)
    parser.add_argument("--test-chk-name", default='ckpt-%s-4800.t7', type=str)
    parser.add_argument('--model-resume-path', default='../models-chks-release', type=str,
                        help="Path to the pre-trained models")
    parser.add_argument('--subset-group', default=0, type=int)

    # Parameters for poisons
    parser.add_argument('--target-label', default=1, type=int)
    parser.add_argument('--poison-label', '-plabel', default=6, type=int,
                        help='label of the poisons, or the target label we want to classify into')
    parser.add_argument('--target-subset', default=6, type=int,
                                    help='model of the car in epfl-gims08 dataset')
    parser.add_argument('--target-num', default=5, type=int)
    parser.add_argument('--target-path', default='../datasets/epfl-gims08/resized/tripod_seq', type=str,
                                    help='path to the external images')
    parser.add_argument('--target-start', default='-1', type=int,
                                    help='first index of the car in epfl-gims08 dataset')
    parser.add_argument('--target-end', default='-1', type=int,
                                    help='last index of the car in epfl-gims08 dataset')

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
    parser.add_argument('--eval-poisons-root', default='', type=str,
                        help="Root folder containing poisons crafted for the targets")
    parser.add_argument('--train-data-path', default='../datasets/CIFAR10_TRAIN_Split.pth', type=str,
                        help='path to the official datasets')
    parser.add_argument('--dset-path', default='datasets', type=str,
                        help='path to the official datasets')

    # Defenses parameters
    parser.add_argument('--knn-defense-k', default=range(1, 21), type=int, nargs="+")
    parser.add_argument('--l2outlier-defense-fraction', default=np.arange(0.02, 0.41, 0.02))

    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    print(args)

    # Set visible CUDA devices
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        cudnn.benchmark = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''

    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    testset = torchvision.datasets.CIFAR10(root=args.dset_path, train=False, download=True, transform=transform_test)

    from torch.utils.data import random_split

    # testset, _ = random_split(testset, (100, 9900))
    # testset.data = testset.data[:1000] # just for the speed

    defenses_dir = '{}/defenses/target-num-{}'.format(args.eval_poisons_root, args.target_num)
    if not os.path.exists(defenses_dir):
        os.makedirs(defenses_dir)

    no_defense_res_path = "{}/no-defense.pickle".format(defenses_dir)
    if os.path.exists(no_defense_res_path):
        no_defense_res = pd.read_pickle(no_defense_res_path)
    else:
        no_defense_res = pd.DataFrame(
            columns=["target_idx", "victim_net", "ite", "poisons_base_indices", "victim_net_res",
                     "victim_net_test_acc", "victim_net_adv_succ"])

    knn_defense_res_path = "{}/knn.pickle".format(defenses_dir)
    if os.path.exists(knn_defense_res_path):
        knn_defense_res = pd.read_pickle(knn_defense_res_path)
    else:
        knn_defense_res = pd.DataFrame(
            columns=["target_idx", "victim_net", "ite", "poisons_base_indices", "deleted_poisons_base_indices",
                     "num_deleted_samples", "k", "victim_net_res", "victim_net_test_acc", "victim_net_adv_succ"])

    l2outlier_defense_res_path = "{}/l2outlier.pickle".format(defenses_dir)
    if os.path.exists(l2outlier_defense_res_path):
        l2outlier_defense_res = pd.read_pickle(l2outlier_defense_res_path)
    else:
        l2outlier_defense_res = pd.DataFrame(
            columns=["target_idx", "victim_net", "ite", "poisons_base_indices", "deleted_poisons_base_indices",
                     "num_deleted_samples", "fraction", "victim_net_res", "victim_net_test_acc", "victim_net_adv_succ"])

    target_subsets = [1, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19]

    for target_subset in target_subsets:

        print("Target_subset: {}".format(target_subset))

        # Get the target images
        _targets, targets_indices, eval_targets, val_targets_indices = \
            fetch_all_external_targets(args.target_label,
                                       args.target_path,
                                       args.target_subset,
                                       args.target_start,
                                       args.target_end,
                                       args.target_num,
                                       transforms=transform_test)

        ites = [1000]

        for ite in ites[::-1]:

            poisons_path = '{}/{}/target-num-{}/{}'.format(args.eval_poisons_root, target_subset, args.target_num,
                                                           "poison_%05d.pth" % (ite - 1))
            print("ITE: {}".format(ite))
            print("Loading poisons from {}".format(poisons_path))
            if not os.path.exists(poisons_path):
                print("We processed all poisons (for all iterations of attack) for target: {}".format(target_subset))
                break
            if args.device == 'cuda':
                state_dict = torch.load(poisons_path)
            else:
                state_dict = torch.load(poisons_path, map_location=torch.device('cpu'))

            poison_tuple_list, base_idx_list = state_dict['poison'], state_dict['idx']
            print("Poisons loaded")
            poisoned_dset = PoisonedDataset(args.train_data_path, subset='others', transform=transform_train,
                                            num_per_label=args.num_per_class, poison_tuple_list=poison_tuple_list,
                                            poison_indices=base_idx_list, subset_group=args.subset_group)
            print("Poisoned dataset created")

            for victim_name in args.target_net:
                print(victim_name)

                l = len(no_defense_res[no_defense_res.ite == ite]
                        [no_defense_res.target_idx == target_subset]
                        [no_defense_res.victim_net == victim_name])
                assert l <= 1
                #if False and l == 0:
                if True:

                    victim_net = load_pretrained_net(victim_name, args.test_chk_name, model_chk_path=args.model_resume_path,
                                                     device=args.device)
                    res = train_network_with_poison(victim_net, eval_targets, poison_tuple_list,
                                                    poisoned_dset, args, testset)

                    victim_res_test_acc = res['clean acc']
                    victim_res_adv_succ = res['attack acc']

                    attack_acc = res['attack acc']
                    
                    new_row = [target_subset, victim_name, ite, base_idx_list,
                               res, victim_res_test_acc, victim_res_adv_succ]
                    new_row = {col: val for col, val in zip(no_defense_res.columns, new_row)}
                    no_defense_res = no_defense_res.append(new_row, ignore_index=True)

                    no_defense_res.to_pickle(no_defense_res_path)
                else:
                    print("Skipping updating no_defense_res for ite: {}, target_idx: {}, victim_name: {}".format(ite, target_subset, victim_name))

                print()
                print("BEFORE DEFENSE FILTERING : ", victim_name, attack_acc)
                avg_attack_acc = 0
                # Now evaluate the victim after knn defense deployed!
                for k in args.knn_defense_k:

                    l = len(knn_defense_res[knn_defense_res.ite == ite]
                            [knn_defense_res.target_idx == target_subset]
                            [knn_defense_res.victim_net == victim_name]
                            [knn_defense_res.k == k])
                    #assert l <= 1
                    #if True and l > 0:
                    #    print("Skipping updating knn_defense_res for ite: {}, target_idx: {}, victim_name: {}, k: {}".format(ite, target_subset, victim_name, k))

                    victim_net = load_pretrained_net(victim_name, args.test_chk_name,
                                                     model_chk_path=args.model_resume_path,
                                                     device=args.device)

                    poisoned_dset_filtered, poison_filtered_tuple_list, deleted_poisons_base_indices, num_deleted_samples = \
                        knn_defense(victim_net, poison_tuple_list, base_idx_list, poisoned_dset, k)

                    res = train_network_with_poison(victim_net, eval_targets, poison_filtered_tuple_list,
                                                    poisoned_dset_filtered, args, testset)
                    victim_res_test_acc = res['clean acc']
                    victim_res_adv_succ = res['attack acc']

                    avg_attack_acc += res['attack acc']
                    new_row = [target_subset, victim_name, ite, base_idx_list, deleted_poisons_base_indices,
                               num_deleted_samples, k, res, victim_res_test_acc, victim_res_adv_succ]
                    new_row = {col: val for col, val in zip(knn_defense_res.columns, new_row)}
                    knn_defense_res = knn_defense_res.append(new_row, ignore_index=True)
                    
                    knn_defense_res.to_pickle(knn_defense_res_path)
                
                print("AFTER DEFENSE FILTERING : ", avg_attack_acc/20.0)

                """
                # Now evaluate the victim after l2-norm-outlier defense deployed
                for fraction in args.l2outlier_defense_fraction:

                    l = len(l2outlier_defense_res[l2outlier_defense_res.ite == ite]
                            [l2outlier_defense_res.target_idx == target_subset]
                            [l2outlier_defense_res.victim_net == victim_name]
                            [l2outlier_defense_res.fraction == fraction])
                    assert l <= 1
                    if l > 0:
                        print("Skipping updating l2outlier_defense_res for ite: {}, target_idx: {}, victim_name: {}, fraction: {}".format(ite, target_subset, victim_name, fraction))

                    victim_net = load_pretrained_net(victim_name, args.test_chk_name,
                                                     model_chk_path=args.model_resume_path,
                                                     device=args.device)

                    poisoned_dset_filtered, poison_filtered_tuple_list, deleted_poisons_base_indices, num_deleted_samples = \
                        l2outlier_defense(victim_net, poison_tuple_list, base_idx_list, poisoned_dset, fraction)

                    res = train_network_with_poison(victim_net, eval_targets, poison_filtered_tuple_list,
                                                    poisoned_dset_filtered, args, testset)
                    victim_res_test_acc = res['clean acc']
                    victim_res_adv_succ = res['attack acc']

                    new_row = [target_subset, victim_name, ite, base_idx_list, deleted_poisons_base_indices,
                               num_deleted_samples, fraction, res, victim_res_test_acc, victim_res_adv_succ]
                    new_row = {col: val for col, val in zip(l2outlier_defense_res.columns, new_row)}
                    l2outlier_defense_res = l2outlier_defense_res.append(new_row, ignore_index=True)

                    l2outlier_defense_res.to_pickle(l2outlier_defense_res_path)
                """

                print("Done with evaluating the defenses for vicitm: {}".format(victim_name))

            print("Done with poisons for ite: {}".format(ite))

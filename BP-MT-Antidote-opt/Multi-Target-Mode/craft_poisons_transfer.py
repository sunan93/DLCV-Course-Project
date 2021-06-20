import random
import sys
import pandas as pd

sys.path.append("../")
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from scipy.spatial.distance import cdist
from collections import Counter
from torch.utils.data import Subset

import argparse
import os
from models import *
from utils import load_pretrained_net, fetch_all_external_targets, \
    fetch_nearest_poison_bases, fetch_poison_bases
from trainer import make_convex_polytope_poisons, train_network_with_poison
from dataloader import PoisonedDataset, FeatureSet

def seed_everything(seed=233):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



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

def knn_defense(net, poison_list, poison_base_indices, poisoned_dset, k, antidote_list, antidote_base_indices):
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

    print("benign indices detected : ", benign_indices)
    poisoned_dset_filtered = Subset(poisoned_dset, indices=benign_indices)
    poison_filtered_tuple_list = []
    deleted_poisons_base_indices = []
    for idx, p in enumerate(poison_list):
        if idx in benign_indices:
            poison_filtered_tuple_list.append(p)
        else:
            deleted_poisons_base_indices.append(poison_base_indices[idx])

    antidote_filtered_tuple_list = []
    deleted_antidote_base_indices = []
    for idx, p in enumerate(antidote_list):
        if idx in benign_indices:
            antidote_filtered_tuple_list.append(p)
        else:
            deleted_antidote_base_indices.append(antidote_base_indices[idx])

    num_deleted_samples = len(poisoned_dset) - len(poisoned_dset_filtered)

    return poisoned_dset_filtered, poison_filtered_tuple_list, deleted_poisons_base_indices, num_deleted_samples, antidote_filtered_tuple_list, deleted_antidote_base_indices



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
    parser.add_argument('--eval-poisons-root', default='', type=str,
                        help="Root folder containing poisons crafted for the targets")

    parser.add_argument('--mode', default='convex', type=str,
                        help='if convex, run the convexpolytope attack proposed by the paper, '
                             'otherwise run the mean method')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--ce_loss_flag', default='False', choices=['True', 'False'], type=str)
    parser.add_argument('--max_loss_flag', default='False', choices=['True', 'False'], type=str)
    parser.add_argument('--feat_reg_flag', default='False', choices=['True', 'False'], type=str)
    parser.add_argument('--multi_flag', default='False', choices=['True', 'False'], type=str)
    parser.add_argument('--antidote_flag', default='False', choices=['True', 'False'], type=str)
    args = parser.parse_args()
    
    if args.ce_loss_flag=='False':
        args.ce_loss_flag = False
    else:
        args.ce_loss_flag = True

    if args.max_loss_flag=='False':
        args.max_loss_flag = False
    else:
        args.max_loss_flag = True

    if args.feat_reg_flag=='False':
        args.feat_reg_flag = False
    else:
        args.feat_reg_flag = True

    if args.multi_flag=='False':
        args.multi_flag = False
    else:
        args.multi_flag = True
   
    if args.antidote_flag=='False':
        args.antidote_flag = False
    else:
        args.antidote_flag = True


    seed_everything()
    print(args.end2end, args.ce_loss_flag, args.max_loss_flag, args.feat_reg_flag, args.multi_flag)
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
    targets = [x for x, _ in _targets]
    # assert len(targets) > 1, "we have only one target, but the multiple_target mode is enabled"
    print("All target indices used in crafting poisons: {}".format(targets_indices))

    # creating result directory and log file
    if args.mode == 'mean':
        chk_path = os.path.join(args.chk_path, 'mean')
    else:
        chk_path = os.path.join(args.chk_path, args.mode)
    if args.net_repeat > 1:
        chk_path = '{}-{}Repeat'.format(chk_path, args.net_repeat)
    chk_path = os.path.join(chk_path, str(args.poison_ites))
    chk_path = os.path.join(chk_path, str(args.target_index))
    chk_path = os.path.join(chk_path, "target-num-{}".format(args.target_num))
    if not os.path.exists(chk_path):
        os.makedirs(chk_path)
    import sys

    sys.stdout = Logger('{}/log.txt'.format(chk_path))

    # Load or craft the poison!
    if args.eval_poison_path != "":
        state_dict = torch.load(args.eval_poison_path)
        poison_tuple_list, base_idx_list = state_dict['poison'], state_dict['idx']
        print("=" * 100)
        print("=" * 100)
        print("Poisons loaded")
        print("Now evaluating on the target nets")
        t = 0
        tt = 0
        assert False, "ToDo: check here!"
    else:
        print(args)
        print("Path: {}".format(chk_path))

        # Otherwise, we craft new poisons
        if args.nearest:
            base_tensor_list, base_idx_list = \
                fetch_nearest_poison_bases(sub_net_list, targets, args.poison_num,
                                           args.poison_label, 500, subset='others',
                                           train_data_path=args.train_data_path, transforms=transform_test, )

        else:
            # just fetch the first poison_num samples
            base_tensor_list, base_idx_list = fetch_poison_bases(args.poison_label, args.poison_num, subset='others',
                                                                 path=args.train_data_path, transforms=transform_test)
        
        #if args.antidote_flag:
        base_antidotes_tensor_list, base_antidotes_idx_list = \
                fetch_nearest_poison_bases(sub_net_list, targets, args.poison_num,
                                           args.target_label, 500, subset='others',
                                           train_data_path=args.train_data_path, transforms=transform_test, )

        base_antidotes_tensor_list = [bt.to(args.device) for bt in base_antidotes_tensor_list]
        antidote_init = base_antidotes_tensor_list
            
        print("Selected antidote image indices: {}".format(base_antidotes_idx_list))
        """
        else:
            base_antidotes_tensor_list=None
            base_antidotes_idx_list=None
            antidote_init = None
        """
        base_tensor_list = [bt.to(args.device) for bt in base_tensor_list]
        print("Selected base image indices: {}".format(base_idx_list))
        print("Target indices: {}".format(targets_indices))
        print("# Targets: {}, # Poisons: {}".format(len(targets), len(base_tensor_list)))

        if args.resume_poison_ite > 0:
            state_dict = torch.load(os.path.join(chk_path, "poison_%05d.pth" % args.resume_poison_ite))
            poison_tuple_list, base_idx_list = state_dict['poison'], state_dict['idx']
            poison_init = [pt.to('cuda') for pt, _ in poison_tuple_list]
            # re-direct the results to the resumed dir...
            chk_path += '-resume'
            if not os.path.exists(chk_path):
                os.makedirs(chk_path)
        else:
            poison_init = base_tensor_list

        import time

        t = time.time()
        """ 
        poison_tuple_list, antidote_tuple_list = \
            make_convex_polytope_poisons(sub_net_list, victim_net, base_tensor_list, base_antidotes_tensor_list, targets,
                                         targets_indices, device=args.device,
                                         opt_method=args.poison_opt,
                                         lr=args.poison_lr, momentum=args.poison_momentum, iterations=args.poison_ites,
                                         epsilon=args.poison_epsilon, decay_ites=args.poison_decay_ites,
                                         mean=torch.Tensor(cifar_mean).reshape(1, 3, 1, 1),
                                         std=torch.Tensor(cifar_std).reshape(1, 3, 1, 1),
                                         decay_ratio=args.poison_decay_ratio, chk_path=chk_path, end2end=args.end2end,
                                         poison_idxes=base_idx_list, antidote_idxs=base_antidotes_idx_list, poison_label=args.poison_label,target_label=args.target_label, 
                                         mode=args.mode,
                                         tol=args.tol, start_ite=args.resume_poison_ite, poison_init=poison_init, antidote_init=antidote_init,
                                         net_repeat=args.net_repeat, max_loss_flag=args.max_loss_flag, 
                                         feat_reg_flag=args.feat_reg_flag,ce_loss_flag=args.ce_loss_flag, multi_flag=args.multi_flag, antidote_flag=False )

        """
        poison_tuple_list2, antidote_tuple_list2 = \
            make_convex_polytope_poisons(sub_net_list, victim_net, base_tensor_list, base_antidotes_tensor_list, targets,
                                         targets_indices, device=args.device,
                                         opt_method=args.poison_opt,
                                         lr=args.poison_lr, momentum=args.poison_momentum, iterations=args.poison_ites,
                                         epsilon=args.poison_epsilon, decay_ites=args.poison_decay_ites,
                                         mean=torch.Tensor(cifar_mean).reshape(1, 3, 1, 1),
                                         std=torch.Tensor(cifar_std).reshape(1, 3, 1, 1),
                                         decay_ratio=args.poison_decay_ratio, chk_path=chk_path, end2end=args.end2end,
                                         poison_idxes=base_idx_list, antidote_idxs=base_antidotes_idx_list, poison_label=args.poison_label,target_label=args.target_label, 
                                         mode=args.mode,
                                         tol=args.tol, start_ite=args.resume_poison_ite, poison_init=poison_init, antidote_init=antidote_init,
                                         net_repeat=args.net_repeat, max_loss_flag=args.max_loss_flag, 
                                         feat_reg_flag=args.feat_reg_flag,ce_loss_flag=args.ce_loss_flag, multi_flag=args.multi_flag, antidote_flag=args.antidote_flag )

        tt = time.time()


    torch.save(poison_tuple_list2, 'poison_list.pt')
    torch.save(antidote_tuple_list2, 'antidote_list.pt')
    torch.save(poison_init, 'poison_init.pt')
    torch.save(antidote_init, 'antidote_init.pt')

    """
    res = []
    num_trials = 1
    print("Evaluating against victims networks")
    
    for vnet, vnet_name in zip(targets_net, args.victim_net):
        attc = 0
        val_attc = 0
        pre_acc = 0
        post_acc = 0
        for i in range(num_trials):
            print(vnet_name)
            attack_acc, attack_val_acc, pre_test_acc, post_test_acc = train_network_with_poison(vnet, targets, targets_indices, poison_tuple_list, antidote_tuple_list,
                                                                   base_idx_list, base_antidotes_idx_list, chk_path,
                                                                   args, save_state=False, eval_targets=eval_targets, load_poisons=0)
            attc += attack_acc
            val_attc += attack_val_acc
            pre_acc += pre_test_acc
            post_acc += post_test_acc
        res.append((attc/num_trials, val_attc/num_trials, pre_acc/num_trials, post_acc/num_trials))
        print("--------")


    print("------SUMMARY WITHOUT POISONS------")
    print("TIME ELAPSED (mins): {}".format(int((tt - t) / 60)))
    print("TARGET INDEX: {}".format(args.target_index))
    for tnet_name, (attack_acc, attack_val_acc, pre_test_acc, post_test_acc) in zip(args.victim_net, res):
        print(tnet_name, attack_acc, attack_val_acc, pre_test_acc, post_test_acc)
   
    """ 
    """
    seed_everything()
    print("Loading the victims networks again :")
    targets_net = []
    for vnet in args.victim_net:
        victim_net = load_pretrained_net(vnet, args.test_chk_name, model_chk_path=args.model_resume_path)
        targets_net.append(victim_net)

    res = []
    num_trials = 1
    print("Evaluating against victims networks")
    for vnet, vnet_name in zip(targets_net, args.victim_net):
        attc = 0
        val_attc = 0
        pre_acc = 0
        post_acc = 0
        for i in range(num_trials):
            print(vnet_name)
            attack_acc, attack_val_acc, pre_test_acc, post_test_acc = train_network_with_poison(vnet, targets, targets_indices, poison_tuple_list, antidote_tuple_list,
                                                                   base_idx_list, base_antidotes_idx_list, chk_path,
                                                                   args, save_state=False, eval_targets=eval_targets, load_poisons=1)
            attc += attack_acc
            val_attc += attack_val_acc
            pre_acc += pre_test_acc
            post_acc += post_test_acc
        res.append((attc/num_trials, val_attc/num_trials, pre_acc/num_trials, post_acc/num_trials))
        print("--------")


    print("------SUMMARY WITH POISONS------")
    print("TIME ELAPSED (mins): {}".format(int((tt - t) / 60)))
    print("TARGET INDEX: {}".format(args.target_index))
    for tnet_name, (attack_acc, attack_val_acc, pre_test_acc, post_test_acc) in zip(args.victim_net, res):
        print(tnet_name, attack_acc, attack_val_acc, pre_test_acc, post_test_acc)


   """
    
    seed_everything()
    print("Loading the victims networks second time :")
    targets_net = []
    for vnet in args.victim_net:
        victim_net = load_pretrained_net(vnet, args.test_chk_name, model_chk_path=args.model_resume_path)
        targets_net.append(victim_net)

    res = []
    num_trials = 1
    print("Evaluating against victims networks")
    for vnet, vnet_name in zip(targets_net, args.victim_net):
        attc = 0
        val_attc = 0
        pre_acc = 0
        post_acc = 0
        for i in range(num_trials):
            print(vnet_name)
            attack_acc, attack_val_acc, pre_test_acc, post_test_acc = train_network_with_poison(vnet, targets, targets_indices, poison_tuple_list2, antidote_tuple_list2,
                                                                   base_idx_list, base_antidotes_idx_list, chk_path,
                                                                   args, save_state=False, eval_targets=eval_targets, load_poisons=1)
            attc += attack_acc
            val_attc += attack_val_acc
            pre_acc += pre_test_acc
            post_acc += post_test_acc
        res.append((attc/num_trials, val_attc/num_trials, pre_acc/num_trials, post_acc/num_trials))
        print("--------")


    print("------SUMMARY POISONS + ANTIDOTES(PRE DEFENSE)------")
    print("TIME ELAPSED (mins): {}".format(int((tt - t) / 60)))
    print("TARGET INDEX: {}".format(args.target_index))
    for tnet_name, (attack_acc, attack_val_acc, pre_test_acc, post_test_acc) in zip(args.victim_net, res):
        print(tnet_name, attack_acc, attack_val_acc, pre_test_acc, post_test_acc)


    print("DEFENSE EVALUATION")

    """
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

    """
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    target_subsets = [args.target_index]


    poisoned_dset = PoisonedDataset(args.train_data_path, subset='others', transform=transform_train,
                                    num_per_label=args.num_per_class, poison_tuple_list=poison_tuple_list2,
                                    poison_indices=base_idx_list, antidote_tuple_list=antidote_tuple_list2,
                                    antidote_indices=base_antidotes_idx_list,subset_group=args.subset_group, load_poisons=1)



    print("Loading the victims networks")
    targets_net = []
    for vnet in args.victim_net:
        victim_net = load_pretrained_net(vnet, args.test_chk_name, model_chk_path=args.model_resume_path)
        targets_net.append(victim_net)

    res = []
    for vnet, vnet_name in zip(targets_net, args.victim_net):
        print(vnet_name)
        poisoned_dset_filtered, poison_filtered_tuple_list, deleted_poisons_base_indices, num_deleted_samples, \
                        antidote_filtered_tuple_list, deleted_antidote_base_indices = \
                        knn_defense(vnet, poison_tuple_list2, base_idx_list, poisoned_dset, 5, antidote_tuple_list2, base_antidotes_idx_list)
        print(deleted_poisons_base_indices, num_deleted_samples)

        filtered_list = []
        for idx in base_idx_list:
            if idx not in deleted_poisons_base_indices:
                filtered_list.append(idx)
        filtered_antidote_list = []
        for idx in base_antidotes_idx_list:
            if idx not in deleted_antidote_base_indices:
                filtered_antidote_list.append(idx)
        print("Filtered lists : ", filtered_list, filtered_antidote_list)
        #attack_acc, attack_val_acc, pre_test_acc, post_test_acc = train_network_with_poison(vnet, targets, targets_indices, poison_filtered_tuple_list,
        #                                                           filtered_list, chk_path,
        #                                                           args, save_state=False, eval_targets=eval_targets)

        attack_acc, attack_val_acc, pre_test_acc, post_test_acc = train_network_with_poison(vnet, targets, targets_indices, poison_filtered_tuple_list, 
                                                                   antidote_filtered_tuple_list,
                                                                   filtered_list, filtered_antidote_list, chk_path,
                                                                   args, save_state=False, eval_targets=eval_targets, load_poisons=1)
        res.append((attack_acc, attack_val_acc, pre_test_acc, post_test_acc, (args.poison_num - len(filtered_list))*100/args.poison_num, (args.poison_num - len(filtered_antidote_list))*100/args.poison_num))



    print("------SUMMARY POISONS + ANTIDOTES(POST DEFENSE)------")
    print("TIME ELAPSED (mins): {}".format(int((tt - t) / 60)))
    print("TARGET INDEX: {}".format(args.target_index))
    for tnet_name, (attack_acc, attack_val_acc, pre_test_acc, post_test_acc, red_poison, red_antidote) in zip(args.victim_net, res):
        print(tnet_name, attack_acc, attack_val_acc, pre_test_acc, post_test_acc, red_poison, red_antidote) 

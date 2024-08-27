#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from random import random
from models.test import test_img, test_mnist7
from models.Fed import FedAvg
from models.Nets import ResNet18, vgg19_bn, vgg19, get_model

from models.MaliciousUpdate import LocalMaliciousUpdate
from models.Update import LocalUpdate
from utils.info import print_exp_details, write_info_to_accfile, get_base_info
from utils.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, dirichlet, mnist7
from utils.defense import fltrust, multi_krum, get_update, RLR, flame, fl_defender, multi
import torch
from torchvision import datasets, transforms
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
import os
import random
import time
import math

from torch.utils.tensorboard import SummaryWriter
matplotlib.use('Agg')


def write_file(filename, accu_list, back_list, args, analyse = False):
    write_info_to_accfile(filename, args)
    f = open(filename, "a")
    f.write("main_task_accuracy=")
    f.write(str(accu_list))
    f.write('\n')
    f.write("backdoor_accuracy=")
    f.write(str(back_list))
    if args.defence == "krum":
        krum_file = filename+"_krum_dis"
        torch.save(args.krum_distance,krum_file)
    if analyse == True:
        need_length = len(accu_list)//10
        acc = accu_list[-need_length:]
        back = back_list[-need_length:]
        best_acc = round(max(acc),2)
        average_back=round(np.mean(back),2)
        best_back=round(max(back),2)
        f.write('\n')
        f.write('BBSR:')
        f.write(str(best_back))
        f.write('\n')
        f.write('ABSR:')
        f.write(str(average_back))
        f.write('\n')
        f.write('max acc:')
        f.write(str(best_acc))
        f.write('\n')
        f.close()
        return best_acc, average_back, best_back
    f.close()


def central_dataset_iid(dataset, dataset_size):
    all_idxs = [i for i in range(len(dataset))]
    central_dataset = set(np.random.choice(
        all_idxs, dataset_size, replace=False))
    return central_dataset

def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def seed_experiment(seed=0):
    # seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #TODO: Do we need deterministic in cudnn ? Double check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    test_mkdir('./'+args.save)
    print_exp_details(args)
    seed_experiment(args.seed)

    if args.defence == 'krum' or args.defence == 'multi-krum':
        krum_writer = SummaryWriter("../hlogs/krum/{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.attack, args.malicious, args.defence, args.heter, args.alpha, args.lr_b, args.lr))
        if math.isclose(args.malicious, 0) == False:
            ba_writer = SummaryWriter("../hlogs/BA/{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.attack, args.malicious, args.defence, args.heter, args.alpha, args.lr_b, args.lr))
        ma_writer = SummaryWriter("../hlogs/MA/{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.attack, args.malicious, args.defence, args.heter, args.alpha, args.lr_b, args.lr))
    elif args.defence == 'RLR':
        RLR_writer = SummaryWriter("../hlogs/RLR/{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.attack, args.malicious, args.defence, args.heter, args.alpha, args.lr_b, args.lr, args.robustLR_threshold))
        if math.isclose(args.malicious, 0) == False:
            ba_writer = SummaryWriter("../hlogs/BA/{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.attack, args.malicious, args.defence, args.heter, args.alpha, args.lr_b, args.lr, args.robustLR_threshold))
        ma_writer = SummaryWriter("../hlogs/MA/{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.attack, args.malicious, args.defence, args.heter, args.alpha, args.lr_b, args.lr, args.robustLR_threshold))
    elif args.defence == 'fl_defender':
        fldefender_file = f'../files/{args.dataset}_{args.attack}_{args.malicious}_{args.defence}_{args.heter}_{args.alpha}_{args.lr_b}_{args.lr}.txt'
        if math.isclose(args.malicious, 0) == False:
            ba_writer = SummaryWriter("../hlogs/BA/{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.attack, args.malicious, args.defence, args.heter, args.alpha, args.lr_b, args.lr))
        ma_writer = SummaryWriter("../hlogs/MA/{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.attack, args.malicious, args.defence, args.heter, args.alpha, args.lr_b, args.lr))
    elif args.defence == 'flame':
        flame_norm_writer = SummaryWriter("../hlogs/flame_norm/{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.attack, args.malicious, args.defence, args.heter, args.alpha, args.lr_b, args.lr))
        if math.isclose(args.malicious, 0) == False:
            ba_writer = SummaryWriter("../hlogs/BA/{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.attack, args.malicious, args.defence, args.heter, args.alpha, args.lr_b, args.lr))
        ma_writer = SummaryWriter("../hlogs/MA/{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.attack, args.malicious, args.defence, args.heter, args.alpha, args.lr_b, args.lr))
    elif args.defence == 'multi':
        if math.isclose(args.malicious, 0) == False:
            ba_writer = SummaryWriter("../hlogs/BA/{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.attack, args.malicious, args.defence, args.heter, args.alpha, args.lr_b, args.lr))
        ma_writer = SummaryWriter("../hlogs/MA/{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.attack, args.malicious, args.defence, args.heter, args.alpha, args.lr_b, args.lr))
    else:
        if math.isclose(args.malicious, 0) == False:
            ba_writer = SummaryWriter("../hlogs/BA/{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.attack, args.malicious, args.defence, args.heter, args.alpha, args.lr_b, args.lr))
        ma_writer = SummaryWriter("../hlogs/MA/{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.attack, args.malicious, args.defence, args.heter, args.alpha, args.lr_b, args.lr))
    
    # trainma_writer = SummaryWriter("../hlogs/TrainMA/{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.attack, args.malicious, args.defence, args.heter, args.alpha, args.lr_b, args.lr))

    if args.heter == "mnist7":
        mnist7_other_writer = SummaryWriter("../hlogs/mnist7/other/{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.attack, args.malicious, args.defence, args.heter, args.alpha, args.lr_b, args.lr))
        mnist7_7_writer = SummaryWriter("../hlogs/mnist7/7/{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.attack, args.malicious, args.defence, args.heter, args.alpha, args.lr_b, args.lr))
        mnist7_bar7_writer = SummaryWriter("../hlogs/mnist7/bar7/{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.attack, args.malicious, args.defence, args.heter, args.alpha, args.lr_b, args.lr))

    if args.loss:
        loss_writer = SummaryWriter("../hlogs/loss/{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.attack, args.malicious, args.defence, args.heter, args.alpha, args.lr_b, args.lr))

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(
            '../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(
            '../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.heter == "iid":
            dict_users = mnist_iid(dataset_train, args.num_users)
        elif args.heter == "label_noniid":
            dict_users = cifar_noniid(np.array([data[1] for data in dataset_train]), args.num_users, 10, args.alpha)
        elif args.heter == "dirichlet":
            dict_users = dirichlet(dataset_train, args.num_users, args.alpha)
        elif args.heter == "mnist7":
            dict_users, test_other_idx, test_7_idx, test_bar7_idx = mnist7(args, dataset_train, dataset_test)
        else:
            exit('Error: unrecognized heterogenity setting')
    elif args.dataset == 'fashion_mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        dataset_train = datasets.FashionMNIST(
            '../data/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST(
            '../data/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = np.load('./data/iid_fashion_mnist.npy', allow_pickle=True).item()
        else:
            dict_users = np.load('./data/non_iid_fashion_mnist.npy', allow_pickle=True).item()
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(
            '../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(
            '../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.heter == "iid":
            # dict_users = np.load('./data/iid_cifar.npy', allow_pickle=True).item()
            dict_users = cifar_iid(dataset_train, args.num_users)
        elif args.heter == "label_noniid":
            # dict_users = np.load('./data/non_iid_cifar.npy', allow_pickle=True).item()
            dict_users = cifar_noniid(np.array([data[1] for data in dataset_train]), args.num_users, 10, args.alpha)
        elif args.heter == "dirichlet":
            dict_users = dirichlet(dataset_train, args.num_users, args.alpha)
        else:
            exit('Error: unrecognized heterogenity setting')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
    num_classes = 10

    # build model
    if args.model == 'VGG' and args.dataset == 'cifar':
        net_glob = vgg19_bn().to(args.device)
    elif args.model == "resnet" and args.dataset == 'cifar':
        net_glob = ResNet18().to(args.device)
    elif args.model == "rlr_mnist" or args.model == "cnn":
        net_glob = get_model('fmnist').to(args.device)
    else:
        exit('Error: unrecognized model')
    
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    
    if math.isclose(args.malicious, 0):
        backdoor_begin_acc = 100
    else:
        backdoor_begin_acc = args.attack_begin  # overtake backdoor_begin_acc then attack
    central_dataset = central_dataset_iid(dataset_test, args.server_dataset)
    base_info = get_base_info(args)
    filename = './'+args.save+'/accuracy_file_{}.txt'.format(base_info)
    
    if args.init != 'None':
        param = torch.load(args.init)
        net_glob.load_state_dict(param)
        print("load init model")

        
    val_acc_list, net_list = [0], []
    backdoor_acculist = [0]

    args.attack_layers=[]
    
    if args.attack == "dba":
        args.dba_sign=0
    if args.defence == "krum" or args.defence == "multi-krum":
        args.krum_distance=[]
    if args.defence == "fl_defender":
        score_history = np.zeros([args.num_users], dtype = float)
    if args.defence == "flame" or "multi":
        excluded_frequency = [0] * args.num_users
    
    adversaries = [i for i in range(args.num_users) if i % num_classes == 1]

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
            w_updates = []
            local_models = []
        # m = max(int(args.frac * args.num_users), 1)
        # if val_acc_list[-1] > backdoor_begin_acc:
        #     attack_number = int(args.malicious * m)
        # else:
        #     attack_number = 0
        # if math.isclose(args.malicious, 0) and args.visualize == True and args.heter != "iid":
        #     if args.heter == 'label_noniid':
        #         idxs_users = [random.randint(10*i, 10*i+9) for i in range(10)]
        #     elif args.heter == 'dirichlet':
        #         idxs_users = [i for i in range(10)]
        #     elif args.heter == 'mnist7':
        #         idxs_users = list(np.random.choice(range(5), 1, replace=False)) + \
        #                 list(np.random.choice(range(5, 100), m - 1, replace=False))
        # elif math.isclose(args.malicious, 0) == False:
        #     idxs_users = list(np.random.choice(adversaries, attack_number, replace=False)) + \
        #                 list(np.random.choice(list(set(range(args.num_users))-set(adversaries)), m - attack_number, replace=False))
        # else:
        #     idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # 不用选客户端了，每轮10个客户端都参与训练
        attack_number = 0
        idxs_users = [i for i in range(10)]
        
        # idxs_attacker = np.random.choice(adversaries, attack_number, replace=False)
        for num_turn, idx in enumerate(idxs_users):
            if attack_number > 0:
                attack = True
            else:
                attack = False
            if attack == True:
                # idx = random.randint(0, int(args.num_users * args.malicious))
                # idx = idxs_attacker[attack_number-1]
                if args.attack == "dba":
                    num_dba_attacker = int(args.num_users * args.malicious)
                    dba_group = num_dba_attacker/4
                    idx = args.dba_sign % (4*dba_group)
                    args.dba_sign+=1
                local = LocalMaliciousUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], order=idx)
                if args.attack == "layerattack_ER_his" or args.attack == "LFA" or args.attack == "LPA":
                    local_model, loss, args.attack_layers = local.train(
                        net=copy.deepcopy(net_glob).to(args.device), test_img = test_img)
                else:
                    local_model, loss = local.train(
                        net=copy.deepcopy(net_glob).to(args.device), test_img = test_img)
                print("client", idx, "--attack--")
                attack_number -= 1
            else:
                local = LocalUpdate(
                    args=args, dataset=dataset_train, idxs=dict_users[idx])
                local_model, loss = local.train(
                    net=copy.deepcopy(net_glob).to(args.device))
            w = local_model.state_dict()
            local_models.append(local_model)
            w_updates.append(get_update(w, w_glob))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        if args.defence == 'avg':  # no defence
            w_glob = FedAvg(w_locals)
            if args.loss:
                train_loss = sum(loss_locals) / len(loss_locals)
        elif args.defence == 'krum':  # single krum
            selected_client = multi_krum(w_updates, 1, args, krum_writer, iter)
            # print(args.krum_distance)
            w_glob = w_locals[selected_client[0]]
            train_loss = loss_locals[selected_client[0]]
            # w_glob = FedAvg([w_locals[i] for i in selected_clinet])
        elif args.defence == 'multi-krum':
            selected_client = multi_krum(w_updates, 1, args, krum_writer, iter, multi_k=True)
            w_glob = FedAvg([w_locals[i] for i in selected_client])
            train_loss = sum([loss_locals[i] for i in selected_client]) / len(selected_client)
        elif args.defence == 'RLR':
            w_glob = RLR(copy.deepcopy(net_glob), w_updates, args, RLR_writer, iter)
        elif args.defence == 'fltrust':
            local = LocalUpdate(
                args=args, dataset=dataset_test, idxs=central_dataset)
            fltrust_norm, loss = local.train(
                net=copy.deepcopy(net_glob).to(args.device))
            fltrust_norm = get_update(fltrust_norm, w_glob)
            w_glob = fltrust(w_updates, fltrust_norm, w_glob, args)
        elif args.defence == 'flame':
            w_glob = flame(w_locals,w_updates,w_glob, args, flame_norm_writer, iter, excluded_frequency)
        elif args.defence == 'fl_defender':
            w_glob = fl_defender(copy.deepcopy(net_glob), copy.deepcopy(local_models), score_history, idxs_users, fldefender_file, iter)
        elif args.defence == 'multi':
            net_glob = multi(copy.deepcopy(local_models), args.device, excluded_frequency)
        else:
            print("Wrong Defense Method")
            os._exit(0)
        
        # copy weight to net_glob
        if args.defence != 'multi':
            net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        if args.loss:
            loss_writer.add_scalar("train_loss", train_loss, iter)

        if iter % 1 == 0:
            if args.heter == "mnist7":
                acc_other, acc_7, acc_bar7 = test_mnist7(net_glob, dataset_test, test_other_idx, test_7_idx, test_bar7_idx, args)
                print("Other classes: {:.2f}".format(acc_other))
                print("Class 7: {:.2f}".format(acc_7))
                print("Class bar7: {:.2f}".format(acc_bar7))
                mnist7_other_writer.add_scalar("MA", acc_other, iter)
                mnist7_7_writer.add_scalar("MA", acc_7, iter)
                mnist7_bar7_writer.add_scalar("MA", acc_bar7, iter)
            else:
                if math.isclose(args.malicious, 0):
                    acc_test, _ = test_img(net_glob, dataset_test, args, test_backdoor=False)
                    # acc_train, _ = test_img(copy.deepcopy(net_glob), dataset_train, args)
                    print("Main accuracy: {:.2f}".format(acc_test))
                    # trainma_writer.add_scalar("trainMA", acc_train, iter)
                    ma_writer.add_scalar("MA", acc_test, iter)
                else:
                    acc_test, _, back_acc = test_img(
                        net_glob, dataset_test, args, test_backdoor=True)
                    print("Main accuracy: {:.2f}".format(acc_test))
                    print("Backdoor accuracy: {:.2f}".format(back_acc))
                    ba_writer.add_scalar("BA", back_acc, iter)
                    ma_writer.add_scalar("MA", acc_test, iter)
                    backdoor_acculist.append(back_acc)
                val_acc_list.append(acc_test.item())

            # write_file(filename, val_acc_list, backdoor_acculist, args)
    
    best_acc, absr, bbsr = write_file(filename, val_acc_list, backdoor_acculist, args, True)
    
    # plot loss curve
    plt.figure()
    plt.xlabel('communication')
    plt.ylabel('accu_rate')
    plt.plot(val_acc_list, label = 'main task(acc:'+str(best_acc)+'%)')
    plt.plot(backdoor_acculist, label = 'backdoor task(BBSR:'+str(bbsr)+'%, ABSR:'+str(absr)+'%)')
    plt.legend()
    title = base_info
    # plt.title(title, y=-0.3)
    plt.title(title)
    plt.savefig('./'+args.save +'/'+ title + '.pdf', format = 'pdf',bbox_inches='tight')
    
    
    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    
    if args.defence == 'flame' or 'multi':
        print("Excluded frquency:", excluded_frequency)
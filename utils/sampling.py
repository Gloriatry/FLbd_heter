#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import random

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset_label, num_clients, num_classes, q):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    proportion = non_iid_distribution_group(dataset_label, num_clients, num_classes, q)
    dict_users = non_iid_distribution_client(proportion, num_clients, num_classes)
    #  output clients' labels information
    # check_data_each_client(dataset_label, dict_users, num_clients, num_classes)
    return dict_users

def non_iid_distribution_group(dataset_label, num_clients, num_classes, q):
    dict_users, all_idxs = {}, [i for i in range(len(dataset_label))]
    for i in range(num_classes):
        dict_users[i] = set([])
    for k in range(num_classes):
        idx_k = np.where(dataset_label == k)[0]
        num_idx_k = len(idx_k)
        
        selected_q_data = set(np.random.choice(idx_k, int(num_idx_k*q) , replace=False))
        dict_users[k] = dict_users[k]|selected_q_data
        idx_k = list(set(idx_k) - selected_q_data)
        all_idxs = list(set(all_idxs) - selected_q_data)
        for other_group in range(num_classes):
            if other_group == k:
                continue
            selected_not_q_data = set(np.random.choice(idx_k, int(num_idx_k*(1-q)/(num_classes-1)) , replace=False))
            dict_users[other_group] = dict_users[other_group]|selected_not_q_data
            idx_k = list(set(idx_k) - selected_not_q_data)
            all_idxs = list(set(all_idxs) - selected_not_q_data)
    print(len(all_idxs),' samples are remained')
    print('random put those samples into groups')
    num_rem_each_group = len(all_idxs) // num_classes
    for i in range(num_classes):
        selected_rem_data = set(np.random.choice(all_idxs, num_rem_each_group, replace=False))
        dict_users[i] = dict_users[i]|selected_rem_data
        all_idxs = list(set(all_idxs) - selected_rem_data)
    print(len(all_idxs),' samples are remained after relocating')
    return dict_users

def non_iid_distribution_client(group_proportion, num_clients, num_classes):
    num_each_group = num_clients // num_classes
    # num_data_each_client = len(group_proportion[0]) // num_each_group
    # dict_users, all_idxs = {}, [i for i in range(num_data_each_client*num_clients)]
    dict_users = {}
    for i in range(num_classes):
        group_data = list(group_proportion[i])
        num_data_each_client = len(group_proportion[i]) // num_each_group
        for j in range(num_each_group):
            selected_data = set(np.random.choice(group_data, num_data_each_client, replace=False))
            dict_users[i*num_each_group+j] = selected_data
            group_data = list(set(group_data) - selected_data)
            # all_idxs = list(set(all_idxs) - selected_data)
    # print(len(all_idxs),' samples are remained')
    return dict_users
def check_data_each_client(dataset_label, client_data_proportion, num_client, num_classes):
    for client in client_data_proportion.keys():
        client_data = dataset_label[list(client_data_proportion[client])]
        print('client', client, 'distribution information:')
        for i in range(num_classes):
            print('class ', i, ':', len(client_data[client_data==i])/len(client_data))

def dirichlet(dataset_train, no_participants, alpha):
    cifar_classes = {}
    for ind, x in enumerate(dataset_train):  # for cifar: 50000; for tinyimagenet: 100000
        _, label = x
        if label in cifar_classes:
            cifar_classes[label].append(ind)
        else:
            cifar_classes[label] = [ind]
    class_size = len(cifar_classes[0])  # for cifar: 5000
    per_participant_list = {i: [] for i in range(no_participants)}
    no_classes = len(cifar_classes.keys())  # for cifar: 10

    image_nums = []
    for n in range(no_classes):
        image_num = []
        random.shuffle(cifar_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array(no_participants * [alpha]))
        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
            image_num.append(len(sampled_list))
            per_participant_list[user].extend(sampled_list)
            cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
        image_nums.append(image_num)
    # self.draw_dirichlet_plot(no_classes,no_participants,image_nums,alpha)
    return per_participant_list

import os
def mnist7(args, dataset_train, dataset_test):
    dict_users = {i: [] for i in range(args.num_users)}
    
    train_7_idx = []
    train_bar7_idx = []
    test_7_idx = []
    test_bar7_idx = []

    dir_train_7 = "../data/mnist7/train/7-0"
    for image_name in os.listdir(dir_train_7):
        if image_name.endswith('.png'):
            train_7_idx.append(int(image_name.split('_')[1].split('.')[0]))
    dir_train_bar7 = "../data/mnist7/train/7-1"
    for image_name in os.listdir(dir_train_bar7):
        if image_name.endswith('.png'):
            train_bar7_idx.append(int(image_name.split('_')[1].split('.')[0]))
    dir_test_7 = "../data/mnist7/test/7-0"
    for image_name in os.listdir(dir_test_7):
        if image_name.endswith('.png'):
            test_7_idx.append(int(image_name.split('_')[1].split('.')[0]))
    dir_test_bar7 = "../data/mnist7/test/7-1"
    for image_name in os.listdir(dir_test_bar7):
        if image_name.endswith('.png'):
            test_bar7_idx.append(int(image_name.split('_')[1].split('.')[0]))
    
    train_other_idx = list(set([i for i in range(len(dataset_train))]) - set(train_7_idx) - set(train_bar7_idx))
    test_other_idx = list(set([i for i in range(len(dataset_test))]) - set(test_7_idx) - set(test_bar7_idx))

    # 总共10个客户端，-7分给第1个客户端，7均匀分给后9个客户端，其余的数字均匀分给10个客户端
    num_items = int(len(train_bar7_idx)/1)
    for i in range(1):
        dict_users[i].extend(np.random.choice(train_bar7_idx, num_items, replace=False))
        train_bar7_idx = list(set(train_bar7_idx) - set(dict_users[i]))
    num_items = int(len(train_7_idx)/9)
    for i in range(1, 10):
        dict_users[i].extend(np.random.choice(train_7_idx, num_items, replace=False))
        train_7_idx = list(set(train_7_idx) - set(dict_users[i]))
    num_items = int(len(train_other_idx)/args.num_users)
    for i in range(args.num_users):
        dict_users[i].extend(np.random.choice(train_other_idx, num_items, replace=False))
        train_other_idx = list(set(train_other_idx) - set(dict_users[i]))

    return dict_users, test_other_idx, test_7_idx, test_bar7_idx

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)

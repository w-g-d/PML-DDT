import json
import os
import xml.etree.ElementTree as ET
from glob import glob
from os.path import join

import numpy as np
import torch
from PIL import Image
from PIL import ImageFile
from torch.utils.data.dataset import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class VOC2007_handler_train(Dataset):
    def __init__(self, X, Y, Z, data_path, transform=None, transform_aug=None):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.transform = transform
        self.transform_aug = transform_aug
        self.data_path = data_path

    def __getitem__(self, index):
        x_ = Image.open(self.data_path + '/JPEGImages/' + self.X[index] + '.jpg').convert('RGB')
        x_aug = self.transform_aug(x_)
        y = self.Y[index]
        z = self.Z[index]
        return x_aug, y, z, index

    def __len__(self):
        return len(self.X)


class VOC2007_handler_test(Dataset):
    def __init__(self, X, Y, data_path, transform=None, random_crops=0):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.random_crops = random_crops
        self.data_path = data_path

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/JPEGImages/' + self.X[index] + '.jpg').convert('RGB')

        if self.random_crops == 0:
            x = self.transform(x)
        else:
            crops = []
            for i in range(self.random_crops):
                crops.append(self.transform(x))
            x = torch.stack(crops)

        y = self.Y[index]

        return x, y, index

    def __len__(self):
        return len(self.X)


def get_VOC2007(train_data_path, test_data_path):
    train_data, train_labels = __voc_dataset_info(train_data_path, 'trainval')
    train_idx = np.arange(train_labels.shape[0])
    np.random.shuffle(train_idx)
    train_data, train_labels = train_data[train_idx], train_labels[train_idx]

    test_data, test_labels = __voc_dataset_info(test_data_path, 'test')
    test_idx = np.arange(test_labels.shape[0])
    np.random.shuffle(test_idx)
    test_data, test_labels = test_data[test_idx], test_labels[test_idx]

    return train_data, train_labels, test_data, test_labels


def __voc_dataset_info(data_path, trainval):
    classes = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
    num_classes = len(classes)
    class_to_ind = dict(zip(classes, range(num_classes)))

    with open(data_path + '/ImageSets/Main/' + trainval + '.txt') as f:
        annotations = f.readlines()

    annotations = [n[:-1] for n in annotations]
    names = []  # image file names
    labels = []
    for af in annotations:
        if len(af) != 6:
            continue
        filename = os.path.join(data_path, 'Annotations', af)
        tree = ET.parse(filename + '.xml')
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes_cl = np.zeros((num_objs), dtype=np.int32)

        for ix, obj in enumerate(objs):
            cls = class_to_ind[obj.find('name').text.lower().strip()]
            boxes_cl[ix] = cls

        lbl = np.zeros(num_classes)
        lbl[boxes_cl] = 1
        labels.append(lbl)
        names.append(af)

    labels = np.array(labels).astype(np.float32)
    labels = labels[:, 1:]

    return np.array(names), np.array(labels)


class COCO2014_handler_test(Dataset):
    def __init__(self, X, Y, data_path, transform=None, random_crops=0):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.random_crops = random_crops
        self.data_path = data_path

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')

        if self.random_crops == 0:
            x = self.transform(x)
        else:
            # apply the (random) transform `random_crops` times and stack the
            # resulting image tensors into a single tensor
            crops = []
            for i in range(self.random_crops):
                crops.append(self.transform(x))
            x = torch.stack(crops)

        y = self.Y[index]

        return x, y, index

    def __len__(self):
        return len(self.X)


class COCO2014_handler_train(Dataset):
    def __init__(self, X, Y, Z, data_path, transform=None, random_crops=0):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.transform = transform
        self.random_crops = random_crops
        self.data_path = data_path

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')

        if self.random_crops == 0:
            x = self.transform(x)
        else:
            # apply the (random) transform `random_crops` times and stack the
            # resulting image tensors into a single tensor
            crops = []
            for i in range(self.random_crops):
                crops.append(self.transform(x))
            x = torch.stack(crops)

        y = self.Y[index]
        z = self.Z[index]

        return x, y, z, index

    def __len__(self):
        return len(self.X)


def _load_coco_annotations(anno_path, num_classes=80):
    """Load image names and binarized multi-hot labels from a COCO-style annotation file."""
    with open(anno_path, 'r') as f:
        img_list = json.load(f)

    names = []
    labels = []
    for item in img_list:
        names.append(item['file_name'])
        lbl = np.zeros(num_classes)
        lbl[item['labels']] = 1
        labels.append(lbl)

    names = np.array(names)
    labels = np.array(labels)

    rand_idxs = np.random.permutation(names.shape[0])
    names = names[rand_idxs]
    labels = labels[rand_idxs].astype(np.float32)

    return names, labels


def get_COCO2014(anno_dir=None):
    """Return (train_data, train_labels, test_data, test_labels) for COCO 2014.

    Args:
        anno_dir: directory containing ``train_anno.json`` and ``val_anno.json``.
            Defaults to the ``coco`` folder next to this file.
    """
    if anno_dir is None:
        anno_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coco')

    train_data, train_labels = _load_coco_annotations(os.path.join(anno_dir, 'train_anno.json'))
    test_data, test_labels = _load_coco_annotations(os.path.join(anno_dir, 'val_anno.json'))

    return train_data, train_labels, test_data, test_labels


nuswide_category_name = {
    0: 'airport', 1: 'animal', 2: 'beach', 3: 'bear', 4: 'birds', 5: 'boats', 6: 'book', 7: 'bridge', 8: 'buildings', 9: 'cars',
    10: 'castle', 11: 'cat', 12: 'cityscape', 13: 'clouds', 14: 'computer', 15: 'coral', 16: 'cow', 17: 'dancing', 18: 'dog', 19: 'earthquake',
    20: 'elk', 21: 'fire', 22: 'fish', 23: 'flags', 24: 'flowers', 25: 'food', 26: 'fox', 27: 'frost', 28: 'garden', 29: 'glacier',
    30: 'grass', 31: 'harbor', 32: 'horses', 33: 'house', 34: 'lake', 35: 'leaf', 36: 'map', 37: 'military', 38: 'moon', 39: 'mountain',
    40: 'nighttime', 41: 'ocean', 42: 'person', 43: 'plane', 44: 'plants', 45: 'police', 46: 'protest', 47: 'railroad', 48: 'rainbow', 49: 'reflection',
    50: 'road', 51: 'rocks', 52: 'running', 53: 'sand', 54: 'sign', 55: 'sky', 56: 'snow', 57: 'soccer', 58: 'sports', 59: 'statue',
    60: 'street', 61: 'sun', 62: 'sunset', 63: 'surf', 64: 'swimmers', 65: 'tattoo', 66: 'temple', 67: 'tiger', 68: 'tower', 69: 'town',
    70: 'toy', 71: 'train', 72: 'tree', 73: 'valley', 74: 'vehicle', 75: 'water', 76: 'waterfall', 77: 'wedding', 78: 'whales', 79: 'window',
    80: 'zebra'
}


class NUS_WIDE_handler_train(Dataset):
    def __init__(self, X, Y, Z, data_path, transform=None, random_crops=0):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.transform = transform
        self.random_crops = random_crops
        self.data_path = data_path

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')

        if self.random_crops == 0:
            x = self.transform(x)
        else:
            # apply the (random) transform `random_crops` times and stack the
            # resulting image tensors into a single tensor
            crops = []
            for i in range(self.random_crops):
                crops.append(self.transform(x))
            x = torch.stack(crops)

        y = self.Y[index]
        z = self.Z[index]

        return x, y, z, index

    def __len__(self):
        return len(self.X)


class NUS_WIDE_handler_test(Dataset):
    def __init__(self, X, Y, data_path, transform=None, random_crops=0):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.random_crops = random_crops
        self.data_path = data_path

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')

        if self.random_crops == 0:
            x = self.transform(x)
        else:
            # apply the (random) transform `random_crops` times and stack the
            # resulting image tensors into a single tensor
            crops = []
            for i in range(self.random_crops):
                crops.append(self.transform(x))
            x = torch.stack(crops)

        y = self.Y[index]

        return x, y, index

    def __len__(self):
        return len(self.X)


def filter_unlabeled_samples(image_paths, labels):
    """Filter out unlabeled samples (all-zero label rows).

    Args:
        image_paths: list of image paths
        labels: label matrix of shape (n_samples, n_classes)

    Returns:
        filtered_image_paths: image paths with unlabeled samples removed
        filtered_labels: label matrix with unlabeled samples removed
    """
    assert len(image_paths) == labels.shape[0], "number of images and labels do not match"

    # indices of rows that are not all-zero
    non_zero_indices = np.where(np.any(labels != 0, axis=1))[0]

    filtered_image_paths = [image_paths[i] for i in non_zero_indices]
    filtered_labels = labels[non_zero_indices, :]

    print(f"Original number of samples: {len(image_paths)}")
    print(f"Number of samples after filtering: {len(filtered_image_paths)}")
    print(f"Number of removed samples: {len(image_paths) - len(filtered_image_paths)}")

    return filtered_image_paths, filtered_labels


def get_NUS_WIDE(root_dir, cache=True):
    try:
        image_filenames_train = np.load(
            join(root_dir, 'image_filenames_train.npy'))
        labels_train = np.load(join(root_dir, 'labels_train.npy'))
        image_filenames_test = np.load(
            join(root_dir, 'image_filenames_test.npy'))
        labels_test = np.load(join(root_dir, 'labels_test.npy'))
    except FileNotFoundError:
        image_filenames_train_path = join(
            root_dir, 'ImageList', 'TrainImagelist.txt')
        image_filenames_test_path = join(
            root_dir, 'ImageList', 'TestImagelist.txt')
        # glob returns a list of all file paths matching the given pattern
        label_train_per_class = sorted(
            glob(join(root_dir, 'Groundtruth', 'TrainTestLabels', 'Labels_*_Train.txt')))
        label_test_per_class = sorted(
            glob(join(root_dir, 'Groundtruth', 'TrainTestLabels', 'Labels_*_Test.txt')))

        # for train
        with open(image_filenames_train_path, 'r') as file:
            image_filenames_train = [line.strip().replace('\\', '/')
                                     for line in file.readlines()]
            image_filenames_train = [join('Flickr', filename)
                                     for filename in image_filenames_train]

        num_samples = len(image_filenames_train)
        num_classes = len(nuswide_category_name)

        print("Training set size before filtering:", num_samples)

        labels_train = np.zeros((num_samples, num_classes), dtype=np.float32)

        for class_index, filename in enumerate(label_train_per_class):
            with open(filename, 'r') as file:
                label_per_class = np.array(
                    [float(line.strip()) for line in file.readlines()],
                    dtype=np.float32)
                label_per_class[label_per_class != 1] = 0
                labels_train[:, class_index] = label_per_class

        image_filenames_train, labels_train = filter_unlabeled_samples(image_filenames_train, labels_train)

        print("Shape after filtering:", labels_train.shape)
        if cache:
            np.save(join(root_dir, 'image_filenames_train.npy'),
                    np.array(image_filenames_train))
            np.save(join(root_dir, 'labels_train.npy'),
                    np.array(labels_train))

        # for test
        with open(image_filenames_test_path, 'r') as file:
            image_filenames_test = [line.strip().replace('\\', '/')
                                    for line in file.readlines()]
            image_filenames_test = [join('Flickr', filename)
                                    for filename in image_filenames_test]

        num_samples = len(image_filenames_test)
        num_classes = len(nuswide_category_name)
        print("Test set size before filtering:", num_samples)

        labels_test = np.zeros((num_samples, num_classes), dtype=np.float32)

        for class_index, filename in enumerate(label_test_per_class):
            with open(filename, 'r') as file:
                label_per_class = np.array(
                    [float(line.strip()) for line in file.readlines()],
                    dtype=np.float32)
                label_per_class[label_per_class != 1] = 0
                labels_test[:, class_index] = label_per_class
        image_filenames_test, labels_test = filter_unlabeled_samples(image_filenames_test, labels_test)
        print("Shape after filtering:", labels_test.shape)
        if cache:
            np.save(join(root_dir, 'image_filenames_test.npy'),
                    np.array(image_filenames_test))
            np.save(join(root_dir, 'labels_test.npy'),
                    np.array(labels_test))

    return image_filenames_train, labels_train, image_filenames_test, labels_test


def generate_noisy_labels(labels, noise_rate=0.2):
    """Generate partial (candidate) labels by randomly flipping negative labels
    to positive with probability ``noise_rate``."""
    N, C = labels.shape

    alpha_mat = np.ones_like(labels) * noise_rate
    rand_mat = np.random.rand(N, C)

    mask = np.zeros((N, C), dtype=np.float64)
    mask[labels != 1] = rand_mat[labels != 1] < alpha_mat[labels != 1]

    plabels = labels.copy()

    plabels[mask == 1] = 1

    return plabels

#!/data/bin/miniconda2/envs/seed-v1.0/bin/python
# coding: utf-8

import os
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import random


def prepare_data(healthy_dir, infected_dir, n_healthy=-1, n_infected=-1, test_size=0.2, val_size=0.1):
    healthy_files = [os.path.join(healthy_dir, f) for f in os.listdir(healthy_dir) if f.endswith('.npy')]
    infected_files = [os.path.join(infected_dir, f) for f in os.listdir(infected_dir) if f.endswith('.npy')]

    if n_healthy == -1:
        n_healthy = len(healthy_files)
    if n_infected == -1:
        n_infected = len(infected_files)

    random.seed(42)
    healthy_files = random.sample(healthy_files, min(n_healthy, len(healthy_files)))
    infected_files = random.sample(infected_files, min(n_infected, len(infected_files)))

    files = healthy_files + infected_files
    labels = [0] * len(healthy_files) + [1] * len(infected_files)

    X_train, X_temp, y_train, y_temp = train_test_split(files, labels, test_size=test_size + val_size, stratify=labels)
    val_size_adjusted = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - val_size_adjusted, stratify=y_temp)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class HyperspectralDataset:
    def __init__(self, file_paths, labels, bands, shape):
        self.file_paths = file_paths
        self.labels = labels
        self.bands = bands
        self.shape = shape

    def _load_slice(self, path):
        path = path.numpy().decode('utf-8')
        arr = np.load(path, mmap_mode='r')
        sliced = arr[:, :, self.bands]
        return sliced.astype(np.float64)

    def get_dataset(self, batch_size=1, shuffle=False):
        dataset = tf.data.Dataset.from_tensor_slices((self.file_paths, self.labels))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.file_paths))

        def load_fn(path, label):
            bands = tf.py_function(func=self._load_slice, inp=[path], Tout=tf.float64)
            bands.set_shape(self.shape)
            return bands, label

        dataset = dataset.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset
    

class HyperspectralTorchDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, labels, bands, shape):
        self.file_paths = np.array(file_paths)
        self.labels = np.array(labels, dtype=np.float32)
        self.bands = bands
        self.shape = shape

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = np.load(path, mmap_mode='r')  # zero copy

        image = image[:, :, self.bands]       # view if self.bands is simple (not fancy indexing)
        image = np.moveaxis(image, -1, 0)     # view, but must assign it!

        image = torch.from_numpy(image)       # zero-copy
        label = self.labels[idx]

        return image, label

    def __len__(self):
        return len(self.file_paths)
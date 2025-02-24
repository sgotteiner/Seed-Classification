#!/usr/bin/env python
# coding: utf-8

import os


def find_paths_of_seeds(directory):
    paths = []
    for dirpath, _, filenames in os.walk(directory):
        for file in filenames:
            if file.endswith(".npy"):
                paths.append(os.path.join(dirpath, file))
    return paths

def seed_rename(old_name):
    split_old_name = old_name.split('\\')
    tray_number = split_old_name[-3][1]
    position = split_old_name[-1].split('.')[0]
    row = int(ord(position[1]) - ord('A'))
    column = ord(position[0]) - ord('0') - 1
    column = column if split_old_name[-3][2] == 'L' else column + 7
    file_type = split_old_name[-1].split('.')[1]
    new_name = f'Tray_{tray_number}_row_{row}_column_{column}.{file_type}'
    return new_name
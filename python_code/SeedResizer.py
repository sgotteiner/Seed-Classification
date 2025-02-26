#!/usr/bin/env python
# coding: utf-8

import numpy as np

from SeedMaskExtractor import find_seed_width_height_area
from SeedFinder import find_paths_of_seeds


def find_max_width_height_area_of_seeds(seeds_folder):
    max_width, max_height, max_area = 0, 0, 0
    for path in find_paths_of_seeds(seeds_folder):
        seed = np.load(path)[:,:,200]
        width, height, _, _, area = find_seed_width_height_area(seed)
        max_width = width if width > max_width else max_width
        max_height = height if height > max_height else max_height
        max_area = area if area > max_area else max_area
    return max_width, max_height, max_area

def save_max_width_height(seeds_path, save_path):
    max_width, max_height, _ = find_max_width_height_area_of_seeds(seeds_path)
    np.save(f'{save_path}\max_width.npy', max_width)
    np.save(f'{save_path}\max_height.npy', max_height)

def create_resized_image(image, max_width, max_height, data_type):
    height, width, bands = image.shape
    resized_image = np.zeros((max_height + 2, max_width + 2, bands), dtype=data_type)  # +2 is for all the seeds to be surrounded by black
    row_offset, column_offset = (max_height - height)//2 + 1, (max_width - width)//2 + 1
    resized_image[row_offset:row_offset + height, column_offset:column_offset + width, :] = image
    return resized_image
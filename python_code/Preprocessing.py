import numpy as np

from SeedFinder import find_paths_of_seeds, seed_rename
from SeedNormalizer import normalize_seed, save_normalization_parameters
from SeedResizer import create_resized_image, save_max_width_height


seeds_path = r'home\ARO.local\collaboration\Itai-tomer\tomer_data'
healthy_seeds_path = seeds_path + r'\Healthy'
infected_seeds_path = seeds_path + r'\Infected'
preprocessed_seeds_path = r'home\ARO.local\collaboration\sagi-tomer-collab\Normalized_Tomato_Seeds'
normalization_parameters_path = preprocessed_seeds_path + r'\normalization_parameters'


def save_preprocessing_parameters(saving_path):
    save_normalization_parameters(healthy_seeds_path, infected_seeds_path, saving_path)
    save_max_width_height(seeds_path, saving_path)

save_preprocessing_parameters(normalization_parameters_path)


def preprocess_seed(seed_path, most_consistent_band, bands_means, bands_stds, max_width, max_height):
    seed = np.load(seed_path)
    normalized_seed = normalize_seed(seed, most_consistent_band, bands_means, bands_stds)
    resized_normalized_seed = create_resized_image(normalized_seed, max_width, max_height, normalized_seed.dtype)
    return resized_normalized_seed

most_consistent_band = np.load(fr'{normalization_parameters_path}\most_consistent_band.npy')
healthy_bands_means = np.load(fr'{normalization_parameters_path}\healthy_illumination_ratio_based_normalized_means.npy')
infected_bands_means = np.load(fr'{normalization_parameters_path}\healthy_illumination_ratio_based_normalized_means.npy')
healthy_bands_stds = np.load(fr'{normalization_parameters_path}\infected_illumination_ratio_based_normalized_stds.npy')
infected_bands_stds = np.load(fr'{normalization_parameters_path}\infected_illumination_ratio_based_normalized_stds.npy')
max_width = np.load(fr'{normalization_parameters_path}\max_width.npy')
max_height = np.load(fr'{normalization_parameters_path}\max_height.npy')

for seed_path in find_paths_of_seeds(healthy_seeds_path):
    preprocessed_seed = preprocess_seed(seed_path, most_consistent_band, healthy_bands_means, healthy_bands_stds, max_width, max_height)
    np.save(preprocessed_seeds_path + f'\Healthy\{seed_rename(seed_path)}', preprocessed_seed)

for seed_path in find_paths_of_seeds(infected_seeds_path):
    preprocessed_seed = preprocess_seed(seed_path, most_consistent_band, infected_bands_means, infected_bands_stds, max_width,max_height)
    np.save(preprocessed_seeds_path + f'\Infected\{seed_rename(seed_path)}', preprocessed_seed)
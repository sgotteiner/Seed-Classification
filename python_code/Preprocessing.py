import numpy as np

from ipynb.fs.full.SeedFinder import healthy_seeds, infected_seeds, seed_rename
from ipynb.fs.full.SeedMaskExtractor import get_seed_masked_image
from ipynb.fs.full.SeedNormalizer import normalize_seed
from ipynb.fs.full.SeedResizer import create_resized_image, find_max_width_height_area_of_seeds


max_width, max_height, _ = find_max_width_height_area_of_seeds('home')

def preprocess_seed(seed_path, most_consistent_band, bands_means, bands_stds):
    seed = np.load(seed_path)
    normalized_seed = normalize_seed(seed, most_consistent_band, bands_means, bands_stds)

    masked_normalized_seed = get_seed_masked_image(seed, 200)
    resized_normalized_seed = create_resized_image(normalized_seed, max_width, max_height, np.int64)
    return resized_normalized_seed


most_consistent_band = np.load(r'normalization_parameters\most_consistent_band.npy')
bands_means = np.load(r'normalization_parameters\healthy_illumination_normalized_band_means.npy')
bands_stds = np.load(r'normalization_parameters\healthy_illumination_normalized_band_stds.npy')
preprocessed_seeds_path = r'home\ARO.local\collaboration\Itai-tomer\tomer_data\Normalized_Tomato_Seeds'

for seed_path in healthy_seeds[:3]:
    preprocessed_seed = preprocess_seed(seed_path, most_consistent_band, bands_means, bands_stds)
    np.save(preprocessed_seeds_path + f'\Healthy\{seed_rename(seed_path)}', preprocess_seed)

for seed_path in infected_seeds[:3]:
    preprocessed_seed = preprocess_seed(seed_path, most_consistent_band, bands_means, bands_stds)
    np.save(preprocessed_seeds_path + f'\Infected\{seed_rename(seed_path)}', preprocess_seed)
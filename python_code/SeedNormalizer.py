#!/usr/bin/env python
# coding: utf-8

import numpy as np

from SeedMaskExtractor import extract_seed_pixels, get_all_seeds_pixels_in_folder


# The most consistant band is the band that changes the least between healthy and infected seeds. This band could be used as a reference band for the illumination normalization. It also takes into account the brightness of the bands. It is calculated as the band_brightness // (1 + band_mean_difference).

def calculate_total_seeds_mean_per_band(seed_pixels):
    return np.mean(seed_pixels, axis=(0))

def find_the_most_consistant_band(healthy_seed_pixels, infected_seed_pixels, brightest_percentile=90):
    """
    This function finds the most consistant band that changes the least between healthy and infected seeds among the brightest bands.
    """
    healthy_seeds_mean_per_band = calculate_total_seeds_mean_per_band(healthy_seed_pixels)
    infected_seeds_mean_per_band = calculate_total_seeds_mean_per_band(infected_seed_pixels)

    spectral_difference = np.abs(healthy_seeds_mean_per_band - infected_seeds_mean_per_band)
    bands_total_brightness = healthy_seeds_mean_per_band + infected_seeds_mean_per_band

    brightness_threshold = np.percentile(bands_total_brightness, brightest_percentile)
    bright_bands = np.where(bands_total_brightness >= brightness_threshold)[0]

    bands_scores = bands_total_brightness[bright_bands] / (1 + spectral_difference[bright_bands])
    most_consistent_band_index = np.argmax(bands_scores)
    return bright_bands[most_consistent_band_index]


# Illumination ratio based normalization reduces quantom efficiency noise by dividing all the bands by a reference band. It is important to save the most consistant band.

def illumination_ratio_based_normalization(data, most_consistent_band):
    epsilon = 1e-6  # Small value to prevent division by zero
    reference_band = data[:, most_consistent_band] + epsilon
    return data / reference_band[:, np.newaxis]

def save_zscore_parameters(illumination_normalized, illumination_normalized_name, saving_path):
    illumination_normalized_band_means = np.mean(illumination_normalized, axis=0, keepdims=True)
    illumination_normalized_band_stds = np.std(illumination_normalized, axis=0, keepdims=True)
    np.save(f'{saving_path}\{illumination_normalized_name}_means.npy', illumination_normalized_band_means)
    np.save(f'{saving_path}\{illumination_normalized_name}_stds.npy', illumination_normalized_band_stds)

def save_normalization_parameters(healthy_seeds_path, infected_seeds_path, saving_path):
    healthy_seed_pixels = get_all_seeds_pixels_in_folder(healthy_seeds_path)
    infected_seed_pixels = get_all_seeds_pixels_in_folder(infected_seeds_path)
    most_consistent_band = find_the_most_consistant_band(healthy_seed_pixels, infected_seed_pixels)
    np.save(f'{saving_path}\most_consistent_band.npy', most_consistent_band)
    healthy_illumination_ratio_based_normalized = illumination_ratio_based_normalization(healthy_seed_pixels, most_consistent_band)
    save_zscore_parameters(healthy_illumination_ratio_based_normalized, 'healthy_illumination_ratio_based_normalized', saving_path)
    infected_illumination_ratio_based_normalized = illumination_ratio_based_normalization(infected_seed_pixels, most_consistent_band)
    save_zscore_parameters(infected_illumination_ratio_based_normalized, 'infected_illumination_ratio_based_normalized', saving_path)


# Zscore reduces noise and shifts the range of the values between a small negative number to a small positive value.
# It is very important to save the mean and the standard deviation. This is not per pixel normalization but per band normalization.
# For this reason mean_vals and std_vals are arrays of shape 1 x bands and not width x height x bands.

def zscore_normalize(pixels, mean_vals, std_vals):
    normalized = (pixels - mean_vals) / (std_vals + 1e-8)  # Avoid division by zero
    return normalized


# Aligns the pixels spectral signature. Not used as a default normalization step.

def apply_msc_and_shift_positive(image, ref_spectrum, is_apply_shift_positive=True):
    corrected_spectra = []
    for spectrum in image:
        corrected = (spectrum - np.mean(spectrum)) / np.std(spectrum)
        corrected = corrected * np.std(ref_spectrum) + np.mean(ref_spectrum)
        corrected_spectra.append(corrected)
    corrected_spectra = np.array(corrected_spectra)
    if is_apply_shift_positive:
        min_value = np.min(corrected_spectra)
        if min_value < 0:
            shift_value = np.abs(min_value)
            corrected_spectra += shift_value
    return corrected_spectra


# Use the normalization on a single seed.

def normalize_seed(seed, most_consistent_band, bands_means, bands_stds):
    seed_pixels, seed_pixels_spatial_indices, shape = extract_seed_pixels(seed)
    seed_illumination_ratio_based_normalization = illumination_ratio_based_normalization(seed_pixels, most_consistent_band)
    normalized_seed_pixels = zscore_normalize(seed_illumination_ratio_based_normalization, bands_means, bands_stds)
    normalized_seed = np.zeros(shape)
    normalized_seed[seed_pixels_spatial_indices[:, 0], seed_pixels_spatial_indices[:, 1], :] = normalized_seed_pixels
    return normalized_seed
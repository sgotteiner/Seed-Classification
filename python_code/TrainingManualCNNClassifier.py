#!/data/bin/miniconda2/envs/seed-v1.0/bin/python
# coding: utf-8

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from pathlib import Path
import json

from DataGenerator import prepare_data, HyperspectralDataset
from ManualCNNClassifier import build_hyperspectral_cnn


seeds_path = Path('home') / 'ARO.local' / 'collaboration' / 'sagi-tomer-collab' / 'Normalized_Tomato_Seeds'
healthy_dir = seeds_path / 'Healthy'
infected_dir = seeds_path / 'Infected'
models_dir = seeds_path / 'Models'

n = 800
bands = [0, 320, 639]
height = np.load(seeds_path / 'normalization_parameters' / 'max_height.npy').item() + 2
width = np.load(seeds_path / 'normalization_parameters' / 'max_width.npy').item() + 2
shape = (height, width, len(bands))
batch_size = 32
(train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = prepare_data(str(healthy_dir), str(infected_dir), n, n)
train_data = HyperspectralDataset(train_files, train_labels, bands, shape).get_dataset(batch_size, shuffle=True)
val_data = HyperspectralDataset(val_files, val_labels, bands, shape).get_dataset(batch_size)
test_data = HyperspectralDataset(test_files, test_labels, bands, shape).get_dataset(batch_size)

model = build_hyperspectral_cnn(shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    steps_per_epoch=len(train_files),
    validation_steps=len(val_files),
    callbacks=callbacks
)

y_pred = model.predict(test_data, steps=len(test_files))
predictions = (y_pred > 0.5).astype(int)
report = classification_report(test_labels, predictions, digits=4, output_dict=True)
cm = confusion_matrix(test_labels, predictions)
print(report)
print(f"Test Accuracy: {report['accuracy']}")

bands_str = '-'.join(map(str, bands)) + f'-bands-{round(report["accuracy"], 3)}-accuracy'
model.save(models_dir / f'ManualCNNClassifier_{bands_str}_model.keras')
with open(models_dir / f'ManualCNNClassifier_{bands_str}_history.json', "w") as f:
    json.dump(history.history, f)
with open(models_dir / f'ManualCNNClassifier_{bands_str}_report.json', "w") as f:
    json.dump(report, f, indent=4)
with open(models_dir / f'ManualCNNClassifier_{bands_str}_cm.json', "w") as f:
    json.dump(cm.tolist(), f)
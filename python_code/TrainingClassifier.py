
import time
total_run_time = time.time()

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import platform
import psutil
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import classification_report, confusion_matrix
from ManualCNNClassifier import HyperspectralCNN
from SpectralSpatialModuleClassifier import SpectralSpatialModuleModel
from SpectralSpatialAttentionModel import SpectralSpatialAttentionModel
from DataGenerator import prepare_data, HyperspectralTorchDataset
import multiprocessing as mp


def print_env():
    print("\n🔍 ENVIRONMENT SUMMARY")
    print("=" * 40)
    print(f"🖥️ Hostname       : {platform.node()}")
    print(f"💾 RAM Available  : {round(psutil.virtual_memory().total / 1e9, 2)} GB")
    print(f"🧠 CPU Cores      : {psutil.cpu_count(logical=True)} (Logical), {psutil.cpu_count(logical=False)} (Physical)")    
    print(f"🧪 CUDA Available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"💻 GPU Count      : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   ↳ GPU {i}       : {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️  No GPU detected or available.")
    print("=" * 40 + "\n")


if __name__ == '__main__':
    # import pytorch inside main in order to let the environment load before torch backends start doing stuff.
    import torch
    from torch.utils.data import DataLoader
    from pytorch_lightning import Trainer
    from pytorch_lightning.profilers import SimpleProfiler
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

    print_env()

    # Paths
    seeds_path = Path('home') / 'ARO.local' / 'collaboration' / 'sagi-tomer-collab' / 'Normalized_Tomato_Seeds'
    healthy_dir = seeds_path / 'Healthy'
    infected_dir = seeds_path / 'Infected'
    models_dir = seeds_path / 'Models'
    logs_dir = Path('home') / 'ARO.local' / 'sagig' / 'Projects' / 'seed_classification'

    # Parameters
    n = 800
    bands = sorted([0, 319, 639])
    height = np.load(seeds_path / 'normalization_parameters' / 'max_height.npy').item() + 2
    width = np.load(seeds_path / 'normalization_parameters' / 'max_width.npy').item() + 2
    shape = (height, width, len(bands))
    batch_size = 32

    # Data splits
    (train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = prepare_data(str(healthy_dir), str(infected_dir), n, n)

    train_dataset = HyperspectralTorchDataset(train_files, train_labels, bands, shape)
    val_dataset = HyperspectralTorchDataset(val_files, val_labels, bands, shape)
    test_dataset = HyperspectralTorchDataset(test_files, test_labels, bands, shape)

    mp_context = 'fork' if 'fork' in mp.get_all_start_methods() else 'spawn'
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() - 2, persistent_workers=True, prefetch_factor=4, pin_memory=True, multiprocessing_context=mp_context)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=os.cpu_count() - 2, persistent_workers=True, prefetch_factor=4, pin_memory=True, multiprocessing_context=mp_context)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=os.cpu_count() - 2, persistent_workers=True, prefetch_factor=4, pin_memory=True, multiprocessing_context=mp_context)

    # Model
    # model = HyperspectralCNN(shape[-1])
    # model = SpectralSpatialModuleModel(shape[-1])
    model = SpectralSpatialAttentionModel(shape[-1])
    model = torch.compile(model, backend="eager")
    model_name = model.__class__.__name__

    # Callbacks
    bands_str = '-'.join(map(str, bands))
    ckpt_path = models_dir / f'{model_name}_{bands_str}'
    checkpoint_cb = ModelCheckpoint(
        dirpath=models_dir,
        filename=f"{model_name}_{bands_str}" + "-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # profiler = SimpleProfiler()
    trainer = Trainer(
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        default_root_dir=logs_dir,
        max_epochs=50,
        callbacks=[early_stop_cb, checkpoint_cb, lr_monitor],
        log_every_n_steps=10,
        accelerator="auto",
        # profiler=profiler
    )

    # Train
    training_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    training_time = time.time() - training_time
    # print(profiler.summary())

    # Test
    model.eval()
    preds, targets = [], []
    for x, y in test_loader:
        with torch.no_grad():
            logits = model(x)
            if isinstance(logits, tuple):
                logits, spec_w, spat_map, fusion_w = logits
            preds.extend((logits > 0.5).int().cpu().numpy().flatten())
            targets.extend(y.cpu().numpy())
    
    # see band importances
    if type(model) is SpectralSpatialModuleModel:
        spectral_weights = model.spectral.weights.detach().cpu().numpy()
        print('Importances:', spectral_weights)

    report = classification_report(targets, preds, digits=4, output_dict=True)
    cm = confusion_matrix(targets, preds)
    print(report)
    print(f"Test Accuracy: {report['accuracy']}")

    # Save results
    accuracy = round(report["accuracy"], 3)
    suffix = f'{bands_str}-bands-{accuracy}-accuracy'
    torch.save(model.state_dict(), models_dir / f'{model_name}_{suffix}.pt')
    with open(models_dir / f'{model_name}_{suffix}_report.json', "w") as f:
        json.dump(report, f, indent=4)
    with open(models_dir / f'{model_name}_{suffix}_cm.json', "w") as f:
        json.dump(cm.tolist(), f)

    total_run_time = time.time() - total_run_time
    print(f'total run time: {total_run_time}', f'training time: {training_time}')
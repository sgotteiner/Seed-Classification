import time
total_run_time = time.time()

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import itertools
import random
from pathlib import Path
from tqdm import tqdm
import csv
from DataGenerator import prepare_data, HyperspectralTorchDataset
from ManualCNNClassifier import HyperspectralMultiCNN

if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping

    # Reproducibility
    pl.seed_everything(42, workers=True)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Paths
    seeds_path = Path('home') / 'ARO.local' / 'collaboration' / 'sagi-tomer-collab' / 'Normalized_Tomato_Seeds'
    healthy_dir = seeds_path / 'Healthy'
    infected_dir = seeds_path / 'Infected'
    models_dir = seeds_path / 'Models'
    logs_dir = Path('home') / 'ARO.local' / 'sagig' / 'Projects' / 'seed_classification'

    # Parameters
    n = 8000
    band_list = list(range(1, 639, 10))
    combo_size = 1
    batch_size = 32
    height = np.load(seeds_path / 'normalization_parameters' / 'max_height.npy').item() + 2
    width = np.load(seeds_path / 'normalization_parameters' / 'max_width.npy').item() + 2

    (train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = prepare_data(
        str(healthy_dir), str(infected_dir), n, n, 0.85)

    mp_context = 'fork' if 'fork' in torch.multiprocessing.get_all_start_methods() else 'spawn'

    combinations = list(itertools.combinations(sorted(band_list), combo_size))
    total_combos = len(combinations)

    # Descriptive file name
    band_str = '-'.join(map(str, band_list[:10]))
    csv_name = f'HyperspectralMultiCNN_bands[{band_str}]_combo{combo_size}_total{total_combos}.csv'
    csv_file = models_dir / csv_name

    # Load previous results (resume if crash)
    done_combos = set()
    if csv_file.exists():
        with open(csv_file, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            done_combos = {tuple(map(int, row[0].split('-'))) for row in reader}

    print(f"🔄 Resuming from saved results ({len(done_combos)} completed, {total_combos - len(done_combos)} left)")

    pbar = tqdm([c for c in combinations if c not in done_combos], total=total_combos - len(done_combos), desc="Training band combinations")
    for bands in pbar:
        shape = (height, width, len(bands))
        train_dataset = HyperspectralTorchDataset(train_files, train_labels, bands, shape)
        val_dataset = HyperspectralTorchDataset(val_files, val_labels, bands, shape)
        test_dataset = HyperspectralTorchDataset(test_files, test_labels, bands, shape)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() - 2,
                                  persistent_workers=True, prefetch_factor=4, pin_memory=True, multiprocessing_context=mp_context)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=os.cpu_count() - 2,
                                persistent_workers=True, prefetch_factor=4, pin_memory=True, multiprocessing_context=mp_context)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=os.cpu_count() - 2,
                                 persistent_workers=True, prefetch_factor=4, pin_memory=True, multiprocessing_context=mp_context)

        model = HyperspectralMultiCNN(len(bands))

        trainer = Trainer(
            precision="16-mixed" if torch.cuda.is_available() else "32-true",
            default_root_dir=logs_dir,
            max_epochs=50,
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")],
            log_every_n_steps=10,
            accelerator="auto"
        )

        trainer.fit(model, train_loader, val_loader)

        model.eval()
        preds, targets = [], []
        for x, y in test_loader:
            with torch.no_grad():
                logits = model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                preds.extend((logits > 0.5).int().cpu().numpy().flatten())
                targets.extend(y.cpu().numpy())

        accuracy = (np.array(preds) == np.array(targets)).mean()
        row = ['-'.join(map(str, bands)), round(accuracy, 4)]

        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(["bands", "accuracy"])
            writer.writerow(row)

        pbar.set_postfix({
            "bands": '-'.join(map(str, bands[:3])),
            "accuracy": accuracy
        })

        print(f"✅ HyperspectralMultiCNN | bands: {bands} | accuracy: {accuracy:.4f}")

    print(f"\n📄 Results saved to {csv_file}")
    print(f"⏱️ Total time: {round(time.time() - total_run_time, 2)} seconds")
import time
total_run_time = time.time()
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import platform
import psutil
import numpy as np
# import pandas as pd
from pathlib import Path
import json
import random
from sklearn.metrics import classification_report, confusion_matrix
from ManualCNNClassifier import HyperspectralCNN, HyperspectralMultiCNN
from SpectralSpatialModuleClassifier import SpectralSpatialModuleModel
from SpectralSpatialAttentionModel import SpectralSpatialSingleHeatmapAttentionModel, SpectralSpatialMultiHeatmapAttentionModel, SpectralSpatialPatchHeatmapAttentionModel
from BandAttentionModel import BandAttentionModel
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
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.profilers import SimpleProfiler
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
    pl.seed_everything(42, workers=True)

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print_env()

    # Paths
    seeds_path = Path('home') / 'ARO.local' / 'collaboration' / 'sagi-tomer-collab' / 'Normalized_Tomato_Seeds'
    healthy_dir = seeds_path / 'Healthy'
    infected_dir = seeds_path / 'Infected'
    models_dir = seeds_path / 'Models'
    logs_dir = Path('home') / 'ARO.local' / 'sagig' / 'Projects' / 'seed_classification'

    # Parameters
    n = 8000
    # bands = sorted(range(1, 639, 63))
    bands = sorted([64, 127, 568])
    height = np.load(seeds_path / 'normalization_parameters' / 'max_height.npy').item() + 2
    width = np.load(seeds_path / 'normalization_parameters' / 'max_width.npy').item() + 2
    shape = (height, width, len(bands))
    batch_size = 32

    # Data splits
    (train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = prepare_data(str(healthy_dir), str(infected_dir), n, n, 0.5)

    train_dataset = HyperspectralTorchDataset(train_files, train_labels, bands, shape)
    val_dataset = HyperspectralTorchDataset(val_files, val_labels, bands, shape)
    test_dataset = HyperspectralTorchDataset(test_files, test_labels, bands, shape)

    mp_context = 'fork' if 'fork' in mp.get_all_start_methods() else 'spawn'
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() - 2, persistent_workers=True, prefetch_factor=4, pin_memory=True, multiprocessing_context=mp_context)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=os.cpu_count() - 2, persistent_workers=True, prefetch_factor=4, pin_memory=True, multiprocessing_context=mp_context)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=os.cpu_count() - 2, persistent_workers=True, prefetch_factor=4, pin_memory=True, multiprocessing_context=mp_context)

    # Model
    # model = HyperspectralCNN(shape[-1])
    # model = HyperspectralMultiCNN(shape[-1])
    # model = SpectralSpatialModuleModel(shape[-1])
    # model = SpectralSpatialSingleHeatmapAttentionModel(shape[-1])
    # model = SpectralSpatialMultiHeatmapAttentionModel(shape[-1])
    # model = SpectralSpatialPatchHeatmapAttentionModel(shape[-1])
    model = BandAttentionModel(shape[-1])

    # model = torch.compile(model, backend="eager")
    # model_name = model._orig_mod.__class__.__name__
    model_name = model.__class__.__name__

    # Callbacks
    bands_str = '-'.join(map(str, bands))
    ckpt_path = models_dir / f'{model_name}_{bands_str}'
    checkpoint_cb = ModelCheckpoint(
        dirpath=models_dir,
        filename=f"{model_name}_{bands_str}_data_{2*n}_seed{seed}" + "-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    csv_logger = CSVLogger(save_dir=logs_dir, name=f"{model_name}_{bands_str}")
    # profiler = SimpleProfiler()
    trainer = Trainer(
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        default_root_dir=logs_dir,
        max_epochs=50,
        callbacks=[early_stop_cb, checkpoint_cb, lr_monitor],
        logger=csv_logger,
        log_every_n_steps=10,
        accelerator="auto",
        # profiler=profiler
    )
    

    # Train
    training_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    training_time = time.time() - training_time
    # print(profiler.summary())

    if trainer.is_global_zero:
        # Test
        model.eval()
        preds, targets = [], []
        for x, y in test_loader:
            with torch.no_grad():
                output = model(x)
                if isinstance(output, tuple):
                    if len(output) == 4:
                        logits, spec_w, spat_map, fusion_w = output
                    elif len(output) == 2:
                        logits, attn_weights = output
                else:
                    logits = output  # if model returns just logits
                # Robust prediction extraction for multiclass and binary
                if logits.shape[-1] == 1:
                    # Binary classification: logits shape [batch, 1]
                    preds.extend((logits > 0.5).int().cpu().numpy().squeeze())
                else:
                    # Multiclass (including 2-class softmax): logits shape [batch, num_classes]
                    preds.extend(logits.argmax(dim=1).cpu().numpy())
                targets.extend(y.cpu().numpy())

        # see band importances
        if type(model) is SpectralSpatialModuleModel:
            spectral_weights = model.spectral.weights.detach().cpu().numpy()
            print('Importances:', spectral_weights)

        metrics_path = csv_logger.experiment.metrics_file_path
        # history = pd.read_csv(metrics_path).to_dict(orient="list")
        report = classification_report(targets, preds, digits=4, output_dict=True)
        cm = confusion_matrix(targets, preds)
        print(report)
        print(f"Test Accuracy: {report['accuracy']}")
        print(bands)
        print(2*n)

        # Save results
        accuracy = round(report["accuracy"], 3)
        suffix = f'{bands_str}-bands-{accuracy}-accuracy'
        torch.save(model.state_dict(), models_dir / f'{model_name}_{suffix}.pt')
        # with open(models_dir / f'{model_name}_{suffix}_history.json', "w") as f:
        #     json.dump(history, f, indent=4)
        with open(models_dir / f'{model_name}_{suffix}_report.json', "w") as f:
            json.dump(report, f, indent=4)
        with open(models_dir / f'{model_name}_{suffix}_cm.json', "w") as f:
            json.dump(cm.tolist(), f)

        total_run_time = time.time() - total_run_time
        print(f'total run time: {total_run_time}', f'training time: {training_time}')
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from NOAA_dataset import NOAATornadoDataset
from unet import UNet


def load_model(model_path, device):
    model = UNet(in_channels=3, num_classes=2).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _prepare_input(x):
    """Same normalization as NOAATornadoDataset. No resize — data is already 256x256."""
    x = x.float()
    x = x / (x.abs().amax(dim=(1, 2), keepdim=True) + 1e-8)
    return x


def _to_numpy_chw(tensor):
    return tensor.detach().cpu().numpy()


@torch.no_grad()
def pred_show_image_grid(data_path, model_path, device, max_samples=None):
    """
    Grid over validation set (last 20% of 2014 data).
    Rows per column (one sample):
      1. CAPE input
      2. True tornado probability
      3. Predicted tornado probability
      4. True significant-tornado probability
      5. Predicted significant-tornado probability
    """
    model = load_model(model_path, device)
    
    cape_dir = os.path.join(data_path, "train", "cape")
    all_files = sorted(f for f in os.listdir(cape_dir) if f.startswith("2014"))
    split_idx = int(len(all_files) * 0.8)
    val_files = all_files[split_idx:]
    
    dataset = NOAATornadoDataset(data_path, test=False, file_list=val_files)

    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    if n == 0:
        raise ValueError("Test dataset is empty. Check ./data/manual_test/ layout.")

    cape_inputs = []
    true_tor    = []
    pred_tor    = []
    true_sig    = []
    pred_sig    = []

    for i in range(n):
        x, y = dataset[i]
        x_batch = x.unsqueeze(0).to(device)

        probs = torch.sigmoid(model(x_batch)).squeeze(0).cpu()  # [2, H, W]
        y_np  = _to_numpy_chw(y)

        cape_inputs.append(x[0].cpu().numpy())   # channel 0 = CAPE
        true_tor.append(y_np[0])
        pred_tor.append(probs[0].numpy())
        true_sig.append(y_np[1])
        pred_sig.append(probs[1].numpy())

    rows = [
        ("CAPE (input)",           cape_inputs, "viridis"),
        ("True tornado prob",      true_tor,    "hot"),
        ("Pred tornado prob",      pred_tor,    "hot"),
        ("True sig. tornado prob", true_sig,    "hot"),
        ("Pred sig. tornado prob", pred_sig,    "hot"),
    ]

    fig, axes = plt.subplots(len(rows), n, figsize=(3 * n, 3 * len(rows)))
    if n == 1:
        axes = axes.reshape(-1, 1)

    for row_idx, (title, images, cmap) in enumerate(rows):
        for col_idx in range(n):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(images[col_idx], cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(title)
            if row_idx == 0:
                ax.set_title(f"sample {col_idx}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def single_sample_inference(data_path, model_path, device, sample_index=0):
    """Run inference on one validation test sample by index."""
    model   = load_model(model_path, device)
    
    cape_dir = os.path.join(data_path, "train", "cape")
    all_files = sorted(f for f in os.listdir(cape_dir) if f.startswith("2014"))
    split_idx = int(len(all_files) * 0.8)
    val_files = all_files[split_idx:]
    
    dataset = NOAATornadoDataset(data_path, test=False, file_list=val_files)

    if sample_index >= len(dataset):
        raise IndexError(f"sample_index {sample_index} out of range (len={len(dataset)})")

    x, y   = dataset[sample_index]
    x_in   = x.unsqueeze(0).to(device)
    probs  = torch.sigmoid(model(x_in)).squeeze(0).cpu().numpy()  # [2, H, W]
    y_np   = _to_numpy_chw(y)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    panels = [
        (x[0].numpy(),  "CAPE (input)",        "viridis", None),
        (y_np[0],       "True tornado",         "hot",     (0, 1)),
        (probs[0],      "Pred tornado",         "hot",     (0, 1)),
        (y_np[1],       "True sig. tornado",    "hot",     (0, 1)),
        (probs[1],      "Pred sig. tornado",    "hot",     (0, 1)),
    ]

    for ax, (img, title, cmap, vrange) in zip(axes.flat[:5], panels):
        vmin, vmax = vrange if vrange else (None, None)
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[1, 2].axis("off")
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def single_day_inference_from_npy(data_path, model_path, device, sample_id, test=False):
    """
    Run inference when you know the filename/id (e.g. '2014-05-18.npy')
    without using dataset index order.
    """
    model         = load_model(model_path, device)
    folder_prefix = "manual_test" if test else "train"

    # Load and guard against NOAA fill values
    cape = np.nan_to_num(
        np.load(os.path.join(data_path, folder_prefix, "cape", sample_id)),
        nan=0.0, posinf=0.0, neginf=0.0)
    cin  = np.nan_to_num(
        np.load(os.path.join(data_path, folder_prefix, "cin",  sample_id)),
        nan=0.0, posinf=0.0, neginf=0.0)
    geo  = np.nan_to_num(
        np.load(os.path.join(data_path, folder_prefix, "geo",  sample_id)),
        nan=0.0, posinf=0.0, neginf=0.0)

    x = torch.from_numpy(np.stack([cape, cin, geo], axis=0))
    x = _prepare_input(x).unsqueeze(0).to(device)   # [1, 3, 256, 256]

    probs = torch.sigmoid(model(x)).squeeze(0).cpu().numpy()  # [2, H, W]

    # Optionally load labels if they exist
    tor_path = os.path.join(data_path, f"{folder_prefix}_masks", "tornado", sample_id)
    sig_path = os.path.join(data_path, f"{folder_prefix}_masks", "sigtor",  sample_id)
    y = None
    if os.path.exists(tor_path) and os.path.exists(sig_path):
        tor = np.load(tor_path)
        sig = np.load(sig_path)
        y   = np.stack([tor, sig], axis=0)   # [2, H, W] — already 256x256 from preprocess

    ncols = 3 if y is not None else 2
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8))

    # CAPE input
    im = axes[0, 0].imshow(x[0, 0].cpu().numpy(), cmap="viridis")
    axes[0, 0].set_title("CAPE (input)")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Predictions
    im = axes[0, 1].imshow(probs[0], cmap="hot", vmin=0, vmax=1)
    axes[0, 1].set_title("Pred tornado")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im = axes[1, 1].imshow(probs[1], cmap="hot", vmin=0, vmax=1)
    axes[1, 1].set_title("Pred sig. tornado")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Ground truth (if available)
    if y is not None:
        im = axes[0, 2].imshow(y[0], cmap="hot", vmin=0, vmax=1)
        axes[0, 2].set_title("True tornado")
        fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

        im = axes[1, 2].imshow(y[1], cmap="hot", vmin=0, vmax=1)
        axes[1, 2].set_title("True sig. tornado")
        fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1, 0].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    DATA_PATH  = "./data"
    MODEL_PATH = "./models/unet.pth"
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    # Grid over all validation samples (limit with max_samples for speed)
    pred_show_image_grid(DATA_PATH, MODEL_PATH, device, max_samples=5)

    # One sample by dataset index
    single_sample_inference(DATA_PATH, MODEL_PATH, device, sample_index=0)

    # One day by filename — update to any 2014 date you have in train/
    # single_day_inference_from_npy(DATA_PATH, MODEL_PATH, device, "2014-11-18.npy")
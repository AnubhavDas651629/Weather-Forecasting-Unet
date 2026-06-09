import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from NOAA_dataset import NOAATornadoDataset
from unet import UNet


def load_coordinates():
    """
    Load lat/lon coordinate matrices from original netCDF datasets
    and interpolate them to (256, 256) matching the preprocessed tensors.
    """
    cape_nc = "/Users/parthapratimdas/Downloads/CAPE 2014.nc"
    tor_nc = "/Users/parthapratimdas/Downloads/Pper Tor 1979-2023.nc"

    # Load target coordinates (CONUS Lambert Grid, size 65x93)
    with xr.open_dataset(tor_nc, use_cftime=False) as ds_t:
        target_lat = ds_t.lat.values
        target_lon = ds_t.lon.values

    # Load input coordinates (Global Grid, size 91x180)
    with xr.open_dataset(cape_nc, use_cftime=False) as ds_i:
        in_lon, in_lat = np.meshgrid(ds_i.lon.values, ds_i.lat.values)

    # Convert lon to -180 to 180 to avoid wrap-around plotting issues in Cartopy
    target_lon = (target_lon + 180) % 360 - 180
    in_lon = (in_lon + 180) % 360 - 180

    # Resize coordinates to (256, 256) using PyTorch bilinear interpolation
    def _resize_coord(arr):
        t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        r = torch.nn.functional.interpolate(
            t, size=(256, 256), mode="bilinear", align_corners=False
        )
        return r.squeeze().numpy()

    in_lon_256 = _resize_coord(in_lon)
    in_lat_256 = _resize_coord(in_lat)
    tgt_lon_256 = _resize_coord(target_lon)
    tgt_lat_256 = _resize_coord(target_lat)

    return in_lon_256, in_lat_256, tgt_lon_256, tgt_lat_256


# Define custom SPC Colormap matching the paper and official NOAA SPC hazard scale
spc_colors = [
    "none",  # 0% - 2% (transparent, letting map land color show through)
    "#388E3C",  # 2% - 5% (green)
    "#8B5A2B",  # 5% - 10% (brown)
    "#FFEB3B",  # 10% - 15% (yellow)
    "#D32F2F",  # 15% - 30% (red)
    "#E91E63",  # 30% - 45% (pink)
    "#8E24AA",  # 45% - 60% (purple)
    "#1A237E",  # 60% - 100% (indigo/blue)
]
spc_levels = [0.0, 0.02, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60, 1.0]
spc_cmap = mcolors.ListedColormap(spc_colors)
spc_norm = mcolors.BoundaryNorm(spc_levels, ncolors=spc_cmap.N)


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
    true_tor = []
    pred_tor = []
    true_sig = []
    pred_sig = []

    for i in range(n):
        x, y = dataset[i]
        x_batch = x.unsqueeze(0).to(device)

        probs = torch.sigmoid(model(x_batch)).squeeze(0).cpu()  # [2, H, W]
        y_np = _to_numpy_chw(y)

        cape_inputs.append(x[0].cpu().numpy())  # channel 0 = CAPE
        true_tor.append(y_np[0])
        pred_tor.append(probs[0].numpy())
        true_sig.append(y_np[1])
        pred_sig.append(probs[1].numpy())

    # Load coordinates
    in_lon, in_lat, tgt_lon, tgt_lat = load_coordinates()

    rows = [
        ("CAPE (input)", cape_inputs, "viridis", None),
        ("True tornado prob", true_tor, spc_cmap, spc_norm),
        ("Pred tornado prob", pred_tor, spc_cmap, spc_norm),
        ("True sig. tornado prob", true_sig, spc_cmap, spc_norm),
        ("Pred sig. tornado prob", pred_sig, spc_cmap, spc_norm),
    ]

    # Map projection for CONUS (Lambert Conformal)
    proj = ccrs.LambertConformal(central_longitude=-95.0, central_latitude=25.0)

    fig = plt.figure(figsize=(4.5 * n, 4 * len(rows)))

    for row_idx, (title, images, cmap, norm) in enumerate(rows):
        for col_idx in range(n):
            ax_idx = row_idx * n + col_idx + 1
            ax = fig.add_subplot(len(rows), n, ax_idx, projection=proj)
            ax.set_extent([-120, -73, 23, 50], crs=ccrs.PlateCarree())

            # Draw geographical base maps (underneath the data)
            ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", zorder=1)
            ax.add_feature(cfeature.OCEAN, facecolor="#e0f2f1", zorder=1)

            img_data = images[col_idx]
            lon_coords = in_lon if row_idx == 0 else tgt_lon
            lat_coords = in_lat if row_idx == 0 else tgt_lat

            if norm is not None:
                im = ax.pcolormesh(
                    lon_coords,
                    lat_coords,
                    img_data,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap,
                    norm=norm,
                    shading="nearest",
                    zorder=2,
                )
            else:
                im = ax.pcolormesh(
                    lon_coords,
                    lat_coords,
                    img_data,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                    shading="nearest",
                    zorder=2,
                )

            # Draw geographical boundaries (on top of the data)
            ax.add_feature(
                cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=3
            )
            ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="gray", zorder=3)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black", zorder=3)

            if col_idx == 0:
                ax.text(
                    -0.08,
                    0.5,
                    title,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    weight="bold",
                    fontsize=11,
                )
            if row_idx == 0:
                ax.set_title(f"sample {col_idx}", weight="bold")

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("grid_output.png")
    print("Saved grid_output.png")


@torch.no_grad()
def single_sample_inference(data_path, model_path, device, sample_index=0):
    """Run inference on one validation test sample by index."""
    model = load_model(model_path, device)

    cape_dir = os.path.join(data_path, "train", "cape")
    all_files = sorted(f for f in os.listdir(cape_dir) if f.startswith("2014"))
    split_idx = int(len(all_files) * 0.8)
    val_files = all_files[split_idx:]

    dataset = NOAATornadoDataset(data_path, test=False, file_list=val_files)

    if sample_index >= len(dataset):
        raise IndexError(
            f"sample_index {sample_index} out of range (len={len(dataset)})"
        )

    x, y = dataset[sample_index]
    x_in = x.unsqueeze(0).to(device)
    probs = torch.sigmoid(model(x_in)).squeeze(0).cpu().numpy()  # [2, H, W]
    y_np = _to_numpy_chw(y)

    in_lon, in_lat, tgt_lon, tgt_lat = load_coordinates()

    fig = plt.figure(figsize=(15, 10))
    proj = ccrs.LambertConformal(central_longitude=-95.0, central_latitude=25.0)

    panels = [
        (x[0].numpy(), "CAPE (input)", "viridis", None, None, in_lon, in_lat),
        (y_np[0], "True tornado", spc_cmap, spc_norm, None, tgt_lon, tgt_lat),
        (probs[0], "Pred tornado", spc_cmap, spc_norm, None, tgt_lon, tgt_lat),
        (y_np[1], "True sig. tornado", spc_cmap, spc_norm, None, tgt_lon, tgt_lat),
        (probs[1], "Pred sig. tornado", spc_cmap, spc_norm, None, tgt_lon, tgt_lat),
    ]

    for idx, (img, title, cmap, norm, vrange, lon_coords, lat_coords) in enumerate(
        panels
    ):
        ax = fig.add_subplot(2, 3, idx + 1, projection=proj)
        ax.set_extent([-120, -73, 23, 50], crs=ccrs.PlateCarree())

        # Add base maps (underneath the data)
        ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", zorder=1)
        ax.add_feature(cfeature.OCEAN, facecolor="#e0f2f1", zorder=1)

        if norm is not None:
            im = ax.pcolormesh(
                lon_coords,
                lat_coords,
                img,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=norm,
                shading="nearest",
                zorder=2,
            )
        else:
            vmin, vmax = vrange if vrange else (None, None)
            im = ax.pcolormesh(
                lon_coords,
                lat_coords,
                img,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="nearest",
                zorder=2,
            )

        # Add base boundaries (on top of the data)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=3)
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="gray", zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black", zorder=3)

        ax.set_title(title, weight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Empty 6th plot
    ax_empty = fig.add_subplot(2, 3, 6)
    ax_empty.axis("off")

    plt.tight_layout()
    plt.savefig(f"single_sample_{sample_index}.png")
    print(f"Saved single_sample_{sample_index}.png")


@torch.no_grad()
def single_day_inference_from_npy(data_path, model_path, device, sample_id, test=False):
    """
    Run inference when you know the filename/id (e.g. '2014-05-18.npy')
    without using dataset index order.
    """
    model = load_model(model_path, device)
    folder_prefix = "manual_test" if test else "train"

    # Load and guard against NOAA fill values
    cape = np.nan_to_num(
        np.load(os.path.join(data_path, folder_prefix, "cape", sample_id)),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    cin = np.nan_to_num(
        np.load(os.path.join(data_path, folder_prefix, "cin", sample_id)),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    geo = np.nan_to_num(
        np.load(os.path.join(data_path, folder_prefix, "geo", sample_id)),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    x = torch.from_numpy(np.stack([cape, cin, geo], axis=0))
    x = _prepare_input(x).unsqueeze(0).to(device)  # [1, 3, 256, 256]

    probs = torch.sigmoid(model(x)).squeeze(0).cpu().numpy()  # [2, H, W]

    # Optionally load labels if they exist
    tor_path = os.path.join(data_path, f"{folder_prefix}_masks", "tornado", sample_id)
    sig_path = os.path.join(data_path, f"{folder_prefix}_masks", "sigtor", sample_id)
    y = None
    if os.path.exists(tor_path) and os.path.exists(sig_path):
        tor = np.load(tor_path)
        sig = np.load(sig_path)
        y = (
            np.stack([tor, sig], axis=0) / 100.0
        )  # [2, H, W] — already 256x256 from preprocess

    in_lon, in_lat, tgt_lon, tgt_lat = load_coordinates()
    proj = ccrs.LambertConformal(central_longitude=-95.0, central_latitude=25.0)

    ncols = 3 if y is not None else 2
    fig = plt.figure(figsize=(5 * ncols, 10))

    def plot_panel(
        ax_idx, title, img, cmap, norm=None, vmin=None, vmax=None, is_cape=False
    ):
        ax = fig.add_subplot(2, ncols, ax_idx, projection=proj)
        ax.set_extent([-120, -73, 23, 50], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", zorder=1)
        ax.add_feature(cfeature.OCEAN, facecolor="#e0f2f1", zorder=1)

        lon = in_lon if is_cape else tgt_lon
        lat = in_lat if is_cape else tgt_lat

        if norm is not None:
            im = ax.pcolormesh(
                lon,
                lat,
                img,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=norm,
                shading="nearest",
                zorder=2,
            )
        else:
            im = ax.pcolormesh(
                lon,
                lat,
                img,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="nearest",
                zorder=2,
            )

        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=3)
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="gray", zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black", zorder=3)

        ax.set_title(title, weight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # CAPE input
    plot_panel(1, "CAPE (input)", x[0, 0].cpu().numpy(), "viridis", is_cape=True)

    # Predictions
    plot_panel(2, "Pred tornado", probs[0], spc_cmap, norm=spc_norm)
    plot_panel(ncols + 2, "Pred sig. tornado", probs[1], spc_cmap, norm=spc_norm)

    # Ground truth (if available)
    if y is not None:
        plot_panel(3, "True tornado", y[0], spc_cmap, norm=spc_norm)
        plot_panel(ncols + 3, "True sig. tornado", y[1], spc_cmap, norm=spc_norm)

    # Turn off unused axes
    ax_empty = fig.add_subplot(2, ncols, ncols + 1)
    ax_empty.axis("off")

    plt.tight_layout()
    plt.savefig(f"day_inference_{sample_id}.png")
    print(f"Saved day_inference_{sample_id}.png")


if __name__ == "__main__":
    DATA_PATH = "./data"
    MODEL_PATH = "./models/unet.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Grid over all validation samples (limit with max_samples for speed)
    pred_show_image_grid(DATA_PATH, MODEL_PATH, device, max_samples=5)

    # One sample by dataset index
    single_sample_inference(DATA_PATH, MODEL_PATH, device, sample_index=0)

    # One day by filename — update to any 2014 date you have in train/
    # single_day_inference_from_npy(DATA_PATH, MODEL_PATH, device, "2014-11-18.npy")

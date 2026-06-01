import os
import numpy as np
import xarray as xr
import torch
from torch.nn.functional import interpolate

# ── Input .nc file paths (2014) ───────────────────────────────────────────────
# Make sure these filenames exactly match what you downloaded from NOAA PSL.
# No backslashes needed — just plain strings with spaces if the filename has them.
CAPE_PATH        = "/Users/parthapratimdas/Downloads/CAPE 2014.nc"
CIN_PATH         = "/Users/parthapratimdas/Downloads/CIN 2014.nc"       # Must be real CIN file, NOT CAPE
HGT_PATH         = "/Users/parthapratimdas/Downloads/HGT Tropo 2014.nc"
TOR_TARGET_PATH  = "/Users/parthapratimdas/Downloads/Pper Tor 1979-2023.nc"
SIGTOR_TARGET_PATH = "/Users/parthapratimdas/Downloads/Pper Sig Tor 1979-2023.nc"

OUTPUT_DIR = "./data"


def create_folders():
    """Create the directory structure expected by NOAA_dataset.py"""
    paths = [
        "train/cape",
        "train/cin",
        "train/geo",
        "train_masks/tornado",
        "train_masks/sigtor",
    ]
    for p in paths:
        os.makedirs(os.path.join(OUTPUT_DIR, p), exist_ok=True)
    print("Directory structure ready.")


def resize_grid(matrix_2d):
    """
    Resize any 2D weather matrix to exactly 256x256 using bilinear interpolation.
    Bilinear is correct for continuous fields (CAPE, CIN, probability maps).
    """
    arr = np.nan_to_num(matrix_2d, nan=0.0, posinf=0.0, neginf=0.0)
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resized = interpolate(tensor, size=(256, 256), mode='bilinear', align_corners=False)
    return resized.squeeze().numpy()


def get_variable(ds):
    """
    Auto-detect the main data variable in a NetCDF file.
    NOAA files often use lowercase names like 'cape', 'cin', 'hgt', 'prob'.
    Falls back gracefully with a clear error if not found.
    """
    # Drop pure coordinate/dimension variables
    candidates = [v for v in ds.data_vars]
    if len(candidates) == 1:
        return candidates[0]
    # Common NOAA variable names in priority order
    for name in ["cape", "cin", "hgt", "prob", "CAPE", "CIN", "HGT", "PROB"]:
        if name in ds.data_vars:
            return name
    raise KeyError(
        f"Cannot auto-detect variable. Available variables: {list(ds.data_vars)}. "
        "Update get_variable() with the correct name."
    )


def main():
    print("Creating directory structure...")
    create_folders()

    # ── Load feature datasets (already 2014-only files) ──────────────────────
    print("Loading 2014 atmospheric feature files...")
    ds_cape = xr.open_dataset(CAPE_PATH)
    ds_cin  = xr.open_dataset(CIN_PATH)
    ds_hgt  = xr.open_dataset(HGT_PATH)

    # ── Load multi-year target files and slice 2014 ───────────────────────────
    print("Loading target files and slicing 2014...")
    ds_tor    = xr.open_dataset(TOR_TARGET_PATH).sel(time="2014")
    ds_sigtor = xr.open_dataset(SIGTOR_TARGET_PATH).sel(time="2014")

    # Auto-detect variable names inside each file
    cape_var   = get_variable(ds_cape)
    cin_var    = get_variable(ds_cin)
    hgt_var    = get_variable(ds_hgt)
    tor_var    = get_variable(ds_tor)
    sigtor_var = get_variable(ds_sigtor)

    print(f"Detected variables  →  cape: '{cape_var}'  cin: '{cin_var}'  "
          f"hgt: '{hgt_var}'  tor: '{tor_var}'  sigtor: '{sigtor_var}'")

    # ── Build fully-aligned date list across ALL five datasets ───────────────
    dates = ds_cape.time.values
    for ds, label in [
        (ds_cin,    "CIN"),
        (ds_hgt,    "HGT"),
        (ds_tor,    "tornado target"),
        (ds_sigtor, "sigtor target"),
    ]:
        dates = np.intersect1d(dates, ds.time.values)

    print(f"Perfectly aligned days across all datasets: {len(dates)}")
    if len(dates) == 0:
        raise RuntimeError(
            "No overlapping dates found. Check that all files cover 2014 "
            "and that time coordinates are in the same format."
        )

    # ── Process each day ──────────────────────────────────────────────────────
    for i, date in enumerate(dates):
        date_str = np.datetime_as_string(date, unit='D')
        print(f"[{i+1}/{len(dates)}] Processing {date_str} ...", end=" ")

        # Extract 2D slice for this day.
        # .sel(time=date) drops the time dimension; .squeeze() removes any
        # leftover size-1 dimensions (e.g. level, ensemble member).
        cape_map   = ds_cape[cape_var].sel(time=date).squeeze().values
        cin_map    = ds_cin[cin_var].sel(time=date).squeeze().values
        hgt_map    = ds_hgt[hgt_var].sel(time=date).squeeze().values
        tor_map    = ds_tor[tor_var].sel(time=date).squeeze().values
        sigtor_map = ds_sigtor[sigtor_var].sel(time=date).squeeze().values

        # resize_grid already calls nan_to_num internally
        np.save(os.path.join(OUTPUT_DIR, "train/cape",           f"{date_str}.npy"),
                resize_grid(cape_map))
        np.save(os.path.join(OUTPUT_DIR, "train/cin",            f"{date_str}.npy"),
                resize_grid(cin_map))
        np.save(os.path.join(OUTPUT_DIR, "train/geo",            f"{date_str}.npy"),
                resize_grid(hgt_map))
        np.save(os.path.join(OUTPUT_DIR, "train_masks/tornado",  f"{date_str}.npy"),
                resize_grid(tor_map))
        np.save(os.path.join(OUTPUT_DIR, "train_masks/sigtor",   f"{date_str}.npy"),
                resize_grid(sigtor_map))

        print("done")

    print(f"\nPreprocessing complete! {len(dates)} days saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
import os
import numpy as np
import xarray as xr
import torch
from torch.nn.functional import interpolate

# ── Input .nc file paths (2014) ───────────────────────────────────────────────
# Plain strings — no backslashes needed before spaces.
CAPE_PATH          = "/Users/parthapratimdas/Downloads/CAPE 2014.nc"
CIN_PATH           = "/Users/parthapratimdas/Downloads/CIN 2014.nc"
HGT_PATH           = "/Users/parthapratimdas/Downloads/HGT Tropo 2014.nc"
TOR_TARGET_PATH    = "/Users/parthapratimdas/Downloads/Pper Tor 1979-2023.nc"
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
    Resize a 2D weather matrix to exactly (256, 256) using bilinear interpolation.
    - nan_to_num runs first to kill any NOAA fill values.
    - Returns a plain (256, 256) numpy float32 array.
    """
    arr = np.nan_to_num(matrix_2d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    tensor  = torch.tensor(arr).unsqueeze(0).unsqueeze(0)           # [1,1,H,W]
    resized = interpolate(tensor, size=(256, 256), mode='bilinear', align_corners=False)
    return resized.squeeze().numpy()                                  # (256,256)


def get_variable(ds):
    """
    Auto-detect the main data variable in a NetCDF dataset.
    Drops pure coordinate variables and tries common NOAA names.
    Raises a clear error listing available names if nothing matches.
    """
    candidates = list(ds.data_vars)
    if len(candidates) == 1:
        return candidates[0]
    for name in ["cape", "cin", "hgt", "prob", "CAPE", "CIN", "HGT", "PROB"]:
        if name in ds.data_vars:
            return name
    raise KeyError(
        f"Cannot auto-detect variable. Available: {candidates}. "
        "Set the variable name manually in get_variable()."
    )


def extract_2d(da, date):
    """
    Safely extract a 2D (lat x lon) slice from a DataArray for a given date.

    Handles the two most common extra dimensions in NOAA reanalysis files:
      - 'level'    : multiple pressure levels  -> take index 0 (surface/lowest)
      - 'ensemble' : ensemble members          -> take index 0
    After dropping those, .squeeze() removes any remaining size-1 dims.
    Raises clearly if the result is still not 2D.
    """
    da = da.sel(time=date)

    if 'level' in da.dims:
        da = da.isel(level=0)
    if 'ensemble' in da.dims:
        da = da.isel(ensemble=0)

    da = da.squeeze()

    if da.ndim != 2:
        raise ValueError(
            f"Expected a 2D slice after selecting time={date} but got shape {da.shape} "
            f"with dims {da.dims}. Add an explicit .isel() for the extra dimension."
        )

    return da.values   # numpy array (H, W)


def save_validated(arr, path, label):
    """
    Resize to (256,256), assert shape, then save as .npy.
    Fails loudly if resize_grid returns an unexpected shape.
    """
    result = resize_grid(arr)
    if result.shape != (256, 256):
        raise ValueError(
            f"resize_grid returned {result.shape} instead of (256, 256) "
            f"for {label}. Check the input array dimensions."
        )
    np.save(path, result)


def main():
    print("Creating directory structure...")
    create_folders()

    # ── Load datasets with use_cftime=False ──────────────────────────────────
    # NOAA files sometimes use cftime objects instead of numpy.datetime64.
    # Forcing use_cftime=False ensures all time axes are numpy.datetime64,
    # so np.intersect1d works correctly across all five datasets.
    print("Loading 2014 atmospheric feature files...")
    ds_cape = xr.open_dataset(CAPE_PATH,          use_cftime=False)
    ds_cin  = xr.open_dataset(CIN_PATH,           use_cftime=False)
    ds_hgt  = xr.open_dataset(HGT_PATH,           use_cftime=False)

    print("Loading target files and slicing 2014...")
    ds_tor    = xr.open_dataset(TOR_TARGET_PATH,    use_cftime=False).sel(time="2014")
    ds_sigtor = xr.open_dataset(SIGTOR_TARGET_PATH, use_cftime=False).sel(time="2014")

    # Auto-detect variable names
    cape_var   = get_variable(ds_cape)
    cin_var    = get_variable(ds_cin)
    hgt_var    = get_variable(ds_hgt)
    tor_var    = get_variable(ds_tor)
    sigtor_var = get_variable(ds_sigtor)

    print(f"Detected variables  ->  "
          f"cape='{cape_var}'  cin='{cin_var}'  hgt='{hgt_var}'  "
          f"tor='{tor_var}'  sigtor='{sigtor_var}'")

    # ── Build fully-aligned date list across ALL five datasets ───────────────
    dates = ds_cape.time.values
    for ds, label in [
        (ds_cin,    "CIN"),
        (ds_hgt,    "HGT"),
        (ds_tor,    "tornado target"),
        (ds_sigtor, "sigtor target"),
    ]:
        dates = np.intersect1d(dates, ds.time.values)

    print(f"Aligned days across all five datasets: {len(dates)}")
    if len(dates) == 0:
        raise RuntimeError(
            "No overlapping dates found. "
            "Check that all .nc files cover 2014 and that use_cftime=False "
            "decoded the time axes correctly. "
            "Try printing ds_cape.time.values[:3] and ds_tor.time.values[:3] "
            "to inspect the formats."
        )

    # ── Process each day ──────────────────────────────────────────────────────
    skipped = []
    for i, date in enumerate(dates):
        date_str = np.datetime_as_string(date, unit='D')
        print(f"[{i+1}/{len(dates)}] {date_str} ...", end=" ", flush=True)

        try:
            cape_map   = extract_2d(ds_cape[cape_var],     date)
            cin_map    = extract_2d(ds_cin[cin_var],       date)
            hgt_map    = extract_2d(ds_hgt[hgt_var],       date)
            tor_map    = extract_2d(ds_tor[tor_var],       date)
            sigtor_map = extract_2d(ds_sigtor[sigtor_var], date)

            save_validated(cape_map,   os.path.join(OUTPUT_DIR, "train/cape",          f"{date_str}.npy"), f"{date_str}/cape")
            save_validated(cin_map,    os.path.join(OUTPUT_DIR, "train/cin",           f"{date_str}.npy"), f"{date_str}/cin")
            save_validated(hgt_map,    os.path.join(OUTPUT_DIR, "train/geo",           f"{date_str}.npy"), f"{date_str}/geo")
            save_validated(tor_map,    os.path.join(OUTPUT_DIR, "train_masks/tornado", f"{date_str}.npy"), f"{date_str}/tor")
            save_validated(sigtor_map, os.path.join(OUTPUT_DIR, "train_masks/sigtor",  f"{date_str}.npy"), f"{date_str}/sigtor")

            print("done")

        except Exception as e:
            print(f"SKIPPED -- {e}")
            skipped.append((date_str, str(e)))

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\nPreprocessing complete!")
    print(f"  Saved  : {len(dates) - len(skipped)} days")
    print(f"  Skipped: {len(skipped)} days")
    if skipped:
        print("\nSkipped days:")
        for d, reason in skipped:
            print(f"  {d}  ->  {reason}")


if __name__ == "__main__":
    main()
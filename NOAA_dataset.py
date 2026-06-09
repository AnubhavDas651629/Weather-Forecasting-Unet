import os

import numpy as np
import torch
from torch.utils.data import Dataset


class NOAATornadoDataset(Dataset):
    def __init__(self, root_path, test=False, file_list=None):
        self.root_path = root_path
        folder_prefix = "manual_test" if test else "train"

        # If no file_list provided, use everything in the cape folder.
        # Passing file_list allows chronological year-based splitting in main.py.
        if file_list is None:
            ids = sorted(os.listdir(os.path.join(root_path, folder_prefix, "cape")))
        else:
            ids = sorted(file_list)

        self.cape_paths = [
            os.path.join(root_path, folder_prefix, "cape", i) for i in ids
        ]
        self.cin_paths = [os.path.join(root_path, folder_prefix, "cin", i) for i in ids]
        self.geo_paths = [os.path.join(root_path, folder_prefix, "geo", i) for i in ids]
        self.tor_paths = [
            os.path.join(root_path, f"{folder_prefix}_masks", "tornado", i) for i in ids
        ]
        self.sigtor_paths = [
            os.path.join(root_path, f"{folder_prefix}_masks", "sigtor", i) for i in ids
        ]

        # Verify all five channels have the same number of files
        counts = {
            "cape": len(self.cape_paths),
            "cin": len(self.cin_paths),
            "geo": len(self.geo_paths),
            "tor": len(self.tor_paths),
            "sigtor": len(self.sigtor_paths),
        }
        if len(set(counts.values())) != 1:
            raise AssertionError(f"Mismatch in file counts across channels: {counts}")

        if len(self.cape_paths) == 0:
            raise RuntimeError(
                f"No files found under {os.path.join(root_path, folder_prefix, 'cape')}. "
                "Run preprocess.py first."
            )

    def __getitem__(self, index):
        # 1. Load pre-scaled 256x256 numpy arrays
        cape = np.load(self.cape_paths[index])
        cin = np.load(self.cin_paths[index])
        geo = np.load(self.geo_paths[index])

        tor_prob = np.load(self.tor_paths[index])
        sigtor_prob = np.load(self.sigtor_paths[index])

        # 2. Guard against NOAA fill values (e.g. 9.96921e+36)
        cape = np.nan_to_num(cape, nan=0.0, posinf=0.0, neginf=0.0)
        cin = np.nan_to_num(cin, nan=0.0, posinf=0.0, neginf=0.0)
        geo = np.nan_to_num(geo, nan=0.0, posinf=0.0, neginf=0.0)

        tor_prob = np.nan_to_num(tor_prob, nan=0.0, posinf=0.0, neginf=0.0)
        sigtor_prob = np.nan_to_num(sigtor_prob, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. Stack into tensors  [C, H, W]
        x = torch.from_numpy(
            np.stack([cape, cin, geo], axis=0)
        ).float()  # [3, 256, 256]
        y = (
            torch.from_numpy(np.stack([tor_prob, sigtor_prob], axis=0)).float() / 100.0
        )  # [2, 256, 256]

        # 4. Per-channel absolute-max normalization for x
        #    Using abs() handles CIN which is negative.
        #    + 1e-8 prevents division by zero on all-zero channels (e.g. calm CIN day).
        channel_maxes = x.abs().amax(dim=(1, 2), keepdim=True)
        x = x / (channel_maxes + 1e-8)

        # 5. Clamp labels to [0, 1] — guards against any out-of-range values
        #    in the Gensini et al. probability files.
        y = y.clamp(0.0, 1.0)

        return x, y

    def __len__(self):
        return len(self.cape_paths)

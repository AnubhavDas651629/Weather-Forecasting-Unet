import os

import numpy as np
import torch
from tqdm import tqdm

from inference import single_day_inference_from_npy


def evaluate_severe_tornado_days(
    data_path, model_path, output_dir="evaluation_plots", threshold_pct=10.0
):
    """
    Finds all days in 2014 where the true tornado probability exceeded `threshold_pct`,
    and generates an inference plot for manual review.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Scan for severe tornado days
    print(
        f"Scanning 2014 dataset for days with > {threshold_pct}% tornado probability..."
    )
    tor_dir = os.path.join(data_path, "train_masks", "tornado")

    severe_days = []
    for file in os.listdir(tor_dir):
        if not file.endswith(".npy"):
            continue

        prob_mask = np.load(os.path.join(tor_dir, file))
        max_prob = prob_mask.max()

        if max_prob >= threshold_pct:
            severe_days.append((file, max_prob))

    # Sort by severity (highest probability first)
    severe_days.sort(key=lambda x: x[1], reverse=True)

    print(
        f"Found {len(severe_days)} severe days. Generating plots for manual review..."
    )

    # 2. Generate and save plots
    for filename, max_prob in tqdm(severe_days, desc="Generating Plots"):
        # We temporarily change the current working directory so the plot saves in the evaluation folder
        original_cwd = os.getcwd()
        os.chdir(output_dir)
        try:
            # Generate the plot for this specific day
            single_day_inference_from_npy(
                data_path=os.path.join(original_cwd, data_path),
                model_path=os.path.join(original_cwd, model_path),
                device=device,
                sample_id=filename,
            )
            # Rename the file to include the probability for easier review
            old_name = f"day_inference_{filename}.png"
            new_name = f"prob_{int(max_prob):02d}pct_{filename}.png"
            if os.path.exists(old_name):
                os.rename(old_name, new_name)
        finally:
            os.chdir(original_cwd)

    print(f"\nDone! All plots have been saved to the '{output_dir}' folder.")
    print(
        "You can now open this folder and manually check how well the model predicted the actual tornadoes."
    )


if __name__ == "__main__":
    evaluate_severe_tornado_days(
        data_path="./data",
        model_path="./models/unet.pth",
        output_dir="./evaluation_plots",
        threshold_pct=10.0,  # Only evaluate days with at least 10% tornado probability
    )

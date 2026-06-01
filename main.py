import os
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
from NOAA_dataset import NOAATornadoDataset


def kl_divergence(pred_probs, target, eps=1e-8):
    """KL(target || pred) averaged over batch, channels, and spatial dims."""
    target = target.clamp(eps, 1.0 - eps)
    pred_probs = pred_probs.clamp(eps, 1.0 - eps)
    kl_pos = target * (target.log() - pred_probs.log())
    kl_neg = (1.0 - target) * ((1.0 - target).log() - (1.0 - pred_probs).log())
    return (kl_pos + kl_neg).mean()


if __name__ == "__main__":
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8          # Reduced from 64 — only ~365 days of 2014 data
    EPOCHS = 10
    DATA_PATH = "./data"
    MODEL_SAVE_PATH = "./models/unet.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Chronological 80/20 split within 2014 ──────────────────────────────
    # Files are named YYYY-MM-DD.npy so sorting gives chronological order.
    cape_dir = os.path.join(DATA_PATH, "train", "cape")
    all_files = sorted(f for f in os.listdir(cape_dir) if f.startswith("2014"))

    if len(all_files) == 0:
        raise RuntimeError(f"No 2014 .npy files found in {cape_dir}. "
                           "Run preprocess.py first.")

    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files   = all_files[split_idx:]
    print(f"2014 days found: {len(all_files)}  |  "
          f"train: {len(train_files)}  val: {len(val_files)}")

    train_dataset = NOAATornadoDataset(DATA_PATH, test=False, file_list=train_files)
    val_dataset   = NOAATornadoDataset(DATA_PATH, test=False, file_list=val_files)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model = UNet(in_channels=3, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Higher weight on rare positive (tornado) pixels
    pos_weight = torch.tensor([50.0, 100.0]).view(1, 2, 1, 1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):

        # --- Train ---
        model.train()
        train_running_loss = 0.0
        for x, y in tqdm(train_dataloader, desc=f"Train {epoch+1}/{EPOCHS}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()

        train_loss = train_running_loss / len(train_dataloader)

        # --- Validate ---
        model.eval()
        val_running_loss = 0.0
        val_kl_total     = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_dataloader, desc=f"Val   {epoch+1}/{EPOCHS}"):
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                val_running_loss += criterion(y_pred, y).item()
                val_kl_total     += kl_divergence(torch.sigmoid(y_pred), y).item()

        val_loss = val_running_loss / len(val_dataloader)
        avg_kl   = val_kl_total     / len(val_dataloader)

        print("-" * 50)
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train BCE : {train_loss:.4f}")
        print(f"  Val BCE   : {val_loss:.4f}")
        print(f"  Val KL    : {avg_kl:.4f}")
        print("-" * 50)

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
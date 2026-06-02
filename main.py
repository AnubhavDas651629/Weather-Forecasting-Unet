import os
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
from NOAA_dataset import NOAATornadoDataset


def bernoulli_entropy(p, eps=1e-7):
    """H(p) = -p*log(p) - (1-p)*log(1-p), safe for p in {0, 1}."""
    p = p.clamp(eps, 1.0 - eps)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


def kl_divergence_from_logits(logits, target, eps=1e-7):
    """
    Numerically stable KL(target || pred) for Bernoulli distributions.

    Uses the identity:  KL(p || q) = BCE(p, q) - H(p)
    where BCE is computed from raw logits (log-sum-exp trick) and
    H(p) is the entropy of the target distribution.

    This avoids 0*log(0) by never computing log(pred_prob) directly.
    """
    # BCE(target, logits) — numerically stable via log-sum-exp
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='mean')
    # H(target) — the entropy of the ground truth labels
    h_target = bernoulli_entropy(target, eps).mean()
    return bce - h_target


def kl_divergence_from_probs(pred_probs, target, eps=1e-7):
    """
    Compute KL(target || pred) from sigmoid probabilities (for reporting).
    Safe version that handles target values at exactly 0 or 1.
    """
    target = target.clamp(eps, 1.0 - eps)
    pred_probs = pred_probs.clamp(eps, 1.0 - eps)
    kl = target * (target.log() - pred_probs.log()) \
       + (1.0 - target) * ((1.0 - target).log() - (1.0 - pred_probs).log())
    return kl.mean()


if __name__ == "__main__":
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8          # Reduced from 64 — only ~365 days of 2014 data
    EPOCHS = 50
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

    # ── Loss: KL divergence (matching the paper) ─────────────────────────
    # KL(target || pred) = BCE(target, logits) - H(target)
    # The paper uses KL divergence as the training objective.
    # We also keep weighted BCE for comparison reporting.
    pos_weight = torch.tensor([5.0, 10.0]).view(1, 2, 1, 1).to(device)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):

        # --- Train (KL divergence loss, matching the paper) ---
        model.train()
        train_kl_total  = 0.0
        train_bce_total = 0.0
        for x, y in tqdm(train_dataloader, desc=f"Train {epoch+1}/{EPOCHS}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            # Primary loss: KL divergence (paper's objective)
            loss = kl_divergence_from_logits(logits, y)
            loss.backward()
            optimizer.step()
            train_kl_total  += loss.item()
            # Secondary metric: weighted BCE (for comparison)
            with torch.no_grad():
                train_bce_total += bce_criterion(logits, y).item()

        train_kl  = train_kl_total  / len(train_dataloader)
        train_bce = train_bce_total / len(train_dataloader)

        # --- Validate ---
        model.eval()
        val_bce_total = 0.0
        val_kl_total  = 0.0
        val_mae_total = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_dataloader, desc=f"Val   {epoch+1}/{EPOCHS}"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                probs  = torch.sigmoid(logits)
                val_bce_total += bce_criterion(logits, y).item()
                val_kl_total  += kl_divergence_from_logits(logits, y).item()
                val_mae_total += (probs - y).abs().mean().item()

        val_bce = val_bce_total / len(val_dataloader)
        val_kl  = val_kl_total  / len(val_dataloader)
        val_mae = val_mae_total / len(val_dataloader)

        print("-" * 50)
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train KL  : {train_kl:.4f}")
        print(f"  Train BCE : {train_bce:.4f}")
        print(f"  Val KL    : {val_kl:.4f}")
        print(f"  Val BCE   : {val_bce:.4f}")
        print(f"  Val MAE   : {val_mae:.6f}")
        print("-" * 50)

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
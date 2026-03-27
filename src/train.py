import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os, sys

sys.path.insert(0, os.path.dirname(__file__))

from model import LandslideEEGMoE, pretrain_loss, finetune_loss
from utils import set_seed, log, ensure_dir

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────
# SAFE LOSS WRAPPER (fixes NaN)
# ─────────────────────────────────────────────
def safe_loss(loss):
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=loss.device, requires_grad=True)
    return loss


# ─────────────────────────────────────────────
# ONE EPOCH
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, mode='finetune'):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in tqdm(loader, desc=f"Training ({mode})", leave=False):
        x = x.to(DEVICE).float()
        y = y.to(DEVICE).long()

        optimizer.zero_grad()

        # ---------------- PRETRAIN ----------------
        if mode == 'pretrain':
            z_rec, z_orig, mask, probs = model(x, mode='pretrain')

            loss = pretrain_loss(z_rec, z_orig, mask, probs)

            # 🔥 stabilize NaNs
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)
            loss = safe_loss(loss)

        # ---------------- FINETUNE ----------------
        else:
            logits, _ = model(x, mode='finetune')

            loss = finetune_loss(logits, y)

            # 🔥 stabilize logits
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e3, neginf=-1e3)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        loss.backward()

        # 🔥 gradient clipping (CRITICAL for MoE stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / max(len(loader), 1)
    acc = correct / total if total > 0 else 0

    return avg_loss, acc


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()

    correct, total = 0, 0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE).float()
            y = y.to(DEVICE).long()

            logits, _ = model(x, mode='finetune')

            logits = torch.nan_to_num(logits, nan=0.0)

            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(y.cpu().numpy())

    return correct / total, np.array(preds_all), np.array(labels_all)


# ─────────────────────────────────────────────
# MAIN TRAIN FUNCTION
# ─────────────────────────────────────────────
def train(train_loader, val_loader, num_channels,
          patch_size=16,
          pretrain_epochs=5,
          finetune_epochs=35,
          lr_pretrain=1e-4,
          lr_finetune=3e-4):

    set_seed(42)
    ensure_dir('results')

    model = LandslideEEGMoE(
        input_channels=num_channels,
        h=patch_size, w=patch_size,
        embed_dim=128,
        hidden_dim=512,
        num_encoder_layers=4,
        nhead=4,
        num_specific_experts=6,
        top_k=2,
        num_shared_experts=2,
        num_classes=2,
        mask_ratio=0.3   # 🔥 reduced (less instability than 0.4)
    ).to(DEVICE)

    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ───── PRETRAIN ─────
    log("Stage 1: Self-supervised pretraining...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_pretrain,
        weight_decay=1e-4
    )

    for epoch in range(pretrain_epochs):
        loss, _ = train_one_epoch(model, train_loader, optimizer, mode='pretrain')
        log(f"Pretrain Epoch {epoch+1}/{pretrain_epochs} | Loss: {loss:.4f}")

    torch.save(model.state_dict(), 'results/pretrained_model.pt')

    # ───── FINETUNE ─────
    log("\nStage 2: Supervised fine-tuning...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_finetune,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, finetune_epochs
    )

    best_val = 0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(finetune_epochs):

        loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, mode='finetune'
        )

        val_acc, _, _ = evaluate(model, val_loader)
        scheduler.step()

        history['train_loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        log(f"Epoch {epoch+1}/{finetune_epochs} | "
            f"Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), 'results/best_model.pt')
            log(f"  ✅ Best model saved: {val_acc:.4f}")

    log(f"\nBest Val Accuracy: {best_val:.4f}")

    np.save('results/history.npy', history)

    return model, history
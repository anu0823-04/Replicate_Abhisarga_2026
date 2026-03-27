import argparse, sys
sys.path.insert(0, 'src')

import numpy as np
import torch
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataloader import get_dataloaders
from create_ground_truth import create_ground_truth
from train import train, evaluate
from utils import ensure_dir, log
from sklearn.metrics import classification_report


# -------------------------
# reproducibility helper
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    ensure_dir('results')

    set_seed(args.seed)

    log(f"Device: {DEVICE}")
    log("=" * 55)
    log("LandslideEEGMoE — Domain-Decoupled MoE for Landslides")
    log("=" * 55)

    # -------------------------
    # Ground truth generation
    # -------------------------
    print("⚡ Generating ground truth...")
    create_ground_truth(base_dir=args.base_dir)

    # -------------------------
    # Data loading
    # -------------------------
    train_loader, val_loader, test_loader, num_ch, num_t, bands = get_dataloaders(
        base_dir=args.base_dir,
        date_folders=['2024-12-11', '2024-12-16'],
        target_h=128,
        target_w=128,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size
    )

    log(f"Channels:{num_ch} | Time steps:{num_t} | Bands:{bands}")

    # -------------------------
    # TRAIN (IMPORTANT FIX HERE)
    # DO NOT PASS device (your train.py doesn't support it)
    # -------------------------
    model, history = train(
        train_loader,
        val_loader,
        num_channels=num_ch,
        patch_size=args.patch_size,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs
    )

    # -------------------------
    # Load best model
    # -------------------------
    model.load_state_dict(
        torch.load('results/best_model.pt', map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    # -------------------------
    # Evaluation
    # -------------------------
    test_acc, preds, labels = evaluate(model, test_loader)

    log(f"\n{'='*40}")
    log(f"🎯 TEST ACCURACY: {test_acc*100:.2f}%")
    log(f"{'='*40}")

    report = classification_report(
        labels,
        preds,
        target_names=['No Landslide', 'Landslide'],
        digits=4
    )

    print(report)

    # -------------------------
    # plots
    # -------------------------
    plt.figure()
    plt.plot(history['train_loss'])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig('results/loss.png', dpi=150)

    plt.figure()
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.legend()
    plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig('results/accuracy.png', dpi=150)

    log("✅ Saved plots")

    # -------------------------
    # summary
    # -------------------------
    with open('results/summary.txt', 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Bands: {bands}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Pretrain epochs: {args.pretrain_epochs}\n")
        f.write(f"Finetune epochs: {args.finetune_epochs}\n\n")
        f.write(report)

    log("✅ Saved summary")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', default='data/Wayanad_validation_data')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pretrain_epochs', type=int, default=5)
    parser.add_argument('--finetune_epochs', type=int, default=35)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)
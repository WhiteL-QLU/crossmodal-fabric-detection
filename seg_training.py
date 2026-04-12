import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.seg_dataset import get_seg_data_loader
from models.seg_models import CrossModalDBGSeg


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def compute_total_loss(logits, gt_mask, bce_loss):
    return bce_loss(logits, gt_mask) + dice_loss(logits, gt_mask)


def evaluate(model, data_loader, device, bce_loss):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for (rgb, freq_img), gt_mask, _, _ in data_loader:
            rgb = rgb.to(device)
            freq_img = freq_img.to(device)
            gt_mask = gt_mask.to(device)

            logits = model(rgb, freq_img)
            loss = compute_total_loss(logits, gt_mask, bce_loss)

            batch_size = rgb.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / max(total_samples, 1)


def train_segmentation(args):
    set_deterministic(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_seg_data_loader(
        split="train",
        class_name=args.class_name,
        dataset_path=args.dataset_path,
        img_size=224,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = get_seg_data_loader(
        split="validation",
        class_name=args.class_name,
        dataset_path=args.dataset_path,
        img_size=224,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    model = CrossModalDBGSeg(image_size=224).to(device)
    model.feature_extractor.device = device

    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
    )
    bce_loss = nn.BCEWithLogitsLoss()

    checkpoint_dir = os.path.join(args.checkpoint_savepath, args.class_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"CM_DBG_Seg_{args.class_name}_{args.epochs_no}ep_best.pth",
    )

    best_val_loss = float("inf")

    for epoch in range(args.epochs_no):
        model.train()
        model.feature_extractor.deep_feature_extractor.eval()

        running_loss = 0.0
        seen_samples = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Training {args.class_name} | epoch {epoch + 1}/{args.epochs_no}",
            leave=False,
        )

        for (rgb, freq_img), gt_mask, _, _ in progress_bar:
            rgb = rgb.to(device)
            freq_img = freq_img.to(device)
            gt_mask = gt_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(rgb, freq_img)
            loss = compute_total_loss(logits, gt_mask, bce_loss)
            loss.backward()
            optimizer.step()

            batch_size = rgb.shape[0]
            running_loss += loss.item() * batch_size
            seen_samples += batch_size

            progress_bar.set_postfix(loss=f"{loss.item():.5f}")

        train_loss = running_loss / max(seen_samples, 1)
        val_loss = evaluate(model, val_loader, device, bce_loss)

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs_no:03d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "model_state_dict": model.state_dict(),
                    "class_name": args.class_name,
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint to: {checkpoint_path}")

    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CM-DBG-Seg on AITEX")
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--checkpoint_savepath", required=True, type=str)
    parser.add_argument("--class_name", required=True, type=str)
    parser.add_argument("--epochs_no", default=50, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--seed", default=3407, type=int)
    train_segmentation(parser.parse_args())

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


@dataclass
class EmbeddingConfig:
    manifest_path: Path = Path("data/processed/manifest.csv")
    out_dir: Path = Path("data/processed/features_raw")
    batch_size: int = 32
    num_workers: int = 0


class ManifestDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, split: str):
        df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)

        if "path_resized" in df.columns and df["path_resized"].notna().all():
            self.paths = df["path_resized"].tolist()
        else:
            self.paths = df["path"].tolist()

        self.labels = df["label_id"].astype(int).tolist()

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        x = self.tf(img)
        y = int(self.labels[idx])
        return x, y


@torch.no_grad()
def extract_embeddings(cfg: EmbeddingConfig) -> None:
    if not cfg.manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {cfg.manifest_path.resolve()}")

    df = pd.read_csv(cfg.manifest_path)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()  # 512-d
    model.eval().to(device)

    for split in ["train", "val", "test"]:
        ds = ManifestDataset(df, split=split)
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

        embs, ys = [], []
        for x, y in dl:
            x = x.to(device)
            z = model(x)  # (B, 512)
            embs.append(z.cpu().numpy())
            ys.append(y.numpy())

        X = np.concatenate(embs, axis=0).astype(np.float32)
        Y = np.concatenate(ys, axis=0).astype(np.int64)

        np.save(cfg.out_dir / f"X_{split}.npy", X)
        np.save(cfg.out_dir / f"y_{split}.npy", Y)
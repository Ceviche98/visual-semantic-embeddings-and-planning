"""
CIFAR-100 Visual Feature Extractor and Cross-Modal Contrastive Projection Head.

Implements:
1. CIFAR100Filtered: Torchvision dataset wrapper resizing 32x32 images to 224x224 and normalizing
   with ImageNet statistics for MobileNetV3-Small calibration.
2. ImageEncoder: Frozen MobileNetV3-Small feature extractor (576-D pooled representation) coupled
   with a trainable two-layer MLP projection head:
   Linear(576 -> 512) -> BatchNorm1d(512) -> ReLU -> Linear(512 -> 64).
3. Symmetric InfoNCE Loss: CLIP-style bidirectional image-to-text and text-to-image cross-entropy
   loss over L2-normalized embeddings scaled by temperature τ.
4. Constrained Cross-Modal Retrieval & PDDL Sanitization: Restricts cosine similarity search to
   the 100 valid CIFAR-100 class prototypes in the 523-word Skip-Gram space and maps compound
   labels (e.g., 'maple_tree' -> 'maple', 'aquarium_fish' -> 'aquarium-fish') to PDDL identifiers.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

from .skipgram import load_trained_model_modified

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECTION_PATH = REPO_ROOT / "models" / "best_cifar100_projection.pth"
DEFAULT_SKIPGRAM_PATH = REPO_ROOT / "models" / "best_skipgram_523words.pth"

VALID_CIFAR_VOCAB: Set[str] = {
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
}

CIFAR_TO_PDDL: Dict[str, str] = {
    "maple_tree": "maple",
    "oak_tree": "oak",
    "palm_tree": "palm",
    "pine_tree": "pine",
    "willow_tree": "willow",
    "aquarium_fish": "aquarium-fish",
    "lawn_mower": "lawn-mower",
    "pickup_truck": "pickup-truck",
    "sweet_pepper": "sweet-pepper",
}


def clean_object_name(name: str) -> str:
    """Convert CIFAR-100 class names into PDDL-compatible object identifiers."""
    if name in CIFAR_TO_PDDL:
        return CIFAR_TO_PDDL[name]
    return name.replace("_", "-")


def get_cifar100_embeddings(
    all_nodes: List[str],
    all_embeddings: np.ndarray,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, List[str]]:
    """Filter the 523-word vocabulary to retain strictly the 100 valid CIFAR-100 class embeddings."""
    valid_indices: List[int] = []
    valid_words: List[str] = []
    for i, word in enumerate(all_nodes):
        if word in VALID_CIFAR_VOCAB:
            valid_indices.append(i)
            valid_words.append(word)
    valid_emb_tensor = torch.tensor(all_embeddings[valid_indices], dtype=torch.float32, device=device)
    return valid_emb_tensor, valid_words


class CIFAR100Filtered(Dataset):
    """
    CIFAR-100 dataset wrapper with 224x224 resizing and ImageNet normalization.
    Downloads CIFAR-100 automatically via torchvision when needed.
    """

    def __init__(self, root: str = "./data", split: str = "train", transform=None):
        assert split in ("train", "val"), "split must be 'train' or 'val'."
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        self.dataset = datasets.CIFAR100(
            root=root,
            train=(split == "train"),
            download=True,
            transform=transform,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]


class ImageEncoder(nn.Module):
    """
    MobileNetV3-Small visual encoder with frozen backbone (576-D) and trainable
    cross-modal MLP projection head mapping visual features into the 64-D Skip-Gram space.
    """

    def __init__(
        self,
        proj_dim: int = 64,
        device: Union[str, torch.device] = "cpu",
        pretrained: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_small(weights=weights)
        except Exception:
            base = models.mobilenet_v3_small(weights=None)

        self.backbone = nn.Sequential(*list(base.children())[:-1]).to(self.device).eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(576, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (backbone_features [B, 576], projected_embeddings [B, proj_dim]).
        """
        with torch.no_grad():
            feats = self.backbone(x).flatten(1)
        out = self.projection(feats)
        return feats, out


def compute_contrastive_loss(
    visual_proj: torch.Tensor,
    text_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Compute symmetric InfoNCE (CLIP-style contrastive) loss for vision-language alignment:
    L = 0.5 * (CE(V @ S^T / τ, y) + CE(S @ V^T / τ, y))
    """
    V = F.normalize(visual_proj, p=2, dim=1)
    S = F.normalize(text_emb, p=2, dim=1)
    logits = torch.matmul(V, S.T) / temperature
    labels = torch.arange(logits.shape[0], device=visual_proj.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0


def filter_dataset_indices(dataset: CIFAR100Filtered, valid_labels: Set[int]) -> List[int]:
    """Return indices of samples whose label is in `valid_labels`."""
    return [i for i, label in enumerate(dataset.dataset.targets) if label in valid_labels]


def create_data_splits(indices: List[int], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
    """Split dataset indices into deterministic train and validation subsets."""
    rng = np.random.default_rng(seed)
    arr = np.array(indices)
    rng.shuffle(arr)
    split_idx = int((1.0 - val_ratio) * len(arr))
    return arr[:split_idx].tolist(), arr[split_idx:].tolist()


def run_epoch(
    model: ImageEncoder,
    dataloader: DataLoader,
    text_emb: torch.Tensor,
    class_words: List[str],
    label_to_word: Dict[int, str],
    optimizer: Optional[torch.optim.Optimizer],
    temperature: float,
    device: Union[str, torch.device],
    mode: str = "train",
) -> Tuple[float, float]:
    """Run one training or evaluation epoch for contrastive vision-language alignment."""
    if mode == "train":
        model.train()
        model.backbone.eval()
    else:
        model.eval()

    total_loss, total_sim, count = 0.0, 0.0, 0
    ctx = torch.enable_grad() if mode == "train" else torch.no_grad()
    word_to_class_idx = {w: i for i, w in enumerate(class_words)}

    with ctx:
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            _, visual_proj = model(images)

            batch_text_idx = [word_to_class_idx[label_to_word[int(l.item())]] for l in labels]
            batch_text_emb = text_emb[batch_text_idx]

            loss = compute_contrastive_loss(visual_proj, batch_text_emb, temperature)
            if mode == "train" and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(images)
            V = F.normalize(visual_proj, p=2, dim=1)
            S = F.normalize(batch_text_emb, p=2, dim=1)
            total_sim += (V * S).sum(dim=1).sum().item()
            count += len(images)

    return total_loss / max(1, count), total_sim / max(1, count)


def train_with_early_stopping(
    model: ImageEncoder,
    dataloaders: Dict[str, DataLoader],
    text_emb: torch.Tensor,
    class_words: List[str],
    label_to_word: Dict[int, str],
    config: Dict,
    device: Union[str, torch.device],
) -> Tuple[Dict[str, List[float]], int, float, float]:
    """Train projection head with CosineAnnealingLR and validation similarity early stopping."""
    optimizer = torch.optim.AdamW(
        model.projection.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_val_loss = float("inf")
    best_val_sim = -float("inf")
    patience_counter, best_epoch = 0, 0
    history: Dict[str, List[float]] = defaultdict(list)

    for epoch in range(1, config["epochs"] + 1):
        train_loss, _ = run_epoch(
            model, dataloaders["train"], text_emb, class_words, label_to_word,
            optimizer, config["temperature"], device, "train",
        )
        val_loss, val_sim = run_epoch(
            model, dataloaders["val"], text_emb, class_words, label_to_word,
            None, config["temperature"], device, "eval",
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_similarity"].append(val_sim)
        history["learning_rate"].append(current_lr)

        if val_sim > best_val_sim:
            best_val_sim, best_val_loss, best_epoch, patience_counter = val_sim, val_loss, epoch, 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_similarity": val_sim,
                    "class_words": class_words,
                    "text_embeddings": text_emb.cpu(),
                    "history": dict(history),
                    "projection_head": model.projection.state_dict(),
                },
                config["save_path"],
            )
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                break

    return dict(history), best_epoch, best_val_sim, best_val_loss


def collect_embeddings(
    model: ImageEncoder,
    dataloader: DataLoader,
    device: Union[str, torch.device],
) -> Tuple[np.ndarray, List[int]]:
    """Collect L2-normalized visual projection embeddings and ground-truth labels across a DataLoader."""
    model.eval()
    all_visual, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            _, visual_proj = model(images)
            all_visual.append(F.normalize(visual_proj, p=2, dim=1).cpu())
            all_labels.extend(labels.tolist())
    return torch.cat(all_visual, dim=0).numpy(), all_labels


def compute_alignment_metrics(
    visual_emb: np.ndarray,
    labels: List[int],
    text_emb: np.ndarray,
    class_words: List[str],
    label_to_word: Dict[int, str],
) -> Tuple[List[Dict], Dict[int, int], Dict[int, int], np.ndarray]:
    """Compute per-class cosine similarity statistics and bidirectional Recall@K (K=1, 5, 10)."""
    class_sims: Dict[str, List[float]] = defaultdict(list)
    word_to_idx = {w: i for i, w in enumerate(class_words)}

    for i, label in enumerate(labels):
        word = label_to_word[label]
        if word in word_to_idx:
            sim = float(np.dot(visual_emb[i], text_emb[word_to_idx[word]]))
            class_sims[word].append(sim)

    stats = sorted(
        [
            {
                "word": word,
                "mean": float(np.mean(sims)),
                "std": float(np.std(sims)),
                "min": float(np.min(sims)),
                "max": float(np.max(sims)),
                "count": len(sims),
            }
            for word, sims in class_sims.items()
        ],
        key=lambda x: x["mean"],
        reverse=True,
    )

    sim_matrix = cosine_similarity(visual_emb, text_emb)
    i2t_recalls = {k: 0 for k in (1, 5, 10)}
    t2i_recalls = {k: 0 for k in (1, 5, 10)}

    for i, label in enumerate(labels):
        word = label_to_word[label]
        if word in word_to_idx:
            correct_idx = word_to_idx[word]
            ranking = np.argsort(-sim_matrix[i])
            for k in i2t_recalls:
                if correct_idx in ranking[:k]:
                    i2t_recalls[k] += 1

    for class_idx, word in enumerate(class_words):
        class_img_idx = [i for i, l in enumerate(labels) if label_to_word[l] == word]
        if class_img_idx:
            ranking = np.argsort(-sim_matrix[:, class_idx])
            for k in t2i_recalls:
                if any(idx in ranking[:k] for idx in class_img_idx):
                    t2i_recalls[k] += 1

    return stats, i2t_recalls, t2i_recalls, sim_matrix


def _resolve_projection_path(projection_path: str) -> str:
    path = Path(projection_path)
    if path.exists():
        return str(path)
    candidate = REPO_ROOT / "models" / path.name
    if candidate.exists():
        return str(candidate)
    return str(DEFAULT_PROJECTION_PATH)


_PROJECTION_CACHE: Dict[Tuple[str, int, str], Tuple[ImageEncoder, Dict]] = {}


def load_visual_projection_model(
    projection_path: str = "best_cifar100_projection.pth",
    proj_dim: int = 64,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[ImageEncoder, Dict]:
    """Load the trained MobileNetV3-Small + MLP projection head checkpoint."""
    resolved_path = _resolve_projection_path(projection_path)
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = (resolved_path, proj_dim, str(dev))
    if cache_key in _PROJECTION_CACHE:
        return _PROJECTION_CACHE[cache_key]

    checkpoint = torch.load(resolved_path, map_location=dev, weights_only=False)

    # Backbone state_dict is stored inside best_cifar100_projection.pth, so pretrained download is unnecessary
    vision_model = ImageEncoder(proj_dim=proj_dim, device=dev, pretrained=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        vision_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        vision_model.load_state_dict(checkpoint)
    vision_model.eval()
    res = (vision_model, (checkpoint if isinstance(checkpoint, dict) else {}))
    _PROJECTION_CACHE[cache_key] = res
    return res


def classify_image_to_pddl_object(
    image_tensor: torch.Tensor,
    skipgram_path: str = "best_skipgram_523words.pth",
    projection_path: str = "best_cifar100_projection.pth",
    top_k: int = 5,
) -> Dict[str, Union[str, float, List[Tuple[str, float]]]]:
    """
    Project a (3, 224, 224) or (1, 3, 224, 224) CIFAR-100 image tensor into the 64-D semantic
    embedding space and retrieve the highest-similarity CIFAR-100 prototype and PDDL identifier.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, skipgram_embeddings, nodes = load_trained_model_modified(
        model_path=skipgram_path,
        vocab_size=523,
        embedding_dim=64,
        dropout=0.3,
    )
    vision_model, _ = load_visual_projection_model(
        projection_path=projection_path,
        proj_dim=64,
        device=device,
    )

    with torch.no_grad():
        img = image_tensor.to(device)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        _, visual_proj = vision_model(img)
        visual_norm = F.normalize(visual_proj, p=2, dim=1)

        cifar_embs, cifar_words = get_cifar100_embeddings(nodes, skipgram_embeddings, device)
        text_norm = F.normalize(cifar_embs, p=2, dim=1)
        similarity = torch.matmul(visual_norm, text_norm.T)[0]

        sorted_indices = torch.argsort(similarity, descending=True)[:top_k].cpu().tolist()
        top_matches = [(cifar_words[idx], float(similarity[idx].item())) for idx in sorted_indices]
        best_raw = top_matches[0][0]
        best_score = top_matches[0][1]
        pddl_name = clean_object_name(best_raw)

    return {
        "raw_class": best_raw,
        "pddl_object": pddl_name,
        "confidence": best_score,
        "top_matches": top_matches,
    }

"""
Skip-Gram with Negative Sampling (SGNS), Text Co-Occurrence Graph Construction,
Semantic Negative Sampling, and (1+λ) Evolution Strategy for Vocabulary Expansion.

Implements:
1. Text Network Construction: Tokenization, rare-word filtering, induced subgraph
   extraction, and inverted co-occurrence distance matrices (523-word Visual Genome + CIFAR-100 vocabulary).
2. Hub-Penalized Positive Sampling: Normalizes edge co-occurrence counts by the
   geometric mean of endpoint node degrees to prevent high-degree prepositions/conjunctions
   from collapsing semantic clusters.
3. Semantic Negative Sampling: Weights negative sample probabilities proportional to
   (1 - cos_sim) from a semantic teacher while masking synonyms above threshold τ = 0.40.
4. SkipGramModel: 64-dimensional dual-embedding architecture with manual label smoothing (α = 0.1),
   dropout regularization (p = 0.3), and AdamW optimization.
5. (1+λ) Evolution Strategy: Three-term fitness optimization (corpus likelihood, norm matching,
   and semantic anchor cosine similarity) for incremental node insertion.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SKIPGRAM_PATH = REPO_ROOT / "models" / "best_skipgram_523words.pth"

# Full 523-word vocabulary combining Visual Genome core concepts and all 100 CIFAR-100 classes
VOCAB_523: List[str] = [
    'a', 'above', 'against', 'air', 'airplane', 'along', 'an', 'and', 'animal', 'apple',
    'aquarium_fish', 'are', 'area', 'arm', 'around', 'at', 'attached', 'baby', 'back', 'background',
    'backpack', 'bag', 'ball', 'banana', 'bananas', 'base', 'baseball', 'basket', 'bat', 'bathroom',
    'beach', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'behind', 'beige', 'being', 'bench',
    'beside', 'between', 'bicycle', 'big', 'bike', 'bird', 'black', 'blanket', 'blonde', 'blue',
    'board', 'boat', 'body', 'book', 'bottle', 'bottom', 'bowl', 'box', 'boy', 'branch',
    'branches', 'bread', 'brick', 'bridge', 'bright', 'broccoli', 'brown', 'building', 'buildings', 'bunch',
    'bus', 'bush', 'bushes', 'butterfly', 'button', 'by', 'cabinet', 'cake', 'camel', 'camera',
    'can', 'cap', 'car', 'carrying', 'cars', 'castle', 'cat', 'caterpillar', 'cattle', 'ceiling',
    'cell', 'cement', 'chain', 'chair', 'cheese', 'child', 'chimpanzee', 'city', 'clear', 'clock',
    'closed', 'cloth', 'cloud', 'clouds', 'cloudy', 'coat', 'cockroach', 'coffee', 'collar', 'color',
    'colored', 'colorful', 'computer', 'concrete', 'container', 'corner', 'couch', 'counter', 'court', 'cover',
    'covered', 'cow', 'cows', 'crab', 'crocodile', 'cup', 'curtain', 'dark', 'design', 'desk',
    'dinosaur', 'dirt', 'dish', 'display', 'distance', 'dog', 'dolphin', 'donut', 'door', 'double',
    'down', 'dress', 'ear', 'ears', 'eating', 'edge', 'elephant', 'elephants', 'empty', 'end',
    'engine', 'eye', 'eyes', 'face', 'feet', 'fence', 'field', 'fire', 'flag', 'flatfish',
    'floor', 'flower', 'flowers', 'flying', 'food', 'foot', 'for', 'forest', 'fork', 'four',
    'fox', 'frame', 'frisbee', 'from', 'front', 'fruit', 'full', 'fur', 'game', 'giraffe',
    'giraffes', 'girl', 'glass', 'glasses', 'glove', 'gold', 'grass', 'gray', 'green', 'grey',
    'ground', 'group', 'growing', 'guy', 'hair', 'hamster', 'hand', 'handle', 'hands', 'hanging',
    'has', 'hat', 'have', 'he', 'head', 'helmet', 'her', 'hill', 'his', 'holding',
    'horse', 'horses', 'hot', 'house', 'hydrant', 'in', 'inside', 'into', 'is', 'it',
    'jacket', 'jeans', 'jet', 'kangaroo', 'keyboard', 'kitchen', 'kite', 'knife', 'lady', 'lamp',
    'laptop', 'large', 'lawn_mower', 'laying', 'leaf', 'leaves', 'left', 'leg', 'legs', 'leopard',
    'letter', 'lettering', 'letters', 'license', 'lid', 'light', 'lights', 'line', 'lines', 'lion',
    'little', 'lizard', 'lobster', 'logo', 'long', 'looking', 'lot', 'luggage', 'made', 'man',
    "man's", 'many', 'maple_tree', 'men', 'metal', 'middle', 'mirror', 'monitor', 'motorcycle', 'mountain',
    'mountains', 'mouse', 'mouth', 'mushroom', 'napkin', 'near', 'neck', 'next', 'no', 'nose',
    'number', 'numbers', 'oak_tree', 'ocean', 'of', 'off', 'old', 'on', 'one', 'open',
    'orange', 'orchid', 'other', 'otter', 'out', 'outside', 'oven', 'over', 'painted', 'pair',
    'palm_tree', 'pants', 'paper', 'park', 'parked', 'parking', 'part', 'passenger', 'patch', 'pavement',
    'pear', 'people', 'person', "person's", 'phone', 'photo', 'pickup_truck', 'picture', 'piece', 'pile',
    'pillow', 'pine_tree', 'pink', 'pizza', 'plain', 'plane', 'plant', 'plastic', 'plate', 'platform',
    'player', 'playing', 'pole', 'poles', 'poppy', 'porcupine', 'possum', 'post', 'pot', 'purple',
    'rabbit', 'raccoon', 'rack', 'racket', 'rail', 'railing', 'ray', 'rear', 'red', 'reflection',
    'remote', 'riding', 'right', 'road', 'rock', 'rocket', 'rocks', 'roof', 'room', 'rose',
    'round', 'row', 'sand', 'sandwich', 'sauce', 'scene', 'screen', 'sea', 'seal', 'seat',
    'section', 'set', 'several', 'shadow', 'shark', 'sheep', 'shelf', 'shirt', 'shoe', 'shoes',
    'short', 'shorts', 'shrew', 'side', 'sidewalk', 'sign', 'signs', 'silver', 'sink', 'sitting',
    'skateboard', 'ski', 'skier', 'skiing', 'skis', 'skunk', 'sky', 'skyscraper', 'sleeve', 'slice',
    'small', 'smiling', 'snail', 'snake', 'snow', 'some', 'spider', 'spoon', 'spot', 'square',
    'squirrel', 'stand', 'standing', 'statue', 'stone', 'stop', 'stove', 'street', 'streetcar', 'stripe',
    'striped', 'stripes', 'stuffed', 'suit', 'suitcase', 'sun', 'sunflower', 'sunglasses', 'surface', 'surfboard',
    'surfer', 'sweater', 'sweet_pepper', 't', 'table', 'tag', 'tail', 'taken', 'tall', 'tan',
    'tank', 'teddy', 'telephone', 'television', 'tennis', 'that', 'the', 'there', 'these', 'this',
    'three', 'through', 'tie', 'tiger', 'tile', 'tire', 'to', 'toilet', 'top', 'towel',
    'tower', 'track', 'tracks', 'tractor', 'traffic', 'train', 'trash', 'tray', 'tree', 'trees',
    'trout', 'truck', 'trunk', 'tulip', 'turtle', 'tv', 'two', 'umbrella', 'under', 'up',
    'van', 'vase', 'vehicle', 'very', 'view', 'visible', 'walking', 'wall', 'wardrobe', 'watch',
    'watching', 'water', 'wave', 'waves', 'wearing', 'wears', 'wet', 'whale', 'wheel', 'wheels',
    'white', 'willow_tree', 'window', 'windows', 'windshield', 'wine', 'wing', 'wire', 'with', 'wolf',
    'woman', "woman's", 'wood', 'wooden', 'word', 'worm', 'worn', 'wrist', 'writing', 'yellow',
    'young', 'zebra', 'zebras'
]


# =============================================================================
# TEXT CO-OCCURRENCE NETWORK UTILITIES
# =============================================================================

def load_from_source(source: str) -> bytes:
    """Load data from file path or URL."""
    parsed = urllib.parse.urlparse(str(source))
    if parsed.scheme in ("http", "https"):
        response = requests.get(str(source), timeout=30)
        response.raise_for_status()
        return response.content
    with open(source, "rb") as f:
        return f.read()


def tokenize_text(text: str, char_like: Optional[List[str]] = None) -> List[str]:
    """
    Tokenize text into individual words and punctuation marks (',' and '.').
    Characters in `char_like` (e.g., apostrophes or underscores) are treated as alphabetic.
    """
    if char_like is None:
        char_like = ["'"]
    text = text.lower()
    tokens: List[str] = []
    current_word: List[str] = []
    punctuation = {',', '.'}

    for char in text:
        if char.isalpha() or char in char_like:
            current_word.append(char)
        elif char in punctuation:
            if current_word:
                tokens.append("".join(current_word))
                current_word = []
            tokens.append(char)
        else:
            if current_word:
                tokens.append("".join(current_word))
                current_word = []

    if current_word:
        tokens.append("".join(current_word))
    return tokens


def replace_rare_tokens(
    tokens: List[str],
    rare_threshold: float = 0.01,
    rare_token: str = "<RARE>",
) -> Tuple[List[str], Set[str], Counter]:
    """Replace infrequent words below `rare_threshold` with `rare_token`, preserving punctuation."""
    punctuation = {',', '.'}
    word_tokens = [t for t in tokens if t not in punctuation]
    total_words = len(word_tokens)
    if total_words == 0:
        return tokens[:], set(), Counter(tokens)

    word_counts = Counter(word_tokens)
    rare_set = {word for word, count in word_counts.items() if count / total_words < rare_threshold}
    new_tokens = [rare_token if word in rare_set else word for word in tokens]
    final_counts = Counter(new_tokens)
    return new_tokens, rare_set, final_counts


def get_text_adjacencies(tokens: List[str]) -> Counter:
    """Count consecutive token bigrams (excluding self-loops)."""
    adjacency_counts: Counter = Counter()
    for i in range(len(tokens) - 1):
        u, v = tokens[i], tokens[i + 1]
        if u != v:
            adjacency_counts[(u, v)] += 1
    return adjacency_counts


def build_unweighted_graph(nodes: List[str], adjacency_counts: Counter) -> nx.Graph:
    """Build an undirected, unweighted NetworkX graph from adjacency counts."""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    node_set = set(nodes)
    for (a, b), _ in adjacency_counts.items():
        if a in node_set and b in node_set and not G.has_edge(a, b):
            G.add_edge(a, b)
    return G


def compute_distance_matrix(
    nodes: List[str],
    adjacency_counts: Counter,
    distance_mode: str = "inverted",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a symmetric distance matrix from directional adjacency counts.
    In 'inverted' mode: distance = (max_count + 1) - symmetrized_count, with zero diagonal.
    """
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    count_matrix = np.zeros((n, n), dtype=float)

    for (a, b), count in adjacency_counts.items():
        if a in node_index and b in node_index:
            count_matrix[node_index[a], node_index[b]] = count

    count_matrix = (count_matrix + count_matrix.T) / 2.0
    np.fill_diagonal(count_matrix, 0.0)

    if distance_mode == "direct":
        distance_matrix = count_matrix.copy()
    elif distance_mode == "inverted":
        max_count = np.max(count_matrix)
        if max_count > 0:
            distance_matrix = (max_count + 1.0) - count_matrix
        else:
            distance_matrix = np.ones_like(count_matrix)
    else:
        raise ValueError("distance_mode must be 'direct' or 'inverted'")

    np.fill_diagonal(distance_matrix, 0.0)
    return distance_matrix, count_matrix


def process_text_network(
    source: str,
    rare_threshold: float = 0.01,
    rare_token: str = "<RARE>",
    distance_mode: str = "inverted",
    verbose: bool = False,
) -> Dict:
    """Complete text network processing pipeline with rare-word thresholding."""
    content = load_from_source(source)
    text = content.decode("utf-8", errors="ignore")
    tokens = tokenize_text(text)
    filtered_tokens, rare_set, token_counts = replace_rare_tokens(tokens, rare_threshold, rare_token)
    nodes = sorted(token_counts.keys())
    adjacency_counts = get_text_adjacencies(filtered_tokens)
    graph = build_unweighted_graph(nodes, adjacency_counts)
    distance_matrix, count_matrix = compute_distance_matrix(nodes, adjacency_counts, distance_mode)

    if verbose:
        print(f"Processed {len(tokens):,} tokens -> {len(nodes):,} vocabulary nodes.")

    return {
        "graph": graph,
        "nodes": nodes,
        "adjacency_counts": adjacency_counts,
        "distance_matrix": distance_matrix,
        "count_matrix": count_matrix,
        "token_counts": token_counts,
        "rare_set": rare_set,
    }


def process_text_network_restricted(
    source: str,
    target_nodes: List[str],
    distance_mode: str = "inverted",
    verbose: bool = True,
) -> Dict:
    """
    Process text network restricted to a target vocabulary list, preserving underscores
    for compound CIFAR-100 classes (e.g., 'palm_tree', 'pickup_truck').
    """
    content = load_from_source(source)
    text = content.decode("utf-8", errors="ignore")
    tokens_raw = tokenize_text(text, char_like=["'", "_"])

    target_set = set(target_nodes)
    tokens: List[str] = []
    for token in tokens_raw:
        if token in target_set:
            tokens.append(token)
        elif "_" in token:
            tokens.extend([t for t in token.split("_") if t])
        else:
            tokens.append(token)

    full_adjacency_counts = get_text_adjacencies(tokens)
    adjacency_counts = Counter(
        {(u, v): count for (u, v), count in full_adjacency_counts.items() if u in target_set and v in target_set}
    )

    graph = build_unweighted_graph(target_nodes, adjacency_counts)
    distance_matrix, count_matrix = compute_distance_matrix(target_nodes, adjacency_counts, distance_mode)
    full_token_counts = Counter(tokens)
    token_counts = Counter({node: full_token_counts[node] for node in target_nodes})

    if verbose:
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    return {
        "graph": graph,
        "nodes": target_nodes,
        "adjacency_counts": adjacency_counts,
        "distance_matrix": distance_matrix,
        "count_matrix": count_matrix,
        "token_counts": token_counts,
        "original_tokens": tokens,
    }


def filter_punctuation_from_network(
    network_data: Dict,
    punctuation_tokens: Set[str] = {'.', ',', '<RARE>', "'"},
) -> Dict:
    """Remove punctuation tokens from network to prevent hub poisoning."""
    original_graph = network_data["graph"]
    original_nodes = network_data["nodes"]
    original_distance_matrix = network_data["distance_matrix"]

    filtered_nodes = [n for n in original_nodes if n not in punctuation_tokens]
    old_indices = [i for i, n in enumerate(original_nodes) if n not in punctuation_tokens]
    filtered_distance_matrix = original_distance_matrix[np.ix_(old_indices, old_indices)]

    filtered_graph = nx.Graph()
    filtered_graph.add_nodes_from(filtered_nodes)
    node_set = set(filtered_nodes)
    for u, v in original_graph.edges():
        if u in node_set and v in node_set:
            filtered_graph.add_edge(u, v)

    return {
        **network_data,
        "graph": filtered_graph,
        "nodes": filtered_nodes,
        "distance_matrix": filtered_distance_matrix,
    }


def compute_semantic_negative_matrix(
    nodes: List[str],
    model_name: str = "all-mpnet-base-v2",
    threshold: float = 0.40,
) -> np.ndarray:
    """
    Create an (N, N) probability distribution matrix for Semantic Negative Sampling.
    Pairs with semantic similarity > `threshold` receive zero probability; remaining pairs
    are weighted proportional to (1 - cosine_similarity). Falls back to uniform off-diagonal
    sampling if `sentence_transformers` is not installed.
    """
    n = len(nodes)
    try:
        from sentence_transformers import SentenceTransformer, util

        teacher = SentenceTransformer(model_name)
        clean_nodes = [w.replace("_", " ") for w in nodes]
        embeddings = teacher.encode(clean_nodes, convert_to_tensor=True)
        sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()

        mask = sim_matrix < threshold
        np.fill_diagonal(mask, False)
        raw_scores = (1.0 - sim_matrix) * mask.astype(np.float32)
        row_sums = raw_scores.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return (raw_scores / row_sums).astype(np.float32)
    except ImportError:
        probs = np.ones((n, n), dtype=np.float32)
        np.fill_diagonal(probs, 0.0)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs


# =============================================================================
# SKIP-GRAM DATASET & MODEL (SGNS)
# =============================================================================

class SkipGramDataset(Dataset):
    """
    Skip-Gram dataset for learning node embeddings from a co-occurrence graph structure
    with degree-penalized positive edge weights and semantic negative sampling.
    """

    def __init__(
        self,
        graph: nx.Graph,
        nodes: List[str],
        distance_matrix: np.ndarray,
        negative_probs_matrix: Optional[np.ndarray] = None,
        num_negative: int = 15,
        context_size: int = 1,
    ):
        super().__init__()
        self.graph = graph
        self.nodes = nodes
        self.node_to_idx = {node: i for i, node in enumerate(nodes)}
        self.vocab_size = len(nodes)
        self.num_negative = num_negative
        self.distance_matrix = distance_matrix

        if negative_probs_matrix is None:
            probs = np.ones((self.vocab_size, self.vocab_size), dtype=np.float32)
            np.fill_diagonal(probs, 0.0)
            row_sums = probs.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            negative_probs_matrix = probs / row_sums

        self.neg_probs_matrix = negative_probs_matrix.astype(np.float32)
        self.contexts = self._build_contexts(context_size)
        self.pairs, self.weights = self._generate_weighted_pairs()
        self._local_rng: Optional[np.random.Generator] = None

    def _build_contexts(self, context_size: int) -> Dict[str, Set[str]]:
        """Build context set for each node (neighbors within `context_size` hops)."""
        contexts: Dict[str, Set[str]] = {}
        for node in self.nodes:
            if node not in self.graph:
                contexts[node] = set()
                continue
            try:
                paths = nx.single_source_shortest_path_length(self.graph, node, cutoff=context_size)
                contexts[node] = {
                    neighbor for neighbor, dist in paths.items() if dist > 0 and neighbor in self.node_to_idx
                }
            except Exception:
                contexts[node] = set()
        return contexts

    def _generate_weighted_pairs(self) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Generate (center_idx, context_idx) pairs and compute degree-penalized importance weights:
        w(u, v) = ((max_D + 1) - D_{uv}) / sqrt(deg(u) * deg(v))
        """
        pairs: List[Tuple[int, int]] = []
        raw_distances: List[float] = []
        degree_dict = dict(self.graph.degree())
        max_val = float(self.distance_matrix.max()) if self.distance_matrix.size > 0 else 1.0

        for center_word, context_words in self.contexts.items():
            if center_word not in self.node_to_idx:
                continue
            center_idx = self.node_to_idx[center_word]
            center_deg = max(float(degree_dict.get(center_word, 1.0)), 1.0)
            for context_word in context_words:
                if context_word in self.node_to_idx:
                    context_idx = self.node_to_idx[context_word]
                    context_deg = max(float(degree_dict.get(context_word, 1.0)), 1.0)
                    pairs.append((center_idx, context_idx))

                    try:
                        d = float(self.distance_matrix[center_idx, context_idx])
                    except IndexError:
                        d = 10.0

                    count = max((max_val + 1.0) - d, 1.0)
                    denom = np.sqrt(center_deg * context_deg)
                    normalized_weight = count / (denom + 1e-5)
                    raw_distances.append(float(normalized_weight))

        if len(pairs) == 0:
            return [], np.array([], dtype=np.float32)

        weights = np.array(raw_distances, dtype=np.float32)
        clip_val = np.percentile(weights, 95) * 3.0
        weights = np.clip(weights, 0.0, clip_val)
        if weights.sum() == 0:
            weights = np.ones_like(weights)
        weights = weights / weights.max()
        return pairs, weights.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[np.int64, np.int64, np.ndarray]:
        if self._local_rng is None:
            worker_info = torch.utils.data.get_worker_info()
            seed = (torch.initial_seed() + worker_info.id) if worker_info else torch.initial_seed()
            self._local_rng = np.random.default_rng(seed)

        center_idx, context_idx = self.pairs[idx]
        current_probs = self.neg_probs_matrix[center_idx].copy()
        current_probs[center_idx] = 0.0
        current_probs[context_idx] = 0.0

        total = current_probs.sum()
        pool = np.nonzero(current_probs)[0]
        replace_flg = len(pool) < self.num_negative
        if total > 0:
            current_probs /= total
            negatives = self._local_rng.choice(
                self.vocab_size,
                size=self.num_negative,
                replace=replace_flg,
                p=current_probs,
            )
        else:
            negatives = self._local_rng.choice(
                self.vocab_size,
                size=self.num_negative,
                replace=(self.vocab_size < self.num_negative),
            )

        return np.int64(center_idx), np.int64(context_idx), negatives.astype(np.int64)

    def get_sample_weights(self) -> np.ndarray:
        return self.weights

    def __len__(self) -> int:
        return len(self.pairs)


class SkipGramModel(nn.Module):
    """
    Skip-Gram model with Negative Sampling (SGNS), dropout regularization, and label smoothing.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, dropout: float = 0.3):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_embeddings()

    def _init_embeddings(self):
        emb_dim = self.center_embeddings.embedding_dim
        scale = 0.5 / emb_dim
        nn.init.uniform_(self.center_embeddings.weight, -scale, scale)
        nn.init.uniform_(self.context_embeddings.weight, -scale, scale)

    def forward(
        self,
        center: torch.Tensor,
        context: torch.Tensor,
        negatives: torch.Tensor,
        apply_dropout: bool = True,
        label_smoothing: float = 0.1,
    ) -> torch.Tensor:
        """Compute per-sample Skip-Gram Negative Sampling loss with optional label smoothing."""
        center_emb = self.dropout(self.center_embeddings(center)) if apply_dropout else self.center_embeddings(center)
        context_emb = self.context_embeddings(context)
        negative_emb = self.context_embeddings(negatives)

        pos_score = torch.sum(center_emb * context_emb, dim=1)
        neg_score = torch.bmm(negative_emb, center_emb.unsqueeze(2)).squeeze(2)

        if label_smoothing > 0:
            pos_loss = -(
                (1.0 - label_smoothing) * F.logsigmoid(pos_score)
                + label_smoothing * F.logsigmoid(-pos_score)
            )
            neg_loss = -(
                (1.0 - label_smoothing) * F.logsigmoid(-neg_score)
                + label_smoothing * F.logsigmoid(neg_score)
            ).sum(dim=1)
        else:
            pos_loss = -F.logsigmoid(pos_score)
            neg_loss = -F.logsigmoid(-neg_score).sum(dim=1)

        return pos_loss + neg_loss

    def get_embeddings(self) -> np.ndarray:
        """Extract learned center embeddings as a float32 NumPy array."""
        return self.center_embeddings.weight.detach().cpu().numpy().astype(np.float32)


def train_embeddings(
    network_data: Dict,
    embedding_dim: int = 64,
    batch_size: int = 2048,
    epochs: int = 100,
    learning_rate: float = 0.1,
    num_negative: int = 8,
    validation_fraction: float = 0.1,
    context_size: int = 4,
    dropout: float = 0.3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    patience: int = 8,
    device: Optional[str] = None,
    save_path: str = "best_model.pth",
    save_plot: bool = False,
) -> Dict:
    """Train Skip-Gram embeddings with semantic negative sampling and degree-penalized positive weights."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    network_data = filter_punctuation_from_network(network_data)
    nodes = network_data["nodes"]
    graph = network_data["graph"]
    distance_matrix = network_data["distance_matrix"]

    all_edges = list(graph.edges())
    rng = np.random.default_rng(42)
    rng.shuffle(all_edges)
    split_idx = int(len(all_edges) * (1.0 - validation_fraction))

    train_graph = nx.Graph()
    train_graph.add_nodes_from(nodes)
    train_graph.add_edges_from(all_edges[:split_idx])

    val_graph = nx.Graph()
    val_graph.add_nodes_from(nodes)
    val_graph.add_edges_from(all_edges[split_idx:])

    sem_probs = compute_semantic_negative_matrix(nodes, threshold=0.40)
    train_dataset = SkipGramDataset(train_graph, nodes, distance_matrix, sem_probs, num_negative, context_size)
    val_dataset = SkipGramDataset(val_graph, nodes, distance_matrix, sem_probs, num_negative, context_size)

    if len(train_dataset) > 0:
        sampler = WeightedRandomSampler(
            weights=train_dataset.get_sample_weights(),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SkipGramModel(len(nodes), embedding_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        for centers, contexts, negs in train_loader:
            centers, contexts, negs = centers.to(device), contexts.to(device), negs.to(device)
            loss = model(centers, contexts, negs, True, label_smoothing).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            count += 1

        train_loss = total_loss / max(1, count)
        train_losses.append(train_loss)

        model.eval()
        total_val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for centers, contexts, negs in val_loader:
                centers, contexts, negs = centers.to(device), contexts.to(device), negs.to(device)
                batch_loss = model(centers, contexts, negs, False, 0.0).mean().item()
                total_val_loss += batch_loss
                val_count += 1

        val_loss = total_val_loss / max(1, val_count)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": best_model_state,
                    "nodes": nodes,
                    "vocab_size": len(nodes),
                    "embedding_dim": embedding_dim,
                },
                save_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    if save_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, "o-", label="Train", linewidth=2)
        plt.plot(val_losses, "s-", label="Validation", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("training_loss.png", dpi=150)
        plt.close()

    return {
        "nodes": nodes,
        "embeddings": model.get_embeddings(),
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


# =============================================================================
# (1+λ) EVOLUTION STRATEGY FOR INCREMENTAL EMBEDDING INSERTION
# =============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def compute_embedding_stats(embeddings: np.ndarray) -> Dict[str, float]:
    """Compute vector norm and standard deviation statistics for fitness evaluation."""
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "global_std": float(np.std(embeddings)),
    }


def extract_word_contexts(
    text_file: str,
    target_words: List[str],
    vocab_set: Set[str],
    window: int = 5,
) -> Dict[str, Counter]:
    """Extract co-occurrence context word frequencies within `window` around `target_words`."""
    contexts = {word: Counter() for word in target_words}
    target_set = set(target_words)
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            tokens = re.findall(r"\b[a-z]+\b", line.lower())
            for idx, token in enumerate(tokens):
                if token in target_set:
                    start = max(0, idx - window)
                    end = min(len(tokens), idx + window + 1)
                    valid_contexts = [
                        t for t in tokens[start:idx] + tokens[idx + 1 : end] if t in vocab_set
                    ]
                    contexts[token].update(valid_contexts)
    return contexts


def initialize_embedding(
    word: str,
    contexts: Dict[str, Counter],
    embeddings: np.ndarray,
    word_to_idx: Dict[str, int],
) -> np.ndarray:
    """Initialize a candidate word embedding using a frequency-weighted average of top-20 context vectors."""
    if not contexts.get(word):
        return np.mean(embeddings, axis=0)

    top_contexts = contexts[word].most_common(20)
    weight_sum = sum(count for _, count in top_contexts)
    vec_sum = np.zeros(embeddings.shape[1], dtype=np.float32)
    for w, count in top_contexts:
        if w in word_to_idx:
            vec_sum += (count / weight_sum) * embeddings[word_to_idx[w]]
    return vec_sum if np.linalg.norm(vec_sum) > 0 else np.mean(embeddings, axis=0)


def precompute_fitness_vectors(
    word: str,
    contexts: Dict[str, Counter],
    embeddings: np.ndarray,
    word_to_idx: Dict[str, int],
    vocab_list: List[str],
    anchors: Dict[str, List[str]],
    num_negatives: int = 15,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """Precompute context, negative, and normalized anchor vectors for fitness evaluation."""
    ctx_vecs, ctx_weights = None, None
    if contexts.get(word):
        vecs, weights = [], []
        for w, count in contexts[word].items():
            if w in word_to_idx:
                vecs.append(embeddings[word_to_idx[w]])
                weights.append(count)
        if vecs:
            ctx_vecs = np.array(vecs, dtype=np.float32)
            ctx_weights = np.array(weights, dtype=np.float32) / sum(weights)

    neg_words = np.random.choice(vocab_list, min(num_negatives, len(vocab_list)), replace=False)
    neg_vecs = np.array([embeddings[word_to_idx[n]] for n in neg_words], dtype=np.float32)

    anchor_vecs = None
    if word in anchors:
        anchor_words = [a for a in anchors[word] if a in word_to_idx]
        if anchor_words:
            anchor_vecs = np.array([embeddings[word_to_idx[a]] for a in anchor_words], dtype=np.float32)
            anchor_vecs = anchor_vecs / (np.linalg.norm(anchor_vecs, axis=1, keepdims=True) + 1e-10)

    return ctx_vecs, ctx_weights, neg_vecs, anchor_vecs


def compute_fitness(
    vec: np.ndarray,
    word: str,
    ctx_vecs: Optional[np.ndarray],
    ctx_weights: Optional[np.ndarray],
    neg_vecs: np.ndarray,
    anchor_vecs: Optional[np.ndarray],
    stats_dict: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """
    Compute three-term fitness score for a candidate word embedding:
    F(v) = w_corpus * L_corpus(v) + w_norm * S_norm(v) + w_anchor * S_anchor(v)
    """
    vec_norm = float(np.linalg.norm(vec))
    L_corpus = 0.0
    if ctx_vecs is not None and ctx_weights is not None:
        L_corpus += float(np.sum(ctx_weights * np.log(sigmoid(np.dot(ctx_vecs, vec)) + 1e-10)))

    L_corpus += float(np.sum(np.log(sigmoid(-np.dot(neg_vecs, vec)) + 1e-10)))
    total_samples = (float(np.sum(ctx_weights)) if (ctx_vecs is not None and ctx_weights is not None) else 0.0) + len(neg_vecs)
    L_corpus_norm = float(sigmoid(np.array(L_corpus / total_samples))) if total_samples > 0 else 0.5

    S_norm = float(np.exp(-((vec_norm - stats_dict["mean_norm"]) ** 2) / (2.0 * (stats_dict["std_norm"] ** 2) + 1e-10)))

    S_anchor = 0.5
    if anchor_vecs is not None:
        S_anchor = float(np.mean(np.dot(anchor_vecs, vec / (vec_norm + 1e-10))))

    return float(
        weights["corpus"] * L_corpus_norm
        + weights["norm"] * S_norm
        + weights["anchor"] * S_anchor
    )


def evolve_embedding(
    word: str,
    contexts: Dict[str, Counter],
    embeddings: np.ndarray,
    word_to_idx: Dict[str, int],
    vocab_list: List[str],
    stats_dict: Dict[str, float],
    anchors: Dict[str, List[str]],
    config: Dict,
) -> np.ndarray:
    """Optimize a single word embedding vector using a (1+λ) Evolution Strategy."""
    dim = embeddings.shape[1]
    mutation_sigma = config["ga_mutation_factor"] * stats_dict["global_std"]
    best_vec = initialize_embedding(word, contexts, embeddings, word_to_idx)
    ctx_vecs, ctx_weights, neg_vecs, anchor_vecs = precompute_fitness_vectors(
        word, contexts, embeddings, word_to_idx, vocab_list, anchors
    )

    for _ in range(config["ga_generations"]):
        population = best_vec + np.random.normal(0.0, mutation_sigma, (config["ga_pop_size"], dim))
        all_candidates = np.vstack([best_vec, population])
        fitness_scores = [
            compute_fitness(vec, word, ctx_vecs, ctx_weights, neg_vecs, anchor_vecs, stats_dict, config["fitness_weights"])
            for vec in all_candidates
        ]
        best_idx = int(np.argmax(fitness_scores))
        best_vec = all_candidates[best_idx].copy()

    return best_vec.astype(np.float32)


# =============================================================================
# MODEL LOADING, SIMILARITY SEARCH & ANALOGY SOLVING
# =============================================================================

def _resolve_checkpoint_path(checkpoint_path: str) -> str:
    path = Path(checkpoint_path)
    if path.exists():
        return str(path)
    candidate = REPO_ROOT / "models" / path.name
    if candidate.exists():
        return str(candidate)
    return str(DEFAULT_SKIPGRAM_PATH)


_SKIPGRAM_CACHE: Dict[Tuple[str, int, int], Tuple[SkipGramModel, np.ndarray, List[str]]] = {}


def load_trained_model_modified(
    model_path: str = "best_skipgram_523words.pth",
    vocab_size: int = 523,
    embedding_dim: int = 64,
    dropout: float = 0.3,
) -> Tuple[SkipGramModel, np.ndarray, List[str]]:
    """Load trained Skip-Gram checkpoint, returning (model, embeddings, nodes)."""
    resolved_path = _resolve_checkpoint_path(model_path)
    cache_key = (resolved_path, vocab_size, embedding_dim)
    if cache_key in _SKIPGRAM_CACHE:
        model, embeddings, nodes = _SKIPGRAM_CACHE[cache_key]
        return model, embeddings.copy(), list(nodes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(resolved_path, map_location=device, weights_only=False)
    if "nodes" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain 'nodes'. Keys found: {list(checkpoint.keys())}")
    nodes = list(checkpoint["nodes"])
    model = SkipGramModel(vocab_size=vocab_size, embedding_dim=embedding_dim, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        embeddings = model.get_embeddings()

    _SKIPGRAM_CACHE[cache_key] = (model, embeddings, nodes)
    return model, embeddings.copy(), list(nodes)


def build_my_embeddings(
    checkpoint_path: str = "best_skipgram_523words.pth",
) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Load and return the trained 523-word Skip-Gram embeddings (Visual Genome + 100 CIFAR-100 classes).

    Returns:
        vocab: Dictionary mapping words to indices {word: index}
        embeddings: Float32 numpy array of shape (523, 64)
    """
    vocab_size = 523
    embedding_dim = 64
    dropout = 0.3
    _, embeddings, nodes = load_trained_model_modified(
        model_path=checkpoint_path,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )
    vocab = {word: idx for idx, word in enumerate(nodes)}
    assert embeddings.shape[0] == len(vocab), "Vocab and embeddings size do not match"
    return vocab, embeddings


def find_similar_words(
    word: str,
    nodes: List[str],
    embeddings: np.ndarray,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """Find top-K most similar words using cosine similarity."""
    if word not in nodes:
        return []
    idx = nodes.index(word)
    target_vec = embeddings[idx]
    vec_norm = np.linalg.norm(target_vec)
    mat_norm = np.linalg.norm(embeddings, axis=1)
    similarities = (embeddings @ target_vec) / (mat_norm * vec_norm + 1e-9)
    top_indices = np.argsort(-similarities)[1 : top_k + 1]
    return [(nodes[i], float(similarities[i])) for i in top_indices]


def solve_analogy(
    word_a: str,
    word_b: str,
    word_c: str,
    nodes: List[str],
    embeddings: np.ndarray,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Solve vector offset word analogies: `word_a` is to `word_b` as `word_c` is to `?` (v_b - v_a + v_c)."""
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    if not all(w in node_to_idx for w in (word_a, word_b, word_c)):
        return []

    # Normalize before vector arithmetic for stable angular offset (3CosAdd)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    normed = embeddings / norms

    va = normed[node_to_idx[word_a]]
    vb = normed[node_to_idx[word_b]]
    vc = normed[node_to_idx[word_c]]
    target_vec = vb - va + vc
    target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-9)

    similarities = normed @ target_vec
    exclude = {node_to_idx[w] for w in (word_a, word_b, word_c)}

    top_indices: List[int] = []
    for idx in np.argsort(-similarities):
        if int(idx) not in exclude:
            top_indices.append(int(idx))
            if len(top_indices) >= top_k:
                break

    return [(nodes[i], float(similarities[i])) for i in top_indices]

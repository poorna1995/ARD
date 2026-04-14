"""
=============================================================================
  LLM Hidden-State Difficulty Estimator
  Based on: "The LLM Already Knows" + LSDF Extension
=============================================================================

  ARCHITECTURE OVERVIEW
  ─────────────────────
  ┌─────────────────────────────────────────────────────────────────┐
  │  Input Question                                                  │
  │       ↓                                                          │
  │  Tokenizer → Token IDs                                           │
  │       ↓                                                          │
  │  Frozen LLM Forward Pass  (NO decoding, zero tokens generated)   │
  │       ↓                                                          │
  │  Hidden States  h¹, h², ..., hᴸ   (all layers or last only)     │
  │       ↓                                                          │
  │  ┌──────────────┐      ┌──────────────────────────────────┐      │
  │  │  Basic Probe │  OR  │  LSDF  (per-layer probes + meta) │      │
  │  └──────────────┘      └──────────────────────────────────┘      │
  │       ↓                                                          │
  │  V(s₀) ∈ [0,1]  →  EASY  (≥ τ)  /  DIFFICULT  (< τ)           │
  └─────────────────────────────────────────────────────────────────┘

  COMPONENTS
  ──────────
  1. HiddenStateExtractor   – wraps any HuggingFace CausalLM, returns
                              per-layer hidden states in one forward pass
  2. ValueProbe             – 2-layer MLP, the core difficulty estimator
  3. LSDFProbe              – L parallel probes + softmax meta-aggregator
                              (Layer-Selective Difficulty Fingerprinting)
  4. TDLambdaTrainer        – trains probes with TD(λ) bootstrapping loss
  5. DifficultyEstimator    – unified inference API (easy/difficult + score)
  6. DifficultyDataset      – torch Dataset for (question, label) pairs
  7. demo()                 – end-to-end runnable example (no GPU needed)

  DEPENDENCIES
  ────────────
  pip install torch transformers datasets tqdm

  USAGE
  ─────
  # Quickstart
  python difficulty_estimator.py

  # In your own code
  from difficulty_estimator import DifficultyEstimator
  est = DifficultyEstimator.from_pretrained("gpt2")   # any HF causal LM
  result = est.predict("What is 2 + 2?")
  print(result)   # {'score': 0.91, 'label': 'EASY', 'fingerprint': [...]}
=============================================================================
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  HIDDEN STATE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class HiddenStateExtractor(nn.Module):
    """
    Wraps a frozen HuggingFace CausalLM and extracts hidden states from
    every transformer layer in a single forward pass (no token generation).

    The LLM weights are always frozen — only the attached probes are trained.

    Args:
        model_name_or_path : HuggingFace model id or local directory
        device             : "cuda", "cpu", or "auto"
        max_length         : token limit for input truncation
        pooling            : "mean" pools over all token positions;
                             "last" uses only the final token's hidden state
    """

    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        device: str = "auto",
        max_length: int = 512,
        pooling: str = "mean",
    ) -> None:
        super().__init__()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_length = max_length
        self.pooling = pooling

        # ── Tokenizer ────────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Model (frozen) ───────────────────────────────────────────────────
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            output_hidden_states=True,   # <-- key: expose all layer states
            torch_dtype=torch.float32,
        ).to(self.device)

        for param in self.llm.parameters():
            param.requires_grad = False
        self.llm.eval()

        # ── Dimension bookkeeping ────────────────────────────────────────────
        self.num_layers: int = self.llm.config.num_hidden_layers  # L
        self.hidden_dim: int = self.llm.config.hidden_size        # d_model

    # ── Core extraction ──────────────────────────────────────────────────────

    @torch.no_grad()
    def extract(
        self,
        texts: List[str],
        layer_indices: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Run one forward pass per batch and return pooled hidden states.

        Args:
            texts         : list of raw question strings
            layer_indices : which layers to return (None = all layers)
                            Layer 0 is the embedding layer output;
                            layers 1..L are transformer block outputs.

        Returns:
            dict mapping layer_index → tensor of shape [batch, hidden_dim]
        """
        if layer_indices is None:
            layer_indices = list(range(self.num_layers + 1))

        # ── Tokenize ─────────────────────────────────────────────────────────
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        attention_mask = enc["attention_mask"]  # [B, seq_len]

        # ── Forward pass ─────────────────────────────────────────────────────
        # hidden_states is a tuple of (L+1) tensors, each [B, seq_len, d_model]
        # index 0 = embedding output, index l = after transformer block l
        outputs = self.llm(**enc)
        all_hidden: Tuple[torch.Tensor, ...] = outputs.hidden_states

        # ── Pool and collect requested layers ────────────────────────────────
        result: Dict[int, torch.Tensor] = {}
        for idx in layer_indices:
            h = all_hidden[idx]                          # [B, seq_len, d]
            result[idx] = self._pool(h, attention_mask)  # [B, d]

        return result

    def _pool(
        self,
        hidden: torch.Tensor,        # [B, seq_len, d]
        mask: torch.Tensor,          # [B, seq_len]  — 1=real token, 0=pad
    ) -> torch.Tensor:               # [B, d]
        """Pool token dimension → single vector per example."""
        if self.pooling == "last":
            # Position of the last real (non-pad) token for each example
            lengths = mask.sum(dim=1) - 1          # [B]
            batch_size = hidden.size(0)
            last_idx = lengths.clamp(min=0)        # safety
            return hidden[torch.arange(batch_size), last_idx]  # [B, d]

        # mean pooling: average over non-pad positions
        mask_f = mask.unsqueeze(-1).float()        # [B, seq, 1]
        return (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  VALUE PROBE  (Basic — probes a single layer)
# ─────────────────────────────────────────────────────────────────────────────

class ValueProbe(nn.Module):
    """
    2-layer MLP that maps one hidden-state vector → V(s₀) ∈ [0,1].

    Architecture:
        Linear(d_model → hidden_dim) → LayerNorm → ReLU → Dropout
        → Linear(hidden_dim → 1) → Sigmoid

    LayerNorm stabilises training when hidden_dim is large (2048–8192).
    Dropout prevents the probe from memorising training set noise.

    Args:
        input_dim  : d_model of the LLM
        hidden_dim : width of the intermediate layer (default 256)
        dropout    : dropout rate (default 0.1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h : hidden state, shape [B, d_model]
        Returns:
            V : difficulty score, shape [B, 1]  ∈ [0,1]
        """
        return self.net(h)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LSDF PROBE  (Layer-Selective Difficulty Fingerprinting)
# ─────────────────────────────────────────────────────────────────────────────

class LSDFProbe(nn.Module):
    """
    L parallel ValueProbes (one per transformer layer) + a learned
    softmax attention meta-aggregator that selects which layers matter.

        V_final = Σ_l  α_l · v_l

    where  α = softmax(MLP_meta([v¹, v², ..., vᴸ]))

    The α weights form the **difficulty fingerprint** — a distribution over
    layers that reveals *what kind* of difficulty the question exhibits:

        Low layers  → syntactic / surface difficulty
        Mid layers  → factual knowledge retrieval difficulty
        High layers → multi-hop / abstract reasoning difficulty

    Args:
        num_layers   : L (number of transformer blocks, NOT including embed)
        input_dim    : d_model
        probe_hidden : hidden width of each per-layer probe
        meta_hidden  : hidden width of the meta-aggregator
        dropout      : applied in each probe
    """

    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        probe_hidden: int = 256,
        meta_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        # ── Per-layer probes (output a scalar each) ──────────────────────────
        self.probes = nn.ModuleList([
            ValueProbe(input_dim, probe_hidden, dropout)
            for _ in range(num_layers)
        ])

        # ── Meta-aggregator: takes the L scalar scores, outputs L weights ───
        # Input  : [B, L]   (concatenated per-layer scores)
        # Output : [B, L]   (softmax attention weights α)
        self.meta = nn.Sequential(
            nn.Linear(num_layers, meta_hidden),
            nn.ReLU(),
            nn.Linear(meta_hidden, num_layers),
        )

    def forward(
        self,
        layer_hiddens: Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            layer_hiddens : dict  layer_idx → [B, d_model]
                            Must contain keys 1..num_layers (skip embed layer 0)

        Returns:
            v_final     : [B, 1]  — aggregated difficulty score
            fingerprint : [B, L]  — α weights per layer (sums to 1 per example)
        """
        # Collect per-layer scores: list of [B, 1] → stack → [B, L]
        scores = torch.cat(
            [self.probes[l](layer_hiddens[l + 1]) for l in range(self.num_layers)],
            dim=1,
        )  # [B, L]

        # Attention weights
        alpha = F.softmax(self.meta(scores), dim=1)  # [B, L]

        # Weighted sum
        v_final = (alpha * scores).sum(dim=1, keepdim=True)  # [B, 1]

        return v_final, alpha


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DATASET
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QAPair:
    question: str
    is_correct: float   # 1.0 = model answered correctly, 0.0 = wrong


class DifficultyDataset(Dataset):
    """
    Wraps a list of QAPair objects.

    Each item returns  (question_str, correctness_label_float).
    Hidden-state extraction happens in the trainer (batched, on GPU).
    """

    def __init__(self, pairs: List[QAPair]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, float]:
        p = self.pairs[idx]
        return p.question, p.is_correct

    @staticmethod
    def collate(batch: List[Tuple[str, float]]):
        questions, labels = zip(*batch)
        return list(questions), torch.tensor(labels, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TD(λ) TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class TDLambdaTrainer:
    """
    Trains a ValueProbe or LSDFProbe with Temporal Difference (TD) loss.

    TD(λ) Loss
    ──────────
    For a sequence of hidden states {s₀, s₁, ..., sT} with terminal
    correctness reward r ∈ {0,1}, the TD error at step t is:

        δₜ = r(sₜ) + γ · V(sₜ₊₁) - V(sₜ)

    In the simplified training setup here, we do not generate full response
    trajectories (that would require sampling). Instead, we use the terminal
    reward directly and minimise:

        L = BCE(V(s₀), r_terminal)   +   λ · ||V(s₀) - target||²

    where target = running exponential average of rewards (EMA bootstrap).
    This approximates TD(λ) bootstrapping without needing trajectory data.

    Args:
        extractor  : HiddenStateExtractor (frozen LLM)
        probe      : ValueProbe or LSDFProbe
        lr         : learning rate
        gamma      : discount factor (for multi-step bootstrapping)
        lambda_td  : λ — weight of the bootstrapped TD term
        use_lsdf   : if True, treats probe as LSDFProbe
    """

    def __init__(
        self,
        extractor: HiddenStateExtractor,
        probe: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lambda_td: float = 0.5,
        use_lsdf: bool = False,
    ) -> None:
        self.extractor = extractor
        self.probe = probe
        self.device = extractor.device
        self.gamma = gamma
        self.lambda_td = lambda_td
        self.use_lsdf = use_lsdf

        self.probe.to(self.device)
        self.optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )

        # EMA of rewards — used as bootstrap target
        self._ema_reward: float = 0.5
        self._ema_alpha: float = 0.05

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        dataset: DifficultyDataset,
        epochs: int = 5,
        batch_size: int = 8,
        val_dataset: Optional[DifficultyDataset] = None,
        save_path: Optional[str] = None,
    ) -> List[Dict]:
        """
        Full training loop.

        Returns list of per-epoch metric dicts:
            [{'epoch': 1, 'train_loss': 0.4, 'val_acc': 0.72}, ...]
        """
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=DifficultyDataset.collate,
        )

        history = []
        best_val_loss = math.inf

        for epoch in range(1, epochs + 1):
            self.probe.train()
            epoch_loss = 0.0
            n_batches = 0

            pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
            for questions, labels in pbar:
                labels = labels.to(self.device)
                loss = self._train_step(questions, labels)
                epoch_loss += loss
                n_batches += 1
                pbar.set_postfix(loss=f"{loss:.4f}")

            self.scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            metrics: Dict = {"epoch": epoch, "train_loss": round(avg_loss, 4)}

            if val_dataset is not None:
                val_metrics = self.evaluate(val_dataset, batch_size)
                metrics.update(val_metrics)
                if val_metrics.get("val_loss", math.inf) < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    if save_path:
                        torch.save(self.probe.state_dict(), save_path)

            history.append(metrics)
            print(
                f"  Epoch {epoch:>3} │ loss={avg_loss:.4f} "
                + (
                    f"│ val_acc={metrics['val_acc']:.3f} "
                    f"│ val_loss={metrics['val_loss']:.4f}"
                    if "val_acc" in metrics
                    else ""
                )
            )

        return history

    def _train_step(
        self,
        questions: List[str],
        labels: torch.Tensor,
    ) -> float:
        """One gradient step over a batch."""
        # ── Extract hidden states ─────────────────────────────────────────────
        if self.use_lsdf:
            layer_indices = list(range(1, self.extractor.num_layers + 1))
        else:
            layer_indices = [self.extractor.num_layers]  # last layer only

        layer_hiddens = self.extractor.extract(questions, layer_indices)

        # ── Forward through probe ─────────────────────────────────────────────
        self.optimizer.zero_grad()

        if self.use_lsdf:
            v_pred, _ = self.probe(layer_hiddens)
        else:
            h = layer_hiddens[self.extractor.num_layers]
            v_pred = self.probe(h)

        v_pred = v_pred.squeeze(1)   # [B]

        # ── TD(λ) loss ────────────────────────────────────────────────────────
        # Term 1: Binary cross-entropy against ground-truth terminal reward
        bce = F.binary_cross_entropy(v_pred, labels)

        # Term 2: Bootstrapped TD target — EMA of past rewards
        #         Acts as a smoothed Bellman target: V_target = γ·EMA + (1-γ)·r
        batch_mean_reward = labels.mean().item()
        self._ema_reward = (
            (1 - self._ema_alpha) * self._ema_reward
            + self._ema_alpha * batch_mean_reward
        )
        td_target = torch.full_like(v_pred, self.gamma * self._ema_reward)
        td_loss = F.mse_loss(v_pred, td_target.detach())

        loss = bce + self.lambda_td * td_loss
        loss.backward()

        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(self.probe.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        dataset: DifficultyDataset,
        batch_size: int = 8,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Returns val_loss, val_acc, val_precision, val_recall, val_f1.
        """
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DifficultyDataset.collate,
        )

        self.probe.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0
        n_batches = 0

        if self.use_lsdf:
            layer_indices = list(range(1, self.extractor.num_layers + 1))
        else:
            layer_indices = [self.extractor.num_layers]

        for questions, labels in loader:
            labels = labels.to(self.device)
            layer_hiddens = self.extractor.extract(questions, layer_indices)

            if self.use_lsdf:
                v_pred, _ = self.probe(layer_hiddens)
            else:
                h = layer_hiddens[self.extractor.num_layers]
                v_pred = self.probe(h)

            v_pred = v_pred.squeeze(1)
            total_loss += F.binary_cross_entropy(v_pred, labels).item()
            n_batches += 1

            all_preds.append((v_pred >= threshold).float())
            all_labels.append(labels)

        preds = torch.cat(all_preds)
        gts = torch.cat(all_labels)

        tp = ((preds == 1) & (gts == 1)).sum().float()
        fp = ((preds == 1) & (gts == 0)).sum().float()
        fn = ((preds == 0) & (gts == 1)).sum().float()

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        acc = (preds == gts).float().mean()

        return {
            "val_loss": round(total_loss / max(n_batches, 1), 4),
            "val_acc": round(acc.item(), 4),
            "val_precision": round(precision.item(), 4),
            "val_recall": round(recall.item(), 4),
            "val_f1": round(f1.item(), 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DIFFICULTY ESTIMATOR  (Unified Inference API)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DifficultyResult:
    """
    Returned by DifficultyEstimator.predict().

    Attributes:
        question    : the original input string
        score       : V(s₀) ∈ [0,1] — higher = easier
        label       : "EASY" or "DIFFICULT"
        fingerprint : optional [L] list of per-layer α weights (LSDF only)
        dominant_layer_range : rough human-readable label for which layer
                               range drives the difficulty (LSDF only)
    """
    question: str
    score: float
    label: str
    fingerprint: Optional[List[float]] = None
    dominant_layer_range: Optional[str] = None

    def __str__(self) -> str:
        lines = [
            f"  Question : {self.question[:80]}{'...' if len(self.question)>80 else ''}",
            f"  Score    : {self.score:.4f}  →  {self.label}",
        ]
        if self.fingerprint is not None:
            top3 = sorted(
                enumerate(self.fingerprint), key=lambda x: x[1], reverse=True
            )[:3]
            top3_str = "  ".join(f"L{i+1}:{w:.3f}" for i, w in top3)
            lines.append(f"  Top-3 Layers: {top3_str}")
        if self.dominant_layer_range:
            lines.append(f"  Difficulty Type : {self.dominant_layer_range}")
        return "\n".join(lines)


class DifficultyEstimator:
    """
    High-level inference API that combines extractor + probe.

    Usage:
        est = DifficultyEstimator.from_pretrained("gpt2")
        result = est.predict("What is the capital of France?")
        print(result)

    Args:
        extractor  : HiddenStateExtractor
        probe      : trained ValueProbe or LSDFProbe
        threshold  : V(s₀) ≥ threshold → EASY, else DIFFICULT  (default 0.5)
        use_lsdf   : whether probe is an LSDFProbe
    """

    def __init__(
        self,
        extractor: HiddenStateExtractor,
        probe: nn.Module,
        threshold: float = 0.5,
        use_lsdf: bool = False,
    ) -> None:
        self.extractor = extractor
        self.probe = probe.to(extractor.device)
        self.probe.eval()
        self.threshold = threshold
        self.use_lsdf = use_lsdf
        self._num_layers = extractor.num_layers

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "gpt2",
        probe_weights: Optional[str] = None,
        threshold: float = 0.5,
        use_lsdf: bool = False,
        device: str = "auto",
        pooling: str = "mean",
    ) -> "DifficultyEstimator":
        """
        Convenience constructor — loads LLM and builds untrained probe.
        If probe_weights is given, loads saved probe state_dict.
        """
        extractor = HiddenStateExtractor(
            model_name, device=device, pooling=pooling
        )
        d = extractor.hidden_dim
        L = extractor.num_layers

        if use_lsdf:
            probe: nn.Module = LSDFProbe(L, d)
        else:
            probe = ValueProbe(d)

        if probe_weights and Path(probe_weights).exists():
            state = torch.load(probe_weights, map_location=extractor.device)
            probe.load_state_dict(state)
            print(f"  ✓ Loaded probe weights from {probe_weights}")

        return cls(extractor, probe, threshold, use_lsdf)

    # ── Predict ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, question: str) -> DifficultyResult:
        """Single-question inference."""
        results = self.predict_batch([question])
        return results[0]

    @torch.no_grad()
    def predict_batch(self, questions: List[str]) -> List[DifficultyResult]:
        """Batched inference — more efficient for multiple questions."""
        if self.use_lsdf:
            layer_indices = list(range(1, self._num_layers + 1))
        else:
            layer_indices = [self._num_layers]

        layer_hiddens = self.extractor.extract(questions, layer_indices)

        if self.use_lsdf:
            v_pred, alpha = self.probe(layer_hiddens)
            alpha_np = alpha.cpu().numpy().tolist()
        else:
            h = layer_hiddens[self._num_layers]
            v_pred = self.probe(h)
            alpha_np = [None] * len(questions)

        scores = v_pred.squeeze(1).cpu().numpy().tolist()
        results = []

        for i, (q, score) in enumerate(zip(questions, scores)):
            label = "EASY" if score >= self.threshold else "DIFFICULT"
            fp = alpha_np[i]
            dom = self._dominant_range(fp) if fp is not None else None
            results.append(DifficultyResult(q, round(score, 4), label, fp, dom))

        return results

    def calibrate_threshold(
        self,
        val_dataset: DifficultyDataset,
        n_thresholds: int = 50,
    ) -> float:
        """
        Grid-search for the threshold τ that maximises F1 on a validation set.
        Updates self.threshold in-place and returns it.
        """
        questions, labels = zip(*[val_dataset[i] for i in range(len(val_dataset))])
        results = self.predict_batch(list(questions))
        scores = torch.tensor([r.score for r in results])
        gts = torch.tensor(labels)

        best_tau, best_f1 = 0.5, 0.0
        for tau in torch.linspace(0.1, 0.9, n_thresholds).tolist():
            preds = (scores >= tau).float()
            tp = ((preds == 1) & (gts == 1)).sum().float()
            fp = ((preds == 1) & (gts == 0)).sum().float()
            fn = ((preds == 0) & (gts == 1)).sum().float()
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            if f1 > best_f1:
                best_f1, best_tau = f1.item(), tau

        self.threshold = best_tau
        print(f"  ✓ Calibrated threshold: τ = {best_tau:.3f}  (val F1 = {best_f1:.4f})")
        return best_tau

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _dominant_range(self, fingerprint: List[float]) -> str:
        """
        Maps peak α weight to a human-readable difficulty-type label
        based on which third of the network it falls in.
        """
        L = len(fingerprint)
        peak_layer = max(range(L), key=lambda i: fingerprint[i])
        third = L // 3
        if peak_layer < third:
            return "SYNTACTIC / SURFACE DIFFICULTY  (lower layers dominant)"
        elif peak_layer < 2 * third:
            return "FACTUAL KNOWLEDGE DIFFICULTY  (middle layers dominant)"
        else:
            return "ABSTRACT / MULTI-HOP DIFFICULTY  (upper layers dominant)"


# ─────────────────────────────────────────────────────────────────────────────
# 7.  DEMO — End-to-End Runnable Example
# ─────────────────────────────────────────────────────────────────────────────

def build_demo_dataset() -> Tuple[DifficultyDataset, DifficultyDataset]:
    """
    Toy dataset with labelled QA pairs.

    Label 1.0 = model answered correctly (easy question).
    Label 0.0 = model answered incorrectly (hard question).

    In production you would:
      1. Run your LLM on thousands of questions with known ground-truth answers
      2. Check each response for correctness (string match, exact match, LLM judge)
      3. Populate QAPair(question=..., is_correct=...)
    """
    train_pairs = [
        # ── Easy (factual, single-hop) ───────────────────────────────────────
        QAPair("What is the capital of France?", 1.0),
        QAPair("What is 2 + 2?", 1.0),
        QAPair("Who wrote Romeo and Juliet?", 1.0),
        QAPair("What color is the sky during a clear day?", 1.0),
        QAPair("How many days are in a week?", 1.0),
        QAPair("What is the chemical symbol for water?", 1.0),
        QAPair("Who painted the Mona Lisa?", 1.0),
        QAPair("What is the largest planet in the solar system?", 1.0),
        QAPair("What does HTTP stand for?", 1.0),
        QAPair("What is the square root of 144?", 1.0),

        # ── Hard (multi-hop, cross-document, temporal constraint) ────────────
        QAPair(
            "A paper about AI regulation submitted to arXiv in June 2022 shows "
            "a figure with three axes each having a label at both ends. "
            "Which of those words describes a type of society in a Physics "
            "and Society article submitted on August 11, 2016?",
            0.0,
        ),
        QAPair(
            "What is the Hausdorff dimension of the boundary of the "
            "Mandelbrot set, and who first proved it?",
            0.0,
        ),
        QAPair(
            "In the 1987 film whose director later won an Oscar for a "
            "different musical, what is the full name of the character "
            "played by the actor who was born in the same city as "
            "the composer of the film's score?",
            0.0,
        ),
        QAPair(
            "Which amino acid is encoded by the most codons, and "
            "what is its three-letter abbreviation?",
            0.0,
        ),
        QAPair(
            "The CEO of the company that acquired DeepMind in 2014 "
            "attended which university for his undergraduate degree?",
            0.0,
        ),
        QAPair("Compute ∫∫∫_V (x²+y²+z²) dV over the unit ball.", 0.0),
        QAPair(
            "What theorem guarantees that the spectral radius of a "
            "non-negative irreducible matrix equals its largest real eigenvalue?",
            0.0,
        ),
        QAPair(
            "In which year did the country that first used paper money "
            "adopt a central banking system modelled on the Bank of England?",
            0.0,
        ),
        QAPair(
            "The author of the Federalist Papers No. 10 also wrote "
            "which constitutional amendment, and what does it protect?",
            0.0,
        ),
        QAPair(
            "Derive the Euler-Lagrange equations for a double pendulum "
            "and find all equilibrium configurations.",
            0.0,
        ),
    ]

    # Split 80/20 for train/val
    split = int(0.8 * len(train_pairs))
    return (
        DifficultyDataset(train_pairs[:split]),
        DifficultyDataset(train_pairs[split:]),
    )


def demo() -> None:
    """
    Full end-to-end pipeline:
      1. Load GPT-2 (smallest model, runs on CPU)
      2. Build toy dataset
      3. Train both basic ValueProbe and LSDFProbe
      4. Run inference on the paper's example query
      5. Print difficulty fingerprint breakdown
    """
    print("\n" + "="*68)
    print("  LLM Hidden-State Difficulty Estimator — Demo")
    print("="*68)

    MODEL_NAME = "gpt2"   # swap to "meta-llama/Llama-3.2-1B" etc. on GPU

    # ── Build dataset ─────────────────────────────────────────────────────────
    print("\n[1/4] Building demo dataset ...")
    train_ds, val_ds = build_demo_dataset()
    print(f"  Train: {len(train_ds)} examples   Val: {len(val_ds)} examples")

    # ── Train Basic Probe ─────────────────────────────────────────────────────
    print("\n[2/4] Training basic ValueProbe (last-layer only) ...")
    estimator_basic = DifficultyEstimator.from_pretrained(
        MODEL_NAME, use_lsdf=False
    )
    trainer_basic = TDLambdaTrainer(
        estimator_basic.extractor,
        estimator_basic.probe,
        lr=3e-4,
        use_lsdf=False,
    )
    trainer_basic.train(train_ds, epochs=3, batch_size=4, val_dataset=val_ds)

    # ── Train LSDF Probe ──────────────────────────────────────────────────────
    print("\n[3/4] Training LSDFProbe (all layers + meta-aggregator) ...")
    estimator_lsdf = DifficultyEstimator.from_pretrained(
        MODEL_NAME, use_lsdf=True
    )
    trainer_lsdf = TDLambdaTrainer(
        estimator_lsdf.extractor,
        estimator_lsdf.probe,
        lr=3e-4,
        use_lsdf=True,
    )
    trainer_lsdf.train(train_ds, epochs=3, batch_size=4, val_dataset=val_ds)

    # ── Inference on representative queries ───────────────────────────────────
    print("\n[4/4] Running inference on test queries ...")

    test_questions = [
        "What is the capital of France?",                 # obviously easy
        "What is 2 + 2?",                                 # obviously easy
        "Who wrote Romeo and Juliet?",                    # easy factual
        (
            "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia."
        ),
        "Compute ∫∫∫_V (x²+y²+z²) dV over the unit ball.",
        (
            "Im researching species that became invasive after people who kept them as pets released them. There’s a certain species of fish that was popularized as a pet by being the main character of the movie Finding Nemo. According to the USGS, where was this fish found as a nonnative species, before the year 2020? I need the answer formatted as the five-digit zip codes of the places the species was found, separated by commas if there is more than one place."
        ),
    ]

    print("\n── Basic ValueProbe Results ──────────────────────────────────")
    for q in test_questions:
        r = estimator_basic.predict(q)
        print(f"\n{r}")

    print("\n── LSDF Probe Results (with Fingerprint) ─────────────────────")
    for q in test_questions:
        r = estimator_lsdf.predict(q)
        print(f"\n{r}")

    print("\n" + "="*68)
    print("  Demo complete.")
    print("="*68 + "\n")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo()
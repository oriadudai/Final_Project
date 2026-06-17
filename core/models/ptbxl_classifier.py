"""ECG pathology classifier for the BCE evaluation metric.

Two modes:
  1. CLEFClassifier (default when clef_encoder provided)
     Uses the pretrained CLEF backbone directly.
     CLEF features → sigmoid → 256-dim soft clinical scores.
     BCE measures consistency in CLEF's clinically-supervised feature space.
     No additional data or downloads needed.

  2. SurrogateECGClassifier (fallback, random weights)
     Used only if no CLEF encoder is available.
     BCE with random weights is not clinically meaningful.

Usage in evaluate_fold():
    from core.models.ptbxl_classifier import build_classifier
    classifier = build_classifier(clef_encoder)     # preferred
    classifier = build_classifier(None)             # fallback surrogate
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, sosfilt, resample


# ---------------------------------------------------------------------------
# Option 1: CLEF-based classifier (preferred)
# ---------------------------------------------------------------------------

class CLEFClassifier(nn.Module):
    """Wraps the frozen pretrained CLEF encoder as a pathology-proxy classifier.

    Forward pass:
      1. Resample 125 Hz → 500 Hz, bandpass 0.67–40 Hz, per-window z-score
         (identical to ClinicalCompositeLoss._to_clef_input)
      2. Run through frozen CLEF backbone → (B, 256) features
      3. Sigmoid → (B, 256) soft clinical scores in [0, 1]

    BCE between CLEFClassifier(true) and CLEFClassifier(pred) measures how
    similar the two signals are in CLEF's clinically-supervised feature space.
    Lower = more consistent. The 256 dimensions are not named pathology classes
    but are grounded in clinical pretraining on 161k ECGs — more meaningful
    than any randomly initialized surrogate.
    """

    def __init__(self, clef_encoder: nn.Module):
        super().__init__()
        self.encoder = clef_encoder          # already frozen
        self._sos = butter(4, [0.67, 40.0], btype="band", fs=500, output="sos")

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, seq_len) at 125 Hz → (B, 1, 5000) at 500 Hz, z-scored."""
        x_np = x.squeeze(1).detach().cpu().numpy()          # (B, seq_len)
        r    = resample(x_np, 5000, axis=1)                 # → 500 Hz
        r    = sosfilt(self._sos, r, axis=1)                # bandpass
        mean = r.mean(axis=1, keepdims=True)
        std  = r.std(axis=1,  keepdims=True) + 1e-8
        r    = (r - mean) / std
        return torch.from_numpy(r.astype(np.float32)).unsqueeze(1).to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, seq_len)  →  (B, 256) sigmoid scores
        with torch.no_grad():
            features = self.encoder(self._preprocess(x))    # (B, 256)
        return torch.sigmoid(features)


# ---------------------------------------------------------------------------
# Option 2: Random surrogate (fallback only)
# ---------------------------------------------------------------------------

class _ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                          nn.BatchNorm1d(out_ch))
            if (in_ch != out_ch or stride != 1) else nn.Identity()
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class SurrogateECGClassifier(nn.Module):
    """Random-weight fallback. BCE with this is NOT clinically meaningful."""

    def __init__(self, num_classes: int = 5, embed_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            _ResBlock1D(32, 64,  stride=2),
            _ResBlock1D(64, 128, stride=2),
            _ResBlock1D(128, embed_dim, stride=2),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.classifier = nn.Sequential(nn.Linear(embed_dim, num_classes), nn.Sigmoid())

    def forward(self, x):
        return self.classifier(self.backbone(x))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class CLEFProbeClassifier(nn.Module):
    """Frozen CLEF encoder (MIMIC-IV pretrained) + small trainable linear probe.

    The encoder never moves — only the 256×num_classes probe head is trained on
    PTB-XL labels.  Because CLEF was trained on MIMIC-IV (same hospital / patient
    population as BIDMC), its features are domain-matched to our test ECGs, so the
    class probabilities this produces are meaningful — unlike the degenerate BCE
    from sigmoid(raw CLEF features) which collapses to ~log(2) for all inputs.
    """

    def __init__(self, clef_encoder: nn.Module, num_classes: int = 5, feature_dim: int = 256):
        super().__init__()
        self._clef_clf = CLEFClassifier(clef_encoder)   # reuses preprocessing
        self.probe = nn.Linear(feature_dim, num_classes)
        self._sigmoid = nn.Sigmoid()
        for p in clef_encoder.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, seq_len) at 125 Hz  →  (B, num_classes) probabilities
        with torch.no_grad():
            x_proc   = self._clef_clf._preprocess(x)
            features = self._clef_clf.encoder(x_proc)   # (B, 256)
        return self._sigmoid(self.probe(features))


def build_classifier(clef_encoder: Optional[nn.Module]) -> nn.Module:
    """Build the best available classifier for BCE evaluation.

    Args:
        clef_encoder: Pretrained frozen CLEF encoder, or None.

    Returns:
        CLEFClassifier if clef_encoder provided, else SurrogateECGClassifier.
    """
    if clef_encoder is not None:
        model = CLEFClassifier(clef_encoder)
        print("BCE classifier: CLEFClassifier (pretrained, 256-dim clinical features)")
    else:
        model = SurrogateECGClassifier()
        print("BCE classifier: SurrogateECGClassifier (random weights — fallback only)")

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


# Keep backward-compatible factory used by existing code
def get_ptbxl_classifier(
    pretrained_path: Optional[str] = None,
    num_classes: int = 5,
    embed_dim: int = 128,
    freeze: bool = True,
) -> nn.Module:
    """Legacy factory — returns random surrogate. Prefer build_classifier()."""
    model = SurrogateECGClassifier(num_classes=num_classes, embed_dim=embed_dim)
    if pretrained_path is not None:
        ckpt  = torch.load(pretrained_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(state, strict=False)
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    model.eval()
    return model

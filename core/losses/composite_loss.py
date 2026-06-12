import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, sosfilt, resample, firwin


def load_clef_encoder(ckpt_path: str, model_size: str = "small", device=None):
    """Load a pretrained CLEF encoder and freeze all parameters.

    Checkpoint download:
      curl -L "https://zenodo.org/records/17572734/files/clef_small.ckpt?download=1" \
           -o models/clef/clef_small.ckpt

    Args:
        ckpt_path:  Path to the .ckpt file from Zenodo.
        model_size: "small" (256-dim), "medium" (1024-dim), or "large" (2048-dim).
        device:     torch.device to place the model on.

    Returns:
        Frozen nn.Module that maps (B, 1, 5000) → (B, feature_dim).
    """
    from clef.baselines.models.CLEF import create_net1d_by_size  # noqa: PLC0415

    if device is None:
        device = torch.device("cpu")
    model = create_net1d_by_size(
        device=device,
        model_size=model_size,
        n_classes=1000,
        linear_prob=False,
        pth=ckpt_path,
        in_channels=1,
    )
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


class ClinicalCompositeLoss(nn.Module):
    """Huber loss + CLEF perceptual feature-matching loss.

    L_total = huber_weight * HuberLoss(pred, true) + lambda_clinical * ||Φ(true) - Φ(pred)||²

    huber_weight defaults to 1.0 (the standard composite loss used for
    training). Setting huber_weight=0.0 gives a pure CLEF feature-matching
    loss -- useful for calibration-loss ablations.

    Φ is a frozen pretrained CLEF encoder. The CLEF preprocessing pipeline
    (resample 125→500 Hz, bandpass 0.67-40 Hz, per-window z-score) runs inside
    this module and is entirely separate from the main preprocessing in
    src/preprocessing.py.

    For the pred path the preprocessing is implemented with differentiable PyTorch
    ops (F.interpolate + F.conv1d with a pre-computed FIR kernel) so that gradients
    flow from clinical_loss back through the CLEF encoder and into ReHeartNet.
    The true path uses numpy/scipy (inside torch.no_grad()) for speed.
    """

    # FIR bandpass parameters (500 Hz, 0.67–40 Hz), matching CLEF training conditions.
    _FIR_NTAPS = 255   # odd → linear-phase; longer = sharper roll-off

    def __init__(
        self,
        clef_encoder: nn.Module,
        lambda_clinical: float = 0.1,
        huber_delta: float = 1.0,
        huber_weight: float = 1.0,
    ):
        super().__init__()
        self.clef_encoder = clef_encoder  # already frozen by load_clef_encoder
        self.lambda_clinical = lambda_clinical
        self.huber_weight = huber_weight
        self.huber = nn.HuberLoss(delta=huber_delta)

        # IIR filter for the non-differentiable true path (500 Hz, 0.67–40 Hz)
        self._sos = butter(4, [0.67, 40.0], btype="band", fs=500, output="sos")

        # FIR kernel for the differentiable pred path.
        # Stored as a buffer so it moves to the correct device with .to(device).
        fir_coeffs = firwin(
            self._FIR_NTAPS, [0.67, 40.0], fs=500, pass_zero=False
        ).astype(np.float32)
        self.register_buffer(
            "_fir_kernel",
            torch.from_numpy(fir_coeffs).view(1, 1, -1),
        )

    # ------------------------------------------------------------------
    # Non-differentiable path (true ECG, always inside torch.no_grad())
    # ------------------------------------------------------------------

    def _to_clef_input_numpy(self, x: torch.Tensor) -> torch.Tensor:
        """Numpy/scipy preprocessing — fast but not differentiable."""
        x_np = x.squeeze(-1).detach().cpu().numpy()     # (B, 1250)
        r = resample(x_np, 5000, axis=1)                # (B, 5000) @ 500 Hz
        r = sosfilt(self._sos, r, axis=1)               # IIR bandpass
        mean = r.mean(axis=1, keepdims=True)
        std  = r.std(axis=1,  keepdims=True) + 1e-8
        r = (r - mean) / std                            # per-window z-score
        return torch.from_numpy(r.astype(np.float32)).unsqueeze(1).to(x.device)

    # ------------------------------------------------------------------
    # Differentiable path (pred ECG — gradients must reach ReHeartNet)
    # ------------------------------------------------------------------

    def _to_clef_input_diff(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable preprocessing — gradients flow back to the model.

        Uses F.interpolate for resampling and F.conv1d with a frozen FIR
        kernel for bandpass filtering.  Both ops are tracked by autograd.

          (B, 1250, 1) at 125 Hz
          → squeeze  → (B, 1, 1250)
          → F.interpolate  → (B, 1, 5000) @ 500 Hz
          → FIR bandpass   → (B, 1, 5000)
          → z-score        → (B, 1, 5000)
        """
        x = x.squeeze(-1).unsqueeze(1)                  # (B, 1, 1250)
        x = F.interpolate(x, size=5000, mode="linear", align_corners=False)  # (B, 1, 5000)

        # FIR conv1d — kernel is (1, 1, n_taps); "same" length via reflect padding
        pad = self._FIR_NTAPS // 2
        x = F.pad(x, (pad, pad), mode="reflect")
        x = F.conv1d(x, self._fir_kernel)               # (B, 1, 5000)

        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1,  keepdim=True) + 1e-8
        x = (x - mean) / std                            # per-window z-score
        return x                                         # (B, 1, 5000)

    # ------------------------------------------------------------------

    def forward(
        self,
        pred_ecg: torch.Tensor,
        true_ecg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_ecg: (B, SEQ_LEN, 1) reconstructed ECG
            true_ecg: (B, SEQ_LEN, 1) ground-truth ECG
        Returns:
            Scalar loss tensor.
        """
        huber_loss = self.huber(pred_ecg, true_ecg)

        # Ground-truth features: no gradients needed
        with torch.no_grad():
            phi_true = self.clef_encoder(self._to_clef_input_numpy(true_ecg))

        # Predicted features: differentiable path — gradients flow through
        # F.interpolate → FIR conv1d → z-score → CLEF encoder → phi_pred
        # and back into ReHeartNet parameters.
        phi_pred = self.clef_encoder(self._to_clef_input_diff(pred_ecg))

        clinical_loss = torch.mean((phi_true - phi_pred) ** 2)
        total = self.huber_weight * huber_loss + self.lambda_clinical * clinical_loss

        # Expose components so callers can log them to wandb
        self.last_huber_loss = float(huber_loss.detach())
        self.last_clef_loss  = float((self.lambda_clinical * clinical_loss).detach())

        return total


if __name__ == "__main__":
    print("=== ClinicalCompositeLoss smoke test (random encoder) ===")
    dummy_encoder = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(5000, 256),
    ).eval()
    for p in dummy_encoder.parameters():
        p.requires_grad = False

    criterion = ClinicalCompositeLoss(dummy_encoder)
    pred = torch.randn(4, 1250, 1, requires_grad=True)
    true = torch.randn(4, 1250, 1)
    loss = criterion(pred, true)
    loss.backward()
    print(f"Loss: {loss.item():.4f}")
    print(f"pred.grad is not None: {pred.grad is not None}")   # must be True

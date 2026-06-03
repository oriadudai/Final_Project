import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, sosfilt, resample


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

    L_total = HuberLoss(pred, true) + lambda_clinical * ||Φ(true) - Φ(pred)||²

    Φ is a frozen pretrained CLEF encoder. The CLEF preprocessing pipeline
    (resample 125→500 Hz, bandpass 0.67-40 Hz, per-window z-score) runs
    inside _to_clef_input and is entirely separate from the main preprocessing
    in src/preprocessing.py.
    """

    def __init__(
        self,
        clef_encoder: nn.Module,
        lambda_clinical: float = 0.1,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.clef_encoder = clef_encoder  # already frozen by load_clef_encoder
        self.lambda_clinical = lambda_clinical
        self.huber = nn.HuberLoss(delta=huber_delta)

        # Build Butterworth bandpass filter coefficients once (500 Hz, 0.67-40 Hz)
        self._sos = butter(4, [0.67, 40.0], btype="band", fs=500, output="sos")

    def _to_clef_input(self, x: torch.Tensor) -> torch.Tensor:
        """Convert model-domain ECG tensor to CLEF-compatible input.

        Pipeline (separate from main preprocessing):
          (B, 1250, 1) at 125 Hz
          → resample to 5000 samples at 500 Hz (same 10-second window)
          → bandpass 0.67-40 Hz at 500 Hz  (matches CLEF training conditions)
          → per-window z-score              (CLEF requirement)
          → (B, 1, 5000) float32 tensor
        """
        # Detach from graph (only phi_pred needs gradients; phi_true does not)
        x_np = x.squeeze(-1).detach().cpu().numpy()          # (B, 1250)
        r = resample(x_np, 5000, axis=1)                     # (B, 5000) @ 500 Hz
        r = sosfilt(self._sos, r, axis=1)                    # bandpass
        mean = r.mean(axis=1, keepdims=True)
        std  = r.std(axis=1,  keepdims=True) + 1e-8
        r = (r - mean) / std                                 # per-window z-score
        return torch.from_numpy(r.astype(np.float32)).unsqueeze(1).to(x.device)

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

        # Ground-truth features: no gradients needed through the encoder
        with torch.no_grad():
            phi_true = self.clef_encoder(self._to_clef_input(true_ecg))

        # Predicted features: gradients flow back into ReHeartNet through here
        phi_pred = self.clef_encoder(self._to_clef_input(pred_ecg))

        clinical_loss = torch.mean((phi_true - phi_pred) ** 2)
        total = huber_loss + self.lambda_clinical * clinical_loss

        # Expose components so callers can log them to wandb
        self.last_huber_loss    = float(huber_loss.detach())
        self.last_clef_loss     = float((self.lambda_clinical * clinical_loss).detach())

        return total


if __name__ == "__main__":
    print("=== ClinicalCompositeLoss smoke test (random encoder) ===")
    # Use a tiny random linear encoder as a stand-in for CLEF
    dummy_encoder = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(5000, 256),
    ).eval()
    for p in dummy_encoder.parameters():
        p.requires_grad = False

    criterion = ClinicalCompositeLoss(dummy_encoder)
    pred = torch.randn(4, 1250, 1)
    true = torch.randn(4, 1250, 1)
    loss = criterion(pred, true)
    print(f"Loss: {loss.item():.4f}")

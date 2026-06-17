"""Pytest coverage for training guards, diagnostic training, and evaluation.

The tests in this file use only synthetic tensors with the project sequence
length and channel layout. They do not touch BIDMC or PTB-XL files.
"""

import sys
import types
import importlib.util
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.setdefault("wfdb", types.SimpleNamespace(rdrecord=None))


import core.config as config
from core.losses.composite_loss import ClinicalCompositeLoss
from core.models.baselines import SimpleLSTMBaseline, get_model
from core.models.ptbxl_classifier import build_classifier
from core.models.ptbxl_classifier import SurrogateECGClassifier
from core.train import train_one_epoch


def _load_local_script(module_name: str, filename: str):
    module_path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


diag_eval = _load_local_script("diag_eval_local", "evaluate_diagnostic_consistency.py")
diag_train = _load_local_script("diag_train_local", "train_diagnostic_classifier.py")


def _make_signal_batch(batch_size: int, seq_len: int, inject_nan: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a synthetic PPG/ECG batch with the project's expected shapes."""
    ppg = torch.randn(batch_size, seq_len, 1)
    ecg = torch.randn(batch_size, seq_len, 1)
    if inject_nan:
        ppg[0, seq_len // 2, 0] = float("nan")
    return ppg, ecg


def _clone_parameters(model: nn.Module) -> List[torch.Tensor]:
    return [param.detach().clone() for param in model.parameters()]


def _params_all_finite(model: nn.Module) -> bool:
    return all(torch.isfinite(param).all().item() for param in model.parameters())


class IdentityReconModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


@pytest.mark.parametrize("model_name", ["lstm", "bilstm", "reheartnet"])
def test_baseline_factory_models_preserve_project_shape(model_name: str) -> None:
    model = get_model(model_name, hidden_size=4).eval()
    ppg, _ = _make_signal_batch(batch_size=2, seq_len=config.SEQ_LEN)

    with torch.no_grad():
        pred = model(ppg)

    assert pred.shape == (2, config.SEQ_LEN, 1)
    assert torch.isfinite(pred).all()


def test_ptbxl_classifier_factory_returns_frozen_classifier_with_expected_output_shape() -> None:
    classifier = build_classifier(None)
    ecg = torch.randn(2, 1, config.SEQ_LEN)

    with torch.no_grad():
        pred = classifier(ecg)

    assert pred.shape == (2, 5)
    assert torch.isfinite(pred).all()
    assert all(not param.requires_grad for param in classifier.parameters())


def test_clinical_composite_loss_runs_on_dummy_data_and_backpropagates() -> None:
    encoder = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(5000, 256),
    ).eval()
    for param in encoder.parameters():
        param.requires_grad = False

    model = SimpleLSTMBaseline(hidden_size=4).train()
    criterion = ClinicalCompositeLoss(encoder, lambda_clinical=0.1, huber_delta=1.0)
    ppg, ecg = _make_signal_batch(batch_size=2, seq_len=config.SEQ_LEN)

    pred = model(ppg)
    loss = criterion(pred, ecg)

    assert torch.isfinite(loss)
    loss.backward()
    assert any(param.grad is not None for param in model.parameters())


def test_train_one_epoch_skips_optimizer_step_on_nan_input_and_preserves_weights(monkeypatch) -> None:
    model = SimpleLSTMBaseline(hidden_size=4).train()
    ppg, ecg = _make_signal_batch(batch_size=1, seq_len=config.SEQ_LEN, inject_nan=True)
    loader = DataLoader(TensorDataset(ppg, ecg), batch_size=1, shuffle=False)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial_params = _clone_parameters(model)

    def forbidden_step(*args, **kwargs):
        raise AssertionError("optimizer.step() should be skipped for non-finite loss")

    monkeypatch.setattr(optimizer, "step", forbidden_step)

    result = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))

    assert result["loss"] == 0.0
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert _params_all_finite(model)
    for param, expected in zip(model.parameters(), initial_params):
        assert torch.equal(param.detach(), expected)


def test_train_one_epoch_detects_post_step_weight_corruption_and_stops_early() -> None:
    model = SimpleLSTMBaseline(hidden_size=4).train()

    ppg_batches = []
    ecg_batches = []
    for _ in range(3):
        ppg, ecg = _make_signal_batch(batch_size=1, seq_len=config.SEQ_LEN)
        ppg_batches.append(ppg.squeeze(0))
        ecg_batches.append(ecg.squeeze(0))

    ppg_tensor = torch.stack(ppg_batches, dim=0)
    ecg_tensor = torch.stack(ecg_batches, dim=0)
    loader = DataLoader(TensorDataset(ppg_tensor, ecg_tensor), batch_size=1, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    original_step = optimizer.step
    call_count = {"value": 0}

    def corrupting_step(*args, **kwargs):
        call_count["value"] += 1
        result = original_step(*args, **kwargs)
        if call_count["value"] == 2:
            with torch.no_grad():
                first_param = next(model.parameters())
                first_param.view(-1)[0] = float("nan")
        return result

    optimizer.step = corrupting_step

    result = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))

    assert call_count["value"] == 2
    assert result["loss"] >= 0.0
    assert not _params_all_finite(model)


def test_train_diagnostic_classifier_single_step_updates_weights_and_saves_checkpoint(tmp_path) -> None:
    model = SurrogateECGClassifier(num_classes=len(diag_train.SUPERCLASSES)).train()
    x = torch.randn(4, 1, config.SEQ_LEN)
    y = torch.randint(0, 2, (4, len(diag_train.SUPERCLASSES))).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial_params = _clone_parameters(model)

    train_loss = diag_train._run_epoch(model, loader, torch.device("cpu"), criterion, optimizer)

    assert torch.isfinite(torch.tensor(train_loss))
    assert any(not torch.equal(param.detach(), expected) for param, expected in zip(model.parameters(), initial_params))

    ckpt_path = tmp_path / "ptbxl_diagnostic_classifier.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "superclasses": diag_train.SUPERCLASSES,
            "val_bce": train_loss,
            "epoch": 1,
        },
        ckpt_path,
    )

    restored = torch.load(str(ckpt_path), map_location="cpu")
    fresh_model = SurrogateECGClassifier(num_classes=len(restored["superclasses"]))
    fresh_model.load_state_dict(restored["state_dict"])
    assert restored["superclasses"] == diag_train.SUPERCLASSES
    assert fresh_model.__class__ is SurrogateECGClassifier


def test_evaluate_diagnostic_consistency_pipeline_computes_scores(tmp_path) -> None:
    classifier_ckpt = tmp_path / "diagnostic_classifier.pt"
    classifier = SurrogateECGClassifier(num_classes=len(diag_train.SUPERCLASSES))
    torch.save(
        {
            "state_dict": classifier.state_dict(),
            "superclasses": diag_train.SUPERCLASSES,
        },
        classifier_ckpt,
    )

    loaded_classifier, superclasses = diag_eval._load_diagnostic_classifier(str(classifier_ckpt))
    assert superclasses == diag_train.SUPERCLASSES
    assert all(not param.requires_grad for param in loaded_classifier.parameters())

    ppg = torch.randn(6, config.SEQ_LEN, 1)
    ecg = ppg.clone()
    loader = DataLoader(TensorDataset(ppg, ecg), batch_size=3, shuffle=False)

    recon_model = IdentityReconModel()
    true_arr, pred_arr = diag_eval._collect_predictions(recon_model, loader, torch.device("cpu"))
    assert true_arr.shape == (6, config.SEQ_LEN)
    assert pred_arr.shape == (6, config.SEQ_LEN)

    diag_kl, flip_rate = diag_eval._diag_consistency(
        true_arr,
        pred_arr,
        loaded_classifier,
        torch.device("cpu"),
        batch_size=3,
    )

    noisy_pred = pred_arr + 0.25 * np.random.randn(*pred_arr.shape).astype(np.float32)
    noisy_kl, noisy_flip_rate = diag_eval._diag_consistency(
        true_arr,
        noisy_pred.astype(np.float32),
        loaded_classifier,
        torch.device("cpu"),
        batch_size=3,
    )

    assert diag_kl >= 0.0
    assert flip_rate == pytest.approx(0.0, abs=1e-6)
    assert noisy_kl >= diag_kl
    assert noisy_flip_rate >= flip_rate

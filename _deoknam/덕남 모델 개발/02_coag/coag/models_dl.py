# -*- coding: utf-8 -*-
"""LSTM/GRU 시퀀스 회귀 모델 (PyTorch, GPU)."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import config as C

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SeqRegressor(nn.Module):
    """다층 LSTM/GRU → 마지막 은닉 상태 → Linear(2출력)."""

    def __init__(self, rnn_type: str, n_features: int,
                 hidden: int = None, layers: int = None, dropout: float = None):
        super().__init__()
        hidden = C.SEQ["hidden"] if hidden is None else hidden
        layers = C.SEQ["layers"] if layers is None else layers
        dropout = C.SEQ["dropout"] if dropout is None else dropout
        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(n_features, hidden, num_layers=layers,
                           batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 2))
        self.hyper = dict(rnn_type=rnn_type, n_features=n_features,
                          hidden=hidden, layers=layers, dropout=dropout)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


def train_dl(rnn_type: str, Xtr, Ytr, Xval, Yval, verbose: bool = False):
    """Adam + MSE, early stopping(val loss, patience). best state 복원."""
    torch.manual_seed(C.RANDOM_STATE)
    model = SeqRegressor(rnn_type, Xtr.shape[2]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=C.SEQ["lr"])
    lossf = nn.MSELoss()
    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
        batch_size=C.SEQ["batch"], shuffle=True,
        generator=torch.Generator().manual_seed(C.RANDOM_STATE))
    Xv = torch.from_numpy(Xval).to(DEVICE)
    Yv = torch.from_numpy(Yval).to(DEVICE)

    best_loss, best_state, patience = np.inf, None, 0
    for epoch in range(C.SEQ["max_epochs"]):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(lossf(model(Xv), Yv))
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= C.SEQ["patience"]:
                break
        if verbose and epoch % 10 == 0:
            print(f"    {rnn_type} epoch {epoch}: val_loss {val_loss:.5f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_dl(model: SeqRegressor, X3d: np.ndarray, batch: int = 4096) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X3d), batch):
            xb = torch.from_numpy(X3d[i:i + batch]).to(DEVICE)
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs) if outs else np.empty((0, 2))


def to_bundle(model: SeqRegressor) -> dict:
    """joblib 저장용 자기완결 dict (state_dict는 CPU)."""
    return {"framework": "torch", "arch": "SeqRegressor",
            "hyperparams": model.hyper,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}


def from_bundle(bundle: dict) -> SeqRegressor:
    h = bundle["hyperparams"]
    model = SeqRegressor(h["rnn_type"], h["n_features"], h["hidden"],
                         h["layers"], h["dropout"])
    model.load_state_dict(bundle["state_dict"])
    return model.to(DEVICE)

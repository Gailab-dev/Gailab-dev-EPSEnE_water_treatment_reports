# -*- coding: utf-8 -*-
"""5개 후보 모델: LinearRegression·XGBoost·RandomForest(태뷸러) / GRU·LSTM(시퀀스)."""
import numpy as np

from . import config as C

try:
    import torch
    from torch import nn
    _TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _TORCH = False
    DEVICE = None


# ---------- 태뷸러 ----------
def train_tabular(name: str, Xtr, ytr, Xva=None, yva=None):
    if name == "LinearRegression":
        from sklearn.linear_model import LinearRegression
        m = LinearRegression()
        m.fit(Xtr, ytr)
    elif name == "XGBoost":
        from xgboost import XGBRegressor
        m = XGBRegressor(**C.XGB)
        if Xva is not None and len(Xva):
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        else:
            m.fit(Xtr, ytr)
    elif name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(**C.RF)
        m.fit(Xtr, ytr)
    else:
        raise ValueError(name)
    return m


def predict_tabular(model, X):
    return model.predict(X)


# ---------- 시퀀스 (torch) ----------
if _TORCH:
    class SeqRegressor(nn.Module):
        """다층 LSTM/GRU → 마지막 은닉 → Linear(1)."""

        def __init__(self, rnn_type, n_features, hidden, layers, dropout):
            super().__init__()
            rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
            self.rnn = rnn_cls(n_features, hidden, num_layers=layers,
                               batch_first=True, dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :]).squeeze(-1)


def train_seq(rnn_type: str, Xtr, ytr, Xva, yva):
    """Adam+MSE, early stopping. y는 표준화(스텝 스케일) 후 학습, 예측시 역변환."""
    assert _TORCH, "torch 미설치 — 시퀀스 모델 불가"
    torch.manual_seed(C.RANDOM_STATE)
    from torch.utils.data import DataLoader, TensorDataset
    ymu, ysd = float(np.mean(ytr)), float(np.std(ytr) or 1.0)
    ztr = (ytr - ymu) / ysd
    model = SeqRegressor(rnn_type, Xtr.shape[2], C.SEQ["hidden"], C.SEQ["layers"],
                         C.SEQ["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=C.SEQ["lr"])
    lossf = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr.astype(np.float32)),
                      torch.from_numpy(ztr.astype(np.float32))),
        batch_size=C.SEQ["batch"], shuffle=True,
        generator=torch.Generator().manual_seed(C.RANDOM_STATE))
    has_va = Xva is not None and len(Xva) > 0
    if has_va:
        Xv = torch.from_numpy(Xva.astype(np.float32)).to(DEVICE)
        zv = torch.from_numpy(((yva - ymu) / ysd).astype(np.float32)).to(DEVICE)
    best, best_state, patience = np.inf, None, 0
    for ep in range(C.SEQ["max_epochs"]):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        if has_va:
            model.eval()
            with torch.no_grad():
                vl = float(lossf(model(Xv), zv))
        else:
            vl = float(loss.detach())
        if vl < best - 1e-6:
            best, best_state, patience = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= C.SEQ["patience"]:
                break
    if best_state:
        model.load_state_dict(best_state)
    model._ymu, model._ysd = ymu, ysd
    return model


def predict_seq(model, X, batch=4096):
    assert _TORCH
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch].astype(np.float32)).to(DEVICE)
            outs.append(model(xb).cpu().numpy())
    z = np.concatenate(outs) if outs else np.empty((0,))
    return z * model._ysd + model._ymu

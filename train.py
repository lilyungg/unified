import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds.append(torch.sigmoid(model(x)).cpu().numpy())
            targets.append(y.cpu().numpy())
    return roc_auc_score(np.concatenate(targets), np.concatenate(preds))


def train_model(
    model:      nn.Module,
    tr_loader:  DataLoader,
    va_loader:  DataLoader,
    device:     torch.device,
    lr:         float = 1e-3,
    max_epochs: int   = 30,
    patience:   int   = 5,
) -> nn.Module:
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    has_reg   = hasattr(model, "reg_loss")

    best_auc, best_state, no_improve = -1.0, None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            if has_reg:
                loss = loss + model.reg_loss()
            loss.backward()
            optimizer.step()

        val_auc = evaluate(model, va_loader, device)

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

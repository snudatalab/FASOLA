import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ==============================================================================
# Utility: Reproducibility
# ==============================================================================

def set_seed(seed=42):
    """
    Set random seed for reproducibility across numpy, torch, and python.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[System] Random Seed set to {seed}")

# ==============================================================================
# Internal Helpers
# ==============================================================================

def calc_metrics(y_true, y_pred):
    """
    Calculate Accuracy and Macro-F1 score.

    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted labels.

    Returns:
        tuple: (accuracy, f1_score) in percentage.
    """
    acc = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, average="macro") * 100
    return acc, f1

@torch.no_grad()
def infer_num_classes(models, loader, device):
    """
    Infer the number of classes C from a single forward pass.

    Args:
        models (list): List of PyTorch models.
        loader (DataLoader): Data loader for inference.
        device (torch.device): Computation device.

    Returns:
        int: Number of output classes.
    """
    m = models[0].to(device)
    m.eval()
    for bx, _ in loader:
        bx = bx.to(device, non_blocking=True)
        out = m(bx)
        return int(out.shape[1])
    raise RuntimeError("Empty loader: cannot infer num_classes.")

@torch.no_grad()
def apply_adabn(models, loader, device):
    """
    Update BatchNorm running statistics (mean/var) using unlabeled target data.
    This creates a copy of the models, leaving the originals intact.

    Args:
        models (list): List of source models.
        loader (DataLoader): Target data loader.
        device (torch.device): Computation device.

    Returns:
        list: List of models adapted with AdaBN.
    """
    print("Applying AdaBN Pre-adaptation (BN stats only)...")
    models = [copy.deepcopy(m).to(device) for m in models]

    for model in models:
        model.eval()
        for mod in model.modules():
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                mod.train()
    for bx, _ in loader:
        bx = bx.to(device, non_blocking=True)
        for model in models:
            _ = model(bx)

    for model in models:
        model.eval()

    return models

# ==============================================================================
# FASOLA: Momentum EM stats
# ==============================================================================

def get_fasola_stats(
    models,
    loader,
    device,
    tau=2.0,
    sharpness=10.0,
    em_iters=5,
    momentum_alpha=0.2,
    use_momentum=True,
):
    """
    Compute FASOLA statistics including p_hat (target prediction prior),
    agreement weights, and the estimated target prior q via Momentum EM.

    Args:
        models (list): List of AdaBN-adapted source models.
        loader (DataLoader): Target data loader.
        device (torch.device): Computation device.
        tau (float): Temperature for posterior adjustment.
        sharpness (float): Sharpness parameter for weighting models.
        em_iters (int): Number of EM iterations.
        momentum_alpha (float): Momentum coefficient for prior update.
        use_momentum (bool): Whether to use momentum in EM updates.

    Returns:
        tuple: (p_hat, weights, q)
    """
    num_models = len(models)
    num_classes = infer_num_classes(models, loader, device)

    # 1) Buffer predictions on target
    all_probs_list = [[] for _ in range(num_models)]
    total_samples = 0

    with torch.no_grad():
        for bx, _ in tqdm(loader, desc="[FASOLA] Momentum EM Stats"):
            bx = bx.to(device, non_blocking=True)
            bs = bx.size(0)
            total_samples += bs
            for i, model in enumerate(models):
                logits = model(bx)
                probs = torch.softmax(logits, dim=1)
                all_probs_list[i].append(probs)

    tensor_probs = torch.stack([torch.cat(all_probs_list[i], dim=0) for i in range(num_models)], dim=0)  # (K,N,C)
    tensor_probs = torch.clamp(tensor_probs, min=1e-12)

    # 2) Fixed stats: p_hat and agreement weights
    p_hat = tensor_probs.sum(dim=1) / float(total_samples)           # (K,C)
    p_hat = torch.clamp(p_hat, min=1e-12)

    agreement_scores = torch.zeros(num_models, device=device)
    for i in range(num_models):
        my_pred = tensor_probs[i].argmax(dim=1)                      # (N,)
        others_idx = [j for j in range(num_models) if j != i]
        others_probs = tensor_probs[others_idx].mean(dim=0)          # (N,C)
        others_label = others_probs.argmax(dim=1)                    # (N,)
        agreement_scores[i] = (my_pred == others_label).float().mean()

    weights = F.softmax(agreement_scores * sharpness, dim=0)         # (K,)
    w_expanded = weights.view(num_models, 1, 1)                      # (K,1,1)

    log_probs = torch.log(tensor_probs)                               # (K,N,C)
    log_p_hat = torch.log(p_hat).view(num_models, 1, num_classes)     # (K,1,C)

    # 3) EM over q (Momentum only)
    q = torch.ones(num_classes, device=device) / float(num_classes)   # init uniform

    for _ in range(em_iters):
        log_q = torch.log(torch.clamp(q, min=1e-12)).view(1, 1, num_classes)  # (1,1,C)

        bias_term = log_p_hat - log_q                                 # (K,1,C)
        adj_logits = log_probs - (tau * bias_term)                    # (K,N,C)
        adj_probs = torch.softmax(adj_logits, dim=2)                  # (K,N,C)

        ensemble_probs = (adj_probs * w_expanded).sum(dim=0)          # (N,C)
        ensemble_probs = torch.clamp(ensemble_probs, min=1e-12)

        q_calc = ensemble_probs.mean(dim=0)
        q_calc = q_calc / q_calc.sum()

        if use_momentum:
            a = momentum_alpha
            q = (1.0 - a) * q + a * q_calc
            q = q / q.sum()
        else:
            q = q_calc

    return p_hat, weights, q

# ==============================================================================
# FASOLA EM runner
# ==============================================================================

def run_fasola(
    original_models,
    loader,
    device,
    tau=2.0,
    sharpness=10.0,
    em_iters=5,
    momentum_alpha=0.2,
    use_momentum=True,
):
    """
    Run the full FASOLA adaptation pipeline.
    
    1. Apply AdaBN (BN stats update only).
    2. Compute p_hat, weights, and estimate q via Momentum EM.
    3. Perform final inference using posterior adjustment and weighted ensemble.

    Args:
        original_models (list): List of pre-trained source models.
        loader (DataLoader): Target data loader.
        device (torch.device): Computation device.
        tau (float): Temperature for posterior adjustment.
        sharpness (float): Sharpness parameter for weighting.
        em_iters (int): Number of EM iterations.
        momentum_alpha (float): Momentum coefficient.
        use_momentum (bool): Whether to use momentum.

    Returns:
        tuple: (accuracy, f1_score)
    """
    # 1) AdaBN
    models = apply_adabn(original_models, loader, device)
    num_models = len(models)
    num_classes = infer_num_classes(models, loader, device)

    # 2) Stats & EM
    p_hat, weights, q = get_fasola_stats(
        models,
        loader,
        device,
        tau=tau,
        sharpness=sharpness,
        em_iters=em_iters,
        momentum_alpha=momentum_alpha,
        use_momentum=use_momentum,
    )

    log_q = torch.log(torch.clamp(q, min=1e-12)).view(1, num_classes)      # (1,C)
    log_p_hat = torch.log(torch.clamp(p_hat, min=1e-12))                    # (K,C)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for bx, by in tqdm(loader, desc="[FASOLA] Final Inference"):
            bx = bx.to(device, non_blocking=True)
            all_targets.append(by.numpy())

            batch_probs_acc = torch.zeros(bx.size(0), num_classes, device=device)

            for i, model in enumerate(models):
                logits = model(bx)                                         # (B,C)
                bias_term = log_p_hat[i].view(1, num_classes) - log_q       # (1,C)

                adj_logits = logits - (tau * bias_term)
                adj_probs = torch.softmax(adj_logits, dim=1)

                batch_probs_acc += adj_probs * weights[i]

            all_preds.append(batch_probs_acc.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    acc = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, average="macro") * 100
    return acc, f1

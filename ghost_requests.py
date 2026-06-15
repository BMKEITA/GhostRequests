# ghost_requests.py  —  Ghost Requests v6
# Corrections vs v5 :
#   [C1] model.forward_loss() → nn.CrossEntropyLoss()(model(X), y)  (×3)
#   [C2] ghost_mug() retourne maintenant (X_ghost, best_alpha, history, mal_grad_rescaled)
#   [C3] Bootstrap Rescaling IQR-clamped integre dans ghost_mug()
#   [C4] Support multi-samples influents (X_inf peut avoir N lignes)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─── HVP & LiSSA ──────────────────────────────────────────────────────────────

def hvp(model, X, y, v):
    """Hessian-Vector Product via autograd double-backward."""
    model.zero_grad()
    # [C1] Remplace model.forward_loss()
    loss      = nn.CrossEntropyLoss()(model(X), y)
    grads     = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])
    gv        = (flat_grad * v.detach()).sum()
    hvp_grads = torch.autograd.grad(gv, model.parameters())
    return torch.cat([g.reshape(-1) for g in hvp_grads]).detach()


def lissa_ihvp(model, X_train, y_train, v, depth=200, scale=50.0, batch_size=32):
    """LiSSA : approximation iterative de l'inverse Hessian-vector product."""
    h = v.clone().detach()
    n = X_train.size(0)
    for _ in range(depth):
        idx = torch.randperm(n)[:batch_size]
        hv  = hvp(model, X_train[idx], y_train[idx], h)
        h   = v + h - hv / scale
    return h / scale


def compute_influences(model, X_local, y_local, X_target, y_target, FL_params):
    """Calcule les scores d'influence de chaque sample local sur la cible."""
    device = FL_params.device
    model.zero_grad()
    # [C1] Remplace model.forward_loss()
    loss_t  = nn.CrossEntropyLoss()(model(X_target.to(device)), y_target.to(device))
    grads_t = torch.autograd.grad(loss_t, model.parameters())
    v       = torch.cat([g.reshape(-1) for g in grads_t]).detach()

    ihvp = lissa_ihvp(
        model,
        X_local.to(device),
        y_local.to(device),
        v,
        depth=getattr(FL_params, 'lissa_depth', 200),
    )

    scores = []
    for i in range(len(X_local)):
        xi = X_local[i:i+1].to(device)
        yi = y_local[i:i+1].to(device)
        model.zero_grad()
        # [C1] Remplace model.forward_loss()
        loss_i   = nn.CrossEntropyLoss()(model(xi), yi)
        g_i      = torch.autograd.grad(loss_i, model.parameters())
        g_i_flat = torch.cat([g.reshape(-1) for g in g_i]).detach()
        scores.append(-(ihvp * g_i_flat).sum().item())

    return np.array(scores)


def isi(model, X_local, y_local, X_target, y_target, FL_params, ratio=0.5):
    """
    Influence Sample Identification (ISI).
    Retourne les indices des samples les plus influents sur la cible.
    """
    scores   = compute_influences(
        model, X_local, y_local, X_target, y_target, FL_params
    )
    n_select = max(1, int(len(scores) * ratio))
    top_idx  = np.argsort(scores)[:n_select]

    # Preferer les samples de même label que la cible
    target_label = y_target.item()
    same_label   = [i for i in top_idx if y_local[i].item() == target_label]
    if len(same_label) == 0:
        same_label = top_idx[:1].tolist()

    return same_label, scores


# ─── Composantes de la Loss de Furtivite ──────────────────────────────────────

def L_norm(grad_mal, benign_stats):
    """Penalise si la norme depasse Q3 des benins."""
    return F.relu(grad_mal.norm() - benign_stats['q3'])


def L_dist(grad_mal, benign_stats):
    """Distance normalisee à la moyenne benigne (z-score²)."""
    mu    = benign_stats['mean_norm']
    sigma = benign_stats['std_norm'] + 1e-8
    return ((grad_mal.norm() - mu) / sigma) ** 2


def L_dist_kl(grad_mal, benign_stats):
    """Approximation KL divergence sur la norme."""
    mu    = benign_stats['mean_norm']
    sigma = benign_stats['std_norm'] + 1e-8
    norm  = grad_mal.norm()
    kl    = (torch.log(torch.tensor(sigma, dtype=torch.float32,
                                    device=grad_mal.device))
             + (1.0 + (norm - mu) ** 2) / (2.0 * sigma ** 2) - 0.5)
    return F.relu(kl)


def L_inf_stealth(grad_mal, benign_stats):
    """Distance à la norme d'influence moyenne."""
    return (grad_mal.norm() - benign_stats.get('mean_influence', 0.0)) ** 2


def L_cosine(grad_mal, mean_benign_grad):
    """Penalise si la direction est opposee aux benins."""
    cos = F.cosine_similarity(
        grad_mal.unsqueeze(0),
        mean_benign_grad.unsqueeze(0)
    )
    return F.relu(0.5 - cos)


def L_layer_dist(grad_mal_layers, benign_layer_stats):
    """Distance normalisee par couche."""
    device = grad_mal_layers[0].device
    total  = torch.zeros(1, device=device)
    for g_layer, stats in zip(grad_mal_layers, benign_layer_stats):
        mu_l    = stats['mean']
        sigma_l = stats['std'] + 1e-8
        norm_l  = g_layer.norm()
        total   = total + ((norm_l - mu_l) / sigma_l) ** 2
    return total / max(len(grad_mal_layers), 1)


def compute_L_stealth(grad_mal, benign_stats,
                      a1=1.0, a2=2.0, a3=0.5, a4=2.0, a5=1.5,
                      mean_benign_grad=None,
                      grad_mal_layers=None, benign_layer_stats=None):
    """Loss de furtivite combinee (ponderee)."""
    total = a1 * L_norm(grad_mal, benign_stats)
    total = total + a2 * L_dist(grad_mal, benign_stats)
    total = total + a3 * L_dist_kl(grad_mal, benign_stats)
    if mean_benign_grad is not None:
        total = total + a4 * L_cosine(grad_mal, mean_benign_grad)
    if grad_mal_layers is not None and benign_layer_stats is not None:
        total = total + a5 * L_layer_dist(grad_mal_layers, benign_layer_stats)
    return total


# ─── [C2/C3] Bootstrap Rescaling IQR-clamped ─────────────────────────────────

def bootstrap_rescale(grad: torch.Tensor, benign_stats: dict) -> torch.Tensor:
    """
    [C3] Rescale le gradient malveillant pour que sa norme soit
    dans l'IQR des gradients benins (Q1 ≤ norme ≤ Q3).

    Cible : mediane des benins, clampee dans [Q1, Q3].
    Garantit GNR ≈ 1.0 et DR_IQR = 0.0.
    """
    # Recuperer les normes brutes si disponibles, sinon reconstruire depuis stats
    if 'all_norms' in benign_stats:
        norms_np = np.array(benign_stats['all_norms'], dtype=np.float32)
    else:
        # Fallback : simuler une distribution gaussienne autour de mean/std
        mu  = benign_stats['mean_norm']
        std = benign_stats['std_norm'] + 1e-8
        q3  = benign_stats.get('q3', mu + std)
        q1  = max(0.0, mu - std)
        norms_np = np.array([q1, mu, q3], dtype=np.float32)

    q1          = float(np.percentile(norms_np, 25))
    q3          = float(np.percentile(norms_np, 75))
    med         = float(np.median(norms_np))
    target_norm = float(np.clip(med, q1, q3))

    current_norm = grad.norm().item()
    if current_norm < 1e-8:
        return grad

    rescaled  = grad * (target_norm / current_norm)
    final_gnr = rescaled.norm().item() / (float(np.mean(norms_np)) + 1e-8)
    print(f"  [Bootstrap] norm: {current_norm:.4f} → {rescaled.norm().item():.4f} "
          f"| target={target_norm:.4f} | GNR={final_gnr:.4f}")
    return rescaled


# ─── Ghost MUG ────────────────────────────────────────────────────────────────

def ghost_mug(model, X_target, y_target, X_inf, y_inf,
              benign_stats, FL_params, mean_benign_grad=None,
              X_benign_pool=None, y_benign_pool=None):
    """
    Ghost MUG v6 — Solo-Optimization + Bootstrap Rescaling [C2/C3]

    Optimise alpha pour construire :
        X_ghost = alpha * X_target + (1 - alpha) * mean(X_inf)
    tel que :
        1. Le modèle classe X_ghost comme y_inf (ASR = 1.0)
        2. Le gradient de X_ghost ressemble aux gradients benins (stealth)
        3. [C3] Le gradient final est rescale dans l'IQR benin

    [C4] X_inf peut contenir plusieurs samples (N, C, H, W) → moyenne utilisee.

    Retourne :
        X_ghost_final    : image ghost (tensor)
        best_alpha       : float, alpha optimal
        history          : liste des loss par step
        mal_grad_rescaled: gradient malveillant rescale (tensor 1D)
    """
    device = next(model.parameters()).device

    X_target = X_target.to(device)
    y_target = y_target.to(device)
    X_inf    = X_inf.to(device)
    y_inf    = y_inf.to(device)
    if mean_benign_grad is not None:
        mean_benign_grad = mean_benign_grad.to(device)

    # [C4] Si plusieurs samples influents, utiliser leur moyenne comme ancre
    if X_inf.dim() == 4 and X_inf.size(0) > 1:
        X_inf_anchor = X_inf.mean(0, keepdim=True)   # (1, C, H, W)
        y_inf_anchor = y_inf[0:1]
    else:
        X_inf_anchor = X_inf[0:1]
        y_inf_anchor = y_inf[0:1]

    lambda_s = getattr(FL_params, 'ghost_lambda_stealth', 1.5)
    n_steps  = getattr(FL_params, 'ghost_mug_steps',     200)
    a4       = getattr(FL_params, 'ghost_alpha4',         2.0)

    # ── Optimisation de alpha ──────────────────────────────────────────────────
    alpha     = torch.tensor([0.5], dtype=torch.float32,
                              device=device, requires_grad=True)
    optimizer = torch.optim.Adam([alpha], lr=0.01)

    history    = []
    best_alpha = 0.5
    best_loss  = float('inf')

    model.eval()
    for step in range(n_steps):
        optimizer.zero_grad()

        a_clamped = alpha.clamp(0.0, 1.0)
        X_ghost   = a_clamped * X_target + (1.0 - a_clamped) * X_inf_anchor

        # Loss 1 : classification correcte de X_ghost → ASR
        L_cls = nn.CrossEntropyLoss()(model(X_ghost), y_inf_anchor)

        # Loss 2 : furtivite du gradient
        model.zero_grad()
        loss_g = nn.CrossEntropyLoss()(model(X_ghost), y_inf_anchor)
        grads  = torch.autograd.grad(
            loss_g, model.parameters(),
            create_graph=True, allow_unused=True
        )
        grad_mal = torch.cat([
            g.reshape(-1) if g is not None
            else torch.zeros(p.numel(), device=device)
            for g, p in zip(grads, model.parameters())
        ])

        # Gradients par couche pour L_layer_dist
        grad_layers = [
            g.reshape(-1).detach() if g is not None
            else torch.zeros(p.numel(), device=device)
            for g, p in zip(grads, model.parameters())
        ]
        layer_stats = benign_stats.get('layer_stats', None)

        L_s = compute_L_stealth(
            grad_mal, benign_stats,
            a1=1.0, a2=2.0, a3=0.5, a4=a4, a5=1.5,
            mean_benign_grad=mean_benign_grad,
            grad_mal_layers=grad_layers,
            benign_layer_stats=layer_stats,
        )

        loss = L_cls + lambda_s * L_s
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        history.append(loss_val)
        if loss_val < best_loss:
            best_loss  = loss_val
            best_alpha = alpha.clamp(0.0, 1.0).item()

    # ── Construction de X_ghost final ─────────────────────────────────────────
    with torch.no_grad():
        X_ghost_final = (best_alpha * X_target
                         + (1.0 - best_alpha) * X_inf_anchor)

    # ── [C2/C3] Calcul du gradient brut puis Bootstrap Rescaling ──────────────
    model.eval()
    model.zero_grad()
    with torch.enable_grad():
        loss_final = nn.CrossEntropyLoss()(
            model(X_ghost_final.detach().requires_grad_(False)),
            y_inf_anchor
        )
        grads_final = torch.autograd.grad(
            loss_final, model.parameters(), allow_unused=True
        )

    mal_grad_raw = torch.cat([
        g.reshape(-1) if g is not None
        else torch.zeros(p.numel(), device=device)
        for g, p in zip(grads_final, model.parameters())
    ]).detach()

    # Rescaling IQR-clamped
    mal_grad_rescaled = bootstrap_rescale(mal_grad_raw, benign_stats)

    return X_ghost_final.to(device), best_alpha, history, mal_grad_rescaled

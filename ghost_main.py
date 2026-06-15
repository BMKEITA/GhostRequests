# ghost_main.py  -  Ghost Requests v6.1
# Fixes v6:
#   [F1] Seeds globales pour reproductibilite
#   [F2] Bootstrap rescaling renforce (IQR-clamped)
#   [F3] lambda_stealth adaptatif par agregateur
#   [F4] MUG iterations adaptatives par dataset
# Fixes v6.1:
#   [P1] fl_train : acc_snap supprime (overhead inutile)
#   [P2] ghost_mug : label majoritaire pour loss_ste (inf_ys coherent)

import os
import copy
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms, models
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────
#  [F1] Seed globale
# ─────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
#  [F3] Lambda stealth adaptatif par agregateur
# ─────────────────────────────────────────────
LAMBDA_STEALTH_MAP = {
    "fedavg":    1.5,
    "median":    1.5,
    "trim-mean": 2.0,
    "krum":      1.0,
}

def get_lambda_stealth(aggregation: str) -> float:
    return LAMBDA_STEALTH_MAP.get(aggregation.lower(), 1.5)


# ─────────────────────────────────────────────
#  [F4] Config MUG adaptative par dataset
# ─────────────────────────────────────────────
MUG_CONFIG = {
    "mnist":   {"n_iter": 80,  "lr": 0.01,  "tol": 1e-4},
    "cifar10": {"n_iter": 200, "lr": 0.005, "tol": 1e-5},
}

def get_mug_config(dataset: str) -> dict:
    key = "cifar10" if "cifar" in dataset.lower() else "mnist"
    return MUG_CONFIG[key]


# ─────────────────────────────────────────────
#  Parametres FL
# ─────────────────────────────────────────────
@dataclass
class FLParams:
    dataset:               str   = "mnist"
    iid:                   bool  = True
    aggregation:           str   = "fedavg"
    n_total_clients:       int   = 100
    n_clients_per_round:   int   = 10
    global_epochs:         int   = 20
    local_epochs:          int   = 5
    local_lr:              float = 0.01
    local_batch_size:      int   = 64
    n_attackers:           int   = 1
    max_samples_per_client: int  = 500
    ghost_ratio:           float = 0.003
    ghost_lambda_stealth:  float = 1.5   # sera ecrase par get_lambda_stealth()
    krum_f:                int   = 1
    trim_ratio:            float = 0.1
    result_dir:            str   = "./ghost_result"
    seed:                  int   = 42
    # Rempli automatiquement
    device:                str   = field(default="cuda" if torch.cuda.is_available() else "cpu", init=False)


# ─────────────────────────────────────────────
#  Modeles
# ─────────────────────────────────────────────
class LeNet5(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), nn.Tanh(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5), nn.Tanh(),
            nn.AvgPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), nn.Tanh(),
            nn.Linear(120, 84),         nn.Tanh(),
            nn.Linear(84, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ResNet18CIFAR(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc = nn.Linear(base.fc.in_features, n_classes)
        self.net = base

    def forward(self, x):
        return self.net(x)


def build_model(params: FLParams) -> nn.Module:
    if params.dataset == "mnist":
        model = LeNet5()
    else:
        model = ResNet18CIFAR()
    total = sum(p.numel() for p in model.parameters())
    print(f" Modele : {'LeNet5' if params.dataset=='mnist' else 'ResNet18CIFAR'} ({total:,} parametres)")
    return model.to(params.device)


# ─────────────────────────────────────────────
#  Donnees
# ─────────────────────────────────────────────
def load_data(params: FLParams):
    if params.dataset == "mnist":
        tf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
        train_ds = datasets.MNIST("./data", train=True,  download=True, transform=tf)
        test_ds  = datasets.MNIST("./data", train=False, download=True, transform=tf)
    else:
        tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
        ])
        train_ds = datasets.CIFAR10("./data", train=True,  download=True, transform=tf_train)
        test_ds  = datasets.CIFAR10("./data", train=False, download=True, transform=tf_test)
    return train_ds, test_ds


def partition_data(train_ds, params: FLParams):
    n = len(train_ds)
    indices = list(range(n))
    random.shuffle(indices)
    size = n // params.n_total_clients

    if params.iid:
        return [indices[i * size:(i + 1) * size] for i in range(params.n_total_clients)]
    else:
        # Non-IID : tri par label, decoupage en shards
        labels = np.array([train_ds[i][1] for i in range(n)])
        sorted_idx = np.argsort(labels).tolist()
        n_shards = params.n_total_clients * 2
        shard_size = n // n_shards
        shards = [sorted_idx[i * shard_size:(i + 1) * shard_size] for i in range(n_shards)]
        random.shuffle(shards)
        client_data = []
        for i in range(params.n_total_clients):
            client_data.append(shards[2 * i] + shards[2 * i + 1])
        return client_data


# ─────────────────────────────────────────────
#  Agregateurs
# ─────────────────────────────────────────────
def aggregate_fedavg(global_model, client_states):
    gsd = global_model.state_dict()
    for k in gsd:
        gsd[k] = torch.stack([cs[k].float() for cs in client_states]).mean(0)
    global_model.load_state_dict(gsd)


def aggregate_median(global_model, client_states):
    gsd = global_model.state_dict()
    for k in gsd:
        gsd[k] = torch.stack([cs[k].float() for cs in client_states]).median(0).values
    global_model.load_state_dict(gsd)


def aggregate_trimmean(global_model, client_states, trim_ratio=0.1):
    gsd = global_model.state_dict()
    n = len(client_states)
    k = max(1, int(n * trim_ratio))
    for key in gsd:
        stacked = torch.stack([cs[key].float() for cs in client_states])
        sorted_t, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_t[k:n - k]
        gsd[key] = trimmed.mean(0)
    global_model.load_state_dict(gsd)


def aggregate_krum(global_model, client_states, f=1):
    n = len(client_states)
    flat = [torch.cat([p.float().reshape(-1) for p in cs.values()]) for cs in client_states]
    scores = []
    for i in range(n):
        dists = sorted([torch.dist(flat[i], flat[j]).item() for j in range(n) if j != i])
        scores.append(sum(dists[:n - f - 2]))
    best = int(np.argmin(scores))
    global_model.load_state_dict(client_states[best])


def aggregate(global_model, client_states, params: FLParams):
    agg = params.aggregation.lower()
    if agg == "fedavg":
        aggregate_fedavg(global_model, client_states)
    elif agg == "median":
        aggregate_median(global_model, client_states)
    elif agg == "trim-mean":
        aggregate_trimmean(global_model, client_states, params.trim_ratio)
    elif agg == "krum":
        aggregate_krum(global_model, client_states, params.krum_f)
    else:
        raise ValueError(f"Agregateur inconnu : {params.aggregation}")


# ─────────────────────────────────────────────
#  Entraînement local
# ─────────────────────────────────────────────
def local_train(global_model, data_indices, train_ds, params: FLParams):
    model = copy.deepcopy(global_model)
    model.train()
    loader = DataLoader(
        Subset(train_ds, data_indices),
        batch_size=params.local_batch_size,
        shuffle=True,
        drop_last=False,
    )
    optimizer = optim.SGD(model.parameters(), lr=params.local_lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    for _ in range(params.local_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(params.device), yb.to(params.device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
    state = copy.deepcopy(model.state_dict())
    del model
    return state


# ─────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, test_ds, params: FLParams) -> float:
    model.eval()
    loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(params.device), yb.to(params.device)
        preds = model(xb).argmax(1)
        correct += (preds == yb).sum().item()
        total   += yb.size(0)
    return correct / total


# ─────────────────────────────────────────────
#  Entraînement FL + Snapshots
#  [P1] acc_snap supprime : overhead inutile
#       Les snapshots stockent uniquement le state_dict
# ─────────────────────────────────────────────
def fl_train(global_model, train_ds, test_ds, params: FLParams):
    client_data = partition_data(train_ds, params)
    snapshots = {}
    snap_fracs = {"snap_25": 0.25, "snap_50": 0.50, "snap_75": 0.75}
    snapped = set()
    acc_final = 0.0

    for epoch in range(1, params.global_epochs + 1):
        # [P1] Snapshots : stockage state_dict uniquement, sans evaluate()
        frac = epoch / params.global_epochs
        for sname, sfrac in snap_fracs.items():
            if sname not in snapped and frac >= sfrac:
                snapshots[sname] = copy.deepcopy(global_model.state_dict())
                print(f"  [Snapshot {sname}] epoch {epoch}/{params.global_epochs} (frac={sfrac:.2f})")
                snapped.add(sname)

        # Selection clients
        chosen = random.sample(range(params.n_total_clients), params.n_clients_per_round)
        client_states = []
        for cid in chosen:
            state = local_train(global_model, client_data[cid], train_ds, params)
            client_states.append(state)
            torch.cuda.empty_cache()

        aggregate(global_model, client_states, params)
        del client_states

        if epoch % 5 == 0 or epoch == params.global_epochs:
            acc = evaluate(global_model, test_ds, params)
            acc_final = acc
            print(f"  Epoch {epoch:3d}/{params.global_epochs} [{params.aggregation}] | AccG = {acc*100:.2f}%")

    print(f" AccG finale : {acc_final*100:.2f}%")

    # Sauvegarde snapshots
    os.makedirs(params.result_dir, exist_ok=True)
    for sname, sstate in snapshots.items():
        torch.save(sstate, os.path.join(params.result_dir, f"{sname}.pt"))
    print(f"  [Snapshots] {list(snapshots.keys())} sauvegardes dans {params.result_dir}/")

    return global_model, acc_final, snapshots


# ─────────────────────────────────────────────
#  ISI - Influence Sample Identification (LiSSA)
# ─────────────────────────────────────────────
def compute_grad(model, x, y, criterion, device):
    model.zero_grad()
    loss = criterion(model(x.unsqueeze(0).to(device)), torch.tensor([y]).to(device))
    loss.backward()
    return torch.cat([
        p.grad.reshape(-1)
        for p in model.parameters()
        if p.grad is not None
    ]).detach()


def lissa_ihvp(model, v, train_loader, criterion, device,
               n_recursion=10, damping=0.01, scale=10.0):
    h = v.clone()
    for _ in range(n_recursion):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            model.zero_grad()
            loss = criterion(model(xb), yb)
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            flat_g = torch.cat([g.reshape(-1) for g in grads])
            hvp = torch.autograd.grad(flat_g.dot(h.detach()), model.parameters())
            flat_hvp = torch.cat([
                g.reshape(-1) for g in hvp
            ]).detach()
            h = v + (1 - damping) * h - flat_hvp / scale
            break  # 1 batch par recursion (vitesse vs qualite IHVP)
    return h


def isi_select(model, train_ds, target_x, target_y, params: FLParams,
               n_recursion=10, n_top: int = 50):
    t0 = time.time()
    device = params.device
    criterion = nn.CrossEntropyLoss()
    model.eval()

    target_grad = compute_grad(model, target_x, target_y, criterion, device)

    loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    ihvp = lissa_ihvp(model, target_grad, loader, criterion, device, n_recursion)

    # Sous-echantillonnage aleatoire pour accelerer (max 500 candidats)
    n_candidates = min(500, len(train_ds))
    candidate_indices = random.sample(range(len(train_ds)), n_candidates)

    scores = []
    for i in candidate_indices:
        xi, yi = train_ds[i]
        gi = compute_grad(model, xi, yi, criterion, device)
        score = float(ihvp.dot(gi).item())
        scores.append((score, i))

    scores.sort(key=lambda x: x[0], reverse=True)
    k = max(1, min(n_top, len(scores)))
    selected = [idx for _, idx in scores[:k]]
    elapsed = time.time() - t0
    print(f"   {len(selected)} samples influents | ISI time : {elapsed:.2f}s")
    return selected, elapsed


# ─────────────────────────────────────────────
#  [F2] Bootstrap Rescaling renforce (IQR-clamped)
# ─────────────────────────────────────────────
def bootstrap_rescale(grad: torch.Tensor, ben_norms: torch.Tensor) -> torch.Tensor:
    """
    [F2] Rescale le gradient malveillant pour que sa norme
    soit dans l'IQR des normes benignes.
    """
    if isinstance(ben_norms, torch.Tensor):
        norms_np = ben_norms.cpu().numpy()
    else:
        norms_np = np.array(ben_norms)

    if len(norms_np) == 0:
        print("  [WARN] bootstrap_rescale: ben_norms vide !")
        return grad

    q1  = float(np.percentile(norms_np, 25))
    q3  = float(np.percentile(norms_np, 75))
    med = float(np.median(norms_np))

    # Cible : mediane des benins (IQR-clamped entre Q1 et Q3)
    target_norm = np.clip(med, q1, q3)

    current_norm = grad.norm().item()
    if current_norm < 1e-8:
        return grad

    scale = target_norm / current_norm
    rescaled = grad * scale

    # Verification
    new_norm = rescaled.norm().item()
    gnr = new_norm / (float(np.mean(norms_np)) + 1e-8)
    print(f"  [Bootstrap] norm: {current_norm:.3f} → {new_norm:.3f} | target={target_norm:.3f} | GNR={gnr:.3f}")

    return rescaled


def compute_benign_norms(global_model, train_ds, client_data,
                          chosen_clients, params: FLParams) -> torch.Tensor:
    """Calcule les normes de gradient des clients benins selectionnes."""
    criterion = nn.CrossEntropyLoss()
    norms = []
    for cid in chosen_clients:
        model = copy.deepcopy(global_model).to(params.device)
        model.train()
        loader = DataLoader(
            Subset(train_ds, client_data[cid]),
            batch_size=params.local_batch_size, shuffle=True
        )
        optimizer = optim.SGD(model.parameters(), lr=params.local_lr, momentum=0.9)
        for xb, yb in loader:
            xb, yb = xb.to(params.device), yb.to(params.device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            norm = torch.cat([p.grad.reshape(-1) for p in model.parameters()
                               if p.grad is not None]).norm().item()
            norms.append(norm)
            break
        del model
        torch.cuda.empty_cache()
    return torch.tensor(norms)


# ─────────────────────────────────────────────
#  Ghost MUG v6.1 - Solo-Optimization
#  [P2] Label majoritaire pour loss_ste (coherence inf_ys)
# ─────────────────────────────────────────────
def ghost_mug(global_model, target_x, target_y, inf_indices,
              train_ds, params, ben_norms=None):
    """
    Ghost MUG v6.1 — Solo-Optimization + Bootstrap Rescaling [F2]

    [P2] loss_ste utilise le label majoritaire des samples influents
         pour une contrainte de furtivite coherente.

    Retourne (best_alpha, mal_grad_rescaled, mug_time)

    Note : l'ASR retourne par ghost_main est binaire (0 ou 1) car
           il est evalue sur un seul sample cible. L'ASR moyen
           multi-cibles est calcule dans run_ablation_ratio().
    """
    t0  = time.time()
    cfg = get_mug_config(params.dataset)
    device = params.device

    criterion = nn.CrossEntropyLoss()
    lam       = params.ghost_lambda_stealth

    # ── Donnees influentes ────────────────────────────────────
    inf_xs = torch.stack([train_ds[i][0] for i in inf_indices]).to(device)
    inf_ys = torch.tensor([train_ds[i][1] for i in inf_indices],
                           dtype=torch.long).to(device)

    target_x_dev = target_x.unsqueeze(0).to(device)
    target_y_dev = torch.tensor([target_y], dtype=torch.long).to(device)

    # [P2] Label majoritaire parmi les samples influents
    majority_label = int(torch.mode(inf_ys).values.item())
    majority_y_dev = torch.tensor([majority_label], dtype=torch.long).to(device)

    # ── Optimisation alpha (Solo-Optimization) ────────────────
    alpha = torch.tensor(0.5, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([alpha], lr=cfg["lr"])

    best_alpha = 0.5
    best_loss  = float("inf")

    for _ in range(cfg["n_iter"]):
        optimizer.zero_grad()

        # X_ghost = alpha * X_target + (1 - alpha) * X_inf_mean
        x_inf_mean = inf_xs.mean(0, keepdim=True)
        alpha_c    = torch.clamp(alpha, 0.0, 1.0)
        x_ghost    = alpha_c * target_x_dev + (1.0 - alpha_c) * x_inf_mean

        # Loss attaque : maximiser proba de la classe cible
        logits_t = global_model(x_ghost)
        loss_atk = criterion(logits_t, target_y_dev)

        # [P2] Loss furtivite : label majoritaire (coherent avec la distribution influente)
        logits_i = global_model(x_inf_mean)
        loss_ste = criterion(logits_i, majority_y_dev)

        loss = loss_atk - lam * loss_ste
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_alpha = float(torch.clamp(alpha, 0.0, 1.0).item())

    # ── Construction X_ghost final ────────────────────────────
    with torch.no_grad():
        x_inf_mean = inf_xs.mean(0, keepdim=True)
        alpha_c    = torch.clamp(torch.tensor(best_alpha, device=device), 0.0, 1.0)
        x_ghost    = alpha_c * target_x_dev + (1.0 - alpha_c) * x_inf_mean
        y_ghost    = target_y_dev

    # ── Calcul gradient malveillant brut ─────────────────────
    model_tmp = copy.deepcopy(global_model)
    model_tmp.train()
    model_tmp.zero_grad()
    loss_raw = criterion(model_tmp(x_ghost), y_ghost)
    loss_raw.backward()
    mal_grad_raw = torch.cat([
        p.grad.reshape(-1)
        for p in model_tmp.parameters()
        if p.grad is not None
    ]).detach()
    del model_tmp
    torch.cuda.empty_cache()

    # ── [F2] Bootstrap Rescaling IQR-clamped ─────────────────
    if ben_norms is not None and len(ben_norms) > 0:
        mal_grad = bootstrap_rescale(mal_grad_raw, ben_norms)
    else:
        print("  [WARN] ben_norms vide — rescaling ignore !")
        mal_grad = mal_grad_raw

    t_mug = time.time() - t0
    print(f"  Alpha optimal : {best_alpha:.4f} | MUG time : {t_mug:.2f}s")
    return best_alpha, mal_grad, t_mug


# ─────────────────────────────────────────────
#  Metriques de furtivite
# ─────────────────────────────────────────────
def compute_stealth_metrics(mal_grad: torch.Tensor,
                             ben_grads: list,
                             ben_norms: torch.Tensor) -> dict:
    mal_norm = mal_grad.norm().item()
    ben_norms_cpu = ben_norms.float().cpu()
    ben_mu  = ben_norms_cpu.mean().item()
    ben_std = ben_norms_cpu.std().item() + 1e-8

    # ── GNR ──────────────────────────────────────────────────
    gnr = mal_norm / (ben_mu + 1e-8)

    # ── DR_IQR ───────────────────────────────────────────────
    q1 = torch.quantile(ben_norms_cpu, 0.25).item()
    q3 = torch.quantile(ben_norms_cpu, 0.75).item()
    in_iqr  = float(q1 <= mal_norm <= q3)
    dr_iqr  = round(1.0 - in_iqr, 3)   # 0.0 = dans l'IQR (bon), 1.0 = hors IQR

    # ── DR_KL ────────────────────────────────────────────────
    dr_kl = float(np.clip(abs(mal_norm - ben_mu) / (3.0 * ben_std), 0.0, 1.0))

    # ── SI ───────────────────────────────────────────────────
    gnr_penalty = float(np.clip(abs(gnr - 1.0) / 2.0, 0.0, 1.0))
    si = 1.0 - (gnr_penalty + dr_iqr + dr_kl) / 3.0
    si = float(np.clip(si, 0.0, 1.0))

    return {
        "gnr":    round(gnr,   3),
        "dr_iqr": round(dr_iqr, 3),
        "dr_kl":  round(dr_kl,  3),
        "si":     round(si,     3),
    }


# ─────────────────────────────────────────────
#  ASR - Attack Success Rate
#  Note : retourne 0.0 ou 1.0 (binaire, single-sample)
#         L'ASR moyen multi-cibles est calcule dans
#         run_ablation_ratio() de run_experiments.py
# ─────────────────────────────────────────────
@torch.no_grad()
def compute_asr(model, target_x, target_y, params: FLParams) -> float:
    model.eval()
    x = target_x.unsqueeze(0).to(params.device)
    pred = model(x).argmax(1).item()
    return 1.0 if pred == target_y else 0.0


# ─────────────────────────────────────────────
#  Injection du gradient malveillant
# ─────────────────────────────────────────────
def inject_malicious_gradient(global_model, mal_grad: torch.Tensor,
                               params: FLParams):
    """
    Reconstruit un state_dict malveillant a partir du gradient plat.
    Simule un client attaquant qui soumet ce state_dict.
    """
    mal_model = copy.deepcopy(global_model)
    offset = 0
    with torch.no_grad():
        for p in mal_model.parameters():
            numel = p.numel()
            delta = mal_grad[offset:offset + numel].view(p.shape).to(params.device)
            p.data -= params.local_lr * delta
            offset += numel
    return mal_model.state_dict()


# ─────────────────────────────────────────────
#  Fonction principale ghost_main
# ─────────────────────────────────────────────
def ghost_main(params: FLParams,
               target_idx: int = 0,
               proxy_state_dict=None,
               verbose: bool = True):
    """
    Execute Ghost Requests v6.1 complet.

    Args:
        params           : FLParams configures
        target_idx       : index du sample cible dans test_ds
        proxy_state_dict : si fourni, utilise ce state_dict comme proxy (Grey-Box)
        verbose          : affichage detaille

    Returns:
        (result_dict, snapshots, poisoned_model)
        result_dict contient : exp, acc_global, peak_asr (binaire 0/1),
                               alpha, gnr, dr_iqr, dr_kl, si, status,
                               isi_time, mug_time, lambda_stealth
    """
    set_seed(params.seed)

    # [F3] Lambda adaptatif
    params.ghost_lambda_stealth = get_lambda_stealth(params.aggregation)

    os.makedirs(params.result_dir, exist_ok=True)
    device = params.device

    exp_name = f"{params.dataset}_{'iid' if params.iid else 'noniid'}_{params.aggregation}"
    exp_name = exp_name.upper()

    print("=" * 60)
    print(f"  Ghost Requests v6.1 - {exp_name}")
    print(f"  Device : {device}")
    print("=" * 60)

    # ── Donnees & modele ──────────────────────────────────────
    train_ds, test_ds = load_data(params)
    global_model = build_model(params)

    # ── Entraînement FL ───────────────────────────────────────
    print(f"\n Entrainement FL ({params.global_epochs} epoques)...")
    global_model, acc_final, snapshots = fl_train(
        global_model, train_ds, test_ds, params
    )

    # ── Proxy (Grey-Box ou White-Box) ─────────────────────────
    if proxy_state_dict is not None:
        proxy_model = build_model(params)
        proxy_model.load_state_dict(proxy_state_dict)
        proxy_model = proxy_model.to(device)
    else:
        proxy_model = global_model

    # ── Cible ─────────────────────────────────────────────────
    target_x, target_y = test_ds[target_idx]
    print(f"\n Cible : index={target_idx}, label={target_y}")

    # ── ISI ───────────────────────────────────────────────────
    print(f"\n ISI (LiSSA)...")
    client_data = partition_data(train_ds, params)
    inf_indices, isi_time = isi_select(
        proxy_model, train_ds, target_x, target_y, params
    )

    # ── Normes benignes (pour bootstrap) ──────────────────────
    chosen_ref = random.sample(range(params.n_total_clients), params.n_clients_per_round)
    ben_norms = compute_benign_norms(
        global_model, train_ds, client_data, chosen_ref, params
    )
    print(f"  [DEBUG] ben_norms shape={ben_norms.shape}, mean={ben_norms.mean():.4f}, min={ben_norms.min():.4f}")

    # ── Ghost MUG ─────────────────────────────────────────────
    print(f"\n Ghost MUG v6.1...")
    best_alpha, mal_grad, mug_time = ghost_mug(
        proxy_model, target_x, target_y,
        inf_indices, train_ds, params,
        ben_norms=ben_norms
    )

    # ── Injection & evaluation ────────────────────────────────
    mal_state = inject_malicious_gradient(global_model, mal_grad, params)

    # Simuler un round avec le gradient malveillant
    chosen_round = random.sample(range(params.n_total_clients), params.n_clients_per_round - 1)
    client_states = []
    for cid in chosen_round:
        state = local_train(global_model, client_data[cid], train_ds, params)
        client_states.append(state)
        torch.cuda.empty_cache()
    client_states.append(mal_state)

    poisoned_model = copy.deepcopy(global_model)
    aggregate(poisoned_model, client_states, params)
    del client_states

    # ASR (binaire : 0.0 ou 1.0 sur sample unique)
    asr = compute_asr(poisoned_model, target_x, target_y, params)

    # Metriques de furtivite
    metrics = compute_stealth_metrics(mal_grad, [], ben_norms.to(device))

    status = "Ok" if metrics["si"] >= 0.700 else "No"

    # ── Affichage ─────────────────────────────────────────────
    print(f"\n   {exp_name}")
    print(f"  Peak ASR : {asr:.3f}   |  AccG : {acc_final*100:.2f}%")
    print(f"  GNR={metrics['gnr']:.3f}  DR_IQR={metrics['dr_iqr']:.3f}"
          f"  DR_KL={metrics['dr_kl']:.3f}  SI={metrics['si']:.3f} {status}")
    print(f"    ISI={isi_time:.2f}s | MUG={mug_time:.2f}s")

    # ── Sauvegarde JSON ───────────────────────────────────────
    result = {
        "exp":        exp_name,
        "acc_global": round(acc_final * 100, 2),
        "peak_asr":   round(asr, 3),
        "alpha":      round(best_alpha, 4),
        "gnr":        metrics["gnr"],
        "dr_iqr":     metrics["dr_iqr"],
        "dr_kl":      metrics["dr_kl"],
        "si":         metrics["si"],
        "status":     status,
        "isi_time":   round(isi_time, 3),
        "mug_time":   round(mug_time, 3),
        "lambda_stealth": params.ghost_lambda_stealth,
    }

    out_path = os.path.join(
        params.result_dir,
        f"ghost_{exp_name.lower()}_metrics.json"
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"   {out_path}")

    return result, snapshots, poisoned_model

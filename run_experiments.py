# run_experiments.py 
# ============================================================
# [F1] Global seeds for full reproducibility 
# [F2] Bootstrap IQR-clamped (applied in ghost_main.py)
# [F3] Adaptive lambda stealth by aggregator 
# [F4] MUG adaptive iterations by dataset 
# [P1] fl_train: acc_snap removes (ghost_main.py) 
# [P2] ghost_mug: majority label (ghost_main.py) 
# [P3] run_ablation_n_attackers: budget actually applied
# ============================================================

import os
import copy
import json
import time
import random
import numpy as np
import torch

from ghost_main import (
    FLParams,
    ghost_main,
    set_seed,
    fl_train,
    load_data,
    build_model,
    evaluate,
    isi_select,
    ghost_mug,
    compute_stealth_metrics,
    compute_benign_norms,
    partition_data,
    inject_malicious_gradient,
    aggregate,
    local_train,
    compute_asr,
    get_lambda_stealth,
    bootstrap_rescale,
)

# ─────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────
RESULT_DIR   = "./ghost_result"
GLOBAL_SEED  = 42
TARGET_IDX   = 0
SI_THRESHOLD = 0.700

# References fixes (paper)
REF_BENIGN = {"acc": 95.12, "gnr": 1.000, "dr_iqr": 0.250, "dr_kl": 0.000, "si": 0.817}
REF_FEDMUA = {"acc": 95.87, "gnr": 2.100, "dr_iqr": 0.850, "dr_kl": 0.600, "si": 0.082}

os.makedirs(RESULT_DIR, exist_ok=True)
set_seed(GLOBAL_SEED)


# ─────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────
def print_env():
    print("=" * 60)
    print(f" PyTorch     : {torch.__version__}")
    print(f" CUDA dispo  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        idx   = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print(f" GPU utilise : {props.name}")
        print(f" VRAM        : {props.total_memory / 1e9:.1f} GB")
    print("=" * 60)
    print(" Modules charges\n")


def status_str(si: float) -> str:
    return "Ok" if si >= SI_THRESHOLD else "No"


def save_json(data, filename: str):
    path = os.path.join(RESULT_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f" Sauvegarde : {path}")
    return path


def build_params(cfg: dict, seed: int = GLOBAL_SEED) -> FLParams:
    """Construit FLParams depuis un dict de config avec lambda adaptatif."""
    p = FLParams(
        dataset          = cfg["dataset"],
        iid              = cfg["iid"],
        aggregation      = cfg["aggregation"],
        global_epochs    = cfg["global_epochs"],
        local_batch_size = cfg.get("local_batch_size", 64),
        local_lr         = cfg.get("local_lr", 0.01),
        result_dir       = RESULT_DIR,
        seed             = seed,
    )
    # [F3] Adaptive lambda applies systematically
    p.ghost_lambda_stealth = get_lambda_stealth(p.aggregation)
    return p


def run_stealth_eval(global_model, train_ds, target_x, target_y,
                     inf_indices, params, ben_norms):
    """
    Evaluate the stealth of the malicious gradient using Ghost MUG. 
    IMPORTANT: ben_norms must be calculated BEFOREHAND via compute_benign_norms
    """
    assert ben_norms is not None and len(ben_norms) > 0, \
        "ben_norms empty! call compute_benign_norms() befor run_stealth_eval"

    best_alpha, mal_grad, mug_time = ghost_mug(
        global_model, target_x, target_y,
        inf_indices, train_ds, params,
        ben_norms=ben_norms,
    )
    metrics = compute_stealth_metrics(mal_grad, [], ben_norms.to(params.device))
    return metrics, mal_grad, mug_time


def run_asr_injection(global_model, mal_grad, client_data,
                      train_ds, target_x, target_y, params):
    """
    Injects the malicious gradient into a FL round and returns the ASR
    """
    mal_state = inject_malicious_gradient(global_model, mal_grad, params)
    chosen = random.sample(
        range(params.n_total_clients),
        params.n_clients_per_round - 1
    )
    c_states = []
    for cid in chosen:
        s = local_train(global_model, client_data[cid], train_ds, params)
        c_states.append(s)
        torch.cuda.empty_cache()
    c_states.append(mal_state)

    poisoned = copy.deepcopy(global_model)
    aggregate(poisoned, c_states, params)
    asr = compute_asr(poisoned, target_x, target_y, params)

    del poisoned, c_states
    torch.cuda.empty_cache()
    return asr


# ─────────────────────────────────────────────────────────────
#  Configurations of 10 main experiences 
# ─────────────────────────────────────────────────────────────
MAIN_EXPERIMENTS = [
    # ── MNIST ──────────────────────────────────────────────
    dict(dataset="mnist",   iid=True,  aggregation="fedavg",
         global_epochs=20,  local_batch_size=64,  local_lr=0.01),
    dict(dataset="mnist",   iid=False, aggregation="fedavg",
         global_epochs=20,  local_batch_size=64,  local_lr=0.01),
    dict(dataset="mnist",   iid=True,  aggregation="median",
         global_epochs=20,  local_batch_size=64,  local_lr=0.01),
    dict(dataset="mnist",   iid=True,  aggregation="trim-mean",
         global_epochs=20,  local_batch_size=64,  local_lr=0.01),
    dict(dataset="mnist",   iid=True,  aggregation="krum",
         global_epochs=20,  local_batch_size=64,  local_lr=0.01),
    # ── CIFAR-10 ───────────────────────────────────────────
    dict(dataset="cifar10", iid=True,  aggregation="fedavg",
         global_epochs=150, local_batch_size=64,  local_lr=0.01),
    dict(dataset="cifar10", iid=False, aggregation="fedavg",
         global_epochs=300, local_batch_size=64,  local_lr=0.005),
    dict(dataset="cifar10", iid=True,  aggregation="median",
         global_epochs=150, local_batch_size=64,  local_lr=0.01),
    dict(dataset="cifar10", iid=True,  aggregation="trim-mean",
         global_epochs=150, local_batch_size=64,  local_lr=0.01),
    dict(dataset="cifar10", iid=True,  aggregation="krum",
         global_epochs=150, local_batch_size=64,  local_lr=0.01),
]


# ─────────────────────────────────────────────────────────────
#  Experiences  (10 configs)
# ─────────────────────────────────────────────────────────────
def run_main_experiments():
    all_results = []

    for cfg in MAIN_EXPERIMENTS:
        iid_tag = "IID" if cfg["iid"] else "NONIID"
        exp_tag = f"{cfg['dataset'].upper()}_{iid_tag}_{cfg['aggregation'].upper()}"
        print("\n" + "-" * 60)
        print(f"  EXP : {exp_tag}")
        print("-" * 60)

        params = build_params(cfg)
        result, snapshots, _ = ghost_main(params, target_idx=TARGET_IDX)
        all_results.append(result)

    return all_results


# ─────────────────────────────────────────────────────────────
#  Ablation 1 : lambda_stealth
# ─────────────────────────────────────────────────────────────
def run_ablation_lambda():
    print("\n" + "─" * 60)
    print("  Ablation 1 : lambda_stealth - MNIST IID FedAvg")
    print("─" * 60)

    lambdas = [0.5, 1.0, 1.5, 2.0, 3.0]
    results = []

    params_base = build_params(
        dict(dataset="mnist", iid=True, aggregation="fedavg",
             global_epochs=20, local_batch_size=64)
    )
    set_seed(GLOBAL_SEED)
    train_ds, test_ds = load_data(params_base)
    global_model      = build_model(params_base)
    global_model, _, _ = fl_train(global_model, train_ds, test_ds, params_base)

    target_x, target_y = test_ds[TARGET_IDX]
    client_data = partition_data(train_ds, params_base)
    chosen_ref  = random.sample(range(params_base.n_total_clients),
                                params_base.n_clients_per_round)
    ben_norms   = compute_benign_norms(
        global_model, train_ds, client_data, chosen_ref, params_base
    )
    inf_indices, _ = isi_select(
        global_model, train_ds, target_x, target_y, params_base
    )

    for lam in lambdas:
        set_seed(GLOBAL_SEED)
        params_lam = copy.deepcopy(params_base)
        params_lam.ghost_lambda_stealth = lam

        metrics, _, _ = run_stealth_eval(
            global_model, train_ds, target_x, target_y,
            inf_indices, params_lam, ben_norms
        )
        st     = status_str(metrics["si"])
        marker = "★" if lam == 1.5 else " "
        print(f"  l={lam:<4}{marker}  | SI={metrics['si']:.3f} | "
              f"GNR={metrics['gnr']:.3f} | DR_IQR={metrics['dr_iqr']:.3f} | "
              f"DR_KL={metrics['dr_kl']:.3f} {st}")
        results.append({"lambda": lam, **metrics, "status": st})

    save_json(results, "ghost_ablation_lambda.json")
    return results


# ─────────────────────────────────────────────────────────────
#  Ablation 2 : Malicious request ratio (multi-target)
# ─────────────────────────────────────────────────────────────
def run_ablation_ratio():
    print("\n" + "─" * 60)
    print("  Ablation 2 : Malicious request ratio - MNIST IID FedAvg")
    print("─" * 60)

    ratios    = [0.001, 0.003, 0.005, 0.007, 0.010]
    n_targets = 5
    results   = []

    params_base = build_params(
        dict(dataset="mnist", iid=True, aggregation="fedavg",
             global_epochs=20, local_batch_size=64)
    )
    set_seed(GLOBAL_SEED)
    train_ds, test_ds = load_data(params_base)
    global_model      = build_model(params_base)
    global_model, _, _ = fl_train(global_model, train_ds, test_ds, params_base)

    client_data = partition_data(train_ds, params_base)
    chosen_ref  = random.sample(range(params_base.n_total_clients),
                                params_base.n_clients_per_round)
    ben_norms   = compute_benign_norms(
        global_model, train_ds, client_data, chosen_ref, params_base
    )

    # 5 cibles aleatoires fixes (meme seed)
    set_seed(GLOBAL_SEED)
    target_indices = random.sample(range(len(test_ds)), n_targets)

    for ratio in ratios:
        set_seed(GLOBAL_SEED)
        params_r = copy.deepcopy(params_base)
        params_r.ghost_ratio            = ratio
        params_r.max_samples_per_client = max(
            1, int(len(train_ds) * ratio / params_r.n_total_clients)
        )

        asr_list    = []
        si_list     = []
        gnr_list    = []
        dr_iqr_list = []
        dr_kl_list  = []
        n_inf_last  = 0

        for tidx in target_indices:
            tx, ty = test_ds[tidx]
            inf_idx, _ = isi_select(global_model, train_ds, tx, ty, params_r)
            n_inf_last  = len(inf_idx)

            metrics, mal_grad, _ = run_stealth_eval(
                global_model, train_ds, tx, ty,
                inf_idx, params_r, ben_norms
            )
            asr = run_asr_injection(
                global_model, mal_grad, client_data,
                train_ds, tx, ty, params_r
            )
            asr_list.append(round(asr, 3))
            si_list.append(metrics["si"])
            gnr_list.append(metrics["gnr"])
            dr_iqr_list.append(metrics["dr_iqr"])
            dr_kl_list.append(metrics["dr_kl"])

        avg_asr   = round(float(np.mean(asr_list)),    3)
        avg_si    = round(float(np.mean(si_list)),     3)
        avg_gnr   = round(float(np.mean(gnr_list)),    3)
        avg_driqr = round(float(np.mean(dr_iqr_list)), 3)
        avg_drkl  = round(float(np.mean(dr_kl_list)),  3)
        st        = status_str(avg_si)
        marker    = "★" if ratio == 0.003 else " "

        print(f"  {ratio*100:.1f}%{marker}  n_inf={n_inf_last:4d} | "
              f"ASR={avg_asr:.3f} (cibles: {asr_list}) | "
              f"SI={avg_si:.3f} | GNR={avg_gnr:.3f} | "
              f"DR_IQR={avg_driqr:.3f} | DR_KL={avg_drkl:.3f} {st}")

        results.append({
            "ratio":          ratio,
            "asr":            avg_asr,
            "si":             avg_si,
            "gnr":            avg_gnr,
            "dr_iqr":         avg_driqr,
            "dr_kl":          avg_drkl,
            "status":         st,
            "asr_per_target": asr_list,
        })

    save_json(results, "ghost_ablation_ratio.json")
    return results


# ─────────────────────────────────────────────────────────────
#  Ablation 3 : Number of attackers
#  [P3] Budget actually applied to params.max_samples_per_client
# ─────────────────────────────────────────────────────────────
def run_ablation_n_attackers():
    print("\n" + "─" * 60)
    print("  Ablation 3 : Number of attackers - MNIST IID FedAvg")
    print("─" * 60)

    n_atk_list = [1, 2, 3, 4]
    results    = []

    for n_atk in n_atk_list:
        print(f"\n  >> n_attackers = {n_atk}")
        params = build_params(
            dict(dataset="mnist", iid=True, aggregation="fedavg",
                 global_epochs=20, local_batch_size=64),
            seed=GLOBAL_SEED
        )
        params.n_attackers = n_atk

        # [P3] Budget par client inversement proportionnel au nombre d'attaquants
        #      ET effectivement applique a max_samples_per_client
        budget = params.max_samples_per_client // max(1, n_atk)
        params.max_samples_per_client = budget  # ← correction P3 : etait manquant

        result, _, _ = ghost_main(params, target_idx=TARGET_IDX)
        st = status_str(result["si"])

        print(f"   n_atk={n_atk} | budget/client={budget} | "
              f"ASR={result['peak_asr']:.3f} | AccG={result['acc_global']:.2f}% | "
              f"SI={result['si']:.3f} | GNR={result['gnr']:.3f} {st}")

        results.append({
            "n_attackers":       n_atk,
            "budget_per_client": budget,
            "asr":               result["peak_asr"],
            "acc":               result["acc_global"],
            "si":                result["si"],
            "gnr":               result["gnr"],
            "dr_iqr":            result["dr_iqr"],
            "dr_kl":             result["dr_kl"],
            "status":            st,
        })

    save_json(results, "ghost_ablation_n_attackers.json")
    return results


# ─────────────────────────────────────────────────────────────
#  Ablation 4 : Rand+Ghost vs ISI+Ghost
# ─────────────────────────────────────────────────────────────
def run_ablation_rand_vs_isi():
    print("\n" + "─" * 60)
    print("  Ablation 4 : Rand+Ghost vs ISI+Ghost - MNIST IID FedAvg")
    print("─" * 60)

    params = build_params(
        dict(dataset="mnist", iid=True, aggregation="fedavg",
             global_epochs=20, local_batch_size=64)
    )
    set_seed(GLOBAL_SEED)
    train_ds, test_ds = load_data(params)
    global_model      = build_model(params)
    global_model, _, _ = fl_train(global_model, train_ds, test_ds, params)

    target_x, target_y = test_ds[TARGET_IDX]
    client_data = partition_data(train_ds, params)
    chosen_ref  = random.sample(range(params.n_total_clients),
                                params.n_clients_per_round)
    ben_norms   = compute_benign_norms(
        global_model, train_ds, client_data, chosen_ref, params
    )

    results = []

    # ── Rand+Ghost ───────────────────────────────────────────
    set_seed(GLOBAL_SEED)
    rand_indices = random.sample(range(len(train_ds)), 50)
    metrics_rand, mal_grad_rand, _ = run_stealth_eval(
        global_model, train_ds, target_x, target_y,
        rand_indices, params, ben_norms
    )
    asr_rand = run_asr_injection(
        global_model, mal_grad_rand, client_data,
        train_ds, target_x, target_y, params
    )
    st_rand = status_str(metrics_rand["si"])
    print(f"  Rand+Ghost         | ASR={asr_rand:.3f} | "
          f"SI={metrics_rand['si']:.3f} | GNR={metrics_rand['gnr']:.3f} | "
          f"DR_IQR={metrics_rand['dr_iqr']:.3f} | "
          f"DR_KL={metrics_rand['dr_kl']:.3f} {st_rand}")
    results.append({
        "method": "Rand+Ghost",
        "asr":    asr_rand,
        **metrics_rand,
        "status": st_rand,
    })

    # ── ISI+Ghost ────────────────────────────────────────────
    set_seed(GLOBAL_SEED)
    isi_indices, _ = isi_select(
        global_model, train_ds, target_x, target_y, params
    )
    metrics_isi, mal_grad_isi, _ = run_stealth_eval(
        global_model, train_ds, target_x, target_y,
        isi_indices, params, ben_norms
    )
    asr_isi = run_asr_injection(
        global_model, mal_grad_isi, client_data,
        train_ds, target_x, target_y, params
    )
    st_isi = status_str(metrics_isi["si"])
    print(f"  ISI+Ghost★         | ASR={asr_isi:.3f} | "
          f"SI={metrics_isi['si']:.3f} | GNR={metrics_isi['gnr']:.3f} | "
          f"DR_IQR={metrics_isi['dr_iqr']:.3f} | "
          f"DR_KL={metrics_isi['dr_kl']:.3f} {st_isi}")
    results.append({
        "method": "ISI+Ghost",
        "asr":    asr_isi,
        **metrics_isi,
        "status": st_isi,
    })

    save_json(results, "ghost_ablation_rand_vs_isi.json")
    return results


# ─────────────────────────────────────────────────────────────
#  Ablation 5 : Resistance defense FedMUA (IQR clipping)
# ─────────────────────────────────────────────────────────────
def run_ablation_defense():
    print("\n" + "─" * 60)
    print("  Ablation 5 : Resistance defense FedMUA - MNIST IID FedAvg")
    print("─" * 60)

    lambda_defs = [1.0, 0.5, 0.1]
    results     = []

    params = build_params(
        dict(dataset="mnist", iid=True, aggregation="fedavg",
             global_epochs=20, local_batch_size=64)
    )
    set_seed(GLOBAL_SEED)
    train_ds, test_ds = load_data(params)
    global_model      = build_model(params)
    global_model, _, _ = fl_train(global_model, train_ds, test_ds, params)

    target_x, target_y = test_ds[TARGET_IDX]
    client_data = partition_data(train_ds, params)
    chosen_ref  = random.sample(range(params.n_total_clients),
                                params.n_clients_per_round)
    ben_norms   = compute_benign_norms(
        global_model, train_ds, client_data, chosen_ref, params
    )

    # Q3 benin
    q3_ben = torch.quantile(ben_norms, 0.75).item()
    print(f"  Q3 benin = {q3_ben:.4f}")

    #Basic malicious gradient (calculates only once)
    set_seed(GLOBAL_SEED)
    inf_indices, _ = isi_select(
        global_model, train_ds, target_x, target_y, params
    )
    _, mal_grad_base, _ = run_stealth_eval(
        global_model, train_ds, target_x, target_y,
        inf_indices, params, ben_norms
    )

    for ldef in lambda_defs:
        # Defense FedMUA : clip si norme > ldef * Q3
        threshold = ldef * q3_ben
        mal_norm  = mal_grad_base.norm().item()

        if mal_norm > threshold:
            clipped = mal_grad_base * (threshold / mal_norm)
        else:
            clipped = mal_grad_base.clone()

        metrics = compute_stealth_metrics(
            clipped, [], ben_norms.to(params.device)
        )
        asr = run_asr_injection(
            global_model, clipped, client_data,
            train_ds, target_x, target_y, params
        )
        st    = status_str(metrics["si"])
        label = "(no defense)" if ldef == 1.0 else ""
        print(f"  ldef={ldef} {label:12s} | "
              f"ASR={asr:.3f} | SI={metrics['si']:.3f} | "
              f"GNR={metrics['gnr']:.3f} | "
              f"DR_IQR={metrics['dr_iqr']:.3f} | "
              f"DR_KL={metrics['dr_kl']:.3f} {st}")
        results.append({
            "lambda_def": ldef,
            "asr":        asr,
            **metrics,
            "status":     st,
        })

    save_json(results, "ghost_ablation_defense.json")
    return results


# ─────────────────────────────────────────────────────────────
#  Ablation 6 : Grey-Box (snapshot proxy)
# ─────────────────────────────────────────────────────────────
def run_ablation_greybox():
    print("\n" + "─" * 60)
    print("  Ablation 6 : Grey-Box (snapshot proxy) - MNIST IID FedAvg")
    print("─" * 60)

    params = build_params(
        dict(dataset="mnist", iid=True, aggregation="fedavg",
             global_epochs=20, local_batch_size=64)
    )
    set_seed(GLOBAL_SEED)
    train_ds, test_ds = load_data(params)
    global_model      = build_model(params)
    global_model, acc_final, snapshots = fl_train(
        global_model, train_ds, test_ds, params
    )

    target_x, target_y = test_ds[TARGET_IDX]
    client_data = partition_data(train_ds, params)
    chosen_ref  = random.sample(range(params.n_total_clients),
                                params.n_clients_per_round)
    ben_norms   = compute_benign_norms(
        global_model, train_ds, client_data, chosen_ref, params
    )

    # Scenarios : (label, state_dict_ou_None, acc_override)
    scenarios = [
        ("Grey-Box 25%",    snapshots.get("snap_25"), None),
        ("Grey-Box 50%",    snapshots.get("snap_50"), None),
        ("Grey-Box 75%",    snapshots.get("snap_75"), None),
        ("White-Box 100%★", None,                     acc_final),
    ]

    results = []

    for label, snap_state, acc_override in scenarios:
        print(f"\n  >> {label}")
        set_seed(GLOBAL_SEED)

        # Construire le modele proxy
        if snap_state is not None:
            proxy = build_model(params)
            proxy.load_state_dict(snap_state)
            proxy = proxy.to(params.device)
            acc_proxy = evaluate(proxy, test_ds, params)
        else:
            # White-box : proxy = modele final
            proxy     = global_model
            acc_proxy = acc_override if acc_override else evaluate(proxy, test_ds, params)

        print(f"     AccProxy = {acc_proxy * 100:.2f}%")

        # ISI sur le proxy
        inf_indices, _ = isi_select(
            proxy, train_ds, target_x, target_y, params
        )

        # MUG sur le proxy
        metrics, mal_grad, _ = run_stealth_eval(
            proxy, train_ds, target_x, target_y,
            inf_indices, params, ben_norms
        )

        # ASR evalue sur le modele FINAL (pas le proxy)
        asr = run_asr_injection(
            global_model, mal_grad, client_data,
            train_ds, target_x, target_y, params
        )

        st = status_str(metrics["si"])
        print(f"  {label:<22} | AccProxy={acc_proxy * 100:.1f}% | "
              f"ASR={asr:.3f} | SI={metrics['si']:.3f} | "
              f"GNR={metrics['gnr']:.3f} | "
              f"DR_IQR={metrics['dr_iqr']:.3f} | "
              f"DR_KL={metrics['dr_kl']:.3f} {st}")

        results.append({
            "scenario":  label,
            "acc_proxy": round(acc_proxy * 100, 1),
            "asr":       asr,
            **metrics,
            "status":    st,
        })

    save_json(results, "ghost_ablation_greybox.json")
    return results


# ─────────────────────────────────────────────────────────────
#  Runtime overhead
# ─────────────────────────────────────────────────────────────
def print_runtime_table(main_results: list):
    print("\n" + "─" * 60)
    print("    Run-time Overhead (ISI + MUG)")
    print("─" * 60)
    print(f"  {'Experience':<30} | {'ISI (s)':>8} | {'MUG (s)':>8}")
    print("  " + "─" * 52)
    for r in main_results:
        exp = r.get("exp", "?")
        print(f"  {exp:<30} | {r.get('isi_time', 0):>8.3f} | "
              f"{r.get('mug_time', 0):>8.3f}")
    save_json(
        [{"exp": r["exp"], "isi_time": r["isi_time"], "mug_time": r["mug_time"]}
         for r in main_results],
        "ghost_runtime.json"
    )


# ─────────────────────────────────────────────────────────────
#  Tableau recapitulatif final
# ─────────────────────────────────────────────────────────────
def print_final_summary(main_results: list, greybox_results: list):
    print("\n" + "=" * 80)
    print("   GHOST REQUESTS v6.1 - RESULTATS COMPLETS")
    print("=" * 80)
    header = (f"  {'Experience':<28} | {'AccG':>6} | {'GNR':>5} | "
              f"{'DR_IQR':>6} | {'DR_KL':>5} | {'SI':>5} | Status")
    print(header)
    print("  " + "─" * 75)

    # References
    print(f"  {'Benign (ref)':<28} | {REF_BENIGN['acc']:>5.2f}% | "
          f"{REF_BENIGN['gnr']:>5.3f} | {REF_BENIGN['dr_iqr']:>6.3f} | "
          f"{REF_BENIGN['dr_kl']:>5.3f} | {REF_BENIGN['si']:>5.3f} |    ref")
    print(f"  {'FedMUA (ref)':<28} | {REF_FEDMUA['acc']:>5.2f}% | "
          f"{REF_FEDMUA['gnr']:>5.3f} | {REF_FEDMUA['dr_iqr']:>6.3f} | "
          f"{REF_FEDMUA['dr_kl']:>5.3f} | {REF_FEDMUA['si']:>5.3f} |    ref")
    print("  " + "─" * 75)

    ok_count = 0
    for r in main_results:
        exp = r.get("exp", "?")
        st  = r.get("status", "No")
        if st == "Ok":
            ok_count += 1
        print(f"  {exp:<28} | {r['acc_global']:>5.2f}% | "
              f"{r['gnr']:>5.3f} | {r['dr_iqr']:>6.3f} | "
              f"{r['dr_kl']:>5.3f} | {r['si']:>5.3f} | {st:>6}")

    print("=" * 80)
    print(f"  Score : {ok_count}/{len(main_results)} experiences SI >= {SI_THRESHOLD}")
    print("=" * 80)

    # Grey-Box summary
    if greybox_results:
        print("\n" + "─" * 80)
        print("   ABLATION 6 - Grey-Box Summary")
        print("─" * 80)
        print(f"  {'Scenario':<22} | {'AccProxy':>8} | {'ASR':>5} | "
              f"{'SI':>5} | {'GNR':>5} | {'DR_IQR':>6} | DR_KL")
        print("  " + "─" * 72)
        for r in greybox_results:
            st = r.get("status", "No")
            print(f"  {r['scenario']:<22} | {r['acc_proxy']:>7.1f}% | "
                  f"{r['asr']:>5.3f} | {r['si']:>5.3f} | "
                  f"{r['gnr']:>5.3f} | {r['dr_iqr']:>6.3f} | "
                  f"{r['dr_kl']:.3f} {st}")
        print("─" * 80)


# ─────────────────────────────────────────────────────────────
#  Point d'entree principal
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_env()

    # ── Experiences principales ───────────────────────────────
    main_results = run_main_experiments()
    print("\n Experiences principales terminees !")

    # ── Ablations ────────────────────────────────────────────
    run_ablation_lambda()
    run_ablation_ratio()
    run_ablation_n_attackers()
    run_ablation_rand_vs_isi()
    run_ablation_defense()
    greybox_results = run_ablation_greybox()

    # ── Tableaux finaux ───────────────────────────────────────
    print_runtime_table(main_results)
    print_final_summary(main_results, greybox_results)

    print("\n All experiments and ablations completed !")

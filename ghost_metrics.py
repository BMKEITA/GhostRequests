# ghost_metrics.py
import torch
import numpy as np
from scipy.stats import entropy

def gradient_norm_ratio(grad_mal, benign_norms):
    """GNR  ratio norme malveillante / moyenne benigne (cible : ~1.0)"""
    if isinstance(grad_mal, torch.Tensor):
        val = grad_mal.norm().item()
    else:
        val = float(grad_mal)
    return val / (np.mean(benign_norms) + 1e-8)

def detection_rate_iqr(grad_norms_mal, benign_norms):
    """DR_IQR  fraction des gradients malveillants au-dessus du Q3 benin (cible : ~0.25)"""
    q3 = np.percentile(benign_norms, 75)
    return float(np.mean(np.array(grad_norms_mal) > q3))

def detection_rate_kl(grad_norms_mal, benign_norms, bins=20):
    """DR_KL  divergence KL entre distributions benigne et malveillante (cible : ~0.0)"""
    all_vals = list(grad_norms_mal) + list(benign_norms)
    range_   = (min(all_vals), max(all_vals))
    p, _ = np.histogram(benign_norms,   bins=bins, range=range_, density=True)
    q, _ = np.histogram(grad_norms_mal, bins=bins, range=range_, density=True)
    return float(entropy(q + 1e-10, p + 1e-10))

def stealth_index(gnr, dr_iqr, dr_kl, w=(0.34, 0.33, 0.33)):
    """
    Stealth Index v2 - GNR + DR_IQR + DR_KL
    Cible : SI > 0.70
    """
    score = w[0]*min(gnr/3.0, 1.0) + w[1]*dr_iqr + w[2]*min(dr_kl/2.0, 1.0)
    return 1.0 - score

def peak_asr(asr_history):
    return max(asr_history) if asr_history else 0.0

def compute_all_metrics(grad_norms_mal, benign_norms,
                        mean_benign_grad, grad_mal_vectors, asr_history):
    gnr    = gradient_norm_ratio(torch.tensor(grad_norms_mal).mean(), benign_norms)
    dr_iqr = detection_rate_iqr(grad_norms_mal, benign_norms)
    dr_kl  = detection_rate_kl(grad_norms_mal, benign_norms)
    return {
        'GNR':      round(gnr,    4),
        'DR_IQR':   round(dr_iqr, 4),
        'DR_KL':    round(dr_kl,  4),
        'SI':       round(stealth_index(gnr, dr_iqr, dr_kl), 4),
        'Peak_ASR': round(peak_asr(asr_history), 4)
    }

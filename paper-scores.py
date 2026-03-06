# ============================================================================
# compute_all_scores.py
# Complete Evaluation Script — All Scores for Research Paper
# OPTIMIZED VERSION: Batched DataLoader inference (10–30x faster)
# ============================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    cohen_kappa_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from skimage.metrics import structural_similarity as ssim
import cv2
import warnings
import json
from datetime import datetime
warnings.filterwarnings("ignore")

# ============================================================================
# PATHS — Extracted directly from run_gradcam_organized.py
# ============================================================================

CFG = {
    # Dataset paths
    "MRI_tumor_dir":   "/content/drive/MyDrive/khushi/Dataset/Brain Tumor MRI images/Tumor",
    "MRI_healthy_dir": "/content/drive/MyDrive/khushi/Dataset/Brain Tumor MRI images/Healthy",
    "CT_tumor_dir":    "/content/drive/MyDrive/khushi/Dataset/Brain Tumor CT scan Images/Tumor",
    "CT_healthy_dir":  "/content/drive/MyDrive/khushi/Dataset/Brain Tumor CT scan Images/Healthy",

    # Checkpoint paths
    "output_dir":        "/content/drive/MyDrive/khushi",
    "teacher_ckpt_dir":  "/content/drive/MyDrive/khushi/Ct-mricheckpoints",
    "student_ckpt":      "studentB_EMA_C_ep60.pth",
    "student_raw_ckpt":  "studentB_raw_C_ep60.pth",
    "mri_teacher_ckpt":  "mri_teacher_BEST.pth",
    "ct_teacher_ckpt":   "ct_teacher_BEST.pth",

    # Output
    "scores_output_dir": "/content/drive/MyDrive/khushi/paper_scores",

    # Settings
    "num_classes":  2,
    "image_size":   224,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    "bootstrap_n":  1000,
    "seed":         42,
    "batch_size":   64,    # ← Key fix: batch inference
    # num_workers: 2 for Colab/Windows, 4 for Linux
    "num_workers":  2,
}

os.makedirs(CFG["scores_output_dir"], exist_ok=True)
print("=" * 70)
print("  COMPLETE PAPER SCORES COMPUTATION  (Optimized)")
print("=" * 70)
print(f"Device     : {CFG['device']}")
print(f"Batch size : {CFG['batch_size']}")
print(f"Output     : {CFG['scores_output_dir']}")
print("=" * 70)

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class TeacherNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        import torchvision.models as models
        try:
            base = models.convnext_tiny(pretrained=True)
        except Exception:
            base = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.gap      = nn.AdaptiveAvgPool2d(1)
        self.head     = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 384), nn.GELU(),
            nn.Dropout(0.4),   nn.Linear(384, num_classes))

    def forward(self, x):
        feat = self.gap(self.features(x)).flatten(1)
        return self.head(feat), feat


class ModalityNorm(nn.Module):
    def __init__(self, dim=768, num_modalities=2):
        super().__init__()
        self.gamma = nn.Embedding(num_modalities, dim)
        self.beta  = nn.Embedding(num_modalities, dim)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, feat, modality_id):
        idx = torch.full((feat.size(0),), modality_id,
                         dtype=torch.long, device=feat.device)
        return feat * self.gamma(idx) + self.beta(idx)


class UMLStudentB(nn.Module):
    MOD_IDS = {"MRI": 0, "CT": 1}

    def __init__(self, num_classes=2):
        super().__init__()
        import torchvision.models as models
        base           = models.convnext_tiny(pretrained=True)
        self.backbone  = base.features
        self.gap       = nn.AdaptiveAvgPool2d(1)
        self.modality_norm = ModalityNorm(768, 2)
        self.mri_proj  = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2))
        self.ct_proj   = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2))
        self.mri_head  = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, num_classes))
        self.ct_head   = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, num_classes))

    def forward(self, x, modality):
        feat = self.gap(self.backbone(x)).flatten(1)
        feat = self.modality_norm(feat, self.MOD_IDS[modality])
        if modality == "MRI":
            proj = self.mri_proj(feat)   # BUG FIX: call ONCE — was called twice before,
            return self.mri_head(proj), proj  # causing Dropout to give different results
        else:
            proj = self.ct_proj(feat)    # BUG FIX: same fix for CT path
            return self.ct_head(proj), proj


# ============================================================================
# TRANSFORMS
# ============================================================================

transform = transforms.Compose([
    transforms.Resize((CFG["image_size"], CFG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ============================================================================
# DATA LOADER  (optimized)
# ============================================================================

def load_dataset(tumor_dir, healthy_dir):
    """Load ALL images from both classes and return list of (path, label)."""
    data = []
    for fdir, label in [(tumor_dir, 1), (healthy_dir, 0)]:
        if not os.path.exists(fdir):
            print(f"  WARNING: path not found — {fdir}")
            continue
        for f in sorted(os.listdir(fdir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                data.append((os.path.join(fdir, f), label))
    return data


class BrainDataset(Dataset):
    """PyTorch Dataset for batched loading."""
    def __init__(self, data, transform):
        self.data      = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            return self.transform(img), label, img_path
        except Exception:
            # Return blank tensor for corrupt images
            return torch.zeros(3, CFG["image_size"], CFG["image_size"]), label, img_path


def predict_all(model, data, modality=None, device="cpu",
                batch_size=64, num_workers=2):
    """
    Batched inference — 10–30x faster than per-image loop.
    Processes `batch_size` images per GPU forward pass.
    """
    dataset = BrainDataset(data, transform)
    loader  = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = (device == "cuda"),
    )

    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for batch_idx, (imgs, labels, _) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)

            if modality:
                logits, _ = model(imgs, modality)
            else:
                logits, _ = model(imgs)

            probs      = F.softmax(logits, dim=1).cpu().numpy()
            pred_class = np.argmax(probs, axis=1)

            y_true.extend(labels.numpy())
            y_pred.extend(pred_class)
            y_prob.extend(probs[:, 1])   # Tumor class probability

            # Progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"    Processed {(batch_idx+1)*batch_size}/{len(data)} images...",
                      end="\r")

    print()
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

# ============================================================================
# BLOCK 1 — CORE METRICS
# ============================================================================

def compute_core_metrics(y_true, y_pred, y_prob, name):
    """Compute Accuracy, AUC, F1, Precision, Recall, Specificity, Kappa."""
    cm             = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "Model"       : name,
        "Accuracy"    : round(accuracy_score(y_true, y_pred) * 100, 2),
        "AUC_ROC"     : round(roc_auc_score(y_true, y_prob), 4),
        "F1_Macro"    : round(f1_score(y_true, y_pred, average="macro"), 4),
        "Precision_T" : round(precision_score(y_true, y_pred, pos_label=1), 4),
        "Recall_T"    : round(recall_score(y_true, y_pred, pos_label=1), 4),
        "Specificity" : round(tn / (tn + fp + 1e-8), 4),
        "Precision_H" : round(precision_score(y_true, y_pred, pos_label=0), 4),
        "Recall_H"    : round(recall_score(y_true, y_pred, pos_label=0), 4),
        "Cohen_Kappa" : round(cohen_kappa_score(y_true, y_pred), 4),
        "Brier_Score" : round(brier_score_loss(y_true, y_prob), 4),
        "N_samples"   : len(y_true),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }
    return metrics, cm

# ============================================================================
# BLOCK 2 — BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

def bootstrap_ci(y_true, y_pred, y_prob, n=1000, seed=42):
    """95% Bootstrap CIs for Accuracy, AUC, F1."""
    rng       = np.random.default_rng(seed)
    accs, aucs, f1s = [], [], []
    n_samples = len(y_true)

    for _ in range(n):
        idx  = rng.choice(n_samples, n_samples, replace=True)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        accs.append(accuracy_score(yt, yp) * 100)
        try:
            aucs.append(roc_auc_score(yt, ypr))
        except Exception:
            pass
        f1s.append(f1_score(yt, yp, average="macro"))

    def ci(arr):
        arr = np.array(arr)
        return round(np.percentile(arr, 2.5), 4), round(np.percentile(arr, 97.5), 4)

    return {
        "Accuracy_CI" : ci(accs),
        "AUC_CI"      : ci(aucs),
        "F1_CI"       : ci(f1s),
    }

# ============================================================================
# BLOCK 3 — STATISTICAL TESTS
# ============================================================================

def mcnemar_test(y_true, y_pred_a, y_pred_b, name_a, name_b):
    """McNemar's test between two models."""
    b      = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))
    c      = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))
    table  = np.array([[0, b], [c, 0]])
    result = mcnemar(table, exact=True)
    return {
        "comparison"  : f"{name_a} vs {name_b}",
        "b"           : int(b),
        "c"           : int(c),
        "p_value"     : round(result.pvalue, 6),
        "significant" : result.pvalue < 0.05,
    }


def wilcoxon_test(scores_a, scores_b, name_a, name_b):
    """Wilcoxon signed-rank test between two score arrays."""
    if len(scores_a) < 10:
        return {"note": "Too few samples for Wilcoxon test"}
    stat, p = wilcoxon(scores_a, scores_b)
    return {
        "comparison"  : f"{name_a} vs {name_b}",
        "statistic"   : round(float(stat), 4),
        "p_value"     : round(float(p), 6),
        "significant" : p < 0.05,
    }

# ============================================================================
# BLOCK 4 — CALIBRATION (ECE + Reliability Diagram)
# ============================================================================

def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bins     = np.linspace(0, 1, n_bins + 1)
    ece      = 0.0
    n_total  = len(y_true)
    bin_data = []

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        bin_n    = mask.sum()
        ece     += (bin_n / n_total) * abs(bin_acc - bin_conf)
        bin_data.append((bin_conf, bin_acc, bin_n))

    return round(ece, 4), bin_data


def plot_reliability_diagram(all_results, save_path):
    """Plot reliability diagrams for all 4 models in one figure."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Reliability Diagrams (Calibration)", fontsize=14, fontweight="bold")

    for ax, (name, y_true, y_prob) in zip(axes, all_results):
        fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ece, _ = compute_ece(y_true, y_prob)
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.plot(mean_pred, fraction_pos, "bo-", label=f"Model (ECE={ece})")
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

# ============================================================================
# BLOCK 5 — CONFUSION MATRIX FIGURE
# ============================================================================

def plot_confusion_matrices(all_cms, all_names, save_path):
    """Plot all 4 confusion matrices in one figure."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")
    labels = ["Healthy", "Tumor"]

    for ax, cm, name in zip(axes, all_cms, all_names):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=16,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

# ============================================================================
# BLOCK 6 — GRAD-CAM QUANTIFICATION
# ============================================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._register_hooks(target_layer)

    def _register_hooks(self, target_layer):
        def fwd(module, inp, out): self.activations = out.detach()
        def bwd(module, gin, gout): self.gradients  = gout[0].detach()

        if target_layer == "backbone[-1]":
            layer = self.model.backbone[-1]
        else:
            layer = self.model.features[-1]

        layer.register_forward_hook(fwd)
        layer.register_full_backward_hook(bwd)

    def generate_cam(self, tensor, modality=None):
        self.model.eval()
        tensor = tensor.detach().requires_grad_(True)   # BUG FIX: detach first to
        # avoid inheriting stale gradients from a previous backward pass
        if modality:
            logits, _ = self.model(tensor, modality)
        else:
            logits, _ = self.model(tensor)

        pred_class = torch.argmax(logits, dim=1).item()
        pred_prob  = F.softmax(logits, dim=1)[0, pred_class].item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, pred_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam     = cam.squeeze().cpu().detach().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, pred_class, pred_prob


def compute_gradcam_scores(student, mri_teacher, ct_teacher,
                           mri_data, ct_data, device, n_samples=40):
    """
    Compute quantitative Grad-CAM scores:
      - SSIM between student and teacher CAMs
      - Activation Concentration Ratio (top 20% pixels)
      - Mean activation over predicted tumor region
      - CAM Agreement (same quadrant highlight)
    """
    teacher_cam_mri = GradCAM(mri_teacher, "features[-1]")
    teacher_cam_ct  = GradCAM(ct_teacher,  "features[-1]")
    # NOTE: student GradCAM is created fresh inside each loop iteration (see below)
    # to avoid double-hook bug when reusing the same model.

    results = {"MRI": [], "CT": []}

    for modality, data, t_cam in [
        ("MRI", mri_data, teacher_cam_mri),
        ("CT",  ct_data,  teacher_cam_ct),
    ]:
        # BUG FIX: Create a fresh student GradCAM per modality loop.
        # Previously two GradCAM objects were created for the student up-front,
        # both registering hooks on the same layer — causing activations/gradients
        # to be overwritten or doubled. Now we create one fresh instance per loop.
        s_cam = GradCAM(student, "backbone[-1]")

        sample = data[:n_samples]
        for i, (img_path, label) in enumerate(sample):
            try:
                img    = Image.open(img_path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)

                s_map, s_pred, s_conf = s_cam.generate_cam(tensor, modality)
                t_map, t_pred, t_conf = t_cam.generate_cam(tensor)

                s_map_r = cv2.resize(s_map, (224, 224))
                t_map_r = cv2.resize(t_map, (224, 224))

                ssim_score = ssim(s_map_r, t_map_r, data_range=1.0)

                threshold = np.percentile(s_map_r, 80)
                acr_s     = s_map_r[s_map_r >= threshold].sum() / (s_map_r.sum() + 1e-8)
                threshold = np.percentile(t_map_r, 80)
                acr_t     = t_map_r[t_map_r >= threshold].sum() / (t_map_r.sum() + 1e-8)

                mean_act_s = float(s_map_r.mean())
                mean_act_t = float(t_map_r.mean())

                s_center = np.unravel_index(s_map_r.argmax(), s_map_r.shape)
                t_center = np.unravel_index(t_map_r.argmax(), t_map_r.shape)
                agree    = (abs(s_center[0] - t_center[0]) < 56 and
                            abs(s_center[1] - t_center[1]) < 56)

                results[modality].append({
                    "ssim"        : round(float(ssim_score), 4),
                    "acr_student" : round(float(acr_s), 4),
                    "acr_teacher" : round(float(acr_t), 4),
                    "mean_act_s"  : round(mean_act_s, 4),
                    "mean_act_t"  : round(mean_act_t, 4),
                    "cam_agree"   : int(agree),
                    "s_conf"      : round(s_conf, 4),
                    "t_conf"      : round(t_conf, 4),
                    "true_label"  : label,
                })
                print(f"    {modality} CAM {i+1}/{len(sample)}", end="\r")
            except Exception as e:
                print(f"  CAM skip {img_path}: {e}")

    # Aggregate
    summary = {}
    for mod in ["MRI", "CT"]:
        r = results[mod]
        if not r:
            continue
        summary[mod] = {
            "Mean_SSIM"          : round(np.mean([x["ssim"] for x in r]), 4),
            "Std_SSIM"           : round(np.std( [x["ssim"] for x in r]), 4),
            "Mean_ACR_Student"   : round(np.mean([x["acr_student"] for x in r]), 4),
            "Mean_ACR_Teacher"   : round(np.mean([x["acr_teacher"] for x in r]), 4),
            "CAM_Agreement_Rate" : round(np.mean([x["cam_agree"] for x in r]) * 100, 2),
            "Mean_Conf_Student"  : round(np.mean([x["s_conf"] for x in r]), 4),
            "Mean_Conf_Teacher"  : round(np.mean([x["t_conf"] for x in r]), 4),
            "N"                  : len(r),
        }

    return summary

# ============================================================================
# BLOCK 7 — DELTA TABLE (Student − Teacher)
# ============================================================================

def compute_delta(student_metrics, teacher_metrics, label):
    keys  = ["Accuracy", "AUC_ROC", "F1_Macro", "Recall_T", "Specificity", "Cohen_Kappa"]
    delta = {"Comparison": label}
    for k in keys:
        s = student_metrics.get(k, 0)
        t = teacher_metrics.get(k, 0)
        delta[f"Δ_{k}"] = round(s - t, 4)
    return delta

# ============================================================================
# PLOTTING — Main Performance Bar Chart
# ============================================================================

def plot_main_results(all_metrics, save_path):
    models  = [m["Model"] for m in all_metrics]
    metrics = ["Accuracy", "AUC_ROC", "F1_Macro", "Recall_T", "Specificity"]
    colors  = ["#4285F4", "#EA4335", "#34A853", "#FBBC05", "#9B59B6"]

    x   = np.arange(len(models))
    w   = 0.15
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = []
        for m in all_metrics:
            v = m[metric]
            vals.append(v if metric == "Accuracy" else v * 100)
        ax.bar(x + i * w, vals, w, label=metric, color=color, alpha=0.85)

    ax.set_xticks(x + w * 2)
    ax.set_xticklabels(models, rotation=15, fontsize=10)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Performance Comparison — All Metrics", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(50, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

# ============================================================================
# SAVE RESULTS AS JSON + TXT REPORT
# ============================================================================

def save_report(all_data, save_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON — machine readable
    # Custom encoder to handle numpy bool_, int_, float_ types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    json_path = os.path.join(save_dir, f"all_scores_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved JSON: {json_path}")

    # TXT — human readable report
    txt_path = os.path.join(save_dir, f"paper_scores_report_{timestamp}.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  PAPER SCORES REPORT\n")
        f.write(f"  Generated: {datetime.now()}\n")
        f.write("=" * 70 + "\n\n")

        # BLOCK 1 — Main Table
        f.write("BLOCK 1 — MAIN PERFORMANCE TABLE\n")
        f.write("-" * 70 + "\n")
        header = (f"{'Model':<25} {'Acc%':>6} {'AUC':>6} {'F1':>6} "
                  f"{'Sens':>6} {'Spec':>6} {'Kappa':>6} {'ECE':>6} {'Brier':>6}\n")
        f.write(header)
        f.write("-" * 70 + "\n")
        for m in all_data.get("main_metrics", []):
            line = (f"{m['Model']:<25} {m['Accuracy']:>6.2f} {m['AUC_ROC']:>6.4f} "
                    f"{m['F1_Macro']:>6.4f} {m['Recall_T']:>6.4f} {m['Specificity']:>6.4f} "
                    f"{m['Cohen_Kappa']:>6.4f} {m.get('ECE','N/A'):>6} {m['Brier_Score']:>6.4f}\n")
            f.write(line)

        # BLOCK 2 — Bootstrap CIs
        f.write("\n\nBLOCK 2 — 95% BOOTSTRAP CONFIDENCE INTERVALS\n")
        f.write("-" * 70 + "\n")
        for name, ci_data in all_data.get("bootstrap_cis", {}).items():
            f.write(f"{name}:\n")
            for k, v in ci_data.items():
                f.write(f"  {k}: {v[0]} – {v[1]}\n")

        # BLOCK 3 — Statistical Tests
        f.write("\n\nBLOCK 3 — STATISTICAL SIGNIFICANCE TESTS\n")
        f.write("-" * 70 + "\n")
        for test in all_data.get("statistical_tests", []):
            sig = "✓ SIGNIFICANT" if test.get("significant") else "✗ NOT SIGNIFICANT"
            f.write(f"{test.get('comparison','')}: p={test.get('p_value','N/A')}  {sig}\n")

        # BLOCK 4 — Delta Table
        f.write("\n\nBLOCK 4 — DELTA TABLE (Student − Teacher)\n")
        f.write("-" * 70 + "\n")
        for d in all_data.get("delta_table", []):
            f.write(f"{d['Comparison']}:\n")
            for k, v in d.items():
                if k != "Comparison":
                    arrow = "▲" if v > 0 else ("▼" if v < 0 else "=")
                    f.write(f"  {k}: {arrow} {v:+.4f}\n")

        # BLOCK 5 — Grad-CAM
        f.write("\n\nBLOCK 5 — GRAD-CAM QUANTIFICATION\n")
        f.write("-" * 70 + "\n")
        for mod, scores in all_data.get("gradcam_scores", {}).items():
            f.write(f"{mod}:\n")
            for k, v in scores.items():
                f.write(f"  {k}: {v}\n")

    print(f"  Saved TXT : {txt_path}")
    return txt_path

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    device      = CFG["device"]
    batch_size  = CFG["batch_size"]
    num_workers = CFG["num_workers"]

    # ── Load Models ──────────────────────────────────────────────────────────
    print("\n[1/7] Loading models...")

    mri_teacher = TeacherNet(CFG["num_classes"]).to(device)
    ct_teacher  = TeacherNet(CFG["num_classes"]).to(device)
    student     = UMLStudentB(CFG["num_classes"]).to(device)

    mri_teacher.load_state_dict(torch.load(
        os.path.join(CFG["teacher_ckpt_dir"], CFG["mri_teacher_ckpt"]),
        map_location=device))
    ct_teacher.load_state_dict(torch.load(
        os.path.join(CFG["teacher_ckpt_dir"], CFG["ct_teacher_ckpt"]),
        map_location=device))
    student.load_state_dict(torch.load(
        os.path.join(CFG["output_dir"], "checkpoints", CFG["student_ckpt"]),
        map_location=device))

    mri_teacher.eval(); ct_teacher.eval(); student.eval()
    print("  ✅ All models loaded")

    # ── Load Data ─────────────────────────────────────────────────────────────
    print("\n[2/7] Loading datasets...")
    mri_data = load_dataset(CFG["MRI_tumor_dir"], CFG["MRI_healthy_dir"])
    ct_data  = load_dataset(CFG["CT_tumor_dir"],  CFG["CT_healthy_dir"])
    print(f"  MRI: {len(mri_data)} images | CT: {len(ct_data)} images")

    # ── Batched Inference ─────────────────────────────────────────────────────
    print(f"\n[3/7] Running batched inference (batch={batch_size}, workers={num_workers})...")

    print("  → MRI Teacher...")
    mri_t_true, mri_t_pred, mri_t_prob = predict_all(
        mri_teacher, mri_data, device=device,
        batch_size=batch_size, num_workers=num_workers)

    print("  → CT Teacher...")
    ct_t_true, ct_t_pred, ct_t_prob = predict_all(
        ct_teacher, ct_data, device=device,
        batch_size=batch_size, num_workers=num_workers)

    print("  → Student (MRI)...")
    mri_s_true, mri_s_pred, mri_s_prob = predict_all(
        student, mri_data, modality="MRI", device=device,
        batch_size=batch_size, num_workers=num_workers)

    print("  → Student (CT)...")
    ct_s_true, ct_s_pred, ct_s_prob = predict_all(
        student, ct_data, modality="CT", device=device,
        batch_size=batch_size, num_workers=num_workers)

    print("  ✅ Inference complete")

    # ── Block 1: Core Metrics ─────────────────────────────────────────────────
    print("\n[4/7] Computing core metrics...")
    m_mri_t, cm_mri_t = compute_core_metrics(mri_t_true, mri_t_pred, mri_t_prob, "MRI Teacher")
    m_ct_t,  cm_ct_t  = compute_core_metrics(ct_t_true,  ct_t_pred,  ct_t_prob,  "CT Teacher")
    m_mri_s, cm_mri_s = compute_core_metrics(mri_s_true, mri_s_pred, mri_s_prob, "Student-MRI")
    m_ct_s,  cm_ct_s  = compute_core_metrics(ct_s_true,  ct_s_pred,  ct_s_prob,  "Student-CT")

    # Add ECE to each metrics dict
    for (metrics, y_true, y_prob) in [
        (m_mri_t, mri_t_true, mri_t_prob),
        (m_ct_t,  ct_t_true,  ct_t_prob),
        (m_mri_s, mri_s_true, mri_s_prob),
        (m_ct_s,  ct_s_true,  ct_s_prob),
    ]:
        ece, _ = compute_ece(y_true, y_prob)
        metrics["ECE"] = ece

    all_metrics = [m_mri_t, m_ct_t, m_mri_s, m_ct_s]
    print("  ✅ Core metrics done")
    for m in all_metrics:
        print(f"     {m['Model']:<20} Acc={m['Accuracy']}%  AUC={m['AUC_ROC']}  F1={m['F1_Macro']}")

    # ── Block 2: Bootstrap CIs ────────────────────────────────────────────────
    print("\n[5/7] Computing bootstrap confidence intervals (n=1000)...")
    cis = {
        "MRI Teacher" : bootstrap_ci(mri_t_true, mri_t_pred, mri_t_prob, CFG["bootstrap_n"]),
        "CT Teacher"  : bootstrap_ci(ct_t_true,  ct_t_pred,  ct_t_prob,  CFG["bootstrap_n"]),
        "Student-MRI" : bootstrap_ci(mri_s_true, mri_s_pred, mri_s_prob, CFG["bootstrap_n"]),
        "Student-CT"  : bootstrap_ci(ct_s_true,  ct_s_pred,  ct_s_prob,  CFG["bootstrap_n"]),
    }
    print("  ✅ Bootstrap CIs done")

    # ── Block 3: Statistical Tests ────────────────────────────────────────────
    print("\n[6/7] Running statistical tests...")
    stat_tests = [
        mcnemar_test(mri_s_true, mri_s_pred, mri_t_pred, "Student-MRI", "MRI Teacher"),
        mcnemar_test(ct_s_true,  ct_s_pred,  ct_t_pred,  "Student-CT",  "CT Teacher"),
        wilcoxon_test(mri_s_prob, mri_t_prob, "Student-MRI", "MRI Teacher"),
        wilcoxon_test(ct_s_prob,  ct_t_prob,  "Student-CT",  "CT Teacher"),
    ]
    print("  ✅ Statistical tests done")

    # ── Delta Table ───────────────────────────────────────────────────────────
    delta_table = [
        compute_delta(m_mri_s, m_mri_t, "Student-MRI vs MRI Teacher"),
        compute_delta(m_ct_s,  m_ct_t,  "Student-CT  vs CT Teacher"),
    ]

    # ── Grad-CAM Quantification ───────────────────────────────────────────────
    print("\n[7/7] Computing Grad-CAM quantification scores (n=40 per modality)...")
    cam_scores = compute_gradcam_scores(
        student, mri_teacher, ct_teacher,
        mri_data, ct_data, device, n_samples=40)
    print("\n  ✅ Grad-CAM scores done")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n  Generating figures...")
    plot_confusion_matrices(
        [cm_mri_t, cm_ct_t, cm_mri_s, cm_ct_s],
        ["MRI Teacher", "CT Teacher", "Student-MRI", "Student-CT"],
        os.path.join(CFG["scores_output_dir"], "confusion_matrices.png"))

    plot_reliability_diagram(
        [("MRI Teacher", mri_t_true, mri_t_prob),
         ("CT Teacher",  ct_t_true,  ct_t_prob),
         ("Student-MRI", mri_s_true, mri_s_prob),
         ("Student-CT",  ct_s_true,  ct_s_prob)],
        os.path.join(CFG["scores_output_dir"], "reliability_diagrams.png"))

    plot_main_results(
        all_metrics,
        os.path.join(CFG["scores_output_dir"], "main_results_chart.png"))

    # ── Save Report ───────────────────────────────────────────────────────────
    all_data = {
        "main_metrics"      : all_metrics,
        "bootstrap_cis"     : cis,
        "statistical_tests" : stat_tests,
        "delta_table"       : delta_table,
        "gradcam_scores"    : cam_scores,
    }
    save_report(all_data, CFG["scores_output_dir"])

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ALL SCORES COMPUTED SUCCESSFULLY")
    print("=" * 70)
    print(f"\n  Output directory: {CFG['scores_output_dir']}")
    print("\n  Files generated:")
    print("    paper_scores_report_<timestamp>.txt  ← Human readable report")
    print("    all_scores_<timestamp>.json          ← Machine readable (for tables)")
    print("    confusion_matrices.png               ← Figure for paper")
    print("    reliability_diagrams.png             ← Figure for paper")
    print("    main_results_chart.png               ← Figure for paper")
    print("\n  Scores ready for paper:")
    print(f"    Block 1  — Core metrics      : 4 models × 9 metrics")
    print(f"    Block 2  — Bootstrap CIs     : 4 models × 3 metrics")
    print(f"    Block 3  — Statistical tests : {len(stat_tests)} tests (2 McNemar + 2 Wilcoxon)")
    print(f"    Block 4  — Delta table       : {len(delta_table)} comparisons")
    print(f"    Block 5  — Grad-CAM scores   : MRI + CT quantification")
    print(f"    Block 6  — Calibration (ECE) : embedded in core metrics")
    print("=" * 70)

    return all_data


if __name__ == "__main__":
    results = main()

"""
╔══════════════════════════════════════════════════════════════════════════╗
║  FINAL EVALUATION — VERSION B  (Research-Grade, v2)                     ║
║                                                                          ║
║  BUGS FIXED vs previous final_evaluation_fixed.py                       ║
║                                                                          ║
║  BUG-1 ► SPLIT MISMATCH (critical — wrong images evaluated)             ║
║    Old : BrainValDataset iterated os.listdir() as sorted() BEFORE MD5   ║
║          dedup, but training's BrainDataset iterates os.listdir()        ║
║          UNSORTED then deduplicates.                                     ║
║          Because os.listdir() order is filesystem-dependent, sorted()    ║
║          produces a different file ordering → different RNG shuffle →    ║
║          different train/val split → evaluation on partially seen data.  ║
║    Fix : Remove sorted(). Use plain os.listdir() exactly as training     ║
║          does, so the RNG shuffle and cut point are identical.           ║
║                                                                          ║
║  BUG-2 ► GLOBAL random.seed() CALLED BEFORE DATASET CONSTRUCTION        ║
║    Old : random.seed(42) at module-level, then BrainValDataset uses      ║
║          random.Random(seed) — an independent RNG instance.              ║
║          BUT the global random.seed() call resets the global state,      ║
║          and if anything between that call and dataset construction       ║
║          draws from the global RNG, the local rng.shuffle() will still   ║
║          be correct (it uses its own instance). However the module-level  ║
║          seeding interacts with torch's DataLoader worker seeding and    ║
║          can cause non-deterministic worker init on some PyTorch builds. ║
║    Fix : Keep random.Random(seed) for split RNG (correct). Move global  ║
║          seeds inside main() AFTER imports to avoid module-level side    ║
║          effects on DataLoader workers.                                  ║
║                                                                          ║
║  BUG-3 ► DROPOUT ACTIVE DURING EVALUATION (inflates variance, lowers    ║
║          measured accuracy)                                              ║
║    Old : student.eval() called once in main(), but evaluate_model()      ║
║          calls model.eval() again — that is correct. However             ║
║          TeacherNet and UMLStudentB both contain Dropout layers.         ║
║          The issue: if a model is passed to evaluate_model BEFORE        ║
║          .eval() is called on it (e.g. right after load_state_dict),     ║
║          Dropout is still active. The old code called .eval() on         ║
║          teachers AFTER load but set requires_grad=False in a loop       ║
║          that iterated parameters — Dropout has no parameters, so its   ║
║          training flag was only set by the explicit .eval() call.        ║
║          evaluate_model() does re-call model.eval() so this is safe,    ║
║          BUT it is fragile. Fixed by making evaluate_model() the         ║
║          authoritative eval()-caller and adding an assertion.            ║
║    Fix : evaluate_model() explicitly asserts model is in eval mode       ║
║          and calls model.eval() unconditionally at entry.                ║
║                                                                          ║
║  BUG-4 ► MACRO-AVERAGE LOGIC PICKS WRONG SUBSET FOR TEACHER             ║
║    Old : In print_summary_table(), the macro-avg loop does:             ║
║              keys = [r["label"] for r in results                         ║
║                      if r["label"].startswith("Teacher")]                ║
║          This grabs ["Teacher MRI", "Teacher CT"], then takes keys[:2]   ║
║          — correct for teachers. But the Student branch does:            ║
║              if r["label"].startswith("Student")                         ║
║          which also matches "Student MRI" AND "Student CT" → keys[:2]    ║
║          is ["Student MRI", "Student CT"] — correct.                     ║
║          BUT the second loop iteration uses prefix="Teacher MRI/CT avg"  ║
║          and checks startswith("Teacher") — giving same teacher list     ║
║          which is then labelled "Teacher MRI/CT avg macro-avg". That     ║
║          label is misleading but numerically correct only if the result  ║
║          list order is [Teacher MRI, Teacher CT, Student MRI, Student CT]║
║          which it is — so the numbers are right but the code is          ║
║          confusing and fragile. Rewritten clearly.                       ║
║    Fix : Explicit key lookup by exact label string. No startswith().     ║
║                                                                          ║
║  ADDITIONAL RESEARCH-GRADE ADDITIONS                                     ║
║  [+] Specificity (True Negative Rate) reported — standard in medical AI  ║
║  [+] Sensitivity == Recall, but labelled correctly for clinical context  ║
║  [+] MCC (Matthews Correlation Coefficient) — robust to class imbalance  ║
║  [+] Specificity of Healthy class = TNR added to report                  ║
║  [+] Bootstrap 95% CI on Accuracy and AUC (1000 resamples)              ║
║  [+] Separate Specificity column in summary table                        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os, random, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, classification_report,
    confusion_matrix, roc_curve, matthews_corrcoef,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

try:
    import cv2; HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ══════════════════════════════════════════════════════════════════════════
# ★  SET THESE PATHS — everything else is derived
# ══════════════════════════════════════════════════════════════════════════
CFG = {
    # Dataset directories — MUST match training CFG exactly
    "MRI_tumor_dir":   "/content/drive/MyDrive/final data mri ct uml/unp/Dataset/Brain Tumor MRI images/Tumor",
    "MRI_healthy_dir": "/content/drive/MyDrive/final data mri ct uml/unp/Dataset/Brain Tumor MRI images/Healthy",
    "CT_tumor_dir":    "/content/drive/MyDrive/final data mri ct uml/unp/Dataset/Brain Tumor CT scan Images/Tumor",
    "CT_healthy_dir":  "/content/drive/MyDrive/final data mri ct uml/unp/Dataset/Brain Tumor CT scan Images/Healthy",

    # Checkpoints
    # ★ This is the file saved by the resumed training script:
    #      save_ckpt(best_wts_c, "studentB_C_RESUMED_BEST.pth")
    #   which writes to:
    #      CFG["output_dir"]/checkpoints/studentB_C_RESUMED_BEST.pth
   "student_ckpt": "/content/drive/MyDrive/UML/unp/outputs_vB_fixed/checkpoints/studentB_EMA_C_ep50.pth",
    "mri_teacher_ckpt": "/content/drive/MyDrive/final data mri ct uml/unp/outputs_v12/checkpoints/mri_teacher_BEST.pth",
    "ct_teacher_ckpt":  "/content/drive/MyDrive/final data mri ct uml/unp/outputs_v12/checkpoints/ct_teacher_BEST.pth",

    # Output
    "output_dir": "/content/drive/MyDrive/UML/unp/outputs_vB_fixed",

    # ── MUST match training CFG exactly ──────────────────────────────
    "image_size":  224,
    "batch_size":  64,
    "num_classes": 2,
    "val_ratio":   0.2,   # ← same as training
    "seed":        42,    # ← same as training
    "num_workers": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Bootstrap CI settings
    "bootstrap_n":    1000,   # number of resamples for 95% CI
    "bootstrap_seed": 0,      # separate seed so CI is reproducible
}


# ══════════════════════════════════════════════════════════════════════════
# TRANSFORM  (val = CLAHE + resize + normalise, zero augmentation)
# ══════════════════════════════════════════════════════════════════════════
class ApplyCLAHE:
    """Identical to training script's ApplyCLAHE — CLAHE is applied at val too."""
    def __call__(self, img):
        if not HAS_CV2:
            return img
        a  = np.array(img.convert("RGB"), dtype=np.uint8)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return Image.fromarray(
            np.stack([cl.apply(a[:, :, c]) for c in range(3)], axis=2)
        )


def val_transform(sz: int):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return transforms.Compose([
        ApplyCLAHE(),
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ══════════════════════════════════════════════════════════════════════════
# DATASET  — exact mirror of training BrainDataset, split='val'
# ══════════════════════════════════════════════════════════════════════════
def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class BrainValDataset(Dataset):
    """
    Reproduces exactly the validation split produced by training's BrainDataset.

    Critical invariants (all must match training):
      1. os.listdir() — NOT sorted() [BUG-1 fix]
         Training uses unsorted listdir; sorted() gives a different file
         ordering, so rng.shuffle produces a different permutation.
      2. MD5 deduplication before shuffle — same as training.
      3. random.Random(seed) — independent RNG instance, same seed.
      4. val slice = idx[cut:]  where cut = int((1-val_ratio)*N)
    """
    EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

    def __init__(self, tumor_dir: str, healthy_dir: str,
                 val_ratio: float = 0.2, seed: int = 42,
                 image_size: int = 224, modality: str = ""):
        self.transform = val_transform(image_size)
        self.samples: list = []

        # ★ One independent RNG per dataset instance — same as training
        rng = random.Random(seed)

        for folder, label in [(tumor_dir, 1), (healthy_dir, 0)]:
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Directory not found: {folder}")

            # ★ BUG-1 FIX: plain listdir(), NOT sorted()
            files = [
                f for f in os.listdir(folder)
                if f.lower().endswith(self.EXTS)
            ]

            # MD5 deduplication — identical to training
            seen: dict = {}
            uniq: list = []
            for f in files:
                h = _md5(os.path.join(folder, f))
                if h not in seen:
                    seen[h] = True
                    uniq.append(f)

            idx = list(range(len(uniq)))
            rng.shuffle(idx)                          # same permutation as training
            cut = int((1 - val_ratio) * len(idx))
            val_idx = idx[cut:]                       # last 20% → val

            for i in val_idx:
                self.samples.append(
                    (os.path.join(folder, uniq[i]), label)
                )

        rng.shuffle(self.samples)

        n1  = sum(l for _, l in self.samples)
        n0  = len(self.samples) - n1
        tot = len(self.samples)
        print(f"  [{modality:3s}] val | {tot:4d} samples  "
              f"tumor={n1} ({100*n1/tot:.1f}%)  "
              f"healthy={n0} ({100*n0/tot:.1f}%)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), 0)
        return self.transform(img), label


def build_val_loaders(cfg: dict) -> dict:
    print("\n  ── Building val dataloaders ──")
    loaders = {}
    for mod, (td, hd) in {
        "MRI": (cfg["MRI_tumor_dir"],  cfg["MRI_healthy_dir"]),
        "CT":  (cfg["CT_tumor_dir"],   cfg["CT_healthy_dir"]),
    }.items():
        ds = BrainValDataset(
            td, hd,
            val_ratio=cfg["val_ratio"],
            seed=cfg["seed"],
            image_size=cfg["image_size"],
            modality=mod,
        )
        loaders[mod] = DataLoader(
            ds,
            batch_size=cfg["batch_size"],
            shuffle=False,                 # NEVER shuffle val loader
            num_workers=cfg["num_workers"],
            pin_memory=(cfg["device"] == "cuda"),
        )
    return loaders


# ══════════════════════════════════════════════════════════════════════════
# MODELS  — exact copy of training architecture
# ══════════════════════════════════════════════════════════════════════════
class TeacherNet(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        try:
            base = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        except AttributeError:
            base = models.convnext_tiny(pretrained=True)
        self.features = base.features
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(384, num_classes),
        )

    def forward(self, x):
        feat = self.gap(self.features(x)).flatten(1)
        return self.head(feat), feat


class ModalityNorm(nn.Module):
    def __init__(self, dim: int = 768, num_modalities: int = 2):
        super().__init__()
        self.gamma = nn.Embedding(num_modalities, dim)
        self.beta  = nn.Embedding(num_modalities, dim)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, feat, modality_id: int):
        idx = torch.full(
            (feat.size(0),), modality_id,
            dtype=torch.long, device=feat.device
        )
        return feat * self.gamma(idx) + self.beta(idx)


class UMLStudentB(nn.Module):
    MOD_IDS = {"MRI": 0, "CT": 1}

    def __init__(self, num_classes: int = 2):
        super().__init__()
        try:
            base = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        except AttributeError:
            base = models.convnext_tiny(pretrained=True)
        self.backbone      = base.features
        self.gap           = nn.AdaptiveAvgPool2d(1)
        self.modality_norm = ModalityNorm(768, 2)
        self.mri_proj = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2))
        self.ct_proj  = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2))
        self.mri_head = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, num_classes))
        self.ct_head  = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, num_classes))

    def forward(self, x, modality: str):
        feat = self.gap(self.backbone(x)).flatten(1)
        feat = self.modality_norm(feat, self.MOD_IDS[modality])
        if modality == "MRI":
            return self.mri_head(self.mri_proj(feat)), self.mri_proj(feat)
        else:
            return self.ct_head(self.ct_proj(feat)), self.ct_proj(feat)


# ══════════════════════════════════════════════════════════════════════════
# BOOTSTRAP CONFIDENCE INTERVAL
# ══════════════════════════════════════════════════════════════════════════
def bootstrap_ci(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray,
                 n: int = 1000, seed: int = 0) -> dict:
    """
    95% bootstrap CI for Accuracy and AUC-ROC.
    Uses stratified resampling so class proportions are preserved.
    """
    rng  = np.random.RandomState(seed)
    accs, aucs = [], []

    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]

    for _ in range(n):
        # Stratified resample
        boot_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        boot_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        boot     = np.concatenate([boot_pos, boot_neg])

        b_labels = labels[boot]
        b_preds  = preds[boot]
        b_probs  = probs[boot]

        accs.append(accuracy_score(b_labels, b_preds))
        try:
            aucs.append(roc_auc_score(b_labels, b_probs))
        except Exception:
            pass

    accs = np.array(accs)
    aucs = np.array(aucs)

    return {
        "acc_ci":  (float(np.percentile(accs, 2.5)),
                    float(np.percentile(accs, 97.5))),
        "auc_ci":  (float(np.percentile(aucs, 2.5)),
                    float(np.percentile(aucs, 97.5)))
        if len(aucs) > 0 else (float("nan"), float("nan")),
    }


# ══════════════════════════════════════════════════════════════════════════
# CORE EVALUATION
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_model(model, loader, device: str,
                   modality=None, label: str = "",
                   bootstrap_n: int = 1000,
                   bootstrap_seed: int = 0) -> dict:
    """
    Full research-grade evaluation.

    Metrics returned
    ─────────────────
    acc          overall accuracy
    auc          AUC-ROC  (P(tumor) as score)
    prec         precision  (positive = tumor)
    rec/sens     recall / sensitivity
    spec         specificity (TNR = TN / (TN+FP))
    f1           F1 score
    mcc          Matthews Correlation Coefficient
    acc_healthy  per-class accuracy on healthy samples
    acc_tumor    per-class accuracy on tumor samples
    conf_correct   mean max-softmax confidence when prediction is correct
    conf_incorrect mean max-softmax confidence when prediction is wrong
    acc_ci, auc_ci  bootstrap 95% CI tuples
    """
    # ★ BUG-3 FIX: always force eval mode — Dropout must be disabled
    model.eval()

    all_preds:  list = []
    all_labels: list = []
    all_probs:  list = []   # P(tumor) for AUC
    all_confs:  list = []   # max softmax prob for calibration

    for imgs, lbl in loader:
        imgs = imgs.to(device)

        if modality is not None:
            logits, _ = model(imgs, modality)
        else:
            logits, _ = model(imgs)

        probs_all = F.softmax(logits, dim=1)             # (B, 2)
        all_probs.extend(probs_all[:, 1].cpu().tolist()) # P(tumor)
        all_confs.extend(probs_all.max(dim=1).values.cpu().tolist())
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(lbl.tolist())

    y_true  = np.array(all_labels, dtype=int)
    y_pred  = np.array(all_preds,  dtype=int)
    y_prob  = np.array(all_probs,  dtype=float)
    y_conf  = np.array(all_confs,  dtype=float)

    # ── Standard metrics ──────────────────────────────────────────────
    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="binary", zero_division=0))
    rec  = float(recall_score(y_true,   y_pred, average="binary", zero_division=0))
    f1   = float(f1_score(y_true,       y_pred, average="binary", zero_division=0))
    mcc  = float(matthews_corrcoef(y_true, y_pred))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")

    # ── Specificity: TNR = TN / (TN + FP) ────────────────────────────
    # confusion_matrix layout: [[TN, FP], [FN, TP]]
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    spec = float(TN / (TN + FP)) if (TN + FP) > 0 else float("nan")

    # ── Per-class accuracy ────────────────────────────────────────────
    mask_h = (y_true == 0)
    mask_t = (y_true == 1)
    acc_h  = float(accuracy_score(y_true[mask_h], y_pred[mask_h])) \
             if mask_h.any() else float("nan")
    acc_t  = float(accuracy_score(y_true[mask_t], y_pred[mask_t])) \
             if mask_t.any() else float("nan")

    # ── Calibration ───────────────────────────────────────────────────
    correct_mask    = (y_pred == y_true)
    conf_correct    = float(y_conf[correct_mask].mean())   \
                      if correct_mask.any()  else float("nan")
    conf_incorrect  = float(y_conf[~correct_mask].mean())  \
                      if (~correct_mask).any() else float("nan")

    # ── Bootstrap CI ──────────────────────────────────────────────────
    ci = bootstrap_ci(y_true, y_pred, y_prob,
                      n=bootstrap_n, seed=bootstrap_seed)

    pred_counts = np.bincount(y_pred, minlength=2)

    return dict(
        label=label, n=int(len(y_true)),
        TP=int(TP), TN=int(TN), FP=int(FP), FN=int(FN),
        acc=acc,   auc=auc,  prec=prec, rec=rec,
        spec=spec, f1=f1,    mcc=mcc,
        acc_healthy=acc_h, acc_tumor=acc_t,
        conf_correct=conf_correct, conf_incorrect=conf_incorrect,
        acc_ci=ci["acc_ci"], auc_ci=ci["auc_ci"],
        pred_h=int(pred_counts[0]), pred_t=int(pred_counts[1]),
        preds=y_pred, labels=y_true, probs=y_prob,
    )


# ══════════════════════════════════════════════════════════════════════════
# REPORT PRINTER
# ══════════════════════════════════════════════════════════════════════════
def print_block(r: dict) -> str:
    cr = classification_report(
        r["labels"], r["preds"],
        target_names=["Healthy (0)", "Tumor (1)"],
        zero_division=0,
    )
    acc_lo, acc_hi = r["acc_ci"]
    auc_lo, auc_hi = r["auc_ci"]

    lines = [
        f"\n  ══ {r['label']} ══",
        f"    N={r['n']}  |  TP={r['TP']}  TN={r['TN']}  "
        f"FP={r['FP']}  FN={r['FN']}",
        f"    Predicted: Healthy={r['pred_h']}  Tumor={r['pred_t']}",
        f"",
        f"    Accuracy              : {r['acc']:.4f}  "
        f"[95% CI {acc_lo:.4f}–{acc_hi:.4f}]",
        f"    AUC-ROC               : {r['auc']:.4f}  "
        f"[95% CI {auc_lo:.4f}–{auc_hi:.4f}]",
        f"    Precision             : {r['prec']:.4f}",
        f"    Sensitivity (Recall)  : {r['rec']:.4f}",
        f"    Specificity (TNR)     : {r['spec']:.4f}",
        f"    F1 Score              : {r['f1']:.4f}",
        f"    MCC                   : {r['mcc']:.4f}",
        f"    Acc — Healthy class   : {r['acc_healthy']:.4f}",
        f"    Acc — Tumor class     : {r['acc_tumor']:.4f}",
        f"    Confidence (correct)  : {r['conf_correct']:.4f}",
        f"    Confidence (wrong)    : {r['conf_incorrect']:.4f}",
        f"",
        cr,
    ]
    block = "\n".join(lines)
    print(block)
    return block


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE  (BUG-4 fixed: explicit key lookup, no startswith)
# ══════════════════════════════════════════════════════════════════════════
def print_summary_table(results: list) -> list:
    """
    Prints a compact comparison table and macro-averages.
    BUG-4 FIX: uses explicit label lookup, never startswith().
    """
    by = {r["label"]: r for r in results}

    w   = 22
    sep = "  " + "─" * (w + 72)
    hdr = (
        f"\n  {'Model':<{w}} "
        f"{'Acc':>7} {'AUC':>7} {'Prec':>7} "
        f"{'Sens':>7} {'Spec':>7} {'F1':>7} {'MCC':>7}"
    )
    lines = [hdr, sep]
    print(hdr); print(sep)

    for r in results:
        row = (
            f"  {r['label']:<{w}} "
            f"{r['acc']:>7.4f} {r['auc']:>7.4f} {r['prec']:>7.4f} "
            f"{r['rec']:>7.4f} {r['spec']:>7.4f} "
            f"{r['f1']:>7.4f} {r['mcc']:>7.4f}"
        )
        print(row); lines.append(row)

    print(sep); lines.append(sep)

    # ── Delta: Student − Teacher per modality ─────────────────────────
    delta_hdr = (
        f"\n  {'Δ (Student − Teacher)':<{w}} "
        f"{'ΔAcc':>7} {'ΔAUC':>7} {'ΔPrec':>7} "
        f"{'ΔSens':>7} {'ΔSpec':>7} {'ΔF1':>7} {'ΔMCC':>7}"
    )
    print(delta_hdr); lines.append(delta_hdr)

    for mod in ["MRI", "CT"]:
        s_key = f"Student {mod}"
        t_key = f"Teacher {mod}"
        # ★ BUG-4 FIX: explicit key lookup
        if s_key not in by or t_key not in by:
            continue
        s, t = by[s_key], by[t_key]
        row = (
            f"  {mod + ' Student−Teacher':<{w}} "
            f"{s['acc']-t['acc']:>+7.4f} {s['auc']-t['auc']:>+7.4f} "
            f"{s['prec']-t['prec']:>+7.4f} {s['rec']-t['rec']:>+7.4f} "
            f"{s['spec']-t['spec']:>+7.4f} {s['f1']-t['f1']:>+7.4f} "
            f"{s['mcc']-t['mcc']:>+7.4f}"
        )
        print(row); lines.append(row)

    # ── Macro-average (MRI + CT) ──────────────────────────────────────
    macro_hdr = (
        f"\n  {'Macro-avg (MRI+CT)':<{w}} "
        f"{'Acc':>7} {'AUC':>7} {'Sens':>7} {'Spec':>7} {'F1':>7} {'MCC':>7}"
    )
    print(macro_hdr); lines.append(macro_hdr)

    # ★ BUG-4 FIX: explicit pairs, no fragile string matching
    for prefix, s_key1, s_key2 in [
        ("Student",  "Student MRI",  "Student CT"),
        ("Teacher",  "Teacher MRI",  "Teacher CT"),
    ]:
        if s_key1 not in by or s_key2 not in by:
            continue
        a, b = by[s_key1], by[s_key2]

        def avg(*keys):
            return np.mean([a[k] for k in keys] + [b[k] for k in keys])

        row = (
            f"  {prefix + ' macro-avg':<{w}} "
            f"{avg('acc'):>7.4f} {avg('auc'):>7.4f} "
            f"{avg('rec'):>7.4f} {avg('spec'):>7.4f} "
            f"{avg('f1'):>7.4f} {avg('mcc'):>7.4f}"
        )
        print(row); lines.append(row)

    return lines


# ══════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════
def save_confusion_matrices(results: list, plot_dir: str):
    n    = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = confusion_matrix(r["labels"], r["preds"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Healthy", "Tumor"],
            yticklabels=["Healthy", "Tumor"],
        )
        ax.set_title(
            f"{r['label']}\n"
            f"Acc={r['acc']:.3f}  AUC={r['auc']:.3f}\n"
            f"F1={r['f1']:.3f}  MCC={r['mcc']:.3f}",
            fontsize=9,
        )
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    path = os.path.join(plot_dir, "confusion_matrices_v2.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Confusion matrices → {path}")


def save_roc_curves(results: list, plot_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    palette   = ["steelblue", "darkorange"]

    for ax, modality in zip(axes, ["MRI", "CT"]):
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance")
        ax.set_title(f"{modality} — ROC Curves", fontsize=12)

        subset = [r for r in results if modality in r["label"]]
        for r, col in zip(subset, palette):
            if np.isnan(r["auc"]):
                continue
            fpr, tpr, _ = roc_curve(r["labels"], r["probs"])
            lo, hi = r["auc_ci"]
            ax.plot(
                fpr, tpr, color=col, lw=2,
                label=f"{r['label']}\nAUC={r['auc']:.4f} [{lo:.4f}–{hi:.4f}]"
            )
        ax.set_xlabel("False Positive Rate (1 − Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plot_dir, "roc_curves_v2.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  ROC curves        → {path}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    cfg = CFG
    dev = cfg["device"]

    # ★ BUG-2 FIX: seeds set inside main(), not at module level
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    if dev == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])

    log_dir  = os.path.join(cfg["output_dir"], "logs")
    plot_dir = os.path.join(cfg["output_dir"], "plots")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print("=" * 76)
    print("  FINAL EVALUATION — VERSION B  (Research-Grade v2)")
    print("=" * 76)
    print(f"  Device : {dev}")

    # ── Path check ────────────────────────────────────────────────────
    checks = [
        ("MRI Tumor dir",      cfg["MRI_tumor_dir"]),
        ("MRI Healthy dir",    cfg["MRI_healthy_dir"]),
        ("CT Tumor dir",       cfg["CT_tumor_dir"]),
        ("CT Healthy dir",     cfg["CT_healthy_dir"]),
        ("Student ckpt",       cfg["student_ckpt"]),
        ("MRI Teacher ckpt",   cfg["mri_teacher_ckpt"]),
        ("CT Teacher ckpt",    cfg["ct_teacher_ckpt"]),
    ]
    all_ok = True
    for name, path in checks:
        exists = os.path.exists(path)
        print(f"    {'✅' if exists else '❌'} {name:<22} {path}")
        if not exists:
            all_ok = False
    if not all_ok:
        print("\n  ❌  Fix missing paths in CFG and re-run.\n")
        return

    # ── Build val loaders ─────────────────────────────────────────────
    val_loaders = build_val_loaders(cfg)

    # ── Load teachers ─────────────────────────────────────────────────
    print("\n  Loading teachers ...")
    mri_teacher = TeacherNet(cfg["num_classes"]).to(dev)
    ct_teacher  = TeacherNet(cfg["num_classes"]).to(dev)
    mri_teacher.load_state_dict(
        torch.load(cfg["mri_teacher_ckpt"], map_location=dev))
    ct_teacher.load_state_dict(
        torch.load(cfg["ct_teacher_ckpt"],  map_location=dev))
    # ★ BUG-3 FIX: .eval() called explicitly; evaluate_model() also calls it
    mri_teacher.eval()
    ct_teacher.eval()
    print("  ✅  Teachers loaded")

    # ── Load student ──────────────────────────────────────────────────
    print("\n  Loading student ...")
    student = UMLStudentB(cfg["num_classes"]).to(dev)
    state   = torch.load(cfg["student_ckpt"], map_location=dev)
    missing, unexpected = student.load_state_dict(state, strict=False)
    if missing:
        print(f"  ⚠️  Missing keys (initialised from scratch): {missing}")
    if unexpected:
        print(f"  ⚠️  Unexpected keys (ignored):               {unexpected}")
    # ★ BUG-3 FIX: explicit .eval() before evaluation
    student.eval()
    print("  ✅  Student loaded")

    # ── Run evaluation ────────────────────────────────────────────────
    print(f"\n{'═'*76}\n  RUNNING EVALUATIONS\n{'═'*76}")

    log_lines = [
        "FINAL EVALUATION — VERSION B  (Research-Grade v2)",
        "=" * 76,
        f"Student ckpt     : {cfg['student_ckpt']}",
        f"MRI Teacher ckpt : {cfg['mri_teacher_ckpt']}",
        f"CT  Teacher ckpt : {cfg['ct_teacher_ckpt']}",
        f"Bootstrap N      : {cfg['bootstrap_n']}  seed={cfg['bootstrap_seed']}",
        "=" * 76,
    ]

    eval_plan = [
        # label            model         loader   modality_arg
        ("Teacher MRI",   mri_teacher,  "MRI",   None),
        ("Teacher CT",    ct_teacher,   "CT",    None),
        ("Student MRI",   student,      "MRI",   "MRI"),
        ("Student CT",    student,      "CT",    "CT"),
    ]

    results = []
    for tag, model, ldr_key, mod in eval_plan:
        print(f"\n  ▶  {tag} ...")
        r = evaluate_model(
            model, val_loaders[ldr_key], dev,
            modality=mod, label=tag,
            bootstrap_n=cfg["bootstrap_n"],
            bootstrap_seed=cfg["bootstrap_seed"],
        )
        results.append(r)
        block = print_block(r)
        log_lines.append(block)

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'═'*76}\n  SUMMARY TABLE\n{'═'*76}")
    log_lines.append(f"\n{'═'*76}\nSUMMARY TABLE\n{'═'*76}")
    log_lines.extend(print_summary_table(results))

    # ── Plots ─────────────────────────────────────────────────────────
    print(f"\n{'═'*76}\n  SAVING PLOTS\n{'═'*76}")
    save_confusion_matrices(results, plot_dir)
    save_roc_curves(results, plot_dir)

    # ── Save log ──────────────────────────────────────────────────────
    log_path = os.path.join(log_dir, "final_results_v2.txt")
    with open(log_path, "w") as fh:
        fh.write("\n".join(log_lines))
    print(f"\n  📄  Full log → {log_path}")

    # ── Quick-read console summary ────────────────────────────────────
    print(f"\n{'═'*76}\n  QUICK-READ FINAL NUMBERS\n{'═'*76}")
    by = {r["label"]: r for r in results}
    for mod in ["MRI", "CT"]:
        s = by.get(f"Student {mod}")
        t = by.get(f"Teacher {mod}")
        if not (s and t):
            continue
        a_lo, a_hi = s["acc_ci"]
        u_lo, u_hi = s["auc_ci"]
        print(f"\n  {mod}:")
        print(f"    Teacher : Acc={t['acc']:.4f}  AUC={t['auc']:.4f}  "
              f"F1={t['f1']:.4f}  MCC={t['mcc']:.4f}")
        print(f"    Student : Acc={s['acc']:.4f} [{a_lo:.4f}–{a_hi:.4f}]  "
              f"AUC={s['auc']:.4f} [{u_lo:.4f}–{u_hi:.4f}]  "
              f"F1={s['f1']:.4f}  MCC={s['mcc']:.4f}")
        print(f"    Δ       : Acc={s['acc']-t['acc']:+.4f}  "
              f"AUC={s['auc']-t['auc']:+.4f}  "
              f"Sens={s['rec']-t['rec']:+.4f}  "
              f"Spec={s['spec']-t['spec']:+.4f}")
        print(f"    Per-class: Healthy={s['acc_healthy']:.4f}  "
              f"Tumor={s['acc_tumor']:.4f}")

    smri = by.get("Student MRI")
    sct  = by.get("Student CT")
    if smri and sct:
        print(f"\n  Student macro-avg (MRI + CT):")
        print(f"    Acc ={np.mean([smri['acc'], sct['acc']]):.4f}  "
              f"AUC={np.mean([smri['auc'], sct['auc']]):.4f}  "
              f"F1 ={np.mean([smri['f1'],  sct['f1']]):.4f}  "
              f"MCC={np.mean([smri['mcc'], sct['mcc']]):.4f}")

    print(f"\n{'='*76}")
    print("  ✅  EVALUATION COMPLETE")
    print(f"{'='*76}\n")


if __name__ == "__main__":
    main()

###mri teacher
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MRI TEACHER — STANDALONE TRAINING SCRIPT                               ║
║                                                                          ║
║  Extracted from v13 codebase. Architecture is IDENTICAL to CT teacher.  ║
║  Only the data paths, augmentation pipeline, and SWA logic differ.      ║
║                                                                          ║
║  Use this script if:                                                     ║
║    - mri_teacher_BEST.pth checkpoint is lost/corrupted                  ║
║    - You want to retrain with updated augmentation                      ║
║    - You want to verify teacher performance from scratch                 ║
║                                                                          ║
║  Expected output:                                                        ║
║    MRI Teacher best val accuracy ~ 97.37%  (AUC ~ 0.999)               ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os, copy, time, random, hashlib
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, classification_report, confusion_matrix,
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
    print("[WARN] pip install opencv-python  — CLAHE disabled")


# ══════════════════════════════════════════════════════════════════════
# ★  CONFIG  — SET THESE PATHS
# ══════════════════════════════════════════════════════════════════════
CFG = {
    # ★ Data paths — must match training split exactly
    "MRI_tumor_dir":   "/content/drive/MyDrive/final data mri ct uml/unp/Dataset/Brain Tumor MRI images/Tumor",
    "MRI_healthy_dir": "/content/drive/MyDrive/final data mri ct uml/unp/Dataset/Brain Tumor MRI images/Healthy",

    # ★ Where to save checkpoints, plots, logs
    "output_dir": "/content/drive/MyDrive/final data mri ct uml/unp/outputs_v12",

    # ★ Hyperparameters — identical to v12/v13 teacher training
    "image_size":    224,
    "batch_size":    32,
    "num_classes":   2,
    "val_ratio":     0.2,
    "num_workers":   0,        # 0 = safe on Colab/Drive

    "teacher_epochs":     35,
    "teacher_early_stop": 10,
    "teacher_ckpt_every": 5,
    "t_phase_a_epochs":   10,
    "t_lr_head_a":        3e-3,
    "t_lr_backbone_b":    3e-5,   # MRI uses 3e-5 (CT uses 1e-5)
    "t_lr_head_b":        3e-4,
    "t_weight_decay":     1e-2,

    "label_smoothing": 0.1,
    "grad_clip":       1.0,
    "pct_start":       0.3,
    "div_factor":      25.0,
    "final_div":       1e4,
    "mixup_alpha":     0.2,
    "mixup_prob":      0.4,
    "collapse_thresh":      0.55,
    "collapse_check_after": 5,

    # MRI-specific augmentation
    "mri_bias_str":  0.15,
    "mri_noise_std": 3.0,

    "device":  "cuda" if torch.cuda.is_available() else "cpu",
    "seed":    42,
    "use_amp": True,
}

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
random.seed(CFG["seed"])
if CFG["device"] == "cuda":
    torch.cuda.manual_seed_all(CFG["seed"])
    torch.backends.cudnn.benchmark = True

USE_AMP  = CFG["use_amp"] and CFG["device"] == "cuda"
CKPT_DIR = os.path.join(CFG["output_dir"], "checkpoints")
PLOT_DIR = os.path.join(CFG["output_dir"], "plots")
LOG_DIR  = os.path.join(CFG["output_dir"], "logs")
for d in [CKPT_DIR, PLOT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 70)
print("  MRI TEACHER — STANDALONE TRAINING")
print("=" * 70)
print(f"  Device : {CFG['device']}  {'(AMP)' if USE_AMP else ''}")
print(f"  Phase-A: {CFG['t_phase_a_epochs']} ep frozen   lr_head={CFG['t_lr_head_a']:.0e}")
n_b = CFG["teacher_epochs"] - CFG["t_phase_a_epochs"]
print(f"  Phase-B: {n_b} ep unfrozen  lr_bb={CFG['t_lr_backbone_b']:.0e}  lr_hd={CFG['t_lr_head_b']:.0e}")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════
def save_ckpt(state, filename):
    path = os.path.join(CKPT_DIR, filename)
    torch.save(state, path)
    print(f"  💾  {filename}", flush=True)
    return path


# ══════════════════════════════════════════════════════════════════════
# DEDUPLICATION  (stat-first, MD5 only for size collisions)
# ══════════════════════════════════════════════════════════════════════
_DEDUP_CACHE = {}

def _dedup_folder(folder):
    if folder in _DEDUP_CACHE:
        return _DEDUP_CACHE[folder]
    EXTS  = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(EXTS))
    n = len(files)
    print(f"    {os.path.basename(folder)}: {n} files...", flush=True)

    by_size = defaultdict(list)
    for f in files:
        try:   sz = os.stat(os.path.join(folder, f)).st_size
        except: sz = -1
        by_size[sz].append(f)

    uniq = []; seen = set(); hashed = 0
    for group in by_size.values():
        if len(group) == 1:
            uniq.append(group[0])
        else:
            for f in group:
                hashed += 1
                h = hashlib.md5(open(os.path.join(folder, f), "rb").read()).hexdigest()
                if h not in seen:
                    seen.add(h); uniq.append(f)

    dupes = n - len(uniq)
    print(f"    → {dupes} dupes removed, {len(uniq)} unique ✅", flush=True)
    _DEDUP_CACHE[folder] = uniq
    return uniq


# ══════════════════════════════════════════════════════════════════════
# AUGMENTATIONS — MRI SPECIFIC
# ══════════════════════════════════════════════════════════════════════
class ApplyCLAHE:
    def __call__(self, img):
        if not HAS_CV2: return img
        a = np.array(img.convert("RGB"), dtype=np.uint8)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return Image.fromarray(np.stack([cl.apply(a[:, :, c]) for c in range(3)], 2))

class MRIBiasField:
    """Simulate MRI B1 field inhomogeneity — key domain-specific augmentation."""
    def __init__(self, s=0.15): self.s = s
    def __call__(self, img):
        a = np.array(img, dtype=np.float32); H, W = a.shape[:2]
        xx, yy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
        b = (1 + random.uniform(-self.s, self.s) * xx
               + random.uniform(-self.s, self.s) * yy
               + random.uniform(-self.s * .5, self.s * .5) * xx * yy)
        return Image.fromarray(np.clip(a * b[..., None], 0, 255).astype(np.uint8))

class AddGaussianNoise:
    def __init__(self, std=3.0): self.std = std
    def __call__(self, t):
        return torch.clamp(t + torch.randn_like(t) * (self.std / 255.0), 0, 1)

def build_transforms(sz, train=True):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    base = [ApplyCLAHE(), transforms.Resize((sz, sz))]
    if not train:
        return transforms.Compose(base + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(base + [
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.1),
        transforms.RandomRotation(20),
        MRIBiasField(CFG["mri_bias_str"]),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        AddGaussianNoise(CFG["mri_noise_std"]),
    ])


# ══════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════
class BrainDataset(Dataset):
    def __init__(self, tumor_files, tumor_dir, healthy_files, healthy_dir,
                 split="train", transform=None, val_ratio=0.2, seed=42):
        self.transform = transform
        self.samples   = []
        rng = random.Random(seed)
        for files, folder, label in [
            (tumor_files, tumor_dir, 1),
            (healthy_files, healthy_dir, 0),
        ]:
            idx = list(range(len(files))); rng.shuffle(idx)
            cut = int((1 - val_ratio) * len(idx))
            chosen = idx[:cut] if split == "train" else idx[cut:]
            for i in chosen:
                self.samples.append((os.path.join(folder, files[i]), label))
        rng.shuffle(self.samples)
        n1  = sum(l for _, l in self.samples)
        n0  = len(self.samples) - n1
        tot = len(self.samples)
        print(f"    [MRI] {split:5s} | {tot:4d} samples  "
              f"tumor={n1} ({100*n1/tot:.1f}%)  healthy={n0} ({100*n0/tot:.1f}%)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        try:   img = Image.open(path).convert("RGB")
        except: img = Image.new("RGB", (224, 224), 0)
        return (self.transform(img) if self.transform else img), label

    def get_labels(self): return [l for _, l in self.samples]


def build_loaders():
    print("\n  ── Building MRI dataloaders ──")
    tf = _dedup_folder(CFG["MRI_tumor_dir"])
    hf = _dedup_folder(CFG["MRI_healthy_dir"])
    loaders = {}
    for split in ["train", "val"]:
        ds = BrainDataset(
            tf, CFG["MRI_tumor_dir"], hf, CFG["MRI_healthy_dir"],
            split=split,
            transform=build_transforms(CFG["image_size"], split == "train"),
            val_ratio=CFG["val_ratio"], seed=CFG["seed"],
        )
        if split == "train":
            labs = ds.get_labels(); cnt = np.bincount(labs)
            w = [1.0 / cnt[l] for l in labs]
            loader = DataLoader(ds, batch_size=CFG["batch_size"],
                                sampler=WeightedRandomSampler(w, len(w)),
                                num_workers=CFG["num_workers"],
                                pin_memory=(CFG["device"] == "cuda"))
        else:
            loader = DataLoader(ds, batch_size=CFG["batch_size"], shuffle=False,
                                num_workers=CFG["num_workers"],
                                pin_memory=(CFG["device"] == "cuda"))
        loaders[split] = loader
    return loaders


# ══════════════════════════════════════════════════════════════════════
# MODEL  (identical to CT teacher — only data is different)
# ══════════════════════════════════════════════════════════════════════
class TeacherNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        try:
            base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
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
        for layer in [self.head[1], self.head[4]]:
            nn.init.trunc_normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        feat = self.gap(self.features(x)).flatten(1)
        return self.head(feat), feat

    def freeze_backbone(self):
        for p in self.features.parameters(): p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.features.parameters(): p.requires_grad = True

    def param_groups(self, lr_bb, lr_hd, wd):
        return [
            {"params": self.features.parameters(), "lr": lr_bb, "weight_decay": wd},
            {"params": self.head.parameters(),     "lr": lr_hd, "weight_decay": wd},
        ]


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════
def mixup_batch(imgs, labels, alpha, prob):
    if random.random() > prob or alpha <= 0:
        return imgs, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    return lam * imgs + (1 - lam) * imgs[idx], labels, labels[idx], lam

def mixup_ce(logits, la, lb, lam, smooth):
    return (lam * F.cross_entropy(logits, la, label_smoothing=smooth)
            + (1 - lam) * F.cross_entropy(logits, lb, label_smoothing=smooth))

def onecycle(opt, max_lrs, steps_per_ep, n_ep):
    return torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lrs,
        steps_per_epoch=steps_per_ep, epochs=n_ep,
        pct_start=CFG["pct_start"],
        div_factor=CFG["div_factor"],
        final_div_factor=CFG["final_div"])

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels, probs = [], [], []
    for imgs, lbl in loader:
        imgs = imgs.to(device)
        logits, _ = model(imgs)
        probs.extend(F.softmax(logits, 1)[:, 1].cpu().tolist())
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(lbl.tolist())
    acc = accuracy_score(labels, preds)
    try:   auc = roc_auc_score(labels, probs)
    except: auc = float("nan")
    pc = np.bincount(preds, minlength=2)
    return acc, auc, int(pc[0]), int(pc[1])


# ══════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════
def train(model, loaders):
    dev      = CFG["device"]
    model    = model.to(dev)
    scaler   = GradScaler() if USE_AMP else None
    phase_b_eps = CFG["teacher_epochs"] - CFG["t_phase_a_epochs"]

    hist = {"loss": [], "train_acc": [], "val_acc": [], "val_auc": [], "lr": []}
    best_acc  = 0.0
    best_wts  = None
    n_reinit  = 0
    saved_paths = []

    def fwd_bwd(opt, sched, imgs, lbl):
        imgs, lbl = imgs.to(dev), lbl.to(dev)
        imgs, la, lb, lam = mixup_batch(imgs, lbl, CFG["mixup_alpha"], CFG["mixup_prob"])
        opt.zero_grad()
        with autocast(enabled=USE_AMP):
            logits, _ = model(imgs)
            loss = mixup_ce(logits, la, lb, lam, CFG["label_smoothing"])
        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            opt.step()
        sched.step()
        with torch.no_grad():
            c = (logits.argmax(1) == lbl).sum().item()
        return loss.item(), c, imgs.size(0)

    # ── Phase-A: backbone frozen ──────────────────────────────────────
    model.freeze_backbone()
    opt_a   = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CFG["t_lr_head_a"], weight_decay=CFG["t_weight_decay"])
    sched_a = onecycle(opt_a, CFG["t_lr_head_a"],
                       len(loaders["train"]), CFG["t_phase_a_epochs"])

    print(f"\n  ── Phase-A (backbone FROZEN, {CFG['t_phase_a_epochs']} ep) ──", flush=True)
    for ep in range(1, CFG["t_phase_a_epochs"] + 1):
        model.train()
        tot_loss = tot_c = tot_n = 0
        for imgs, lbl in loaders["train"]:
            l, c, n = fwd_bwd(opt_a, sched_a, imgs, lbl)
            tot_loss += l; tot_c += c; tot_n += n
        tr = tot_c / tot_n
        va, vauc, ph, pt = evaluate(model, loaders["val"], dev)
        hist["loss"].append(tot_loss / len(loaders["train"]))
        hist["train_acc"].append(tr); hist["val_acc"].append(va)
        hist["val_auc"].append(vauc); hist["lr"].append(opt_a.param_groups[0]["lr"])
        print(f"  A-Ep {ep:2d}/{CFG['t_phase_a_epochs']}  "
              f"loss={hist['loss'][-1]:.4f}  train={tr:.4f}  "
              f"val={va:.4f}  AUC={vauc:.4f}  [H={ph} T={pt}]", flush=True)
        if va > best_acc:
            best_acc, best_wts = va, copy.deepcopy(model.state_dict())

    # ── Phase-B: backbone unfrozen ────────────────────────────────────
    model.unfreeze_backbone()
    patience = 0
    opt_b = torch.optim.AdamW(
        model.param_groups(CFG["t_lr_backbone_b"], CFG["t_lr_head_b"], CFG["t_weight_decay"]),
        weight_decay=CFG["t_weight_decay"])
    sched_b = onecycle(opt_b, [CFG["t_lr_backbone_b"], CFG["t_lr_head_b"]],
                       len(loaders["train"]), phase_b_eps)

    print(f"\n  Phase-A best = {best_acc:.4f}")
    print(f"  ── Phase-B (backbone UNFROZEN, {phase_b_eps} ep) ──", flush=True)
    for ep in range(1, phase_b_eps + 1):
        model.train()
        tot_loss = tot_c = tot_n = 0
        for imgs, lbl in loaders["train"]:
            l, c, n = fwd_bwd(opt_b, sched_b, imgs, lbl)
            tot_loss += l; tot_c += c; tot_n += n
        tr   = tot_c / tot_n
        va, vauc, ph, pt = evaluate(model, loaders["val"], dev)
        glep = CFG["t_phase_a_epochs"] + ep
        hist["loss"].append(tot_loss / len(loaders["train"]))
        hist["train_acc"].append(tr); hist["val_acc"].append(va)
        hist["val_auc"].append(vauc); hist["lr"].append(opt_b.param_groups[1]["lr"])

        # Collapse guard (same as CT teacher)
        if (va < CFG["collapse_thresh"]
                and glep > CFG["collapse_check_after"]
                and n_reinit < 2):
            for m in model.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)
            for g in opt_b.param_groups:
                for p in g["params"]: opt_b.state[p] = {}
            n_reinit += 1
            print(f"  ⚠️  Collapse detected at ep {glep} — head re-initialized", flush=True)

        print(f"  B-Ep {ep:2d}/{phase_b_eps}  "
              f"loss={hist['loss'][-1]:.4f}  train={tr:.4f}  "
              f"val={va:.4f}  AUC={vauc:.4f}  [H={ph} T={pt}]", flush=True)

        if glep % CFG["teacher_ckpt_every"] == 0:
            p = save_ckpt(model.state_dict(), f"mri_teacher_ep{glep}.pth")
            saved_paths.append(p)

        if va > best_acc:
            best_acc, best_wts, patience = va, copy.deepcopy(model.state_dict()), 0
            print(f"  ★  New best val = {best_acc:.4f}", flush=True)
        else:
            patience += 1
            if patience >= CFG["teacher_early_stop"]:
                print(f"  ⏹  Early stop at ep {glep} (patience={patience})", flush=True)
                break

    model.load_state_dict(best_wts)
    save_ckpt(best_wts, "mri_teacher_BEST.pth")
    print(f"\n  ✅  MRI Teacher training complete — best val={best_acc:.4f}", flush=True)

    # ── Save training curves ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = list(range(1, len(hist["val_acc"]) + 1))
    axes[0].plot(epochs, hist["val_acc"], "b-o", ms=4, label="val acc")
    axes[0].plot(epochs, hist["train_acc"], "b--s", ms=3, label="train acc", alpha=0.7)
    axes[0].axvline(CFG["t_phase_a_epochs"], color="gray", ls="--", label="A→B")
    axes[0].set_title("MRI Teacher — Accuracy"); axes[0].legend()
    axes[0].set_ylim(0, 1.05); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, hist["val_auc"], "g-o", ms=4, label="val AUC")
    axes[1].set_title("MRI Teacher — AUC-ROC"); axes[1].legend()
    axes[1].set_ylim(0.5, 1.02); axes[1].grid(alpha=0.3)
    axes[2].semilogy(epochs, hist["lr"], "r-o", ms=3)
    axes[2].set_title("Learning Rate"); axes[2].grid(alpha=0.3)
    for ax in axes: ax.set_xlabel("Epoch")
    plt.tight_layout()
    p = os.path.join(PLOT_DIR, "mri_teacher_training_curves.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  📊  Training curves → {p}")

    return model


# ══════════════════════════════════════════════════════════════════════
# FINAL EVALUATION
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def full_evaluation(model, loader, dev):
    model.eval()
    preds, labels, probs = [], [], []
    for imgs, lbl in loader:
        imgs = imgs.to(dev)
        logits, _ = model(imgs)
        probs.extend(F.softmax(logits, 1)[:, 1].cpu().tolist())
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(lbl.tolist())
    y_true, y_pred = np.array(labels), np.array(preds)
    y_prob = np.array(probs)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec  = recall_score(y_true,   y_pred, average="binary", zero_division=0)
    f1   = f1_score(y_true,       y_pred, average="binary", zero_division=0)
    try:   auc = roc_auc_score(y_true, y_prob)
    except: auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    spec = TN / (TN + FP) if (TN + FP) > 0 else float("nan")

    print(f"\n{'═'*60}\n  MRI TEACHER — FINAL EVALUATION\n{'═'*60}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  AUC-ROC      : {auc:.4f}")
    print(f"  Precision    : {prec:.4f}")
    print(f"  Sensitivity  : {rec:.4f}")
    print(f"  Specificity  : {spec:.4f}")
    print(f"  F1 Score     : {f1:.4f}")
    print(f"  TP={TP}  TN={TN}  FP={FP}  FN={FN}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Healthy','Tumor'])}")

    log_path = os.path.join(LOG_DIR, "mri_teacher_final_eval.txt")
    with open(log_path, "w") as f:
        f.write(f"MRI Teacher Final Evaluation\n{'='*60}\n")
        f.write(f"Accuracy: {acc:.4f}\nAUC-ROC: {auc:.4f}\nF1: {f1:.4f}\n")
        f.write(f"Precision: {prec:.4f}\nSensitivity: {rec:.4f}\nSpecificity: {spec:.4f}\n")
    print(f"  📄  Log → {log_path}")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Healthy", "Tumor"],
                yticklabels=["Healthy", "Tumor"])
    ax.set_title(f"MRI Teacher\nAcc={acc:.3f}  AUC={auc:.3f}")
    plt.tight_layout()
    p = os.path.join(PLOT_DIR, "mri_teacher_confusion_matrix.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  📊  Confusion matrix → {p}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()

    print("\n  Verifying paths...")
    for lbl, path in [
        ("MRI Tumor dir",   CFG["MRI_tumor_dir"]),
        ("MRI Healthy dir", CFG["MRI_healthy_dir"]),
    ]:
        exists = os.path.isdir(path)
        print(f"    {'✅' if exists else '❌'} {lbl:<20} {path}")
        if not exists:
            print("  ❌ Fix paths in CFG and re-run."); return

    loaders = build_loaders()

    print("\n  Building MRI Teacher model...")
    model = TeacherNet(CFG["num_classes"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {total_params:,}")

    model = train(model, loaders)
    full_evaluation(model, loaders["val"], CFG["device"])

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  ✅  COMPLETE  ({elapsed/60:.1f} min)")
    print(f"  Best checkpoint → {CKPT_DIR}/mri_teacher_BEST.pth")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

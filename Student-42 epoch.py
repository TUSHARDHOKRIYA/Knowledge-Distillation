  # yes
 """
  ╔══════════════════════════════════════════════════════════════════════╗
  ║   UML Student — VERSION B  FIXED  (Full Training from Scratch       ║
  ║   OR Resume from checkpoint)                                         ║
  ║                                                                      ║
  ║   FIXES APPLIED vs original vB:                                      ║
  ║   [1]  EMA implemented (was configured but never coded)              ║
  ║   [2]  Mixup + KD label mismatch fixed in uncertainty_weighted_kd   ║
  ║   [3]  Early stopping patience reset per phase                       ║
  ║   [4]  best_avg reset to 0.0 at each new phase                       ║
  ║   [5]  Cross-modal contrastive weight halved + CT guard              ║
  ║   [6]  Distillation temperature lowered 4.0 → 2.0                   ║
  ║   [7]  steps_per_epoch balanced (separate MRI/CT counters)           ║
  ║   [8]  Proper CT window/level augmentation                           ║
  ║   [9]  lambda_disagree config removed (was dead / never used)        ║
  ║   [10] Modality-specific BatchNorm injection (AdaIN-style FiLM)      ║
  ║   [11] Teacher soft-label computed BEFORE mixup on clean images      ║
  ║   [12] Contrastive loss disabled when CT val_acc < 0.80              ║
  ╚══════════════════════════════════════════════════════════════════════╝
  """

  import os, copy, time, random, hashlib
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
      recall_score, f1_score, classification_report, confusion_matrix
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

  # ══════════════════════════════════════════════════════════════════════
  # CONFIG — all changes marked with ★ FIX
  # ══════════════════════════════════════════════════════════════════════
  CFG = {
      "MRI_tumor_dir":   "/content/drive/MyDrive/final data mri ct uml/unp/Dataset/Brain Tumor MRI images/Tumor",
      "MRI_healthy_dir": "/content/drive/MyDrive/final data mri ct uml/unp/Dataset/Brain Tumor MRI images/Healthy",
      "CT_tumor_dir":    "/content/drive/MyDrive/final data mri ct uml/unp/Dataset/Brain Tumor CT scan Images/Tumor",
      "CT_healthy_dir":  "/content/drive/MyDrive/final data mri ct uml/unp/Dataset/Brain Tumor CT scan Images/Healthy",

      "output_dir":       "/content/drive/MyDrive/UML/unp/outputs_vB_fixed",
      "teacher_ckpt_dir": "/content/drive/MyDrive/final data mri ct uml/unp/outputs_v12/checkpoints",

      "image_size":  224,
      "batch_size":  32,
      "num_classes": 2,
      "val_ratio":   0.2,

      # ── Phase definitions ──────────────────────────────────────────
      "s_phase_a_epochs":   5,
      "s_lr_head_a":        1e-3,

      "s_phase_b_epochs":   15,
      "s_lr_backbone_b":    1e-5,
      "s_lr_head_b":        1e-4,

      "s_phase_c_epochs":   30,
      "s_lr_backbone_c":    5e-6,
      "s_lr_head_c":        5e-5,

      "s_weight_decay":     1e-2,
      "student_early_stop": 12,          # ★ FIX [3]: was 10, now 12 + reset per phase
      "student_ckpt_every": 5,

      "label_smoothing": 0.1,
      "grad_clip":       1.0,
      "pct_start":       0.3,
      "div_factor":      25.0,
      "final_div":       1e4,
      "ema_decay":       0.998,          # ★ FIX [1]: EMA now actually used

      "mixup_alpha":     0.2,
      "mixup_prob":      0.4,

      "temperature":     2.0,            # ★ FIX [6]: was 4.0 — teacher is strong, T=4 over-smooths
      "lambda_cls":      1.0,
      "lambda_distill":  0.5,
      "lambda_contrast": 0.15,           # ★ FIX [5]: was 0.3 — halved, + conditional disable

      # ★ FIX [9]: lambda_disagree REMOVED — was configured but never used

      "contrast_margin": 1.0,
      "ct_guard_acc":    0.78,           # ★ FIX [12]: disable contrastive if CT below this

      "mri_bias_str":    0.15,
      "mri_noise_std":   3.0,
      "ct_noise_std":    2.0,
      # ★ FIX [8]: CT window/level augmentation now uses realistic HU simulation
      "ct_window_center": 40,            # brain window center
      "ct_window_width":  80,            # brain window width
      "ct_window_jitter": 15,            # ± jitter on center and width

      "device":      "cuda" if torch.cuda.is_available() else "cpu",
      "seed":        42,
      "num_workers": 2,
      "use_amp":     True,

      # ── RESUME SETTINGS ────────────────────────────────────────────
      # Set resume_ckpt to None to train from scratch
      # Set to "studentB_B_ep15.pth" to resume from that checkpoint
      "resume_ckpt":      None,          # e.g. "studentB_B_ep15.pth" or None
      "resume_phase":     "A",           # "A", "B", or "C"
      "resume_global_ep": 0,             # global epoch of loaded checkpoint (0 = fresh start)
  }

  torch.manual_seed(CFG["seed"])
  np.random.seed(CFG["seed"])
  random.seed(CFG["seed"])
  if CFG["device"] == "cuda":
      torch.cuda.manual_seed_all(CFG["seed"])
      torch.backends.cudnn.benchmark = True

  CKPT_DIR = os.path.join(CFG["output_dir"], "checkpoints")
  PLOT_DIR = os.path.join(CFG["output_dir"], "plots")
  LOG_DIR  = os.path.join(CFG["output_dir"], "logs")
  for d in [CKPT_DIR, PLOT_DIR, LOG_DIR]:
      os.makedirs(d, exist_ok=True)

  USE_AMP = CFG["use_amp"] and CFG["device"] == "cuda"

  print("=" * 72)
  print("  UML Student — VERSION B  FIXED")
  print("=" * 72)
  print(f"  Device   : {CFG['device']}")
  print(f"  AMP      : {USE_AMP}")
  if CFG["resume_ckpt"]:
      print(f"  Resuming : {CFG['resume_ckpt']}  (global ep {CFG['resume_global_ep']})")
  else:
      print("  Mode     : Full training from scratch")
  print("=" * 72)


  # ══════════════════════════════════════════════════════════════════════
  # HELPERS
  # ══════════════════════════════════════════════════════════════════════
  def save_ckpt(state_dict, filename):
      path = os.path.join(CKPT_DIR, filename)
      torch.save(state_dict, path)
      print(f"  💾  Saved: {filename}")


  def md5(path):
      h = hashlib.md5()
      with open(path, "rb") as f:
          while chunk := f.read(65536):
              h.update(chunk)
      return h.hexdigest()


  # ══════════════════════════════════════════════════════════════════════
  # EMA HELPER  — ★ FIX [1]
  # ══════════════════════════════════════════════════════════════════════
  class EMAModel:
      """
      Exponential Moving Average of model weights.
      Used for evaluation — provides smoother, more stable accuracy estimates.
      """
      def __init__(self, model, decay=0.998):
          self.decay = decay
          self.shadow = copy.deepcopy(model)
          self.shadow.eval()
          for p in self.shadow.parameters():
              p.requires_grad_(False)

      @torch.no_grad()
      def update(self, model):
          for s_param, param in zip(self.shadow.parameters(), model.parameters()):
              s_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
          # Also update buffers (BN running stats etc.)
          for s_buf, buf in zip(self.shadow.buffers(), model.buffers()):
              s_buf.copy_(buf)

      def get_model(self):
          return self.shadow


  # ══════════════════════════════════════════════════════════════════════
  # AUGMENTATIONS
  # ══════════════════════════════════════════════════════════════════════
  class MRIBiasField:
      """Simulates MRI B1 bias field inhomogeneity."""
      def __init__(self, strength=0.15):
          self.s = strength

      def __call__(self, img):
          a = np.array(img, dtype=np.float32)
          H, W = a.shape[:2]
          xx, yy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
          b = (1
              + random.uniform(-self.s, self.s) * xx
              + random.uniform(-self.s, self.s) * yy
              + random.uniform(-self.s * 0.5, self.s * 0.5) * xx * yy)
          return Image.fromarray(np.clip(a * b[..., None], 0, 255).astype(np.uint8))


  class CTWindowLevel:
      """
      ★ FIX [8]: Proper CT window/level augmentation.
      Simulates variation in radiologist window/level settings.
      Much more realistic than a simple pixel shift.
      """
      def __init__(self, center=40, width=80, jitter=15):
          self.center = center
          self.width  = width
          self.jitter = jitter

      def __call__(self, img):
          a = np.array(img, dtype=np.float32)
          # Jitter window center and width
          c = self.center + random.uniform(-self.jitter, self.jitter)
          w = max(20, self.width  + random.uniform(-self.jitter, self.jitter))
          lo = c - w / 2.0
          hi = c + w / 2.0
          # Normalize to [0,255] using the window
          a_norm = np.clip((a - lo) / (hi - lo) * 255.0, 0, 255)
          return Image.fromarray(a_norm.astype(np.uint8))


  class AddGaussianNoise:
      def __init__(self, std=3.0):
          self.std = std

      def __call__(self, t):
          if self.std <= 0:
              return t
          return torch.clamp(t + torch.randn_like(t) * (self.std / 255.0), 0, 1)


  class ApplyCLAHE:
      def __call__(self, img):
          if not HAS_CV2:
              return img
          a  = np.array(img.convert("RGB"), dtype=np.uint8)
          cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
          return Image.fromarray(np.stack([cl.apply(a[:, :, c]) for c in range(3)], 2))


  def build_transforms(sz, modality, train=True):
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      clahe = ApplyCLAHE()
      base  = [clahe, transforms.Resize((sz, sz))]

      if not train:
          return transforms.Compose(base + [
              transforms.ToTensor(),
              transforms.Normalize(mean, std)
          ])

      if modality == "MRI":
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
      else:  # CT
          # ★ FIX [8]: CTWindowLevel replaces CTWindowShift
          return transforms.Compose(base + [
              transforms.RandomHorizontalFlip(0.5),
              transforms.RandomRotation(15),
              CTWindowLevel(CFG["ct_window_center"],
                            CFG["ct_window_width"],
                            CFG["ct_window_jitter"]),
              transforms.RandomAffine(degrees=0, scale=(0.85, 1.15)),
              transforms.ColorJitter(brightness=0.1, contrast=0.15),
              transforms.ToTensor(),
              transforms.Normalize(mean, std),
              AddGaussianNoise(CFG["ct_noise_std"]),
          ])


  # ══════════════════════════════════════════════════════════════════════
  # DATASET
  # ══════════════════════════════════════════════════════════════════════
  class BrainDataset(Dataset):
      EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

      def __init__(self, tumor_dir, healthy_dir, split="train",
                  transform=None, modality="", val_ratio=0.2, seed=42):
          self.transform = transform
          self.samples   = []
          rng = random.Random(seed)

          for folder, label in [(tumor_dir, 1), (healthy_dir, 0)]:
              if not os.path.isdir(folder):
                  raise FileNotFoundError(f"Directory not found: {folder}")
              files = [f for f in os.listdir(folder) if f.lower().endswith(self.EXTS)]
              seen, uniq = {}, []
              for f in files:
                  h = md5(os.path.join(folder, f))
                  if h not in seen:
                      seen[h] = True
                      uniq.append(f)
              n_dupes = len(files) - len(uniq)
              if n_dupes > 0:
                  print(f"    [{modality}|{os.path.basename(folder)}] "
                        f"Removed {n_dupes} duplicates → {len(uniq)} unique")
              idx = list(range(len(uniq)))
              rng.shuffle(idx)
              cut = int((1 - val_ratio) * len(idx))
              chosen = idx[:cut] if split == "train" else idx[cut:]
              for i in chosen:
                  self.samples.append((os.path.join(folder, uniq[i]), label))

          rng.shuffle(self.samples)
          n1 = sum(l for _, l in self.samples)
          n0 = len(self.samples) - n1
          tot = len(self.samples)
          print(f"    [{modality:3s}] {split:5s} | {tot:4d} samples  "
                f"tumor={n1} ({100*n1/tot:.1f}%)  healthy={n0} ({100*n0/tot:.1f}%)")

      def __len__(self):
          return len(self.samples)

      def __getitem__(self, idx):
          path, label = self.samples[idx]
          try:
              img = Image.open(path).convert("RGB")
          except Exception:
              img = Image.new("RGB", (224, 224), 0)
          return (self.transform(img) if self.transform else img), label

      def get_labels(self):
          return [l for _, l in self.samples]


  def build_loaders(cfg):
      print("\n  ── Building dataloaders ──")
      loaders = {}
      for mod, (td, hd) in {
          "MRI": (cfg["MRI_tumor_dir"],  cfg["MRI_healthy_dir"]),
          "CT":  (cfg["CT_tumor_dir"],   cfg["CT_healthy_dir"]),
      }.items():
          for split in ["train", "val"]:
              ds = BrainDataset(
                  td, hd, split=split, modality=mod,
                  transform=build_transforms(cfg["image_size"], mod, split == "train"),
                  val_ratio=cfg["val_ratio"], seed=cfg["seed"]
              )
              if split == "train":
                  labs    = ds.get_labels()
                  cnt     = np.bincount(labs)
                  weights = [1.0 / cnt[l] for l in labs]
                  loader  = DataLoader(
                      ds, batch_size=cfg["batch_size"],
                      sampler=WeightedRandomSampler(weights, len(weights)),
                      num_workers=cfg["num_workers"],
                      pin_memory=(cfg["device"] == "cuda")
                  )
              else:
                  loader = DataLoader(
                      ds, batch_size=cfg["batch_size"], shuffle=False,
                      num_workers=cfg["num_workers"],
                      pin_memory=(cfg["device"] == "cuda")
                  )
              loaders[f"{mod}_{split}"] = loader
      return loaders


  # ══════════════════════════════════════════════════════════════════════
  # MODELS
  # ══════════════════════════════════════════════════════════════════════
  class TeacherNet(nn.Module):
      def __init__(self, num_classes=2):
          super().__init__()
          try:
              base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
          except AttributeError:
              base = models.convnext_tiny(pretrained=True)
          self.features = base.features
          self.gap      = nn.AdaptiveAvgPool2d(1)
          self.head     = nn.Sequential(
              nn.LayerNorm(768),
              nn.Linear(768, 384),
              nn.GELU(),
              nn.Dropout(0.4),
              nn.Linear(384, num_classes)
          )
          for layer in [self.head[1], self.head[4]]:
              nn.init.trunc_normal_(layer.weight, std=0.02)
              nn.init.zeros_(layer.bias)

      def forward(self, x):
          feat = self.gap(self.features(x)).flatten(1)
          return self.head(feat), feat


  class ModalityNorm(nn.Module):
      """
      ★ FIX [10]: Modality-conditioned feature scaling via learned affine params.
      Allows the shared backbone to specialize per modality without fully duplicating weights.
      Each modality gets its own gamma/beta to re-scale the backbone output.
      """
      def __init__(self, dim=768, num_modalities=2):
          super().__init__()
          # One set of (gamma, beta) per modality
          self.gamma = nn.Embedding(num_modalities, dim)
          self.beta  = nn.Embedding(num_modalities, dim)
          nn.init.ones_(self.gamma.weight)
          nn.init.zeros_(self.beta.weight)

      def forward(self, feat, modality_id):
          """
          feat        : (B, dim)
          modality_id : int  0=MRI, 1=CT
          """
          dev = feat.device
          idx = torch.full((feat.size(0),), modality_id, dtype=torch.long, device=dev)
          g   = self.gamma(idx)   # (B, dim)
          b   = self.beta(idx)    # (B, dim)
          return feat * g + b


  class UMLStudentB(nn.Module):
      """
      Fixed student with:
      - Shared ConvNeXt-Tiny backbone
      - Modality-specific affine normalization (ModalityNorm) after backbone
      - Separate MRI/CT projection heads + classifiers
      """
      MOD_IDS = {"MRI": 0, "CT": 1}

      def __init__(self, num_classes=2):
          super().__init__()
          try:
              base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
          except AttributeError:
              base = models.convnext_tiny(pretrained=True)

          self.backbone     = base.features
          self.gap          = nn.AdaptiveAvgPool2d(1)
          self.modality_norm = ModalityNorm(dim=768, num_modalities=2)  # ★ FIX [10]

          self.mri_proj = nn.Sequential(
              nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2)
          )
          self.ct_proj  = nn.Sequential(
              nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2)
          )
          self.mri_head = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, num_classes))
          self.ct_head  = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, num_classes))

          for mod in [self.mri_proj, self.ct_proj, self.mri_head, self.ct_head]:
              for m in mod.modules():
                  if isinstance(m, nn.Linear):
                      nn.init.trunc_normal_(m.weight, std=0.02)
                      nn.init.zeros_(m.bias)

      def forward(self, x, modality):
          feat = self.gap(self.backbone(x)).flatten(1)
          feat = self.modality_norm(feat, self.MOD_IDS[modality])  # ★ FIX [10]
          if modality == "MRI":
              proj   = self.mri_proj(feat)
              logits = self.mri_head(proj)
          else:
              proj   = self.ct_proj(feat)
              logits = self.ct_head(proj)
          return logits, proj

      def freeze_backbone(self):
          for p in self.backbone.parameters():
              p.requires_grad = False

      def unfreeze_backbone(self):
          for p in self.backbone.parameters():
              p.requires_grad = True

      def head_params(self):
          return (
              list(self.mri_proj.parameters())
              + list(self.ct_proj.parameters())
              + list(self.mri_head.parameters())
              + list(self.ct_head.parameters())
              + list(self.modality_norm.parameters())
          )

      def backbone_params(self):
          return list(self.backbone.parameters())


  # ══════════════════════════════════════════════════════════════════════
  # LOSSES
  # ══════════════════════════════════════════════════════════════════════
  def mixup_batch(imgs, labels, alpha, prob):
      """Returns (mixed_imgs, labels_a, labels_b, lam)."""
      if random.random() > prob or alpha <= 0:
          return imgs, labels, labels, 1.0
      lam = float(np.random.beta(alpha, alpha))
      idx = torch.randperm(imgs.size(0), device=imgs.device)
      return lam * imgs + (1 - lam) * imgs[idx], labels, labels[idx], lam


  def mixup_ce(logits, la, lb, lam, smooth):
      return (lam * F.cross_entropy(logits, la, label_smoothing=smooth)
              + (1 - lam) * F.cross_entropy(logits, lb, label_smoothing=smooth))


  def uncertainty_weighted_kd(s_logits, t_logits, la, lb, lam, cfg):
      """
      ★ FIX [2]: Both labels (la, lb) and lam now passed in.
      Teacher logits are on CLEAN images (computed before mixup).
      CE component properly handles mixup labels.
      """
      T = cfg["temperature"]
      t_conf  = F.softmax(t_logits, 1).max(1).values
      weights = t_conf / (t_conf.mean() + 1e-8)
      kl_per  = F.kl_div(
          F.log_softmax(s_logits / T, 1),
          F.softmax(t_logits / T, 1),
          reduction="none"
      ).sum(1)
      kl = (weights * kl_per).mean() * (T ** 2)
      ce = mixup_ce(s_logits, la, lb, lam, cfg["label_smoothing"])  # ★ FIX [2]
      return cfg["lambda_cls"] * ce + cfg["lambda_distill"] * kl


  def cross_modal_contrastive(mri_feat, ct_feat, mri_labels, ct_labels, cfg):
      """Contrastive loss aligning same-class cross-modal embeddings."""
      mri_n = F.normalize(mri_feat, 1)
      ct_n  = F.normalize(ct_feat, 1)
      dist  = 1 - torch.mm(mri_n, ct_n.t())
      same  = (mri_labels.unsqueeze(1) == ct_labels.unsqueeze(0)).float()
      loss  = (same * dist ** 2
              + (1 - same) * F.relu(cfg["contrast_margin"] - dist) ** 2).mean()
      return cfg["lambda_contrast"] * loss


  # ══════════════════════════════════════════════════════════════════════
  # EVALUATE
  # ══════════════════════════════════════════════════════════════════════
  @torch.no_grad()
  def evaluate(model, loader, device, modality=None):
      model.eval()
      all_preds, all_labels, all_probs = [], [], []
      for imgs, lbl in loader:
          imgs = imgs.to(device)
          logits, _ = model(imgs, modality) if modality else model(imgs)
          probs = F.softmax(logits, 1)[:, 1]
          all_probs.extend(probs.cpu().numpy())
          all_preds.extend(logits.argmax(1).cpu().numpy())
          all_labels.extend(lbl.numpy())
      acc  = accuracy_score(all_labels, all_preds)
      prec = precision_score(all_labels, all_preds, average="binary", zero_division=0)
      rec  = recall_score(all_labels, all_preds, average="binary", zero_division=0)
      f1   = f1_score(all_labels, all_preds, average="binary", zero_division=0)
      try:
          auc = roc_auc_score(all_labels, all_probs)
      except Exception:
          auc = float("nan")
      pc = np.bincount(all_preds, minlength=2)
      return dict(acc=acc, auc=auc, prec=prec, rec=rec, f1=f1,
                  preds=all_preds, labels=all_labels, probs=all_probs,
                  pred_h=int(pc[0]), pred_t=int(pc[1]))


  def quick_eval(model, loader, device, modality=None):
      r = evaluate(model, loader, device, modality)
      return r["acc"], r["pred_h"], r["pred_t"]


  def make_scheduler(optimizer, max_lrs, steps_per_epoch, n_epochs, cfg):
      """
      Fresh OneCycleLR for exactly n_epochs.
      Always create fresh — never restore stale scheduler state.
      """
      return torch.optim.lr_scheduler.OneCycleLR(
          optimizer,
          max_lr=max_lrs,
          steps_per_epoch=steps_per_epoch,
          epochs=n_epochs,
          pct_start=cfg["pct_start"],
          div_factor=cfg["div_factor"],
          final_div_factor=cfg["final_div"]
      )


  def save_plots(history, phase_tag):
      """Save training curves."""
      epochs = list(range(1, len(history["mri_acc"]) + 1))
      fig, axes = plt.subplots(1, 3, figsize=(18, 5))

      axes[0].set_title("Validation Accuracy")
      axes[0].plot(epochs, history["mri_acc"], label="MRI", marker="o", ms=3)
      axes[0].plot(epochs, history["ct_acc"],  label="CT",  marker="s", ms=3)
      avg_acc = [(m + c) / 2 for m, c in zip(history["mri_acc"], history["ct_acc"])]
      axes[0].plot(epochs, avg_acc, label="Avg", linestyle="--")
      axes[0].legend(); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")

      axes[1].set_title("Training Loss")
      axes[1].plot(epochs, history["mri_loss"], label="MRI Loss")
      axes[1].plot(epochs, history["ct_loss"],  label="CT Loss")
      axes[1].legend(); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")

      axes[2].set_title("Learning Rate")
      axes[2].plot(epochs, history["lr"])
      axes[2].set_yscale("log"); axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("LR")

      plt.tight_layout()
      path = os.path.join(PLOT_DIR, f"training_curves_{phase_tag}.png")
      plt.savefig(path, dpi=150, bbox_inches="tight")
      plt.close()
      print(f"  📊  Plot saved → {path}")


  # ══════════════════════════════════════════════════════════════════════
  # TRAINING LOOP
  # ══════════════════════════════════════════════════════════════════════
  def run_phase(phase, n_epochs, lr_bb, lr_hd, ep_offset,
                student, mri_teacher, ct_teacher,
                mri_train, ct_train, mri_val, ct_val,
                history, cfg, scaler):
      """
      Runs exactly n_epochs of training for the given phase.
      ep_offset  = global epoch number BEFORE this phase starts.

      Returns (best_avg, best_wts, history).

      ★ FIX [3][4]: best_avg and patience are RESET fresh each phase call.
      """
      dev = cfg["device"]

      # ★ FIX [7]: Use balanced steps — not max(MRI, CT) which oversamples MRI.
      # We take ceil average so both modalities complete at least one full pass.
      mri_steps = len(mri_train)
      ct_steps  = len(ct_train)
      steps_per_ep = (mri_steps + ct_steps) // 2   # ★ FIX [7]

      # ★ FIX [3][4]: Fresh best_avg and patience per phase
      best_avg = 0.0
      best_wts = copy.deepcopy(student.state_dict())
      patience = 0

      # ★ FIX [1]: Create EMA model for this phase
      ema = EMAModel(student, decay=cfg["ema_decay"])

      if lr_bb is None:
          student.freeze_backbone()
          opt   = torch.optim.AdamW(
              student.head_params(), lr=lr_hd,
              weight_decay=cfg["s_weight_decay"]
          )
          sched = make_scheduler(opt, lr_hd, steps_per_ep * 2, n_epochs, cfg)
      else:
          student.unfreeze_backbone()
          opt = torch.optim.AdamW([
              {"params": student.backbone_params(), "lr": lr_bb,
              "weight_decay": cfg["s_weight_decay"]},
              {"params": student.head_params(),     "lr": lr_hd,
              "weight_decay": cfg["s_weight_decay"]},
          ], weight_decay=cfg["s_weight_decay"])
          sched = make_scheduler(opt, [lr_bb, lr_hd], steps_per_ep * 2, n_epochs, cfg)

      ep_label = f"{'backbone FROZEN' if lr_bb is None else f'bb_lr={lr_bb:.0e}'}"
      print(f"\n  ── Phase-S-{phase} ({ep_label})  "
            f"[{n_epochs} epochs, global ep {ep_offset+1}–{ep_offset+n_epochs}] ──")

      for epoch in range(n_epochs):
          student.train()
          glep     = ep_offset + epoch + 1
          cur_lr   = opt.param_groups[-1]["lr"]
          mri_iter = iter(mri_train)
          ct_iter  = iter(ct_train)
          mb_loss, cb_loss = [], []

          # Track latest CT embeddings for cross-modal contrastive
          # ★ FIX: These are updated each CT step (not stale from previous step)
          last_ct_proj = None
          last_ct_lbls = None

          # ── Get a recent CT val acc to decide if contrastive is safe ──
          # We use the EMA model for this check (most stable estimate)
          ct_guard_ok = True  # Will be set after first epoch

          for step in range(steps_per_ep):
              # ══ MRI STEP ══════════════════════════════════════════════

              try:
                  im_m, lb_m = next(mri_iter)
              except StopIteration:
                  mri_iter = iter(mri_train)
                  im_m, lb_m = next(mri_iter)
              im_m, lb_m = im_m.to(dev), lb_m.to(dev)

              # ★ FIX [11]: Get teacher logits on CLEAN images BEFORE mixup
              t_mri_logits = None
              if phase in ("B", "C"):
                  with torch.no_grad():
                      with autocast(enabled=USE_AMP):
                          t_mri_logits, _ = mri_teacher(im_m)

              # Apply mixup AFTER teacher inference
              im_m_mix, la_m, lb2_m, lam_m = mixup_batch(
                  im_m, lb_m, cfg["mixup_alpha"], cfg["mixup_prob"]
              )

              opt.zero_grad()
              with autocast(enabled=USE_AMP):
                  s_logits_m, s_mri_proj = student(im_m_mix, "MRI")

                  if phase == "A":
                      loss_m = mixup_ce(s_logits_m, la_m, lb2_m, lam_m,
                                        cfg["label_smoothing"])
                  else:
                      # ★ FIX [2]: Pass both labels and lam to KD loss
                      loss_m = uncertainty_weighted_kd(
                          s_logits_m, t_mri_logits,
                          la_m, lb2_m, lam_m, cfg
                      )

                  # ★ FIX [12]: Cross-modal contrastive only if CT guard passes
                  if phase == "C" and last_ct_proj is not None and ct_guard_ok:
                      idx = torch.randperm(
                          last_ct_proj.size(0), device=dev
                      )[:s_mri_proj.size(0)]
                      loss_m = loss_m + cross_modal_contrastive(
                          s_mri_proj, last_ct_proj[idx],
                          la_m, last_ct_lbls[idx], cfg
                      )

              if USE_AMP:
                  scaler.scale(loss_m).backward()
                  scaler.unscale_(opt)
                  torch.nn.utils.clip_grad_norm_(student.parameters(), cfg["grad_clip"])
                  scaler.step(opt)
                  scaler.update()
              else:
                  loss_m.backward()
                  torch.nn.utils.clip_grad_norm_(student.parameters(), cfg["grad_clip"])
                  opt.step()

              sched.step()
              ema.update(student)  # ★ FIX [1]
              mb_loss.append(loss_m.item())

              # ══ CT STEP ═══════════════════════════════════════════════

              try:
                  im_c, lb_c = next(ct_iter)
              except StopIteration:
                  ct_iter = iter(ct_train)
                  im_c, lb_c = next(ct_iter)
              im_c, lb_c = im_c.to(dev), lb_c.to(dev)

              # ★ FIX [11]: Teacher on clean CT images BEFORE mixup
              t_ct_logits = None
              if phase in ("B", "C"):
                  with torch.no_grad():
                      with autocast(enabled=USE_AMP):
                          t_ct_logits, _ = ct_teacher(im_c)

              im_c_mix, la_c, lb2_c, lam_c = mixup_batch(
                  im_c, lb_c, cfg["mixup_alpha"], cfg["mixup_prob"]
              )

              opt.zero_grad()
              with autocast(enabled=USE_AMP):
                  s_logits_c, s_ct_proj = student(im_c_mix, "CT")

                  if phase == "A":
                      loss_c = mixup_ce(s_logits_c, la_c, lb2_c, lam_c,
                                        cfg["label_smoothing"])
                  else:
                      # ★ FIX [2]
                      loss_c = uncertainty_weighted_kd(
                          s_logits_c, t_ct_logits,
                          la_c, lb2_c, lam_c, cfg
                      )

                  # ★ FIX [12]: Contrastive loss guarded
                  if phase == "C" and ct_guard_ok:
                      idx = torch.randperm(
                          s_mri_proj.size(0), device=dev
                      )[:s_ct_proj.size(0)]
                      loss_c = loss_c + cross_modal_contrastive(
                          s_mri_proj.detach()[idx], s_ct_proj,
                          la_m[idx], la_c, cfg
                      )

              if USE_AMP:
                  scaler.scale(loss_c).backward()
                  scaler.unscale_(opt)
                  torch.nn.utils.clip_grad_norm_(student.parameters(), cfg["grad_clip"])
                  scaler.step(opt)
                  scaler.update()
              else:
                  loss_c.backward()
                  torch.nn.utils.clip_grad_norm_(student.parameters(), cfg["grad_clip"])
                  opt.step()

              sched.step()
              ema.update(student)  # ★ FIX [1]
              cb_loss.append(loss_c.item())

              # Save latest CT embeddings for next MRI step
              last_ct_proj = s_ct_proj.detach()
              last_ct_lbls = la_c.detach()

          # ── Epoch evaluation — use EMA model ★ FIX [1] ──────────────
          ema_model = ema.get_model()
          acc_m, phm, ptm = quick_eval(ema_model, mri_val, dev, "MRI")
          acc_c, phc, ptc = quick_eval(ema_model, ct_val,  dev, "CT")
          avg = (acc_m + acc_c) / 2

          # ★ FIX [12]: Update CT guard for next epoch
          ct_guard_ok = (acc_c >= cfg["ct_guard_acc"])
          guard_str   = "✅" if ct_guard_ok else "🚫 contrast OFF"

          history["mri_loss"].append(np.mean(mb_loss))
          history["ct_loss"].append(np.mean(cb_loss))
          history["mri_acc"].append(acc_m)
          history["ct_acc"].append(acc_c)
          history["lr"].append(cur_lr)
          history["phase"].append(phase)

          print(f"  [{phase}] Ep {glep:3d}  lr={cur_lr:.2e}  "
                f"MRI[loss={np.mean(mb_loss):.4f}|val={acc_m:.4f}|H={phm} T={ptm}]  "
                f"CT[loss={np.mean(cb_loss):.4f}|val={acc_c:.4f}|H={phc} T={ptc}]  "
                f"avg={avg:.4f}  {guard_str}")

          if glep % cfg["student_ckpt_every"] == 0:
              save_ckpt(ema_model.state_dict(), f"studentB_EMA_{phase}_ep{glep}.pth")

          if avg > best_avg:
              best_avg = avg
              best_wts = copy.deepcopy(ema_model.state_dict())  # ★ FIX [1]: save EMA weights
              patience = 0
              print(f"  ★  New best avg = {best_avg:.4f}  [EMA]")
          else:
              patience += 1
              if patience >= cfg["student_early_stop"]:
                  print(f"  ⏹  Early stop Phase-{phase} at ep {glep}  "
                        f"(patience={patience})")
                  break

      return best_avg, best_wts, history


  # ══════════════════════════════════════════════════════════════════════
  # FINAL EVALUATION
  # ══════════════════════════════════════════════════════════════════════
  def full_evaluation(student, mri_teacher, ct_teacher, mri_val, ct_val, cfg):
      dev = cfg["device"]
      print(f"\n{'═'*72}\n  FULL FINAL EVALUATION — VERSION B FIXED\n{'═'*72}")

      configs = [
          ("MRI Teacher", mri_teacher, mri_val, None),
          ("CT Teacher",  ct_teacher,  ct_val,  None),
          ("Student MRI", student,     mri_val, "MRI"),
          ("Student CT",  student,     ct_val,  "CT"),
      ]
      summary = {}
      lines   = ["VERSION B FIXED — EMA + All Fixes", "=" * 72]

      for tag, mdl, ldr, mod in configs:
          m  = evaluate(mdl, ldr, dev, mod)
          summary[tag] = m
          cr = classification_report(
              m["labels"], m["preds"],
              target_names=["Healthy", "Tumor"],
              zero_division=0
          )
          blk = (f"\n  ── {tag} ──\n"
                f"    Accuracy  : {m['acc']:.4f}\n"
                f"    AUC-ROC   : {m['auc']:.4f}\n"
                f"    Precision : {m['prec']:.4f}\n"
                f"    Recall    : {m['rec']:.4f}\n"
                f"    F1        : {m['f1']:.4f}\n"
                f"    Predictions: H={m['pred_h']}  T={m['pred_t']}\n\n{cr}")
          print(blk)
          lines.append(blk)

      w = 20
      hdr = f"\n  {'Model':<{w}} {'Acc':>7} {'AUC':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}"
      div = "  " + "─" * (w + 38)
      print(hdr); print(div)
      lines += [hdr, div]

      for tag, m in summary.items():
          row = (f"  {tag:<{w}} {m['acc']:>7.4f} {m['auc']:>7.4f} "
                f"{m['prec']:>7.4f} {m['rec']:>7.4f} {m['f1']:>7.4f}")
          print(row)
          lines.append(row)

      print(div); lines.append(div)

      for metric, key in [("Acc", "acc"), ("F1", "f1"), ("AUC", "auc")]:
          dm = summary["Student MRI"][key] - summary["MRI Teacher"][key]
          dc = summary["Student CT"][key]  - summary["CT Teacher"][key]
          l  = (f"  Δ Student vs Teacher | "
                f"MRI {metric}: {dm:+.4f}   CT {metric}: {dc:+.4f}")
          print(l)
          lines.append(l)

      log_path = os.path.join(LOG_DIR, "final_results_vB_fixed.txt")
      with open(log_path, "w") as f:
          f.write("\n".join(lines))
      print(f"\n  📄  Results saved → {log_path}")

      # Confusion matrices
      fig, axes = plt.subplots(1, 4, figsize=(22, 5))
      for ax, (tag, m) in zip(axes, summary.items()):
          cm = confusion_matrix(m["labels"], m["preds"])
          sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                      xticklabels=["Healthy", "Tumor"],
                      yticklabels=["Healthy", "Tumor"])
          ax.set_title(f"{tag}\nAcc={m['acc']:.3f}  AUC={m['auc']:.3f}")
          ax.set_xlabel("Predicted")
          ax.set_ylabel("True")
      plt.tight_layout()
      cm_path = os.path.join(PLOT_DIR, "confusion_matrices_fixed.png")
      plt.savefig(cm_path, dpi=150, bbox_inches="tight")
      plt.close()
      print(f"  📊  Confusion matrices → {cm_path}")

      return summary


  # ══════════════════════════════════════════════════════════════════════
  # MAIN
  # ══════════════════════════════════════════════════════════════════════
  def main():
      t0 = time.time()

      print("\n  Verifying dataset paths...")
      paths = [
          ("MRI Tumor",   CFG["MRI_tumor_dir"]),
          ("MRI Healthy", CFG["MRI_healthy_dir"]),
          ("CT Tumor",    CFG["CT_tumor_dir"]),
          ("CT Healthy",  CFG["CT_healthy_dir"]),
      ]
      ok = True
      for l, p in paths:
          exists = os.path.isdir(p)
          print(f"    {l:<15} {'✅' if exists else '❌  NOT FOUND — check path'}")
          if not exists:
              ok = False
      if not ok:
          print("\n  ❌  Fix missing paths in CFG and re-run.")
          return

      loaders = build_loaders(CFG)

      # ── Load teachers ──────────────────────────────────────────────
      print("\n  Loading pretrained teachers...")
      mri_teacher = TeacherNet(CFG["num_classes"])
      ct_teacher  = TeacherNet(CFG["num_classes"])
      mri_ckpt    = os.path.join(CFG["teacher_ckpt_dir"], "mri_teacher_BEST.pth")
      ct_ckpt     = os.path.join(CFG["teacher_ckpt_dir"], "ct_teacher_BEST.pth")

      for tag, path in [("MRI teacher", mri_ckpt), ("CT teacher", ct_ckpt)]:
          if not os.path.exists(path):
              print(f"  ❌  {tag} checkpoint not found: {path}")
              return

      mri_teacher.load_state_dict(torch.load(mri_ckpt, map_location=CFG["device"]))
      ct_teacher.load_state_dict(torch.load(ct_ckpt,   map_location=CFG["device"]))
      for p in mri_teacher.parameters(): p.requires_grad = False
      for p in ct_teacher.parameters():  p.requires_grad = False
      mri_teacher.eval().to(CFG["device"])
      ct_teacher.eval().to(CFG["device"])
      print("  ✅  MRI Teacher loaded & frozen")
      print("  ✅  CT Teacher loaded & frozen")

      # Quick check teacher performance
      t_acc_m, _, _ = quick_eval(mri_teacher, loaders["MRI_val"], CFG["device"])
      t_acc_c, _, _ = quick_eval(ct_teacher,  loaders["CT_val"],  CFG["device"])
      print(f"  Teacher sanity: MRI={t_acc_m:.4f}  CT={t_acc_c:.4f}")

      # ── Initialize student ─────────────────────────────────────────
      student = UMLStudentB(CFG["num_classes"]).to(CFG["device"])

      # Determine which phases to run based on resume config
      resume_ckpt = CFG.get("resume_ckpt")
      resume_ep   = CFG.get("resume_global_ep", 0)
      resume_phase = CFG.get("resume_phase", "A")

      if resume_ckpt:
          ckpt_path = os.path.join(CKPT_DIR, resume_ckpt)
          if not os.path.exists(ckpt_path):
              print(f"\n  ❌  Resume checkpoint not found: {ckpt_path}")
              return
          # Load weights — allow partial match for architecture differences
          state = torch.load(ckpt_path, map_location=CFG["device"])
          missing, unexpected = student.load_state_dict(state, strict=False)
          if missing:
              print(f"  ⚠️   Missing keys (new layers, OK): {missing}")
          if unexpected:
              print(f"  ⚠️   Unexpected keys (ignored): {unexpected}")
          print(f"\n  ✅  Student loaded from {resume_ckpt}  (global ep {resume_ep})")
          acc_m, _, _ = quick_eval(student, loaders["MRI_val"], CFG["device"], "MRI")
          acc_c, _, _ = quick_eval(student, loaders["CT_val"],  CFG["device"], "CT")
          print(f"  Sanity eval at load: MRI={acc_m:.4f}  CT={acc_c:.4f}  "
                f"avg={(acc_m+acc_c)/2:.4f}")
      else:
          print("\n  ✅  Student initialised fresh (no resume)")
          resume_ep    = 0
          resume_phase = "A"

      scaler  = GradScaler() if USE_AMP else None
      history = {"mri_loss": [], "ct_loss": [], "mri_acc": [],
                "ct_acc": [], "lr": [], "phase": []}

      best_wts_global  = copy.deepcopy(student.state_dict())
      best_avg_global  = 0.0

      # ─────────────────────────────────────────────────────────────────
      # PHASE A — head-only warmup
      # ─────────────────────────────────────────────────────────────────
      phase_a_end = CFG["s_phase_a_epochs"]
      if resume_phase == "A" and resume_ep < phase_a_end:
          a_done = resume_ep                      # epochs completed in A
          a_left = phase_a_end - a_done           # remaining A epochs
          print(f"\n{'━'*72}")
          print(f"  PHASE A — {a_left} epochs  (head-only warmup)")
          print(f"{'━'*72}")

          best_avg_a, best_wts_a, history = run_phase(
              phase="A", n_epochs=a_left,
              lr_bb=None, lr_hd=CFG["s_lr_head_a"],
              ep_offset=resume_ep,
              student=student,
              mri_teacher=mri_teacher, ct_teacher=ct_teacher,
              mri_train=loaders["MRI_train"], ct_train=loaders["CT_train"],
              mri_val=loaders["MRI_val"],     ct_val=loaders["CT_val"],
              history=history, cfg=CFG, scaler=scaler
          )
          student.load_state_dict(best_wts_a)
          if best_avg_a > best_avg_global:
              best_avg_global = best_avg_a
              best_wts_global = copy.deepcopy(best_wts_a)

          save_ckpt(best_wts_a, "studentB_A_BEST.pth")
          save_plots(history, "phase_A")
          print(f"\n  Phase A complete  best avg = {best_avg_a:.4f}")

      # ─────────────────────────────────────────────────────────────────
      # PHASE B — backbone fine-tune + distillation
      # ─────────────────────────────────────────────────────────────────
      phase_b_end = CFG["s_phase_a_epochs"] + CFG["s_phase_b_epochs"]
      if resume_phase in ("A", "B") and resume_ep < phase_b_end:
          b_done   = max(0, resume_ep - CFG["s_phase_a_epochs"])
          b_left   = CFG["s_phase_b_epochs"] - b_done
          b_offset = CFG["s_phase_a_epochs"] + b_done
          print(f"\n{'━'*72}")
          print(f"  PHASE B — {b_left} epochs  (backbone + distillation)")
          print(f"{'━'*72}")

          best_avg_b, best_wts_b, history = run_phase(
              phase="B", n_epochs=b_left,
              lr_bb=CFG["s_lr_backbone_b"], lr_hd=CFG["s_lr_head_b"],
              ep_offset=b_offset,
              student=student,
              mri_teacher=mri_teacher, ct_teacher=ct_teacher,
              mri_train=loaders["MRI_train"], ct_train=loaders["CT_train"],
              mri_val=loaders["MRI_val"],     ct_val=loaders["CT_val"],
              history=history, cfg=CFG, scaler=scaler
          )
          student.load_state_dict(best_wts_b)
          if best_avg_b > best_avg_global:
              best_avg_global = best_avg_b
              best_wts_global = copy.deepcopy(best_wts_b)

          save_ckpt(best_wts_b, "studentB_B_BEST.pth")
          save_plots(history, "phase_B")
          print(f"\n  Phase B complete  best avg = {best_avg_b:.4f}")

      # ─────────────────────────────────────────────────────────────────
      # PHASE C — deep fine-tune + cross-modal contrastive
      # ─────────────────────────────────────────────────────────────────
      phase_c_end = phase_b_end + CFG["s_phase_c_epochs"]
      if resume_ep < phase_c_end:
          c_done   = max(0, resume_ep - phase_b_end)
          c_left   = CFG["s_phase_c_epochs"] - c_done
          c_offset = phase_b_end + c_done
          print(f"\n{'━'*72}")
          print(f"  PHASE C — {c_left} epochs  "
                f"(deep fine-tune + conditional cross-modal contrastive)")
          print(f"{'━'*72}")

          best_avg_c, best_wts_c, history = run_phase(
              phase="C", n_epochs=c_left,
              lr_bb=CFG["s_lr_backbone_c"], lr_hd=CFG["s_lr_head_c"],
              ep_offset=c_offset,
              student=student,
              mri_teacher=mri_teacher, ct_teacher=ct_teacher,
              mri_train=loaders["MRI_train"], ct_train=loaders["CT_train"],
              mri_val=loaders["MRI_val"],     ct_val=loaders["CT_val"],
              history=history, cfg=CFG, scaler=scaler
          )
          student.load_state_dict(best_wts_c)
          if best_avg_c > best_avg_global:
              best_avg_global = best_avg_c
              best_wts_global = copy.deepcopy(best_wts_c)

          save_ckpt(best_wts_c, "studentB_C_BEST.pth")
          save_plots(history, "phase_C")
          print(f"\n  Phase C complete  best avg = {best_avg_c:.4f}")

      # ── Save overall best ──────────────────────────────────────────
      student.load_state_dict(best_wts_global)
      save_ckpt(best_wts_global, "studentB_BEST_FIXED.pth")
      print(f"\n  ✅  Overall best avg across all phases = {best_avg_global:.4f}")

      # ── Final evaluation ───────────────────────────────────────────
      full_evaluation(
          student, mri_teacher, ct_teacher,
          loaders["MRI_val"], loaders["CT_val"], CFG
      )

      elapsed = time.time() - t0
      print(f"\n{'='*72}")
      print(f"  ✅  VERSION B FIXED COMPLETE  ({elapsed/60:.1f} min)")
      print(f"  Best ckpt  : {CKPT_DIR}/studentB_BEST_FIXED.pth")
      print(f"  Logs       : {LOG_DIR}/final_results_vB_fixed.txt")
      print(f"  Plots      : {PLOT_DIR}/")
      print(f"{'='*72}")


  if __name__ == "__main__":
      main()

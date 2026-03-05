# ============================================================================
# run_gradcam_organized.py - Organized Grad-CAM with Proper Folder Structure
# ============================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import cv2
import random
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

CFG = {
    # Dataset paths
    "MRI_tumor_dir": "/content/drive/MyDrive/khushi/Dataset/Brain Tumor MRI images/Tumor",
    "MRI_healthy_dir": "/content/drive/MyDrive/khushi/Dataset/Brain Tumor MRI images/Healthy",
    "CT_tumor_dir": "/content/drive/MyDrive/khushi/Dataset/Brain Tumor CT scan Images/Tumor",
    "CT_healthy_dir": "/content/drive/MyDrive/khushi/Dataset/Brain Tumor CT scan Images/Healthy",

    # Checkpoint paths
    "output_dir": "/content/drive/MyDrive/khushi",
    "teacher_ckpt_dir": "/content/drive/MyDrive/khushi/Ct-mricheckpoints",
    "student_ckpt": "studentB_EMA_C_ep60.pth",
    "mri_teacher_ckpt": "mri_teacher_BEST.pth",
    "ct_teacher_ckpt": "ct_teacher_BEST.pth",

    # Model settings
    "num_classes": 2,
    "image_size": 224,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

print("="*72)
print("  ORGANIZED GRAD-CAM ANALYSIS")
print("  Folder Structure: MRI_Healthy, MRI_Tumor, CT_Healthy, CT_Tumor")
print("="*72)
print(f"Device: {CFG['device']}")
print("="*72)

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class TeacherNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        try:
            import torchvision.models as models
            base = models.convnext_tiny(pretrained=True)
        except:
            base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768,384), nn.GELU(),
            nn.Dropout(0.4), nn.Linear(384,num_classes))

    def forward(self, x):
        feat = self.gap(self.features(x)).flatten(1)
        return self.head(feat), feat

class ModalityNorm(nn.Module):
    def __init__(self, dim=768, num_modalities=2):
        super().__init__()
        self.gamma = nn.Embedding(num_modalities, dim)
        self.beta = nn.Embedding(num_modalities, dim)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, feat, modality_id):
        dev = feat.device
        idx = torch.full((feat.size(0),), modality_id, dtype=torch.long, device=dev)
        return feat * self.gamma(idx) + self.beta(idx)

class UMLStudentB(nn.Module):
    MOD_IDS = {"MRI": 0, "CT": 1}

    def __init__(self, num_classes=2):
        super().__init__()
        import torchvision.models as models
        base = models.convnext_tiny(pretrained=True)
        self.backbone = base.features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.modality_norm = ModalityNorm(768, 2)
        self.mri_proj = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768,256), nn.GELU(), nn.Dropout(0.2))
        self.ct_proj = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768,256), nn.GELU(), nn.Dropout(0.2))
        self.mri_head = nn.Sequential(nn.Dropout(0.4), nn.Linear(256,num_classes))
        self.ct_head = nn.Sequential(nn.Dropout(0.4), nn.Linear(256,num_classes))

    def forward(self, x, modality):
        feat = self.gap(self.backbone(x)).flatten(1)
        feat = self.modality_norm(feat, self.MOD_IDS[modality])
        if modality == "MRI":
            proj = self.mri_proj(feat)
            logits = self.mri_head(proj)
        else:
            proj = self.ct_proj(feat)
            logits = self.ct_head(proj)
        return logits, proj

# ============================================================================
# GRAD-CAM IMPLEMENTATION - FIXED VERSION
# ============================================================================

class GradCAM:
    def __init__(self, model, target_layer='backbone[-1]'):  # Changed default to backbone for student
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # Find target layer - handle both TeacherNet and UMLStudentB
        module = self.model

        # Check model type and set appropriate target layer
        if hasattr(self.model, 'backbone') and 'features' in self.target_layer:
            # For student model, map 'features' to 'backbone'
            target = self.target_layer.replace('features', 'backbone')
        else:
            target = self.target_layer

        # Navigate to the target layer
        if target == 'backbone[-1]':
            module = module.backbone[-1]
        elif target == 'features[-1]':
            if hasattr(module, 'features'):
                module = module.features[-1]
            elif hasattr(module, 'backbone'):
                module = module.backbone[-1]
        else:
            for part in target.split('.'):
                if part == '-1':
                    module = module[-1]
                else:
                    module = getattr(module, part)

        module.register_forward_hook(forward_hook)
        module.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None, modality=None):
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass - handle both TeacherNet and UMLStudentB
        if modality:  # Student model with modality
            logits, _ = self.model(input_tensor, modality)
        else:  # Teacher model without modality
            logits, _ = self.model(input_tensor)

        probs = F.softmax(logits, dim=1)
        pred_prob, pred_class = torch.max(probs, dim=1)

        if target_class is None:
            target_class = pred_class.item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Weighted combination
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, pred_class.item(), pred_prob.item()

def overlay_heatmap(image, heatmap, alpha=0.5):
    """Overlay heatmap on image"""
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()

    # Ensure 3-channel
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] > 3:
        img = img[:, :, :3]

    # Resize heatmap to match image
    h, w = img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay
    overlayed = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlayed, heatmap_resized

# ============================================================================
# DATA LOADING
# ============================================================================

def load_images_by_category(tumor_dir, healthy_dir, samples_per_class=10, seed=42):
    """Load images and organize by category"""
    random.seed(seed)

    categories = {
        'tumor': [],
        'healthy': []
    }

    # Load tumor images
    if os.path.exists(tumor_dir):
        tumor_files = [f for f in os.listdir(tumor_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        selected_tumor = random.sample(tumor_files, min(samples_per_class, len(tumor_files)))
        for f in selected_tumor:
            categories['tumor'].append((os.path.join(tumor_dir, f), 1))

    # Load healthy images
    if os.path.exists(healthy_dir):
        healthy_files = [f for f in os.listdir(healthy_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        selected_healthy = random.sample(healthy_files, min(samples_per_class, len(healthy_files)))
        for f in selected_healthy:
            categories['healthy'].append((os.path.join(healthy_dir, f), 0))

    return categories

# ============================================================================
# TRANSFORMS
# ============================================================================

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((CFG["image_size"], CFG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_comparison_image(original_img, student_overlay, teacher_overlay,
                          student_pred, teacher_pred, true_label,
                          student_conf, teacher_conf, filename):
    """Create a single comparison image with 3 panels"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Original with true label
    axes[0].imshow(np.array(original_img))
    true_label_str = "TUMOR" if true_label == 1 else "HEALTHY"
    axes[0].set_title(f"Original Image\nTRUE: {true_label_str}",
                     fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Student
    axes[1].imshow(student_overlay)
    student_label = "TUMOR" if student_pred == 1 else "HEALTHY"
    student_color = 'green' if student_pred == true_label else 'red'
    student_status = "✓ CORRECT" if student_pred == true_label else "✗ INCORRECT"
    axes[1].set_title(f"STUDENT\nPRED: {student_label}\nConf: {student_conf:.3f}\n{student_status}",
                     fontsize=14, fontweight='bold', color=student_color)
    axes[1].axis('off')

    # Panel 3: Teacher
    axes[2].imshow(teacher_overlay)
    teacher_label = "TUMOR" if teacher_pred == 1 else "HEALTHY"
    teacher_color = 'green' if teacher_pred == true_label else 'red'
    teacher_status = "✓ CORRECT" if teacher_pred == true_label else "✗ INCORRECT"
    axes[2].set_title(f"TEACHER\nPRED: {teacher_label}\nConf: {teacher_conf:.3f}\n{teacher_status}",
                     fontsize=14, fontweight='bold', color=teacher_color)
    axes[2].axis('off')

    plt.suptitle(f"File: {filename}", fontsize=12, y=0.98)
    plt.tight_layout()

    return fig

def create_heatmap_comparison(original_img, student_heatmap, teacher_heatmap,
                            student_pred, teacher_pred, true_label,
                            student_conf, teacher_conf, filename):
    """Create a detailed comparison with heatmaps"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    true_label_str = "TUMOR" if true_label == 1 else "HEALTHY"

    # Row 1: Overlays
    # Original
    axes[0, 0].imshow(np.array(original_img))
    axes[0, 0].set_title(f"Original\nTRUE: {true_label_str}", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Student overlay
    student_overlay, _ = overlay_heatmap(original_img, student_heatmap)
    axes[0, 1].imshow(student_overlay)
    student_label = "TUMOR" if student_pred == 1 else "HEALTHY"
    student_color = 'green' if student_pred == true_label else 'red'
    axes[0, 1].set_title(f"Student Overlay\nPRED: {student_label} ({student_conf:.3f})",
                        fontsize=12, fontweight='bold', color=student_color)
    axes[0, 1].axis('off')

    # Teacher overlay
    teacher_overlay, _ = overlay_heatmap(original_img, teacher_heatmap)
    axes[0, 2].imshow(teacher_overlay)
    teacher_label = "TUMOR" if teacher_pred == 1 else "HEALTHY"
    teacher_color = 'green' if teacher_pred == true_label else 'red'
    axes[0, 2].set_title(f"Teacher Overlay\nPRED: {teacher_label} ({teacher_conf:.3f})",
                        fontsize=12, fontweight='bold', color=teacher_color)
    axes[0, 2].axis('off')

    # Row 2: Raw heatmaps
    # Empty
    axes[1, 0].axis('off')

    # Student heatmap
    im1 = axes[1, 1].imshow(student_heatmap, cmap='jet', vmin=0, vmax=1)
    axes[1, 1].set_title("Student Attention Map", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Teacher heatmap
    im2 = axes[1, 2].imshow(teacher_heatmap, cmap='jet', vmin=0, vmax=1)
    axes[1, 2].set_title("Teacher Attention Map", fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.suptitle(f"Grad-CAM Analysis - {filename}\nTrue: {true_label_str}",
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    return fig

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_organized_gradcam():
    """Run Grad-CAM analysis with organized folder structure"""

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create parent folder
    parent_dir = os.path.join(CFG["output_dir"], f"gradcam_results_{timestamp}")
    os.makedirs(parent_dir, exist_ok=True)

    # Create subfolders
    subfolders = [
        "MRI_Healthy",
        "MRI_Tumor",
        "CT_Healthy",
        "CT_Tumor"
    ]

    for folder in subfolders:
        os.makedirs(os.path.join(parent_dir, folder), exist_ok=True)
        # Create subfolders for different visualizations
        os.makedirs(os.path.join(parent_dir, folder, "1_comparison"), exist_ok=True)
        os.makedirs(os.path.join(parent_dir, folder, "2_heatmaps"), exist_ok=True)
        os.makedirs(os.path.join(parent_dir, folder, "3_overlays_only"), exist_ok=True)

    print(f"\n📁 Created folder structure in: {parent_dir}")

    # Load models
    print("\n📦 Loading trained models...")

    # Load teachers
    mri_teacher = TeacherNet(CFG["num_classes"]).to(CFG["device"])
    ct_teacher = TeacherNet(CFG["num_classes"]).to(CFG["device"])

    mri_teacher.load_state_dict(torch.load(
        os.path.join(CFG["teacher_ckpt_dir"], CFG["mri_teacher_ckpt"]),
        map_location=CFG["device"]))
    ct_teacher.load_state_dict(torch.load(
        os.path.join(CFG["teacher_ckpt_dir"], CFG["ct_teacher_ckpt"]),
        map_location=CFG["device"]))

    mri_teacher.eval()
    ct_teacher.eval()
    print("  ✅ Teachers loaded")

    # Load student
    student = UMLStudentB(CFG["num_classes"]).to(CFG["device"])
    student_ckpt_path = os.path.join(CFG["output_dir"], "checkpoints", CFG["student_ckpt"])
    student.load_state_dict(torch.load(student_ckpt_path, map_location=CFG["device"]))
    student.eval()
    print("  ✅ Student loaded")

    # Initialize Grad-CAM with correct layer names
    # For student: use 'backbone[-1]' (since it has backbone attribute)
    # For teachers: use 'features[-1]' (since they have features attribute)
    student_cam_mri = GradCAM(student, 'backbone[-1]')
    student_cam_ct = GradCAM(student, 'backbone[-1]')
    teacher_cam_mri = GradCAM(mri_teacher, 'features[-1]')
    teacher_cam_ct = GradCAM(ct_teacher, 'features[-1]')

    # Load samples (10 per category)
    print("\n📸 Loading sample images (10 per category)...")
    mri_categories = load_images_by_category(CFG["MRI_tumor_dir"], CFG["MRI_healthy_dir"], samples_per_class=10)
    ct_categories = load_images_by_category(CFG["CT_tumor_dir"], CFG["CT_healthy_dir"], samples_per_class=10)

    print(f"\nMRI Samples:")
    print(f"  MRI_Tumor: {len(mri_categories['tumor'])} images")
    print(f"  MRI_Healthy: {len(mri_categories['healthy'])} images")
    print(f"\nCT Samples:")
    print(f"  CT_Tumor: {len(ct_categories['tumor'])} images")
    print(f"  CT_Healthy: {len(ct_categories['healthy'])} images")

    # Process MRI Healthy
    print("\n🧠 Processing MRI Healthy samples...")
    for idx, (img_path, true_label) in enumerate(mri_categories['healthy']):
        process_and_save(img_path, true_label, student_cam_mri, teacher_cam_mri,
                        "MRI", "Healthy", idx, parent_dir)

    # Process MRI Tumor
    print("\n🧠 Processing MRI Tumor samples...")
    for idx, (img_path, true_label) in enumerate(mri_categories['tumor']):
        process_and_save(img_path, true_label, student_cam_mri, teacher_cam_mri,
                        "MRI", "Tumor", idx, parent_dir)

    # Process CT Healthy
    print("\n🫁 Processing CT Healthy samples...")
    for idx, (img_path, true_label) in enumerate(ct_categories['healthy']):
        process_and_save(img_path, true_label, student_cam_ct, teacher_cam_ct,
                        "CT", "Healthy", idx, parent_dir)

    # Process CT Tumor
    print("\n🫁 Processing CT Tumor samples...")
    for idx, (img_path, true_label) in enumerate(ct_categories['tumor']):
        process_and_save(img_path, true_label, student_cam_ct, teacher_cam_ct,
                        "CT", "Tumor", idx, parent_dir)

    # Create summary HTML
    create_summary_html(parent_dir, timestamp, mri_categories, ct_categories)

    print(f"\n✅ Analysis complete! Results saved to: {parent_dir}")
    return parent_dir

def process_and_save(img_path, true_label, student_cam, teacher_cam,
                    modality, category, idx, parent_dir):
    """Process a single image and save all visualizations"""

    try:
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(CFG["device"])

        # Get filename without extension
        filename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(filename)[0]

        # Generate CAMs
        student_cam_map, student_pred, student_conf = student_cam.generate_cam(
            img_tensor, modality=modality)
        teacher_cam_map, teacher_pred, teacher_conf = teacher_cam.generate_cam(img_tensor)

        # Create overlay images
        student_overlay, student_heatmap = overlay_heatmap(img, student_cam_map)
        teacher_overlay, teacher_heatmap = overlay_heatmap(img, teacher_cam_map)

        # Determine folder name (MRI_Healthy, MRI_Tumor, etc.)
        folder_name = f"{modality}_{category}"
        target_folder = os.path.join(parent_dir, folder_name)

        # 1. Save comparison image (3 panels)
        fig1 = create_comparison_image(img, student_overlay, teacher_overlay,
                                      student_pred, teacher_pred, true_label,
                                      student_conf, teacher_conf, filename)
        comparison_path = os.path.join(target_folder, "1_comparison",
                                      f"{idx+1:02d}_{name_without_ext}_comparison.png")
        fig1.savefig(comparison_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig1)

        # 2. Save detailed heatmap comparison
        fig2 = create_heatmap_comparison(img, student_cam_map, teacher_cam_map,
                                       student_pred, teacher_pred, true_label,
                                       student_conf, teacher_conf, filename)
        heatmap_path = os.path.join(target_folder, "2_heatmaps",
                                   f"{idx+1:02d}_{name_without_ext}_heatmaps.png")
        fig2.savefig(heatmap_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig2)

        # 3. Save overlays only (individual images)
        # Student overlay only
        fig3, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(student_overlay)
        student_label = "TUMOR" if student_pred == 1 else "HEALTHY"
        status = "✓" if student_pred == true_label else "✗"
        ax.set_title(f"Student: {student_label} ({student_conf:.3f}) {status}",
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        student_only_path = os.path.join(target_folder, "3_overlays_only",
                                        f"{idx+1:02d}_{name_without_ext}_student.png")
        plt.savefig(student_only_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig3)

        # Teacher overlay only
        fig4, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(teacher_overlay)
        teacher_label = "TUMOR" if teacher_pred == 1 else "HEALTHY"
        status = "✓" if teacher_pred == true_label else "✗"
        ax.set_title(f"Teacher: {teacher_label} ({teacher_conf:.3f}) {status}",
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        teacher_only_path = os.path.join(target_folder, "3_overlays_only",
                                        f"{idx+1:02d}_{name_without_ext}_teacher.png")
        plt.savefig(teacher_only_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig4)

        print(f"    ✅ {idx+1:2d}: {filename[:30]}...")

    except Exception as e:
        print(f"    ❌ Error processing {img_path}: {str(e)}")

def create_summary_html(parent_dir, timestamp, mri_categories, ct_categories):
    """Create a summary HTML file with folder structure"""

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Grad-CAM Analysis - Organized Results</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; background: #f0f2f5; }}
        h1 {{ color: #1a73e8; border-bottom: 3px solid #1a73e8; padding-bottom: 10px; }}
        h2 {{ color: #0d47a1; margin-top: 30px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .folder-structure {{ background: #1e1e1e; color: #d4d4d4; padding: 20px; border-radius: 10px; font-family: 'Courier New', monospace; line-height: 1.6; }}
        .folder {{ color: #569cd6; }}
        .file {{ color: #ce9178; }}
        .section {{ background: white; padding: 25px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .badge {{ display: inline-block; padding: 5px 10px; border-radius: 20px; font-weight: bold; margin: 2px; }}
        .badge-green {{ background: #e6f4ea; color: #0f9d58; }}
        .badge-blue {{ background: #e8f0fe; color: #1a73e8; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: white; padding: 15px; border-radius: 8px; text-align: center; }}
        .stat-number {{ font-size: 28px; font-weight: bold; color: #1a73e8; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Grad-CAM Analysis Results</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Analysis ID: {timestamp}</p>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{len(mri_categories['healthy'])}</div>
                <div>MRI Healthy Samples</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(mri_categories['tumor'])}</div>
                <div>MRI Tumor Samples</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(ct_categories['healthy'])}</div>
                <div>CT Healthy Samples</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(ct_categories['tumor'])}</div>
                <div>CT Tumor Samples</div>
            </div>
        </div>

        <div class="section">
            <h2>📁 Folder Structure</h2>
            <div class="folder-structure">
                <span class="folder">📂 gradcam_results_{timestamp}/</span><br>
                │<br>
                ├── <span class="folder">📂 MRI_Healthy/</span><br>
                │   ├── <span class="folder">📂 1_comparison/</span> - 10 images (student vs teacher 3-panel)<br>
                │   ├── <span class="folder">📂 2_heatmaps/</span> - 10 images (detailed with raw heatmaps)<br>
                │   └── <span class="folder">📂 3_overlays_only/</span> - 20 images (10 student + 10 teacher)<br>
                │<br>
                ├── <span class="folder">📂 MRI_Tumor/</span><br>
                │   ├── <span class="folder">📂 1_comparison/</span> - 10 images<br>
                │   ├── <span class="folder">📂 2_heatmaps/</span> - 10 images<br>
                │   └── <span class="folder">📂 3_overlays_only/</span> - 20 images<br>
                │<br>
                ├── <span class="folder">📂 CT_Healthy/</span><br>
                │   ├── <span class="folder">📂 1_comparison/</span> - 10 images<br>
                │   ├── <span class="folder">📂 2_heatmaps/</span> - 10 images<br>
                │   └── <span class="folder">📂 3_overlays_only/</span> - 20 images<br>
                │<br>
                └── <span class="folder">📂 CT_Tumor/</span><br>
                    ├── <span class="folder">📂 1_comparison/</span> - 10 images<br>
                    ├── <span class="folder">📂 2_heatmaps/</span> - 10 images<br>
                    └── <span class="folder">📂 3_overlays_only/</span> - 20 images<br>
            </div>
        </div>

        <div class="section">
            <h2>📊 File Naming Convention</h2>
            <ul>
                <li><span class="badge badge-blue">01_filename_comparison.png</span> - 3-panel comparison (Original | Student | Teacher)</li>
                <li><span class="badge badge-blue">01_filename_heatmaps.png</span> - Detailed view with raw attention maps</li>
                <li><span class="badge badge-blue">01_filename_student.png</span> - Student overlay only</li>
                <li><span class="badge badge-blue">01_filename_teacher.png</span> - Teacher overlay only</li>
            </ul>
            <p>Numbers (01-10) indicate the sample index in each category.</p>
        </div>

        <div class="section">
            <h2>🎨 Color Coding</h2>
            <ul>
                <li><span class="badge badge-green">Green text</span> - Correct prediction (matches ground truth)</li>
                <li><span class="badge" style="background: #fce8e6; color: #d93025;">Red text</span> - Incorrect prediction</li>
                <li><span class="badge" style="background: #fff3e0; color: #f57c00;">✓</span> - Correct prediction marker</li>
                <li><span class="badge" style="background: #fce8e6; color: #d93025;">✗</span> - Incorrect prediction marker</li>
            </ul>
        </div>

        <div class="section">
            <h2>📈 Total Files Generated</h2>
            <ul>
                <li><strong>4 categories</strong> × 10 samples = 40 samples total</li>
                <li>Per sample: 3 visualization types</li>
                <li><strong>Total images:</strong> 160 files (40 comparison + 40 heatmaps + 80 overlays)</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

    html_path = os.path.join(parent_dir, "00_index.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    print(f"\n📄 Summary HTML saved to: {html_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run organized Grad-CAM analysis
    parent_dir = run_organized_gradcam()

    print("\n" + "="*72)
    print("✅ ORGANIZED GRAD-CAM ANALYSIS COMPLETE!")
    print("="*72)
    print(f"\nParent folder: {parent_dir}")
    print("\nFolder structure:")
    print("  📂 MRI_Healthy/")
    print("  📂 MRI_Tumor/")
    print("  📂 CT_Healthy/")
    print("  📂 CT_Tumor/")
    print("\nEach folder contains:")
    print("  📂 1_comparison/     - 3-panel (Original | Student | Teacher)")
    print("  📂 2_heatmaps/       - Detailed with raw attention maps")
    print("  📂 3_overlays_only/  - Individual overlays")
    print("\nTotal: 4 categories × 10 samples × 3 visualizations = 120 images")
    print("="*72)

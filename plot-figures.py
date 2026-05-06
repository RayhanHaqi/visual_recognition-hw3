import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

FIGS = Path(__file__).resolve().parent / "report" / "figure"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Figure 1: CodaBench Progress ──────────────────────────────────
experiments = [
    "v1\nResNet-50", "v2\nResNet-50", "ConvNeXt\nno RPN",
    "ConvNeXt\nv1", "ConvNeXt\nv2", "ConvNeXt\nv3",
    "ConvNeXt\nv4", "ConvNeXt\nv5", "ConvNeXt\nv6",
    "ConvNeXt\nv7", "ConvNeXt\nv8", "ConvNeXt\nv9",
]
scores = [0.3484, 0.4504, 0.0981, 0.5220, 0.5287, 0.5741,
          0.5510, 0.5631, 0.5811, 0.5587, 0.5656, 0.6110]
colors = ["#999999"] * 11 + ["#D62728"]

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(range(len(scores)), scores, color=colors, edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(scores)))
ax.set_xticklabels(experiments, fontsize=7)
ax.set_ylabel("CodaBench AP50", fontsize=11)
ax.set_title("CodaBench AP50 Progression Across Model Iterations", fontsize=13, fontweight="bold")
ax.set_ylim(0, 0.68)
for i, (s, c) in enumerate(zip(scores, colors)):
    ax.text(i, s + 0.008, f"{s:.4f}", ha="center", fontsize=7, fontweight="bold" if c == "#D62728" else "normal")
ax.grid(axis="y", linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "codabench_progress.png", dpi=150)
plt.close(fig)
print("✓ codabench_progress.png")


# ── Figure 2: Training Curves (run17) ─────────────────────────────
CSV = Path(__file__).resolve().parent / "log" / "convnext_bs2_lr2e-4_wd2e-3_ep250_run17.csv"
df = pd.read_csv(CSV)
epochs = df["epoch"].values

fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

# Panel 1: loss components (use first epoch as initial spike anchor)
ax = axes[0]
for key, label, c in [("loss_classifier", "Classif.", "#1f77b4"),
                       ("loss_box_reg", "Box Reg.", "#ff7f0e"),
                       ("loss_mask", "Mask", "#2ca02c"),
                       ("loss_objectness", "Objectn.", "#d62728"),
                       ("loss_rpn_box_reg", "RPN Reg.", "#9467bd")]:
    ax.plot(epochs[1:], df[key].values[1:], label=label, color=c, linewidth=0.8, alpha=0.85)
ax.set_ylabel("Loss")
ax.set_title("Training Loss Components (Run 17: lr=2e-4 wd=2e-3 epochs=250 pct=0.5)")
ax.legend(fontsize=7, ncol=5, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.25)

# Panel 2: total loss + grad norm
ax2 = axes[1]
ax2.plot(epochs[1:], df["train_loss"].values[1:], color="#1f77b4", linewidth=1.2, label="Train Loss")
ax2.set_ylabel("Loss", color="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#1f77b4")
ax2.grid(True, linestyle="--", alpha=0.25)

ax2b = ax2.twinx()
gn = df["grad_norm"].values[1:]
gn = np.where(np.isinf(gn) | np.isnan(gn), np.nan, gn)
ax2b.plot(epochs[1:], gn, color="#d62728", linewidth=0.7, alpha=0.6, label="Grad Norm")
ax2b.set_ylabel("Gradient Norm", color="#d62728")
ax2b.tick_params(axis="y", labelcolor="#d62728")

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

# Panel 3: LR schedule
ax3 = axes[2]
ax3.plot(epochs, df["lr"].values, color="#2ca02c", linewidth=1.5)
ax3.set_ylabel("Learning Rate")
ax3.set_xlabel("Epoch")
ax3.set_title("OneCycleLR Schedule (pct_start=0.5, lr=2e-4, 250 epochs)")
ax3.grid(True, linestyle="--", alpha=0.25)

# Mark key epochs
for ep, label, idx in [(25, "ep25", 25), (50, "ep50", 50), (100, "ep100", 100),
                      (125, "LR peak", 125), (150, "BEST 0.6110", 150),
                      (200, "ep200", 200), (249, "ep250", 249)]:
    lr_at = df["lr"].values[idx]
    ax3.annotate(label, (ep, lr_at), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=7,
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.6))

fig.tight_layout()
fig.savefig(FIGS / "training_curves.png", dpi=150)
plt.close(fig)
print("✓ training_curves.png")

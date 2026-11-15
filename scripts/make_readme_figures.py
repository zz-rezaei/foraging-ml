"""
Create the figures shown in README.md.

The figures are built from files that the notebooks save, so no model is
re-trained here:
  data/processed/trials_clean.csv                          (notebook 01)
  results/tables/02_behavior_classification_results.csv    (notebook 02)
  results/tables/02_per_class_all_models.csv               (notebook 02)
  results/tables/03_*.csv                                  (notebook 03)

Run from the project root after the notebooks:
    python scripts/make_readme_figures.py

Design note: GitHub shrinks every image to the width of its page column
(about 880 pixels). A wide figure is therefore shrunk a lot and its text
becomes tiny. These figures are kept narrow (at most ~11 inches) and use
large fonts, so the text stays readable after shrinking.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import visualization as viz

TABLES = ROOT / "results" / "tables"
OUT = ROOT / "results" / "figures" / "readme"
OUT.mkdir(parents=True, exist_ok=True)

viz.set_plot_style()
plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
    "axes.titleweight": "bold",
    "figure.facecolor": "white", "savefig.facecolor": "white",   # readable in GitHub dark mode too
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------------------
# Figure 1: behavior by condition (per-mouse proportions, mean ± SEM across mice)
# ---------------------------------------------------------------------------
trials = pd.read_csv(ROOT / "data" / "processed" / "trials_clean.csv")
per_mouse = (trials.groupby(["Status", "Id", "Condition", "SizeCategory"])["Behavior"]
             .value_counts(normalize=True).unstack(fill_value=0)
             .reindex(columns=viz.BEHAVIOR_ORDER, fill_value=0).reset_index())
mean = per_mouse.groupby(["Status", "Condition", "SizeCategory"])[viz.BEHAVIOR_ORDER].mean()
sem = per_mouse.groupby(["Status", "Condition", "SizeCategory"])[viz.BEHAVIOR_ORDER].sem()

cells = [(c, s) for c in viz.CONDITION_ORDER for s in viz.SIZE_ORDER]
fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharey=True)
for ax, behavior in zip(axes.flat, viz.BEHAVIOR_ORDER):
    for g_index, group in enumerate(viz.GROUP_ORDER):
        x = np.arange(len(cells)) + (g_index - 0.5) * 0.2
        y = [mean.loc[(group, c, s), behavior] for c, s in cells]
        e = [sem.loc[(group, c, s), behavior] for c, s in cells]
        ax.errorbar(x, y, yerr=e, color=viz.GROUP_COLORS[group], marker="o", markersize=8,
                    linewidth=2.8, capsize=3, label=group)
    for boundary in [1.5, 3.5, 5.5]:
        ax.axvline(boundary, color=viz.NEUTRAL_LIGHT, linewidth=1.2)
    ax.set_title(behavior)
    ax.set_xticks(np.arange(len(cells)))
    ax.set_xticklabels(["S", "L"] * 4)
    for i, condition in enumerate(viz.CONDITION_ORDER):
        ax.text(2 * i + 0.5, -0.12, condition.replace("-", "\n"), ha="center", va="top",
                fontsize=14, transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1)
for ax in axes[:, 0]:
    ax.set_ylabel("Proportion of trials")
axes[1, 1].legend(loc="upper right")
fig.text(0.5, -0.02, "S = small pellet (sizes 1–2), L = large pellet (sizes 3–4); mean ± SEM across mice",
         ha="center", fontsize=14, color=viz.NEUTRAL_DARK)
fig.subplots_adjust(hspace=0.55, wspace=0.12)
save(fig, "fig1_behavior_by_condition")

# ---------------------------------------------------------------------------
# Figure 2: heatmap of per-behavior recall for every model (held-out mice)
# ---------------------------------------------------------------------------
per_class = pd.read_csv(TABLES / "02_per_class_all_models.csv")
model_rows = [
    ("Task", "Baseline (class frequencies)", "Baseline (always 'Eat')"),
    ("Task", "Logistic regression", "Logistic reg.: task"),
    ("Task + History", "Logistic regression", "Logistic reg.: + previous trial"),
    ("Task + History + Experience", "Logistic regression", "Logistic reg.: + experience"),
    ("Task + History + Experience", "Decision tree", "Decision tree"),
    ("Task + History + Experience", "Random forest", "Random forest"),
    ("Task + History + Experience", "Gradient boosting", "Gradient boosting"),
]
columns = viz.BEHAVIOR_ORDER + ["Balanced\naccuracy"]

fig, axes = plt.subplots(2, 1, figsize=(10, 11.5))
for ax, group in zip(axes, viz.GROUP_ORDER):
    matrix = np.zeros((len(model_rows), len(columns)))
    for r, (features, model, _) in enumerate(model_rows):
        rows = per_class[(per_class["group"] == group) & (per_class["features"] == features)
                         & (per_class["model"] == model)].set_index("behavior")
        recalls = [rows.loc[b, "recall"] for b in viz.BEHAVIOR_ORDER]
        matrix[r, :4] = recalls
        matrix[r, 4] = np.mean(recalls)          # balanced accuracy = mean recall over classes
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = matrix[r, c]
            ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=15,
                    fontweight="bold" if c == 4 else "normal",
                    color="white" if value > 0.6 else "black")
    ax.axvline(3.5, color="white", linewidth=6)          # separate the summary column
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(model_rows)))
    ax.set_yticklabels([label for _, _, label in model_rows])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{group} mice", pad=58, color=viz.GROUP_COLORS[group])
fig.text(0.5, 0.02, "Recall = fraction of each behavior's trials predicted correctly, on mice the model never saw.\n"
         "Balanced accuracy = mean recall over the four behaviors (chance = 0.25).",
         ha="center", fontsize=13, color=viz.NEUTRAL_DARK)
fig.subplots_adjust(hspace=0.42, bottom=0.08)
save(fig, "fig2_model_heatmap")

# ---------------------------------------------------------------------------
# Figure 3: balanced accuracy with confidence intervals (uncertainty view)
# ---------------------------------------------------------------------------
results = pd.read_csv(TABLES / "02_behavior_classification_results.csv")
rows = model_rows[1:]
fig, ax = plt.subplots(figsize=(10, 6.5))
for g_index, group in enumerate(viz.GROUP_ORDER):
    for r_index, (features, model, label) in enumerate(rows):
        row = results[(results["group"] == group) & (results["features"] == features)
                      & (results["model"] == model)].iloc[0]
        low, high = [float(v) for v in row["balanced_accuracy_CI"].strip("[]").split(",")]
        y = r_index + (g_index - 0.5) * 0.32
        ax.plot([low, high], [y, y], color=viz.GROUP_COLORS[group], linewidth=3.5)
        ax.scatter(row["balanced_accuracy"], y, color=viz.GROUP_COLORS[group], s=110, zorder=3,
                   label=group if r_index == 0 else None)
ax.axvline(0.25, color=viz.NEUTRAL_DARK, linestyle="--", linewidth=1.5)
ax.text(0.257, 2.5, "chance", va="center", color=viz.NEUTRAL_DARK, fontsize=15)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([label for _, _, label in rows])
ax.invert_yaxis()
ax.set_xlim(0.2, 0.65)
ax.set_xlabel("Balanced accuracy on held-out mice\n(95% confidence interval over mice)")
ax.grid(axis="x", color=viz.NEUTRAL_LIGHT)
ax.grid(axis="y", visible=False)
ax.legend(loc="center right")
save(fig, "fig3_behavior_prediction_ci")

# ---------------------------------------------------------------------------
# Figure 4: Control vs AD classification from mouse-level features
# ---------------------------------------------------------------------------
group_results = pd.read_csv(TABLES / "03_group_classification_results.csv")
permutation = pd.read_csv(TABLES / "03_permutation_tests.csv").set_index("feature set")
sets = [
    ("Overall proportions", "Overall\nproportions"),
    ("Short-soft choices", "Choices in\nshort-soft"),
    ("Distance sensitivity", "Distance\nsensitivity*"),
    ("Movement (Eat trials)", "Movement\n(Eat trials)"),
    ("Choices + Movement", "Choices +\nmovement"),
]
fig, ax = plt.subplots(figsize=(10, 6.5))
for i, (name, label) in enumerate(sets):
    row = group_results[(group_results["feature set"] == name)
                        & (group_results["model"] == "Logistic regression")].iloc[0]
    null95 = permutation.loc[name, "null 95th percentile"]
    p_value = permutation.loc[name, "p-value"]
    ax.bar(i, null95, width=0.65, color=viz.NEUTRAL_LIGHT, zorder=1,
           label="95% of AUCs with shuffled labels" if i == 0 else None)
    color = viz.GROUP_COLORS["AD"] if p_value < 0.05 else viz.NEUTRAL_DARK
    ax.errorbar(i, row["AUC mean"], yerr=row["AUC SD (splits)"], fmt="o", color=color,
                markersize=14, capsize=5, linewidth=2.5, zorder=3)
    p_text = "p < 0.01" if p_value < 0.01 else f"p = {p_value:.2f}"
    ax.text(i, row["AUC mean"] + row["AUC SD (splits)"] + 0.02, f"{row['AUC mean']:.2f}\n{p_text}",
            ha="center", va="bottom", fontsize=15, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
ax.axhline(0.5, color=viz.NEUTRAL_DARK, linestyle="--", linewidth=1.2)
ax.set_xticks(range(len(sets)))
ax.set_xticklabels([label for _, label in sets])
ax.set_ylim(0.3, 1.12)
ax.set_yticks(np.arange(0.3, 1.01, 0.1))
ax.set_ylabel("ROC-AUC")
ax.legend(loc="upper left")
fig.text(0.5, -0.06, "*36 mice that ran both distances. Logistic regression, repeated 5-fold CV;\n"
         "error bars = SD over 20 random splits; orange = better than chance (permutation test).",
         ha="center", fontsize=13, color=viz.NEUTRAL_DARK)
save(fig, "fig4_group_classification")

# ---------------------------------------------------------------------------
# Figure 5: three mouse-level features that separate the groups
# ---------------------------------------------------------------------------
features = pd.read_csv(TABLES / "03_mouse_features.csv", index_col="Id")
panels = [
    ("soft_distance_effect_NoLeave", "NoLeave increase,\nshort → long"),
    ("ss_Carry", "Carry proportion\n(short-soft)"),
    ("ss_size_effect_Carry", "Extra carrying,\nlarge vs small"),
]
fig, axes = plt.subplots(1, 3, figsize=(11, 5.5))
rng = np.random.default_rng(0)
for ax, (column, title) in zip(axes, panels):
    for g_index, group in enumerate(viz.GROUP_ORDER):
        values = features.loc[features["Status"] == group, column].dropna()
        jitter = rng.uniform(-0.14, 0.14, size=len(values))
        ax.scatter(g_index + jitter, values, s=80, alpha=0.85, color=viz.GROUP_COLORS[group],
                   edgecolor="white", linewidth=0.8)
        ax.hlines(values.median(), g_index - 0.3, g_index + 0.3, color=viz.NEUTRAL_DARK, linewidth=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{g}\nn={features.loc[features['Status'] == g, column].notna().sum()}"
                        for g in viz.GROUP_ORDER])
    ax.set_xlim(-0.6, 1.6)
    ax.set_title(title, fontsize=16)
fig.text(0.5, -0.04, "Dots = mice, bars = medians", ha="center", fontsize=13, color=viz.NEUTRAL_DARK)
fig.tight_layout()
save(fig, "fig5_mouse_features")

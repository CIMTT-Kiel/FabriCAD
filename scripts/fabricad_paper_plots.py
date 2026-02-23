"""
Generate publication-quality figures for the FabriCAD dataset paper.

Targets Elsevier two-column journal format.
- Single-column width:  ~90 mm  (3.54 in)
- Double-column width: ~190 mm  (7.48 in)

All figures are saved as PDF **and** PNG (300 dpi) in reports/figures/.
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fabricad.constants import PATHS

DATA_RAW = PATHS.DATA_RAW
FIG_DIR = PATHS.REPORT_FIGURES
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Elsevier style configuration
# ---------------------------------------------------------------------------
# Elsevier recommends Times / Helvetica; we use serif for body-text harmony.
SINGLE_COL = 3.54  # inches
DOUBLE_COL = 7.48  # inches

COLORS = {
    "primary": "#2c3e50",
    "secondary": "#7f8c8d",
    "accent1": "#2980b9",
    "accent2": "#e74c3c",
    "accent3": "#27ae60",
    "accent4": "#f39c12",
    "removal": "#c0392b",
    "addition": "#27ae60",
    "neutral": "#34495e",
    "light_bg": "#ecf0f1",
}

# Ordered color palette for categorical plots
PALETTE = [
    COLORS["accent1"],
    COLORS["accent2"],
    COLORS["accent3"],
    COLORS["accent4"],
    "#8e44ad",
    "#1abc9c",
    "#d35400",
    "#2c3e50",
    "#16a085",
    "#c0392b",
]

# Step name translations (German -> English)
STEP_NAMES = {
    "liefern": "Delivery",
    "fräsen": "Milling",
    "bohren": "Drilling",
    "schweißen": "Welding",
    "drehen": "Turning",
    "schleifen": "Grinding",
    "prüfen": "Inspection",
    "kontrollieren": "Quality control",
    "entgraten": "Deburring",
    "markieren": "Marking",
    "sägen": "Sawing",
    "biegen": "Bending",
    "stanzen": "Punching",
    "nieten": "Riveting",
    "lackieren": "Coating",
    "initWorkingstep": "Init",
}

MATERIAL_NAMES = {
    "Baustahl": "Structural steel",
    "Aluminiumlegierungen": "Aluminum alloy",
    "Edelstahl": "Stainless steel",
    "Kupferlegierungen": "Copper alloy",
    "Gusseisen": "Cast iron",
}


def _translate_step(name: str) -> str:
    return STEP_NAMES.get(name, name.capitalize())


def _translate_material(name: str) -> str:
    return MATERIAL_NAMES.get(name, name)


def setup_mpl():
    """Apply Elsevier-compatible matplotlib defaults."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.4,
        "axes.grid": False,
        "pdf.fonttype": 42,  # TrueType for editable text in PDF
        "ps.fonttype": 42,
    })


def save_fig(fig, name: str):
    """Save figure as PDF and PNG."""
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)
    print(f"  -> saved {name}.pdf / .png")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset() -> dict:
    """Read all plan.csv and plan_metadata.csv files, return aggregated data."""
    folders = sorted([f for f in DATA_RAW.iterdir() if f.is_dir()])
    print(f"Found {len(folders)} samples in {DATA_RAW}")

    all_steps = []
    plan_stats = []
    step_volume_changes = []
    step_details = []
    materials = []
    dimensions = []
    features_all = []

    for folder in tqdm(folders, desc="Reading plan data"):
        plan_file = folder / "plan.csv"
        meta_file = folder / "plan_metadata.csv"
        feat_file = folder / "interim" / "substeps" / "features.csv"

        if not plan_file.exists():
            continue

        try:
            df = pd.read_csv(plan_file, sep=";")
            init_vol = df.loc[df["Nr."] == 0, "Volumen[mm^3]"].iloc[0] if len(df[df["Nr."] == 0]) > 0 else None
            df_steps = df[df["Nr."] >= 1].copy()

            if len(df_steps) == 0 or init_vol is None:
                continue

            steps_list = df_steps["Schritt"].tolist()
            all_steps.extend(steps_list)

            end_vol = df_steps["Volumen[mm^3]"].iloc[-1]
            total_cost = df_steps["Kosten[($)]"].sum()
            total_dur = df_steps["Dauer[min]"].sum()

            plan_stats.append({
                "id": folder.name,
                "init_volume": init_vol,
                "end_volume": end_vol,
                "volume_diff": end_vol - init_vol,
                "total_cost": total_cost,
                "total_duration": total_dur,
                "n_steps": len(df_steps),
                "steps_sequence": " → ".join([_translate_step(s) for s in steps_list]),
            })

            # Per-step details
            prev_vol = init_vol
            for _, row in df_steps.iterrows():
                vol_diff = row["Volumen[mm^3]"] - prev_vol
                step_details.append({
                    "step": row["Schritt"],
                    "step_en": _translate_step(row["Schritt"]),
                    "cost": row["Kosten[($)]"],
                    "duration": row["Dauer[min]"],
                    "volume_change": vol_diff,
                    "qualification": row["Qualifikation"],
                    "workplace": row["Arbeitsplatz"],
                })
                prev_vol = row["Volumen[mm^3]"]

            # Metadata
            if meta_file.exists():
                meta = pd.read_csv(meta_file, sep=";")
                if len(meta) > 0:
                    materials.append(meta["Material"].iloc[0])
                    dim_str = meta["Abmaße"].iloc[0]
                    try:
                        parts = [float(x) for x in str(dim_str).split("x")]
                        if len(parts) == 3:
                            dimensions.append(sorted(parts, reverse=True))
                    except (ValueError, AttributeError):
                        pass

            # Features
            if feat_file.exists():
                try:
                    feat_df = pd.read_csv(feat_file, sep=";")
                    for _, row in feat_df.iterrows():
                        features_all.append({
                            "sample": folder.name,
                            "main_step": row["Arbeitsschritt"],
                            "substep": row["Subschritt"],
                            "description": row["Kurztext"],
                            "volume": row["Volumen[mm^3]"],
                            "cost": row["Kosten[($)]"],
                            "duration": row["Dauer[min]"],
                        })
                except Exception:
                    pass

        except Exception as e:
            print(f"  Warning: {folder.name}: {e}")
            continue

    return {
        "plan_stats": pd.DataFrame(plan_stats),
        "step_counts": Counter(all_steps),
        "step_details": pd.DataFrame(step_details),
        "materials": materials,
        "dimensions": np.array(dimensions) if dimensions else np.array([]),
        "features": pd.DataFrame(features_all) if features_all else pd.DataFrame(),
        "n_total_steps": len(all_steps),
    }


# ---------------------------------------------------------------------------
# Figure 1: Manufacturing step frequency  (single column)
# ---------------------------------------------------------------------------
def fig_step_frequency(data: dict):
    """Horizontal bar chart of manufacturing step frequencies."""
    counts = data["step_counts"]
    # Remove init step
    counts.pop("initWorkingstep", None)

    df = pd.DataFrame(counts.items(), columns=["step", "count"])
    df["step_en"] = df["step"].map(_translate_step)
    df = df.sort_values("count", ascending=True)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    bars = ax.barh(df["step_en"], df["count"], color=COLORS["accent1"], height=0.65)

    ax.set_xlabel("Frequency")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))

    for bar, val in zip(bars, df["count"]):
        label = f"{val/1000:.1f}k" if val >= 1000 else str(val)
        ax.text(bar.get_width() + ax.get_xlim()[1] * 0.01, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=6)

    ax.set_xlim(right=ax.get_xlim()[1] * 1.12)
    save_fig(fig, "fig1_step_frequency")


# ---------------------------------------------------------------------------
# Figure 2: Distribution of number of steps per plan (single column)
# ---------------------------------------------------------------------------
def fig_steps_per_plan(data: dict):
    """Bar chart showing how many plans have N manufacturing steps."""
    ps = data["plan_stats"]
    dist = ps["n_steps"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    ax.bar(dist.index, dist.values, color=COLORS["accent1"], edgecolor="white", linewidth=0.3, width=0.75)

    ax.set_xlabel("Number of manufacturing steps")
    ax.set_ylabel("Number of plans")
    ax.set_xticks(dist.index)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))

    for x_val, y_val in zip(dist.index, dist.values):
        label = f"{y_val/1000:.1f}k" if y_val >= 1000 else str(y_val)
        ax.text(x_val, y_val + dist.values.max() * 0.02, label,
                ha="center", va="bottom", fontsize=6)

    ax.set_ylim(top=dist.values.max() * 1.12)
    save_fig(fig, "fig2_steps_per_plan")


# ---------------------------------------------------------------------------
# Figure 3: Material distribution (single column)
# ---------------------------------------------------------------------------
def fig_material_distribution(data: dict):
    """Pie or bar chart showing material type distribution."""
    mat_counts = Counter(data["materials"])
    if len(mat_counts) == 0:
        print("  Skipping material distribution (no metadata)")
        return

    df = pd.DataFrame(mat_counts.items(), columns=["material", "count"])
    df["material_en"] = df["material"].map(_translate_material)
    df = df.sort_values("count", ascending=True)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    bars = ax.barh(df["material_en"], df["count"], color=PALETTE[:len(df)], height=0.6)

    ax.set_xlabel("Number of samples")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))

    for bar, val in zip(bars, df["count"]):
        label = f"{val/1000:.1f}k" if val >= 1000 else str(val)
        ax.text(bar.get_width() + ax.get_xlim()[1] * 0.01, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=6)

    ax.set_xlim(right=ax.get_xlim()[1] * 1.15)
    save_fig(fig, "fig3_material_distribution")


# ---------------------------------------------------------------------------
# Figure 4: Average cost and duration per step type (double column)
# ---------------------------------------------------------------------------
def fig_cost_duration_per_step(data: dict):
    """Grouped horizontal bar chart: avg cost and avg duration per step type."""
    sd = data["step_details"]
    # Exclude non-value-adding steps
    sd_filtered = sd[~sd["step"].isin(["liefern", "initWorkingstep"])].copy()

    agg = sd_filtered.groupby("step_en").agg(
        avg_cost=("cost", "mean"),
        avg_duration=("duration", "mean"),
        count=("cost", "count"),
    ).reset_index()
    agg = agg.sort_values("avg_cost", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.6), sharey=True)

    y_pos = np.arange(len(agg))
    h = 0.6

    # Cost
    ax1.barh(y_pos, agg["avg_cost"], height=h, color=COLORS["accent4"])
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(agg["step_en"])
    ax1.set_xlabel("Avg. cost (USD)")
    ax1.set_title("Cost per step", fontsize=8, fontweight="bold")

    for i, val in enumerate(agg["avg_cost"]):
        ax1.text(val + ax1.get_xlim()[1] * 0.02, i, f"${val:.2f}", va="center", fontsize=6)
    ax1.set_xlim(right=ax1.get_xlim()[1] * 1.22)

    # Duration
    ax2.barh(y_pos, agg["avg_duration"], height=h, color=COLORS["accent1"])
    ax2.set_xlabel("Avg. duration (min)")
    ax2.set_title("Duration per step", fontsize=8, fontweight="bold")

    for i, val in enumerate(agg["avg_duration"]):
        ax2.text(val + ax2.get_xlim()[1] * 0.02, i, f"{val:.1f}", va="center", fontsize=6)
    ax2.set_xlim(right=ax2.get_xlim()[1] * 1.22)

    fig.subplots_adjust(wspace=0.08)
    save_fig(fig, "fig4_cost_duration_per_step")


# ---------------------------------------------------------------------------
# Figure 5: Volume change per step type (single column)
# ---------------------------------------------------------------------------
def fig_volume_change_per_step(data: dict):
    """Diverging bar chart showing avg material removal/addition per step."""
    sd = data["step_details"]
    sd_filtered = sd[~sd["step"].isin(["liefern", "prüfen", "kontrollieren", "initWorkingstep"])].copy()

    agg = sd_filtered.groupby("step_en").agg(
        avg_vol=("volume_change", "mean"),
        count=("volume_change", "count"),
    ).reset_index()
    agg = agg.sort_values("avg_vol")

    colors = [COLORS["removal"] if v < 0 else COLORS["addition"] for v in agg["avg_vol"]]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    ax.barh(agg["step_en"], agg["avg_vol"], color=colors, height=0.6)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="-")
    ax.set_xlabel("Avg. volume change (mm³)")

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if abs(x) >= 1e6 else (f"{x/1e3:.0f}k" if abs(x) >= 1e3 else f"{x:.0f}")
    ))

    save_fig(fig, "fig5_volume_change_per_step")


# ---------------------------------------------------------------------------
# Figure 6: Total cost distribution (single column)
# ---------------------------------------------------------------------------
def fig_total_cost_distribution(data: dict):
    """Histogram of total manufacturing cost per plan."""
    ps = data["plan_stats"]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    ax.hist(ps["total_cost"], bins=40, color=COLORS["accent4"], edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Total cost per plan (USD)")
    ax.set_ylabel("Number of plans")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))

    # Add descriptive statistics as text
    mean_c = ps["total_cost"].mean()
    med_c = ps["total_cost"].median()
    ax.axvline(mean_c, color=COLORS["accent2"], linestyle="--", linewidth=0.8, label=f"Mean: ${mean_c:.1f}")
    ax.axvline(med_c, color=COLORS["accent3"], linestyle="--", linewidth=0.8, label=f"Median: ${med_c:.1f}")
    ax.legend(frameon=False, loc="upper right")

    save_fig(fig, "fig6_total_cost_distribution")


# ---------------------------------------------------------------------------
# Figure 7: Total duration distribution (single column)
# ---------------------------------------------------------------------------
def fig_total_duration_distribution(data: dict):
    """Histogram of total manufacturing duration per plan."""
    ps = data["plan_stats"]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    ax.hist(ps["total_duration"], bins=40, color=COLORS["accent1"], edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Total duration per plan (min)")
    ax.set_ylabel("Number of plans")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))

    mean_d = ps["total_duration"].mean()
    med_d = ps["total_duration"].median()
    ax.axvline(mean_d, color=COLORS["accent2"], linestyle="--", linewidth=0.8, label=f"Mean: {mean_d:.1f} min")
    ax.axvline(med_d, color=COLORS["accent3"], linestyle="--", linewidth=0.8, label=f"Median: {med_d:.1f} min")
    ax.legend(frameon=False, loc="upper right")

    save_fig(fig, "fig7_total_duration_distribution")


# ---------------------------------------------------------------------------
# Figure 8: Initial volume distribution (single column)
# ---------------------------------------------------------------------------
def fig_initial_volume(data: dict):
    """Histogram of initial workpiece volumes."""
    ps = data["plan_stats"]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    ax.hist(ps["init_volume"] / 1e6, bins=40, color=COLORS["neutral"], edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Initial workpiece volume (×10⁶ mm³)")
    ax.set_ylabel("Number of plans")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))

    save_fig(fig, "fig8_initial_volume")


# ---------------------------------------------------------------------------
# Figure 9: Workpiece dimension scatter (single column)
# ---------------------------------------------------------------------------
def fig_workpiece_dimensions(data: dict):
    """Scatter plot of workpiece dimensions (length vs width, colored by height)."""
    dims = data["dimensions"]
    if len(dims) == 0:
        print("  Skipping dimensions plot (no data)")
        return

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.8))
    sc = ax.scatter(dims[:, 0], dims[:, 1], c=dims[:, 2], s=3, alpha=0.3,
                    cmap="viridis", rasterized=True)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Height (mm)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ax.set_xlabel("Length (mm)")
    ax.set_ylabel("Width (mm)")

    save_fig(fig, "fig9_workpiece_dimensions")


# ---------------------------------------------------------------------------
# Figure 10: Process sequence patterns – most common sequences (single column)
# ---------------------------------------------------------------------------
def fig_sequence_patterns(data: dict):
    """Show the top-N most frequent manufacturing step sequences."""
    ps = data["plan_stats"]
    seq_counts = ps["steps_sequence"].value_counts().head(12).iloc[::-1]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.6))

    # Shorten labels for readability
    labels = []
    for seq in seq_counts.index:
        parts = seq.split(" → ")
        # Abbreviate: take first 3 letters
        abbr = " → ".join([p[:4] + "." if len(p) > 5 else p for p in parts])
        labels.append(abbr)

    ax.barh(range(len(seq_counts)), seq_counts.values, color=COLORS["accent1"], height=0.65)
    ax.set_yticks(range(len(seq_counts)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Number of plans")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))

    for i, val in enumerate(seq_counts.values):
        label = f"{val/1000:.1f}k" if val >= 1000 else str(val)
        ax.text(val + ax.get_xlim()[1] * 0.01, i, label, va="center", fontsize=6)

    ax.set_xlim(right=ax.get_xlim()[1] * 1.1)
    save_fig(fig, "fig10_sequence_patterns")


# ---------------------------------------------------------------------------
# Figure 11: Feature-level statistics (double column, if data available)
# ---------------------------------------------------------------------------
def fig_feature_statistics(data: dict):
    """Box plots of feature volume, cost, and duration distributions."""
    feat = data["features"]
    if feat.empty:
        print("  Skipping feature statistics (no data)")
        return

    # Identify feature types from description keywords
    def classify_feature(desc: str) -> str:
        desc_lower = str(desc).lower()
        if "bohrung" in desc_lower or "bohru" in desc_lower:
            return "Bore/Hole"
        elif "tasche" in desc_lower:
            return "Pocket"
        elif "stufe" in desc_lower:
            return "Step/Shoulder"
        elif "schlitz" in desc_lower or "nut" in desc_lower:
            return "Slot/Groove"
        elif "schweiß" in desc_lower or "geschweißt" in desc_lower:
            return "Weld"
        elif "fase" in desc_lower:
            return "Chamfer"
        else:
            return "Other"

    feat = feat.copy()
    feat["feature_type"] = feat["description"].apply(classify_feature)

    type_counts = feat["feature_type"].value_counts()
    top_types = type_counts[type_counts >= 50].index.tolist()
    feat_filtered = feat[feat["feature_type"].isin(top_types)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8))

    # Volume distribution per feature type
    types_sorted = feat_filtered.groupby("feature_type")["volume"].median().sort_values().index
    box_data_vol = [feat_filtered[feat_filtered["feature_type"] == t]["volume"].dropna().values / 1e3
                    for t in types_sorted]
    bp1 = ax1.boxplot(box_data_vol, vert=False, labels=types_sorted,
                      patch_artist=True, widths=0.5,
                      flierprops={"markersize": 1.5, "alpha": 0.3},
                      medianprops={"color": COLORS["accent2"], "linewidth": 1})
    for patch in bp1["boxes"]:
        patch.set_facecolor(COLORS["accent1"])
        patch.set_alpha(0.6)
    ax1.set_xlabel("Volume (×10³ mm³)")
    ax1.set_title("Volume per feature type", fontsize=8, fontweight="bold")

    # Duration distribution per feature type
    box_data_dur = [feat_filtered[feat_filtered["feature_type"] == t]["duration"].dropna().values
                    for t in types_sorted]
    bp2 = ax2.boxplot(box_data_dur, vert=False, labels=types_sorted,
                      patch_artist=True, widths=0.5,
                      flierprops={"markersize": 1.5, "alpha": 0.3},
                      medianprops={"color": COLORS["accent2"], "linewidth": 1})
    for patch in bp2["boxes"]:
        patch.set_facecolor(COLORS["accent4"])
        patch.set_alpha(0.6)
    ax2.set_xlabel("Duration (min)")
    ax2.set_title("Duration per feature type", fontsize=8, fontweight="bold")
    ax2.set_yticklabels([])

    fig.subplots_adjust(wspace=0.08)
    save_fig(fig, "fig11_feature_statistics")


# ---------------------------------------------------------------------------
# Summary statistics table (LaTeX)
# ---------------------------------------------------------------------------
def generate_summary_table(data: dict):
    """Print a LaTeX-formatted summary statistics table."""
    ps = data["plan_stats"]
    sd = data["step_details"]
    n_samples = len(ps)
    n_steps_total = data["n_total_steps"]

    rows = [
        ("Total samples", f"{n_samples:,}"),
        ("Total manufacturing steps", f"{n_steps_total:,}"),
        ("Unique step types", f"{len(data['step_counts']):,}"),
        ("Avg. steps per plan", f"{ps['n_steps'].mean():.1f} ± {ps['n_steps'].std():.1f}"),
        ("Avg. total cost (USD)", f"{ps['total_cost'].mean():.2f} ± {ps['total_cost'].std():.2f}"),
        ("Avg. total duration (min)", f"{ps['total_duration'].mean():.1f} ± {ps['total_duration'].std():.1f}"),
        ("Avg. init. volume (mm³)", f"{ps['init_volume'].mean()/1e6:.2f}M ± {ps['init_volume'].std()/1e6:.2f}M"),
    ]

    if data["materials"]:
        rows.append(("Unique materials", f"{len(set(data['materials']))}"))

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS (for paper text / table)")
    print("=" * 60)
    for label, value in rows:
        print(f"  {label:40s} {value}")

    # Also save as LaTeX
    tex_path = FIG_DIR / "summary_table.tex"
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Summary statistics of the FabriCAD dataset.}\n")
        f.write("\\label{tab:summary}\n")
        f.write("\\begin{tabular}{lr}\n")
        f.write("\\toprule\n")
        f.write("Metric & Value \\\\\n")
        f.write("\\midrule\n")
        for label, value in rows:
            f.write(f"{label} & {value} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"  -> saved summary_table.tex")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    setup_mpl()
    print("Loading FabriCAD dataset...")
    data = load_dataset()
    ps = data["plan_stats"]
    print(f"Loaded {len(ps)} plans with {data['n_total_steps']} total steps.\n")

    print("Generating figures...")
    fig_step_frequency(data)
    fig_steps_per_plan(data)
    fig_material_distribution(data)
    fig_cost_duration_per_step(data)
    fig_volume_change_per_step(data)
    fig_total_cost_distribution(data)
    fig_total_duration_distribution(data)
    fig_initial_volume(data)
    fig_workpiece_dimensions(data)
    fig_sequence_patterns(data)
    fig_feature_statistics(data)
    generate_summary_table(data)

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()

"""
Exploration Graphs — train_set.csv
All 86 features are StandardScaler-transformed; target credit_card_holder_flag is raw (0/1).

Output: exploration_graphs_train.pdf  (multi-page, one figure group per page)

Usage (run from folder containing train_set.csv):
    python exploration_graphs_train.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
_candidates = [os.path.join(BASE_DIR, "train_set.csv"),
               os.path.join(os.getcwd(), "train_set.csv")]
INPUT_FILE  = next((p for p in _candidates if os.path.exists(p)), _candidates[0])
OUTPUT_FILE = os.path.join(os.path.dirname(INPUT_FILE), "exploration_graphs_train.pdf")

# ── Theme ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi":     130,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
C0 = "#2E75B6"   # blue  – Target = 0
C1 = "#C0392B"   # red   – Target = 1
PALETTE = [C0, C1]

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(INPUT_FILE)
print(f"  {df.shape[0]:,} rows × {df.shape[1]} columns")

TARGET = "credit_card_holder_flag"
df0 = df[df[TARGET] == 0]
df1 = df[df[TARGET] == 1]
N0, N1 = len(df0), len(df1)

# ── Column groups ─────────────────────────────────────────────────────────────
DUMMY_PREFIXES = [
    "gender_", "education_level_", "employment_status_", "marital_status_",
    "primary_account_type_", "account_status_", "loyalty_tier_",
    "risk_tier_", "state_",
]
dummy_cols = [c for c in df.columns if any(c.startswith(p) for p in DUMMY_PREFIXES)]
flag_cols  = [c for c in df.columns
              if c not in dummy_cols + [TARGET]
              and df[c].nunique() <= 3]
continuous_cols = [c for c in df.columns
                   if c not in dummy_cols + flag_cols + [TARGET]]

# Group dummies by original variable
dummy_groups: dict[str, list[str]] = {}
for c in dummy_cols:
    for p in DUMMY_PREFIXES:
        if c.startswith(p):
            dummy_groups.setdefault(p.rstrip("_"), []).append(c)
            break

print(f"  Continuous: {len(continuous_cols)}  |  "
      f"Flags: {len(flag_cols)}  |  "
      f"Dummy groups: {len(dummy_groups)}  |  "
      f"Dummy cols: {len(dummy_cols)}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def add_fig_title(fig, title):
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

def save_page(pdf, fig):
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

def hist_with_target(ax, col, bins=35, title=None):
    """Overlapping histograms for target=0 vs target=1 (scaled data)."""
    ax.hist(df0[col], bins=bins, alpha=0.55, color=C0, label=f"Non-holder (n={N0:,})",
            density=True, edgecolor="none")
    ax.hist(df1[col], bins=bins, alpha=0.70, color=C1, label=f"Holder (n={N1:,})",
            density=True, edgecolor="none")
    ax.set_title(title or col.replace("_", " ").title())
    ax.set_xlabel("Scaled Value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7, framealpha=0.5)

def box_by_target(ax, col, title=None):
    """Side-by-side boxplots by target class."""
    data = [df0[col].dropna().values, df1[col].dropna().values]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for patch, color in zip(bp["boxes"], [C0, C1]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Non-holder\n(0)", "Holder\n(1)"])
    ax.set_title(title or col.replace("_", " ").title())
    ax.set_ylabel("Scaled Value")

# ── PDF ───────────────────────────────────────────────────────────────────────
print("Generating PDF …")
with PdfPages(OUTPUT_FILE) as pdf:

    # =========================================================================
    # PAGE 1 — TARGET DISTRIBUTION
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    add_fig_title(fig, "Page 1 · Target Variable Distribution")

    ax = axes[0]
    counts = [N0, N1]
    bars = ax.bar(["Non-holder (0)", "Holder (1)"], counts,
                  color=[C0, C1], edgecolor="white", width=0.5)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{cnt:,}\n({cnt/len(df):.1%})", ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Class Counts")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(counts) * 1.18)

    ax = axes[1]
    wedges, texts, autotexts = ax.pie(
        counts, labels=["Non-holder (0)", "Holder (1)"],
        colors=[C0, C1], autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.5)
    )
    for at in autotexts:
        at.set_fontsize(11); at.set_fontweight("bold")
    ax.set_title(f"Class Balance  (ratio {N0//N1}:1)")

    save_page(pdf, fig)
    print("  ✅  Page 1 — Target Distribution")

    # =========================================================================
    # PAGES 2–3 — CONTINUOUS FEATURES: DISTRIBUTIONS (density histograms)
    # =========================================================================
    CORE_CONT = [
        "age", "annual_income", "account_tenure_years", "investable_assets_total",
        "checking_balance", "savings_balance", "retirement_balance", "brokerage_balance",
        "monthly_transaction_count", "monthly_spend_amt", "avg_transaction_amount",
        "digital_engagement_score", "mobile_app_logins_monthly", "credit_score",
        "debt_to_income_ratio", "credit_utilization_ratio", "total_open_tradelines",
        "nps_score", "customer_service_calls_12mo", "atm_withdrawals_monthly",
        "overdraft_count_12mo", "total_products_owned", "household_size",
        "mobile_txn_pct", "online_txn_pct", "branch_txn_pct",
    ]
    CORE_CONT = [c for c in CORE_CONT if c in df.columns]

    # Split into two pages of 12 each
    for page_num, chunk in enumerate(
        [CORE_CONT[:13], CORE_CONT[13:]], start=2
    ):
        ncols = 3
        nrows = int(np.ceil(len(chunk) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.5))
        axes = axes.flatten()
        add_fig_title(fig, f"Page {page_num} · Continuous Features — "
                          f"Density Distribution by Target Class")
        for i, col in enumerate(chunk):
            hist_with_target(axes[i], col)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        save_page(pdf, fig)
        print(f"  ✅  Page {page_num} — Continuous histograms ({len(chunk)} cols)")

    # =========================================================================
    # PAGE 4 — CONTINUOUS FEATURES: BOX PLOTS BY TARGET
    # =========================================================================
    KEY_BOX = [
        "annual_income", "investable_assets_total", "monthly_spend_amt",
        "monthly_transaction_count", "avg_transaction_amount", "credit_score",
        "digital_engagement_score", "debt_to_income_ratio", "credit_utilization_ratio",
        "total_products_owned", "account_tenure_years", "nps_score",
    ]
    KEY_BOX = [c for c in KEY_BOX if c in df.columns]

    ncols, nrows = 4, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
    axes = axes.flatten()
    add_fig_title(fig, "Page 4 · Key Features — Box Plots by Target Class")
    for i, col in enumerate(KEY_BOX):
        box_by_target(axes[i], col)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    save_page(pdf, fig)
    print("  ✅  Page 4 — Box plots")

    # =========================================================================
    # PAGE 5 — MEAN FEATURE VALUE BY TARGET (bar charts, top diverging)
    # =========================================================================
    means0 = df0[continuous_cols].mean()
    means1 = df1[continuous_cols].mean()
    diff   = (means1 - means0).abs().sort_values(ascending=False)
    top20  = diff.head(20).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(15, 9))
    add_fig_title(fig, "Page 5 · Top 20 Features — Mean Difference by Target Class")

    ax = axes[0]
    x = np.arange(len(top20))
    w = 0.38
    bars0 = ax.barh(x + w/2, means0[top20], w, color=C0, alpha=0.8, label="Non-holder (0)")
    bars1 = ax.barh(x - w/2, means1[top20], w, color=C1, alpha=0.8, label="Holder (1)")
    ax.set_yticks(x)
    ax.set_yticklabels([c.replace("_", " ") for c in top20], fontsize=8)
    ax.set_xlabel("Mean (Scaled Value)")
    ax.set_title("Mean Value per Class")
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    ax = axes[1]
    raw_diff = means1[top20] - means0[top20]
    colors   = [C1 if v > 0 else C0 for v in raw_diff]
    ax.barh(x, raw_diff, color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels([c.replace("_", " ") for c in top20], fontsize=8)
    ax.set_xlabel("Mean(Holder) − Mean(Non-holder)")
    ax.set_title("Difference in Means  (red = higher for holders)")
    ax.invert_yaxis()

    save_page(pdf, fig)
    print("  ✅  Page 5 — Mean difference chart")

    # =========================================================================
    # PAGE 6 — FLAG COLUMNS (proportion by target)
    # =========================================================================
    FLAG_COLS = [
        "has_credit_card", "has_mortgage", "has_auto_loan", "has_personal_loan",
        "has_home_equity_loan", "has_cd", "has_money_market", "has_safe_deposit_box",
        "direct_deposit_flag", "paper_statements_flag", "autopay_enrolled_flag",
        "bankruptcy_flag", "collections_flag", "charge_off_flag",
    ]
    FLAG_COLS = [c for c in FLAG_COLS if c in df.columns]

    # Since flags are scaled, we compare mean scaled value as a proxy
    nrows = int(np.ceil(len(FLAG_COLS) / 3))
    fig, axes = plt.subplots(nrows, 3, figsize=(14, nrows * 3.2))
    axes = axes.flatten()
    add_fig_title(fig, "Page 6 · Binary Flag Columns — "
                       "Mean Scaled Value by Target Class\n"
                       "(higher mean = more '1' encoded rows in that class)")
    for i, col in enumerate(FLAG_COLS):
        m0 = df0[col].mean()
        m1 = df1[col].mean()
        axes[i].bar(["Non-holder (0)", "Holder (1)"], [m0, m1],
                    color=[C0, C1], edgecolor="white", alpha=0.85, width=0.5)
        axes[i].set_title(col.replace("_", " ").replace("has ", "").title(), fontsize=9)
        axes[i].set_ylabel("Mean scaled value")
        for xi, v in enumerate([m0, m1]):
            axes[i].text(xi, v + abs(v)*0.03, f"{v:.3f}", ha="center", fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    save_page(pdf, fig)
    print("  ✅  Page 6 — Binary flags")

    # =========================================================================
    # PAGE 7 — DUMMY VARIABLE GROUPS (proportion by target)
    # =========================================================================
    PLOT_GROUPS = {k: v for k, v in dummy_groups.items() if k != "state"}
    n_groups    = len(PLOT_GROUPS)
    ncols       = 3
    nrows       = int(np.ceil(n_groups / ncols))
    fig, axes   = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes        = axes.flatten()
    add_fig_title(fig, "Page 7 · One-Hot Dummy Groups — "
                       "Mean Scaled Value by Target Class")
    for i, (grp_name, cols) in enumerate(PLOT_GROUPS.items()):
        ax      = axes[i]
        labels  = [c.replace(grp_name + "_", "").replace("_", " ") for c in cols]
        x       = np.arange(len(cols))
        w       = 0.35
        m0      = [df0[c].mean() for c in cols]
        m1      = [df1[c].mean() for c in cols]
        ax.bar(x - w/2, m0, w, color=C0, alpha=0.8, label="Non-holder")
        ax.bar(x + w/2, m1, w, color=C1, alpha=0.8, label="Holder")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(grp_name.replace("_", " ").title())
        ax.set_ylabel("Mean scaled value")
        ax.legend(fontsize=7)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    save_page(pdf, fig)
    print("  ✅  Page 7 — Dummy variable groups")

    # =========================================================================
    # PAGE 8 — STATE DUMMY DISTRIBUTION (horizontal bar)
    # =========================================================================
    state_cols = dummy_groups.get("state", [])
    if state_cols:
        labels_s = [c.replace("state_", "") for c in state_cols]
        m0_s = np.array([df0[c].mean() for c in state_cols])
        m1_s = np.array([df1[c].mean() for c in state_cols])
        order = np.argsort(m1_s - m0_s)
        labels_s = [labels_s[o] for o in order]
        m0_s = m0_s[order]; m1_s = m1_s[order]

        fig, ax = plt.subplots(figsize=(10, 10))
        add_fig_title(fig, "Page 8 · State Dummies — "
                           "Mean Scaled Value by Target Class")
        y = np.arange(len(labels_s))
        w = 0.38
        ax.barh(y + w/2, m0_s, w, color=C0, alpha=0.8, label="Non-holder (0)")
        ax.barh(y - w/2, m1_s, w, color=C1, alpha=0.8, label="Holder (1)")
        ax.set_yticks(y)
        ax.set_yticklabels(labels_s, fontsize=9)
        ax.set_xlabel("Mean Scaled Value")
        ax.legend(fontsize=9)
        ax.axvline(0, color="black", linewidth=0.6)
        save_page(pdf, fig)
        print("  ✅  Page 8 — State dummies")

    # =========================================================================
    # PAGE 9 — CORRELATION HEATMAP (continuous features)
    # =========================================================================
    corr = df[continuous_cols].corr()
    fig, ax = plt.subplots(figsize=(18, 15))
    add_fig_title(fig, "Page 9 · Correlation Heatmap — Continuous Features")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        annot=False, linewidths=0.2,
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    save_page(pdf, fig)
    print("  ✅  Page 9 — Correlation heatmap")

    # =========================================================================
    # PAGE 10 — SCATTER PLOTS: KEY CONTINUOUS PAIRS
    # =========================================================================
    PAIRS = [
        ("annual_income",        "monthly_spend_amt"),
        ("credit_score",         "debt_to_income_ratio"),
        ("monthly_transaction_count", "monthly_spend_amt"),
        ("digital_engagement_score",  "mobile_app_logins_monthly"),
        ("investable_assets_total",   "retirement_balance"),
        ("account_tenure_years",      "total_products_owned"),
    ]
    PAIRS = [(a, b) for a, b in PAIRS if a in df.columns and b in df.columns]

    ncols = 2; nrows = int(np.ceil(len(PAIRS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 4.5))
    axes = axes.flatten()
    add_fig_title(fig, "Page 10 · Scatter Plots — Key Feature Pairs by Target Class")
    for i, (xa, xb) in enumerate(PAIRS):
        ax = axes[i]
        # Sample for speed
        samp0 = df0.sample(min(1500, N0), random_state=42)
        samp1 = df1.sample(min(1500, N1), random_state=42)
        ax.scatter(samp0[xa], samp0[xb], alpha=0.25, s=8, color=C0, label="Non-holder")
        ax.scatter(samp1[xa], samp1[xb], alpha=0.45, s=12, color=C1, label="Holder")
        r = df[xa].corr(df[xb])
        ax.set_xlabel(xa.replace("_", " "))
        ax.set_ylabel(xb.replace("_", " "))
        ax.set_title(f"r = {r:.3f}")
        ax.legend(fontsize=7, markerscale=2)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    save_page(pdf, fig)
    print("  ✅  Page 10 — Scatter plots")

    # =========================================================================
    # PAGE 11 — VIOLIN PLOTS: HIGH-SIGNAL FEATURES
    # =========================================================================
    HIGH_SIGNAL = [
        "monthly_transaction_count", "total_products_owned",
        "digital_engagement_score",  "annual_income",
        "credit_score",              "monthly_spend_amt",
    ]
    HIGH_SIGNAL = [c for c in HIGH_SIGNAL if c in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    add_fig_title(fig, "Page 11 · High-Signal Features — Violin Plots by Target Class")
    for i, col in enumerate(HIGH_SIGNAL):
        ax = axes[i]
        data_plot = pd.DataFrame({
            "value": pd.concat([df0[col], df1[col]]),
            "target": ["Non-holder"] * N0 + ["Holder"] * N1,
        })
        sns.violinplot(data=data_plot, x="target", y="value",
                       palette=[C0, C1], ax=ax, inner="quartile",
                       hue="target", legend=False)
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel(""); ax.set_ylabel("Scaled Value")
    save_page(pdf, fig)
    print("  ✅  Page 11 — Violin plots")

    # =========================================================================
    # PAGE 12 — FEATURE VARIANCE COMPARISON (holders vs non-holders)
    # =========================================================================
    var0 = df0[continuous_cols].var()
    var1 = df1[continuous_cols].var()
    var_ratio = (var1 / var0.replace(0, np.nan)).sort_values(ascending=False)
    top_var = var_ratio.head(20)

    fig, ax = plt.subplots(figsize=(11, 8))
    add_fig_title(fig, "Page 12 · Top 20 Features by Variance Ratio "
                       "(Holders ÷ Non-holders)")
    colors = [C1 if v > 1 else C0 for v in top_var.values]
    y = np.arange(len(top_var))
    ax.barh(y, top_var.values, color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(1.0, color="black", linewidth=1, linestyle="--", label="Ratio = 1 (equal)")
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in top_var.index], fontsize=9)
    ax.set_xlabel("Variance(Holder) ÷ Variance(Non-holder)")
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    save_page(pdf, fig)
    print("  ✅  Page 12 — Variance ratio")

print(f"\n🎉  Saved → {OUTPUT_FILE}")

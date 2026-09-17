"""
exploration_graphs.py
─────────────────────
Generates a multi-page PDF of exploratory graphs for the banking dataset.

Usage
─────
Place this file in the same folder as banking_dataset_sanitized.csv, then run:
    python exploration_graphs.py

Output
──────
    output/exploration_graphs.pdf   (12 chart sections + cover page)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "banking_dataset_sanitized.csv")
OUT_PDF   = os.path.join(BASE_DIR, "output", "exploration_graphs.pdf")
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
TARGET     = "credit_card_holder_flag"
EXCLUDE    = ["customer_id", "zip_code", "account_open_date", "has_credit_card", TARGET]
BLUE       = "#4C72B0"
ORANGE     = "#DD8452"
TITLE_SIZE = 13

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

num_cols  = [c for c in df.select_dtypes(include=[np.number]).columns if c not in EXCLUDE]
flag_cols = [c for c in num_cols if df[c].dropna().isin([0, 1]).all()]
cont_cols = [c for c in num_cols if c not in flag_cols]
cat_cols  = [c for c in df.select_dtypes(include=["object", "string"]).columns if c not in EXCLUDE]

# Label column used by seaborn (palette needs string keys in v0.13)
df["_target_lbl"] = df[TARGET].map({0: "Non-Holder", 1: "Holder"})
PAL = {"Non-Holder": BLUE, "Holder": ORANGE}

sns.set_theme(style="whitegrid", palette="muted")


def save_fig(pdf, fig):
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
with pdf_backend.PdfPages(OUT_PDF) as pdf:

    # ── COVER ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#2C3E50")
    kw = dict(ha="center", transform=fig.transFigure)
    fig.text(0.5, 0.62, "Banking Dataset", fontsize=36, fontweight="bold", color="white", **kw)
    fig.text(0.5, 0.52, "Exploratory Data Analysis", fontsize=22, color="#AED6F1", **kw)
    fig.text(0.5, 0.42, f"{len(df):,} customers  ·  {len(df.columns)} features",
             fontsize=14, color="#BDC3C7", **kw)
    fig.text(0.5, 0.30,
             f"Target: {TARGET}   |   Class 1 (Holder): {df[TARGET].mean()*100:.1f}%",
             fontsize=13, color="#F0B27A", **kw)
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)

    # ── 1. TARGET DISTRIBUTION ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("1  ·  Target Variable Distribution", fontsize=TITLE_SIZE + 2, fontweight="bold")
    counts = df[TARGET].value_counts().sort_index()
    axes[0].bar(["Non-Holder (0)", "Holder (1)"], counts.values,
                color=[BLUE, ORANGE], edgecolor="white", linewidth=0.8)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 30, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=10)
    axes[0].set_title("Count by Class")
    axes[0].set_ylabel("Count")
    axes[1].pie(counts.values, labels=["Non-Holder", "Holder"],
                colors=[BLUE, ORANGE], autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=1.5))
    axes[1].set_title("Class Proportion")
    save_fig(pdf, fig)

    # ── 2. CONTINUOUS — HISTOGRAMS + KDE vs TARGET ────────────────────────────
    key_cont = [c for c in ["age", "annual_income", "account_tenure_years",
                             "investable_assets_total", "retirement_balance",
                             "monthly_spend_amt", "avg_transaction_amount",
                             "credit_score", "debt_to_income_ratio",
                             "credit_utilization_ratio", "digital_engagement_score",
                             "monthly_transaction_count"] if c in df.columns]
    NC = 3
    NR = int(np.ceil(len(key_cont) / NC))
    fig, axes = plt.subplots(NR, NC, figsize=(14, NR * 3.2))
    fig.suptitle("2  ·  Continuous Features — Distribution by Target Class",
                 fontsize=TITLE_SIZE + 1, fontweight="bold", y=1.01)
    axes = axes.flatten()
    for i, col in enumerate(key_cont):
        for val, color, lbl in [(0, BLUE, "Non-Holder"), (1, ORANGE, "Holder")]:
            sns.histplot(df.loc[df[TARGET] == val, col], ax=axes[i],
                         kde=True, color=color, alpha=0.45, label=lbl,
                         stat="density", bins=30, linewidth=0)
        axes[i].set_title(col.replace("_", " ").title(), fontsize=10)
        axes[i].set_xlabel("")
        axes[i].legend(fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    save_fig(pdf, fig)

    # ── 3. BOX PLOTS ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(NR, NC, figsize=(14, NR * 3.2))
    fig.suptitle("3  ·  Continuous Features — Box Plots by Target Class",
                 fontsize=TITLE_SIZE + 1, fontweight="bold", y=1.01)
    axes = axes.flatten()
    for i, col in enumerate(key_cont):
        sns.boxplot(data=df, x="_target_lbl", y=col, ax=axes[i],
                    hue="_target_lbl", palette=PAL, legend=False,
                    order=["Non-Holder", "Holder"],
                    width=0.5, linewidth=0.8,
                    flierprops=dict(marker="o", markersize=2, alpha=0.3))
        axes[i].set_title(col.replace("_", " ").title(), fontsize=10)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    save_fig(pdf, fig)

    # ── 4. CATEGORICAL — HOLDER RATE BAR CHARTS ───────────────────────────────
    key_cat = [c for c in ["gender", "education_level", "employment_status",
                            "marital_status", "primary_account_type",
                            "account_status", "loyalty_tier", "risk_tier"]
               if c in df.columns]
    NR2 = int(np.ceil(len(key_cat) / 2))
    fig, axes = plt.subplots(NR2, 2, figsize=(14, NR2 * 3.8))
    fig.suptitle("4  ·  Categorical Features — Holder Rate by Category",
                 fontsize=TITLE_SIZE + 1, fontweight="bold", y=1.01)
    axes = axes.flatten()
    for i, col in enumerate(key_cat):
        rate = (df.groupby(col)[TARGET].mean() * 100).sort_values(ascending=False)
        bars = axes[i].bar(rate.index, rate.values,
                           color=sns.color_palette("muted", len(rate)), edgecolor="white")
        for bar, v in zip(bars, rate.values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                         f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
        axes[i].set_title(col.replace("_", " ").title(), fontsize=10)
        axes[i].set_ylabel("Holder Rate (%)")
        axes[i].tick_params(axis="x", rotation=30)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    save_fig(pdf, fig)

    # ── 5. BINARY FLAGS — ADOPTION & HOLDER RATE ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("5  ·  Binary Product / Flag Columns", fontsize=TITLE_SIZE + 2, fontweight="bold")
    overall = (df[flag_cols].mean() * 100).sort_values(ascending=True)
    axes[0].barh(overall.index, overall.values, color=BLUE, edgecolor="white")
    for i2, v in enumerate(overall.values):
        axes[0].text(v + 0.3, i2, f"{v:.1f}%", va="center", fontsize=8)
    axes[0].set_title("Overall Adoption Rate (%)")
    axes[0].set_xlabel("% of Customers")
    flag_rates = df.groupby(TARGET)[flag_cols].mean().T * 100
    flag_rates.columns = ["Non-Holder", "Holder"]
    flag_rates = flag_rates.sort_values("Holder", ascending=True)
    x = np.arange(len(flag_rates))
    w = 0.35
    axes[1].barh(x - w / 2, flag_rates["Non-Holder"], w, label="Non-Holder",
                 color=BLUE, edgecolor="white")
    axes[1].barh(x + w / 2, flag_rates["Holder"], w, label="Holder",
                 color=ORANGE, edgecolor="white")
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(flag_rates.index, fontsize=8)
    axes[1].set_xlabel("% of Group")
    axes[1].set_title("Flag Rate: Holder vs Non-Holder")
    axes[1].legend()
    save_fig(pdf, fig)

    # ── 6. CORRELATION HEATMAP ────────────────────────────────────────────────
    corr_cols = [c for c in cont_cols if c in df.columns] + [TARGET]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", ax=ax,
                cmap="RdBu_r", center=0, linewidths=0.4,
                annot_kws={"size": 7}, vmin=-1, vmax=1)
    ax.set_title("6  ·  Correlation Heatmap — Continuous Features",
                 fontsize=TITLE_SIZE + 1, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    save_fig(pdf, fig)

    # ── 7. FEATURE CORRELATIONS WITH TARGET ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    tc = df[num_cols + [TARGET]].corr()[TARGET].drop(TARGET)
    tc = tc.reindex(tc.abs().sort_values(ascending=True).index)
    ax.barh(tc.index, tc.values,
            color=[ORANGE if v > 0 else BLUE for v in tc.values], edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"7  ·  Feature Correlations with {TARGET}",
                 fontsize=TITLE_SIZE + 1, fontweight="bold")
    ax.set_xlabel("Pearson Correlation")
    ax.tick_params(axis="y", labelsize=8)
    save_fig(pdf, fig)

    # ── 8. FINANCIAL BALANCES ─────────────────────────────────────────────────
    bal_cols = [c for c in ["checking_balance", "savings_balance", "retirement_balance",
                             "brokerage_balance", "investable_assets_total"] if c in df.columns]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("8  ·  Financial Balances Overview", fontsize=TITLE_SIZE + 2, fontweight="bold")
    medians = df.groupby("_target_lbl")[bal_cols].median().T
    medians.plot(kind="bar", ax=axes[0], color=[BLUE, ORANGE], edgecolor="white")
    axes[0].set_title("Median Balance by Target Class")
    axes[0].set_ylabel("Median ($)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    axes[0].legend()
    sns.violinplot(data=df, x="_target_lbl", y="annual_income", ax=axes[1],
                   hue="_target_lbl", palette=PAL, legend=False,
                   order=["Non-Holder", "Holder"], inner="quartile", linewidth=0.8)
    axes[1].set_title("Annual Income by Target Class")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Annual Income ($)")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    save_fig(pdf, fig)

    # ── 9. CREDIT PROFILE KDE ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("9  ·  Credit Profile by Target Class", fontsize=TITLE_SIZE + 2, fontweight="bold")
    for ax, col, title in zip(axes,
                               ["credit_score", "debt_to_income_ratio", "credit_utilization_ratio"],
                               ["Credit Score", "Debt-to-Income Ratio", "Credit Utilization Ratio"]):
        for val, color, lbl in [(0, BLUE, "Non-Holder"), (1, ORANGE, "Holder")]:
            sns.kdeplot(df.loc[df[TARGET] == val, col], ax=ax,
                        color=color, label=lbl, fill=True, alpha=0.35)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.legend(fontsize=8)
    save_fig(pdf, fig)

    # ── 10. TRANSACTION BEHAVIOUR — SCATTER PLOTS ─────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("10 ·  Transaction Behaviour — Scatter Plots",
                 fontsize=TITLE_SIZE + 2, fontweight="bold")
    pairs = [("monthly_transaction_count", "monthly_spend_amt"),
             ("digital_engagement_score",  "mobile_app_logins_monthly"),
             ("atm_withdrawals_monthly",   "overdraft_count_12mo"),
             ("mobile_txn_pct",            "online_txn_pct")]
    for ax, (xc, yc) in zip(axes.flatten(), pairs):
        for val, color, lbl in [(0, BLUE, "Non-Holder"), (1, ORANGE, "Holder")]:
            sub = df[df[TARGET] == val].sample(
                min(500, (df[TARGET] == val).sum()), random_state=42)
            ax.scatter(sub[xc], sub[yc], c=color, alpha=0.35, s=12, label=lbl)
        ax.set_xlabel(xc.replace("_", " ").title(), fontsize=9)
        ax.set_ylabel(yc.replace("_", " ").title(), fontsize=9)
        ax.set_title(f"{xc.replace('_',' ').title()} vs {yc.replace('_',' ').title()}", fontsize=9)
        ax.legend(fontsize=8)
    save_fig(pdf, fig)

    # ── 11. DEMOGRAPHIC PROFILE ───────────────────────────────────────────────
    df["tenure_band"] = pd.cut(df["account_tenure_years"],
                                bins=[0, 5, 10, 15, 20, 30],
                                labels=["0-5", "5-10", "10-15", "15-20", "20+"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("11 ·  Demographic Profile", fontsize=TITLE_SIZE + 2, fontweight="bold")
    for val, color, lbl in [(0, BLUE, "Non-Holder"), (1, ORANGE, "Holder")]:
        sns.histplot(df.loc[df[TARGET] == val, "age"], ax=axes[0],
                     kde=True, color=color, alpha=0.5, label=lbl,
                     stat="density", bins=25, linewidth=0)
    axes[0].set_title("Age Distribution")
    axes[0].legend()
    hs = df.groupby("household_size")[TARGET].mean() * 100
    axes[1].bar(hs.index, hs.values, color=BLUE, edgecolor="white")
    axes[1].set_title("Holder Rate by Household Size")
    axes[1].set_xlabel("Household Size")
    axes[1].set_ylabel("Holder Rate (%)")
    ten = df.groupby("tenure_band", observed=True)[TARGET].mean() * 100
    axes[2].bar(ten.index.astype(str), ten.values, color=BLUE, edgecolor="white")
    axes[2].set_title("Holder Rate by Account Tenure")
    axes[2].set_xlabel("Tenure Band (yrs)")
    axes[2].set_ylabel("Holder Rate (%)")
    save_fig(pdf, fig)

    # ── 12. LOYALTY & RISK TIER ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("12 ·  Loyalty & Risk Tier Analysis", fontsize=TITLE_SIZE + 2, fontweight="bold")
    for ax, col, order, title in zip(
        axes,
        ["loyalty_tier", "risk_tier"],
        [["Bronze", "Silver", "Gold", "Platinum"],
         ["Deep Subprime", "Subprime", "Near Prime", "Prime", "Prime Plus"]],
        ["Holder Rate by Loyalty Tier", "Holder Rate by Risk Tier"]
    ):
        present = [o for o in order if o in df[col].unique()]
        rate = df.groupby(col)[TARGET].mean().reindex(present) * 100
        ax.bar(rate.index, rate.values,
               color=sns.color_palette("Blues_d", len(rate)), edgecolor="white")
        for i3, v in enumerate(rate.values):
            ax.text(i3, v + 0.2, f"{v:.1f}%", ha="center", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel("Holder Rate (%)")
        ax.tick_params(axis="x", rotation=20)
    save_fig(pdf, fig)

print(f"\n✅ PDF saved to: {OUT_PDF}")

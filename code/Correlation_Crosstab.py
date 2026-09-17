# -*- coding: utf-8 -*-
"""
Created on Sat May  2 19:05:13 2026

@author: o_khi
"""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("banking_dataset_sanitized.csv")

target = "credit_card_holder_flag"

# ── Column groups ──────────────────────────────────────────────────────────
int_cols = [
    "age", "household_size", "annual_income",
    "investable_assets_total", "checking_balance", "savings_balance",
    "retirement_balance", "brokerage_balance", "has_credit_card",
    "has_mortgage", "has_auto_loan", "has_personal_loan",
    "has_home_equity_loan", "has_cd", "has_money_market",
    "has_safe_deposit_box", "total_products_owned",
    "monthly_transaction_count", "monthly_spend_amt",
    "avg_transaction_amount", "atm_withdrawals_monthly",
    "overdraft_count_12mo", "direct_deposit_flag",
    "digital_engagement_score", "mobile_app_logins_monthly",
    "paper_statements_flag", "autopay_enrolled_flag",
    "customer_service_calls_12mo", "nps_score", "credit_score",
    "delinquency_30d_count", "delinquency_90d_count",
    "bankruptcy_flag", "collections_flag", "total_open_tradelines",
    "charge_off_flag",
]

obj_cols = [
    "gender", "state", "education_level", "employment_status",
    "marital_status", "primary_account_type", "account_status",
    "loyalty_tier", "risk_tier",
]


# ══════════════════════════════════════════════════════════════════════════
# 1. CORRELATION METRICS — target vs integer columns
# ══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("CORRELATION METRICS: credit_card_holder_flag vs Integer Columns")
print("=" * 72)

records = []
for col in int_cols:
    clean = df[[target, col]].dropna()
    x, y  = clean[target], clean[col]

    pb_r,  pb_p  = stats.pointbiserialr(x, y)   # best for binary vs continuous
    sp_r,  sp_p  = stats.spearmanr(x, y)         # robust to non-normality / ordinal
    pe_r,  pe_p  = stats.pearsonr(x, y)          # linear baseline

    records.append({
        "feature"           : col,
        "point_biserial_r"  : round(pb_r, 4),
        "pb_p_value"        : round(pb_p, 4),
        "spearman_r"        : round(sp_r, 4),
        "spearman_p"        : round(sp_p, 4),
        "pearson_r"         : round(pe_r, 4),
        "pearson_p"         : round(pe_p, 4),
    })

corr_df = (
    pd.DataFrame(records)
      .set_index("feature")
      .sort_values("point_biserial_r", key=abs, ascending=False)
)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 120)
print(corr_df.to_string())


# ══════════════════════════════════════════════════════════════════════════
# 2. CROSSTABS — target vs object / categorical columns
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("CROSSTABS: credit_card_holder_flag vs Object Columns")
print("=" * 72)

for col in obj_cols:
    # Raw counts with row totals
    ct = pd.crosstab(
        df[col],
        df[target],
        margins=True,
        margins_name="Total",
    )
    # Row-percentage version (more readable when group sizes differ)
    ct_pct = pd.crosstab(
        df[col],
        df[target],
        normalize="index",
    ).mul(100).round(2)
    ct_pct.columns = [f"{c}_pct%" for c in ct_pct.columns]

    print(f"\n{'─' * 60}")
    print(f"  {col}")
    print(f"{'─' * 60}")
    print(ct.to_string())
    print("\n  Row %:")
    print(ct_pct.to_string())

    # Chi-square test of independence
    ct_raw = pd.crosstab(df[col], df[target])
    chi2, p, dof, _ = stats.chi2_contingency(ct_raw)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"\n  Chi²={chi2:.4f}  p={p:.4e}  dof={dof}  {sig}")
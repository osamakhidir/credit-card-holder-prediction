# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:00:40 2026

@author: o_khi
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── 1. Load data ───────────────────────────────────────────────────────────
df = pd.read_csv("banking_dataset_sanitized.csv")

# ── 2. Drop non-feature columns ────────────────────────────────────────────
drop_cols = [
    "customer_id",          # identifier
    "account_open_date",    # raw date string
    "zip_code",             # nominal integer
   # "has_credit_card",
]
target = "credit_card_holder_flag"

# ── 3. Encode object columns (One-Hot Encoding) ────────────────────────────
obj_cols = [
    "gender", "education_level", "employment_status", "marital_status",
    "primary_account_type", "account_status", "loyalty_tier", "risk_tier",
    "state",
]

df_model = df.drop(columns=drop_cols)
df_model = pd.get_dummies(df_model, columns=obj_cols, drop_first=True)

# ── 4. Split features and target ───────────────────────────────────────────
X = df_model.drop(columns=[target])
y = df_model[target]

# Fill any nulls with column median
X = X.fillna(X.median(numeric_only=True))

# ── 5. Train (80%) / Validation (20%) Split ────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.20,       # 20% validation
    random_state=42,      # reproducibility
    stratify=y            # preserve class ratio in both splits
)


# ── 6. Scale features ──────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit ONLY on training data
X_val_sc   = scaler.transform(X_val)         # apply same scale to validation

# Convert scaled arrays back to DataFrames (optional but readable)
X_train_sc = pd.DataFrame(X_train_sc, columns=X.columns, index=X_train.index)
X_val_sc   = pd.DataFrame(X_val_sc,   columns=X.columns, index=X_val.index)

# ── 7. Verify split sizes and class balance ────────────────────────────────
print("=" * 55)
print("DATASET SPLIT SUMMARY")
print("=" * 55)

total = len(X)
print(f"\n  Total records   : {total:,}")
print(f"  Training set    : {len(X_train):,} rows  ({len(X_train)/total*100:.1f}%)")
print(f"  Validation set  : {len(X_val):,}  rows  ({len(X_val)/total*100:.1f}%)")
print(f"  Features        : {X.shape[1]:,} columns")

print("\n── Class Distribution (credit_card_holder_flag) ──────")
for split_name, y_split in [("Training", y_train), ("Validation", y_val)]:
    counts = y_split.value_counts().sort_index()
    pcts   = y_split.value_counts(normalize=True).sort_index().mul(100).round(1)
    print(f"\n  {split_name}:")
    for cls in counts.index:
        label = "Has Card" if cls == 1 else "No Card "
        print(f"    {label} ({cls}): {counts[cls]:,}  ({pcts[cls]}%)")

# ── 8. Save splits to CSV (optional) ──────────────────────────────────────
X_train_sc.assign(**{target: y_train}).to_csv("train_set.csv", index=False)
X_val_sc.assign(**{target: y_val}).to_csv("validation_set.csv", index=False)

print("\n  ✅ Saved: train_set.csv")
print("  ✅ Saved: validation_set.csv")
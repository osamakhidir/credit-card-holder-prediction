import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("banking_dataset_sanitized.csv")

print("=" * 60)
print("BANKING DATASET - DESCRIPTIVE STATISTICS")
print("=" * 60)

# -----------------------------------------------------------
# 1. DATASET OVERVIEW
# -----------------------------------------------------------
print("\n📋 DATASET OVERVIEW")
print(f"  Rows    : {df.shape[0]:,}")
print(f"  Columns : {df.shape[1]}")

print("\nColumn data types:")
print(df.dtypes.to_string())

print("\nMissing values per column:")
missing = df.isnull().sum()
missing = missing[missing > 0]
print(missing.to_string() if not missing.empty else "  None")

# -----------------------------------------------------------
# 2. NUMERIC COLUMNS - SUMMARY STATISTICS
# -----------------------------------------------------------
print("\n" + "=" * 60)
print("📊 NUMERIC COLUMNS — SUMMARY STATISTICS")
print("=" * 60)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Drop ID-like columns
exclude = ["customer_id", "zip_code"]
numeric_cols = [c for c in numeric_cols if c not in exclude]

stats = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
stats["range"]    = stats["max"] - stats["min"]
stats["skewness"] = df[numeric_cols].skew()
stats["kurtosis"] = df[numeric_cols].kurt()
stats["cv_%"]     = (stats["std"] / stats["mean"] * 100).round(2)   # coefficient of variation

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)
print(stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max",
             "range", "skewness", "kurtosis", "cv_%"]])

# -----------------------------------------------------------
# 3. CATEGORICAL COLUMNS — FREQUENCY COUNTS
# -----------------------------------------------------------
print("\n" + "=" * 60)
print("🗂  CATEGORICAL COLUMNS — VALUE COUNTS")
print("=" * 60)

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
exclude_cat = ["customer_id", "account_open_date"]
cat_cols = [c for c in cat_cols if c not in exclude_cat]

for col in cat_cols:
    vc = df[col].value_counts(dropna=False)
    pct = (vc / len(df) * 100).round(1)
    summary = pd.DataFrame({"count": vc, "pct_%": pct})
    print(f"\n  {col} ({df[col].nunique()} unique values):")
    print(summary.to_string())

# -----------------------------------------------------------
# 4. BINARY / FLAG COLUMNS — PROPORTION SUMMARY
# -----------------------------------------------------------
print("\n" + "=" * 60)
print("🚩 BINARY FLAG COLUMNS — PROPORTION TRUE")
print("=" * 60)

flag_cols = [c for c in numeric_cols if df[c].dropna().isin([0, 1]).all()]
flag_summary = pd.DataFrame({
    "count_1": df[flag_cols].sum(),
    "pct_%":   (df[flag_cols].mean() * 100).round(1)
}).sort_values("pct_%", ascending=False)
print(flag_summary.to_string())

# -----------------------------------------------------------
# 5. KEY FINANCIAL METRICS — DEEPER LOOK
# -----------------------------------------------------------
print("\n" + "=" * 60)
print("💰 KEY FINANCIAL METRICS")
print("=" * 60)

financial_cols = [
    "annual_income", "investable_assets_total",
    "checking_balance", "savings_balance",
    "retirement_balance", "brokerage_balance",
    "monthly_spend_amt", "avg_transaction_amount",
    "credit_score", "debt_to_income_ratio", "credit_utilization_ratio"
]
financial_cols = [c for c in financial_cols if c in df.columns]

fin_stats = df[financial_cols].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
print(fin_stats.to_string())

# -----------------------------------------------------------
# 6. CORRELATIONS — TOP PAIRS
# -----------------------------------------------------------
print("\n" + "=" * 60)
print("🔗 TOP 15 ABSOLUTE CORRELATIONS (numeric columns)")
print("=" * 60)

corr_matrix = df[numeric_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
top_corr = (upper.stack()
                 .reset_index()
                 .rename(columns={"level_0": "col_a", "level_1": "col_b", 0: "abs_corr"})
                 .sort_values("abs_corr", ascending=False)
                 .head(15))
print(top_corr.to_string(index=False))

print("\n✅ Done.")

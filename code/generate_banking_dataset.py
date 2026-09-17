"""
generate_banking_dataset.py
============================
Generates a sanitized synthetic banking dataset with:
  - 4% credit card holders / 96% non-credit card holders
  - Binary credit card holder flag
  - Transaction data
  - Demographic data
  - Customer data
  - Investable asset data
  - Product ownership & behavioral data
  - Credit risk data
  - Credit score
  - Customer account information

Output: banking_dataset_sanitized.xlsx  (3 sheets)
        banking_dataset_sanitized.csv

Requirements: pandas, numpy, openpyxl
    pip install pandas numpy openpyxl
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Population split ───────────────────────────────────────────────────────────
N = 5_0000
CC_RATE = 0.04                          # 4 % credit-card holders
cc_count = round(N * CC_RATE)           # 200
non_cc_count = N - cc_count             # 4 800

cc_flag = np.array([1] * cc_count + [0] * non_cc_count)
np.random.shuffle(cc_flag)

is_cc = cc_flag == 1                    # boolean mask – reused throughout


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – CUSTOMER / DEMOGRAPHIC DATA
# ══════════════════════════════════════════════════════════════════════════════
customer_ids = [f"CUST{i:06d}" for i in range(1, N + 1)]

genders = np.random.choice(
    ["M", "F", "Non-Binary"], N, p=[0.49, 0.49, 0.02]
)

ages = np.where(
    is_cc,
    np.random.normal(42, 10, N).clip(21, 80),
    np.random.normal(38, 12, N).clip(18, 85),
).astype(int)

states = np.random.choice(
    ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI",
     "WA", "AZ", "CO", "MN", "WI", "MO", "TN", "IN", "MD", "VA"],
    N,
)
zip_codes = [f"{np.random.randint(10000, 99999):05d}" for _ in range(N)]

education = np.random.choice(
    ["High School", "Some College", "Bachelor", "Master", "PhD", "Trade School"],
    N, p=[0.25, 0.20, 0.30, 0.15, 0.05, 0.05],
)
employment = np.random.choice(
    ["Employed", "Self-Employed", "Retired", "Unemployed", "Part-Time"],
    N, p=[0.55, 0.12, 0.18, 0.07, 0.08],
)
marital = np.random.choice(
    ["Single", "Married", "Divorced", "Widowed"], N, p=[0.35, 0.45, 0.15, 0.05]
)
household_size = np.random.choice([1, 2, 3, 4, 5, 6], N, p=[0.28, 0.32, 0.17, 0.14, 0.06, 0.03])

annual_income = np.where(
    is_cc,
    np.random.lognormal(np.log(85_000), 0.5, N),
    np.random.lognormal(np.log(52_000), 0.6, N),
).clip(18_000, 500_000).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – CUSTOMER ACCOUNT INFORMATION
# ══════════════════════════════════════════════════════════════════════════════
open_dates = [
    datetime(2000, 1, 1) + timedelta(days=int(np.random.randint(0, 8_000)))
    for _ in range(N)
]
tenure_yrs = [
    round((datetime(2025, 4, 30) - d).days / 365.25, 1) for d in open_dates
]
primary_acct = np.random.choice(
    ["Checking", "Savings", "Money Market"], N, p=[0.55, 0.30, 0.15]
)
acct_status = np.random.choice(
    ["Active", "Dormant", "Restricted"], N, p=[0.92, 0.06, 0.02]
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – INVESTABLE ASSET DATA
# ══════════════════════════════════════════════════════════════════════════════
investable_assets = np.where(
    is_cc,
    np.random.lognormal(np.log(75_000), 0.8, N),
    np.random.lognormal(np.log(15_000), 1.0, N),
).clip(0, 2_000_000).astype(int)

checking_bal = np.random.lognormal(np.log(3_500), 0.8, N).clip(0, 150_000).astype(int)

savings_bal = np.where(
    is_cc,
    np.random.lognormal(np.log(18_000), 0.7, N),
    np.random.lognormal(np.log(5_000),  0.9, N),
).clip(0, 500_000).astype(int)

retirement_bal = np.where(
    is_cc,
    np.random.lognormal(np.log(120_000), 0.9, N),
    np.random.lognormal(np.log(35_000),  1.1, N),
).clip(0, 3_000_000).astype(int)

has_brokerage = np.random.choice([0, 1], N, p=[0.60, 0.40])
brokerage_bal = (
    np.where(
        is_cc,
        np.random.lognormal(np.log(40_000), 1.0, N),
        np.random.lognormal(np.log(5_000),  1.2, N),
    ).clip(0, 1_500_000).astype(int)
    * has_brokerage
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – PRODUCT OWNERSHIP
# ══════════════════════════════════════════════════════════════════════════════
has_mortgage     = np.random.choice([0, 1], N, p=[0.55, 0.45])
has_auto_loan    = np.random.choice([0, 1], N, p=[0.58, 0.42])
has_personal_ln  = np.random.choice([0, 1], N, p=[0.78, 0.22])
has_heloc        = np.where(is_cc,
                            np.random.choice([0, 1], N, p=[0.65, 0.35]),
                            np.random.choice([0, 1], N, p=[0.82, 0.18]))
has_cd           = np.random.choice([0, 1], N, p=[0.75, 0.25])
has_money_mkt    = np.random.choice([0, 1], N, p=[0.70, 0.30])
has_safe_deposit = np.random.choice([0, 1], N, p=[0.85, 0.15])

total_products = (
    cc_flag + has_mortgage + has_auto_loan + has_personal_ln
    + has_heloc + has_cd + has_money_mkt + has_safe_deposit + 1   # +1 for primary acct
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – TRANSACTION DATA
# ══════════════════════════════════════════════════════════════════════════════
monthly_txn_count = np.where(
    is_cc, np.random.poisson(45, N), np.random.poisson(22, N)
)
monthly_spend = np.where(
    is_cc,
    np.random.lognormal(np.log(3_200), 0.5, N),
    np.random.lognormal(np.log(1_400), 0.6, N),
).clip(100, 50_000).astype(int)

avg_txn_amt = (monthly_spend / monthly_txn_count.clip(1)).astype(int)
atm_withdrawals = np.random.poisson(3, N)

mobile_pct = np.random.beta(2, 3, N).round(2)
online_pct = np.random.beta(3, 3, N).round(2)
branch_pct = (1 - mobile_pct - online_pct).clip(0, 1).round(2)

overdrafts_12mo = np.where(
    is_cc,
    np.random.choice([0,1,2,3,4,5], N, p=[0.75,0.12,0.07,0.04,0.01,0.01]),
    np.random.choice([0,1,2,3,4,5], N, p=[0.55,0.18,0.12,0.08,0.05,0.02]),
)
direct_deposit = np.random.choice([0, 1], N, p=[0.30, 0.70])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – BEHAVIORAL DATA
# ══════════════════════════════════════════════════════════════════════════════
digital_engagement  = np.random.randint(1, 101, N)
app_logins_monthly  = np.random.poisson(12, N)
paper_statements    = np.random.choice([0, 1], N, p=[0.65, 0.35])
autopay             = np.where(is_cc,
                               np.random.choice([0,1], N, p=[0.25,0.75]),
                               np.random.choice([0,1], N, p=[0.55,0.45]))
cs_calls_12mo       = np.random.choice(
    [0,1,2,3,4,5,6], N, p=[0.30,0.25,0.20,0.12,0.07,0.04,0.02]
)
nps_score           = np.random.choice(
    range(11), N,
    p=[0.02,0.02,0.03,0.04,0.06,0.08,0.12,0.18,0.22,0.14,0.09],
)
loyalty_tier        = np.random.choice(
    ["Bronze","Silver","Gold","Platinum"], N, p=[0.40,0.30,0.20,0.10]
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – CREDIT SCORE & CREDIT RISK DATA
# ══════════════════════════════════════════════════════════════════════════════
credit_score = np.where(
    is_cc,
    np.random.normal(730, 55, N).clip(580, 850),
    np.random.normal(680, 70, N).clip(450, 850),
).astype(int)

dti = np.where(
    is_cc, np.random.beta(3,7,N), np.random.beta(4,6,N)
).round(2)

delinq_30d = np.where(
    is_cc,
    np.random.choice([0,1,2,3], N, p=[0.88,0.07,0.03,0.02]),
    np.random.choice([0,1,2,3], N, p=[0.78,0.12,0.06,0.04]),
)
delinq_90d = np.where(
    is_cc,
    np.random.choice([0,1,2], N, p=[0.94,0.04,0.02]),
    np.random.choice([0,1,2], N, p=[0.87,0.09,0.04]),
)
bankruptcy     = np.random.choice([0,1], N, p=[0.97,0.03])
collections    = np.where(is_cc,
                          np.random.choice([0,1], N, p=[0.95,0.05]),
                          np.random.choice([0,1], N, p=[0.88,0.12]))
tradelines     = np.random.randint(1, 20, N)
credit_util    = np.where(
    is_cc, np.random.beta(2,5,N), np.random.beta(3,4,N)
).round(2)
charge_off     = np.random.choice([0,1], N, p=[0.97,0.03])

risk_tier = np.select(
    [credit_score >= 750, credit_score >= 700,
     credit_score >= 650, credit_score >= 600],
    ["Prime Plus", "Prime", "Near Prime", "Subprime"],
    default="Deep Subprime",
)


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════
df = pd.DataFrame({
    # ── Customer / Demographic ────────────────────────────────────────────────
    "customer_id":              customer_ids,
    "credit_card_holder_flag":  cc_flag,
    "gender":                   genders,
    "age":                      ages,
    "state":                    states,
    "zip_code":                 zip_codes,
    "education_level":          education,
    "employment_status":        employment,
    "marital_status":           marital,
    "household_size":           household_size,
    "annual_income":            annual_income,
    # ── Account Information ───────────────────────────────────────────────────
    "account_open_date":        [d.strftime("%Y-%m-%d") for d in open_dates],
    "account_tenure_years":     tenure_yrs,
    "primary_account_type":     primary_acct,
    "account_status":           acct_status,
    # ── Investable Assets ─────────────────────────────────────────────────────
    "investable_assets_total":  investable_assets,
    "checking_balance":         checking_bal,
    "savings_balance":          savings_bal,
    "retirement_balance":       retirement_bal,
    "brokerage_balance":        brokerage_bal,
    # ── Product Ownership ─────────────────────────────────────────────────────
    "has_credit_card":          cc_flag,
    "has_mortgage":             has_mortgage,
    "has_auto_loan":            has_auto_loan,
    "has_personal_loan":        has_personal_ln,
    "has_home_equity_loan":     has_heloc,
    "has_cd":                   has_cd,
    "has_money_market":         has_money_mkt,
    "has_safe_deposit_box":     has_safe_deposit,
    "total_products_owned":     total_products,
    # ── Transaction Data ──────────────────────────────────────────────────────
    "monthly_transaction_count": monthly_txn_count,
    "monthly_spend_amt":         monthly_spend,
    "avg_transaction_amount":    avg_txn_amt,
    "atm_withdrawals_monthly":   atm_withdrawals,
    "mobile_txn_pct":            mobile_pct,
    "online_txn_pct":            online_pct,
    "branch_txn_pct":            branch_pct,
    "overdraft_count_12mo":      overdrafts_12mo,
    "direct_deposit_flag":       direct_deposit,
    # ── Behavioral ────────────────────────────────────────────────────────────
    "digital_engagement_score":  digital_engagement,
    "mobile_app_logins_monthly": app_logins_monthly,
    "paper_statements_flag":     paper_statements,
    "autopay_enrolled_flag":     autopay,
    "customer_service_calls_12mo": cs_calls_12mo,
    "nps_score":                 nps_score,
    "loyalty_tier":              loyalty_tier,
    # ── Credit Score & Risk ───────────────────────────────────────────────────
    "credit_score":              credit_score,
    "debt_to_income_ratio":      dti,
    "delinquency_30d_count":     delinq_30d,
    "delinquency_90d_count":     delinq_90d,
    "bankruptcy_flag":           bankruptcy,
    "collections_flag":          collections,
    "total_open_tradelines":     tradelines,
    "credit_utilization_ratio":  credit_util,
    "charge_off_flag":           charge_off,
    "risk_tier":                 risk_tier,
})


# ══════════════════════════════════════════════════════════════════════════════
# BUILD EXCEL WORKBOOK  (3 sheets)
# ══════════════════════════════════════════════════════════════════════════════
wb = Workbook()

# ── Shared style helpers ───────────────────────────────────────────────────────
HDR_FONT   = Font(name="Arial", bold=True, color="FFFFFF", size=10)
HDR_FILL   = PatternFill("solid", start_color="1F4E79")
ALT_FILL   = PatternFill("solid", start_color="D6E4F0")
LEFT_ALIGN = Alignment(horizontal="left",   vertical="center")
CTR_ALIGN  = Alignment(horizontal="center", vertical="center", wrap_text=True)
_side      = Side(style="thin", color="AAAAAA")
BORDER     = Border(left=_side, right=_side, top=_side, bottom=_side)


def write_header(ws, row, col, label, width=20):
    cell = ws.cell(row=row, column=col, value=label)
    cell.font      = HDR_FONT
    cell.fill      = HDR_FILL
    cell.alignment = CTR_ALIGN
    cell.border    = BORDER
    ws.column_dimensions[get_column_letter(col)].width = width


def shade_row(ws, row_idx, n_cols):
    if row_idx % 2 == 0:
        for c in range(1, n_cols + 1):
            ws.cell(row=row_idx, column=c).fill = ALT_FILL
    for c in range(1, n_cols + 1):
        cell = ws.cell(row=row_idx, column=c)
        cell.border    = BORDER
        cell.alignment = LEFT_ALIGN


# ── Sheet 1 : Main Dataset ─────────────────────────────────────────────────────
ws1 = wb.active
ws1.title = "Banking Dataset"
ws1.freeze_panes = "A2"
ws1.row_dimensions[1].height = 40

for ci, col in enumerate(df.columns, 1):
    write_header(ws1, 1, ci, col)

for ri, row in enumerate(df.itertuples(index=False), 2):
    for ci, val in enumerate(row, 1):
        ws1.cell(row=ri, column=ci, value=val)
    shade_row(ws1, ri, len(df.columns))


# ── Sheet 2 : Data Dictionary ─────────────────────────────────────────────────
ws2 = wb.create_sheet("Data Dictionary")

dict_headers = ["Field Name", "Data Type", "Description", "Category"]
col_widths    = [30, 22, 60, 20]
for ci, (h, w) in enumerate(zip(dict_headers, col_widths), 1):
    write_header(ws2, 1, ci, h, width=w)

DICT_ROWS = [
    ("customer_id",              "String",         "Unique customer identifier (CUST + 6-digit number)",                    "Customer"),
    ("credit_card_holder_flag",  "Binary (0/1)",   "1 = holds a bank credit card, 0 = does not (4 % prevalence)",          "Customer"),
    ("gender",                   "Categorical",    "Customer gender: M, F, Non-Binary",                                    "Demographic"),
    ("age",                      "Integer",        "Customer age in years",                                                 "Demographic"),
    ("state",                    "Categorical",    "US state of primary residence (2-letter code)",                         "Demographic"),
    ("zip_code",                 "String",         "5-digit ZIP code of residence",                                         "Demographic"),
    ("education_level",          "Categorical",    "Highest level of education attained",                                   "Demographic"),
    ("employment_status",        "Categorical",    "Current employment classification",                                     "Demographic"),
    ("marital_status",           "Categorical",    "Marital status",                                                        "Demographic"),
    ("household_size",           "Integer",        "Number of people in household",                                         "Demographic"),
    ("annual_income",            "Integer ($)",    "Self-reported or estimated annual gross income",                        "Demographic"),
    ("account_open_date",        "Date YYYY-MM-DD","Date primary account was opened",                                       "Account"),
    ("account_tenure_years",     "Float",          "Years since account opening (as of Apr 2025)",                          "Account"),
    ("primary_account_type",     "Categorical",    "Checking, Savings, or Money Market",                                    "Account"),
    ("account_status",           "Categorical",    "Active, Dormant, or Restricted",                                        "Account"),
    ("investable_assets_total",  "Integer ($)",    "Total investable assets across all bank accounts",                      "Investable Assets"),
    ("checking_balance",         "Integer ($)",    "Current checking account balance",                                      "Investable Assets"),
    ("savings_balance",          "Integer ($)",    "Current savings account balance",                                       "Investable Assets"),
    ("retirement_balance",       "Integer ($)",    "IRA / 401k / retirement account balance",                               "Investable Assets"),
    ("brokerage_balance",        "Integer ($)",    "Taxable brokerage balance (0 if no account)",                           "Investable Assets"),
    ("has_credit_card",          "Binary (0/1)",   "Mirrors credit_card_holder_flag — product ownership view",              "Product Ownership"),
    ("has_mortgage",             "Binary (0/1)",   "Holds a mortgage loan with the bank",                                   "Product Ownership"),
    ("has_auto_loan",            "Binary (0/1)",   "Holds an auto loan",                                                    "Product Ownership"),
    ("has_personal_loan",        "Binary (0/1)",   "Holds a personal/unsecured loan",                                       "Product Ownership"),
    ("has_home_equity_loan",     "Binary (0/1)",   "Holds a HELOC or home equity loan",                                     "Product Ownership"),
    ("has_cd",                   "Binary (0/1)",   "Holds a certificate of deposit",                                        "Product Ownership"),
    ("has_money_market",         "Binary (0/1)",   "Holds a money market account",                                          "Product Ownership"),
    ("has_safe_deposit_box",     "Binary (0/1)",   "Rents a safe deposit box",                                              "Product Ownership"),
    ("total_products_owned",     "Integer",        "Total number of bank products held",                                    "Product Ownership"),
    ("monthly_transaction_count","Integer",        "Average number of transactions per month",                              "Transaction"),
    ("monthly_spend_amt",        "Integer ($)",    "Total monthly debit + credit spend",                                    "Transaction"),
    ("avg_transaction_amount",   "Integer ($)",    "Average dollar value per transaction",                                  "Transaction"),
    ("atm_withdrawals_monthly",  "Integer",        "Average ATM cash withdrawals per month",                                "Transaction"),
    ("mobile_txn_pct",           "Float 0-1",      "Share of transactions via mobile channel",                              "Transaction"),
    ("online_txn_pct",           "Float 0-1",      "Share of transactions via online/web channel",                          "Transaction"),
    ("branch_txn_pct",           "Float 0-1",      "Share of transactions at branch",                                       "Transaction"),
    ("overdraft_count_12mo",     "Integer",        "Number of overdraft events in past 12 months",                          "Transaction"),
    ("direct_deposit_flag",      "Binary (0/1)",   "Payroll or government direct deposit active",                           "Transaction"),
    ("digital_engagement_score", "Integer 1-100",  "Composite digital engagement index",                                    "Behavioral"),
    ("mobile_app_logins_monthly","Integer",        "Average monthly mobile app logins",                                     "Behavioral"),
    ("paper_statements_flag",    "Binary (0/1)",   "Customer receives paper statements",                                    "Behavioral"),
    ("autopay_enrolled_flag",    "Binary (0/1)",   "Enrolled in automatic payment for any loan",                            "Behavioral"),
    ("customer_service_calls_12mo","Integer",      "Inbound service calls in past 12 months",                               "Behavioral"),
    ("nps_score",                "Integer 0-10",   "Net Promoter Score from most recent survey",                            "Behavioral"),
    ("loyalty_tier",             "Categorical",    "Bronze / Silver / Gold / Platinum",                                     "Behavioral"),
    ("credit_score",             "Integer",        "FICO-equivalent credit score (450–850)",                                "Credit Risk"),
    ("debt_to_income_ratio",     "Float 0-1",      "Total monthly debt payments / gross monthly income",                    "Credit Risk"),
    ("delinquency_30d_count",    "Integer",        "Number of 30+ day delinquencies on record",                             "Credit Risk"),
    ("delinquency_90d_count",    "Integer",        "Number of 90+ day delinquencies on record",                             "Credit Risk"),
    ("bankruptcy_flag",          "Binary (0/1)",   "Prior bankruptcy on record",                                            "Credit Risk"),
    ("collections_flag",         "Binary (0/1)",   "Account(s) sent to collections",                                       "Credit Risk"),
    ("total_open_tradelines",    "Integer",        "Number of open credit accounts across all lenders",                     "Credit Risk"),
    ("credit_utilization_ratio", "Float 0-1",      "Revolving credit used / total revolving limit",                         "Credit Risk"),
    ("charge_off_flag",          "Binary (0/1)",   "Prior charge-off with any lender",                                      "Credit Risk"),
    ("risk_tier",                "Categorical",    "Prime Plus / Prime / Near Prime / Subprime / Deep Subprime",            "Credit Risk"),
]

for ri, row in enumerate(DICT_ROWS, 2):
    for ci, val in enumerate(row, 1):
        cell = ws2.cell(row=ri, column=ci, value=val)
        cell.border    = BORDER
        cell.alignment = LEFT_ALIGN
        if ri % 2 == 0:
            cell.fill = ALT_FILL


# ── Sheet 3 : Summary Statistics ──────────────────────────────────────────────
ws3 = wb.create_sheet("Summary Statistics")

for ci, (h, w) in enumerate(zip(
    ["Metric", "CC Holders (4%)", "Non-CC Holders (96%)"],
    [34, 24, 24]
), 1):
    write_header(ws3, 1, ci, h, width=w)

cc_df = df[df.credit_card_holder_flag == 1]
nc_df = df[df.credit_card_holder_flag == 0]

SUMMARY = [
    ("Record Count",                    len(cc_df),                                len(nc_df)),
    ("Avg Age",                         round(cc_df.age.mean(), 1),                round(nc_df.age.mean(), 1)),
    ("Avg Annual Income",               f"${cc_df.annual_income.mean():,.0f}",     f"${nc_df.annual_income.mean():,.0f}"),
    ("Avg Credit Score",                round(cc_df.credit_score.mean(), 1),       round(nc_df.credit_score.mean(), 1)),
    ("Avg Investable Assets",           f"${cc_df.investable_assets_total.mean():,.0f}", f"${nc_df.investable_assets_total.mean():,.0f}"),
    ("Avg Monthly Spend",               f"${cc_df.monthly_spend_amt.mean():,.0f}", f"${nc_df.monthly_spend_amt.mean():,.0f}"),
    ("Avg Monthly Transactions",        round(cc_df.monthly_transaction_count.mean(), 1), round(nc_df.monthly_transaction_count.mean(), 1)),
    ("Avg Total Products Owned",        round(cc_df.total_products_owned.mean(), 1), round(nc_df.total_products_owned.mean(), 1)),
    ("Avg Account Tenure (yrs)",        round(cc_df.account_tenure_years.mean(), 1), round(nc_df.account_tenure_years.mean(), 1)),
    ("Avg NPS Score",                   round(cc_df.nps_score.mean(), 1),          round(nc_df.nps_score.mean(), 1)),
    ("Avg Digital Engagement Score",    round(cc_df.digital_engagement_score.mean(), 1), round(nc_df.digital_engagement_score.mean(), 1)),
    ("% With Mortgage",                 f"{cc_df.has_mortgage.mean()*100:.1f}%",   f"{nc_df.has_mortgage.mean()*100:.1f}%"),
    ("% With Auto Loan",                f"{cc_df.has_auto_loan.mean()*100:.1f}%",  f"{nc_df.has_auto_loan.mean()*100:.1f}%"),
    ("Avg Debt-to-Income Ratio",        round(cc_df.debt_to_income_ratio.mean(), 2), round(nc_df.debt_to_income_ratio.mean(), 2)),
    ("Avg Credit Utilization",          round(cc_df.credit_utilization_ratio.mean(), 2), round(nc_df.credit_utilization_ratio.mean(), 2)),
    ("% With Bankruptcy",               f"{cc_df.bankruptcy_flag.mean()*100:.1f}%", f"{nc_df.bankruptcy_flag.mean()*100:.1f}%"),
    ("% With Charge-Off",               f"{cc_df.charge_off_flag.mean()*100:.1f}%", f"{nc_df.charge_off_flag.mean()*100:.1f}%"),
    ("% Direct Deposit Active",         f"{cc_df.direct_deposit_flag.mean()*100:.1f}%", f"{nc_df.direct_deposit_flag.mean()*100:.1f}%"),
    ("Avg Overdrafts (12 mo)",          round(cc_df.overdraft_count_12mo.mean(), 2), round(nc_df.overdraft_count_12mo.mean(), 2)),
    ("Avg Savings Balance",             f"${cc_df.savings_balance.mean():,.0f}",   f"${nc_df.savings_balance.mean():,.0f}"),
    ("Avg Retirement Balance",          f"${cc_df.retirement_balance.mean():,.0f}", f"${nc_df.retirement_balance.mean():,.0f}"),
]

for ri, row in enumerate(SUMMARY, 2):
    for ci, val in enumerate(row, 1):
        cell = ws3.cell(row=ri, column=ci, value=val)
        cell.border    = BORDER
        cell.alignment = LEFT_ALIGN
        if ri % 2 == 0:
            cell.fill = ALT_FILL


# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
XLSX_PATH = "banking_dataset_sanitized.xlsx"
CSV_PATH  = "banking_dataset_sanitized.csv"

wb.save(XLSX_PATH)
df.to_csv(CSV_PATH, index=False)

# ── Console summary ────────────────────────────────────────────────────────────
print("=" * 60)
print("  Banking Dataset Generation Complete")
print("=" * 60)
print(f"  Total records  : {N:,}")
print(f"  CC holders     : {cc_flag.sum():,}  ({cc_flag.mean()*100:.1f}%)")
print(f"  Non-CC holders : {(1-cc_flag).sum():,}  ({(1-cc_flag).mean()*100:.1f}%)")
print(f"  Columns        : {len(df.columns)}")
print(f"  Excel output   : {XLSX_PATH}")
print(f"  CSV output     : {CSV_PATH}")
print("=" * 60)

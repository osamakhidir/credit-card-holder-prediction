import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    brier_score_loss, log_loss, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve,
    precision_score, recall_score, f1_score, matthews_corrcoef
)
from sklearn.calibration import calibration_curve

# ── 1. Load & split ───────────────────────────────────────────
df = pd.read_csv("train_set.csv")

# Drop has_credit_card — scaled copy of target (data leakage)
X = df.drop(columns=['has_credit_card', 

'credit_card_holder_flag',
'savings_balance',
'autopay_enrolled_flag',
'investable_assets_total',
'monthly_spend_amt',
'overdraft_count_12mo',
'risk_tier_Prime Plus',
'has_home_equity_loan',
'risk_tier_Subprime',
'account_tenure_years',
'checking_balance',
'digital_engagement_score',
'mobile_txn_pct',
'online_txn_pct',
'risk_tier_Prime',
'total_open_tradelines',
'mobile_app_logins_monthly',
'branch_txn_pct',
'nps_score',
'delinquency_30d_count',
'customer_service_calls_12mo',
'atm_withdrawals_monthly',
'household_size',
'delinquency_90d_count',
'collections_flag',
'risk_tier_Near Prime',
'has_mortgage',
'has_auto_loan',
'has_money_market',
'has_cd',
'gender_M',
'has_personal_loan',
'loyalty_tier_Silver',
'marital_status_Married',
'marital_status_Single',
'monthly_transaction_count',
'has_safe_deposit_box',
'paper_statements_flag',
'direct_deposit_flag',
'primary_account_type_Savings',
'education_level_High School',
'employment_status_Retired',
'primary_account_type_Money Market',
'loyalty_tier_Gold',
'education_level_Some College',
'education_level_Master',
'employment_status_Part-Time',
'state_IN',
'state_TX',
'state_MI',
'loyalty_tier_Platinum',
'account_status_Dormant',
'employment_status_Self-Employed',
'education_level_PhD',
'state_MD',
'state_VA',
'state_IL',
'state_WI',
'state_CO',
'bankruptcy_flag',
'state_CA',
'state_NY',
'education_level_Trade School',
'state_OH',
'state_MN',
'state_PA',
'state_NC',
'state_FL',
'state_MO',
'gender_Non-Binary',
'marital_status_Widowed',
'employment_status_Unemployed',
'state_WA',
'state_GA',
'state_TN',
'charge_off_flag',
'account_status_Restricted'

], errors='ignore')
y = df['credit_card_holder_flag']
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 2. Train Random Forest ────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',  # handles 92%/8% imbalance
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred     = rf.predict(X_test)
y_prob     = rf.predict_proba(X_test)[:, 1]
y_prob_all = rf.predict_proba(X)[:, 1]   # full dataset for output file

# ── 3. Summary metrics ────────────────────────────────────────
acc  = accuracy_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)
ap   = average_precision_score(y_test, y_prob)
bs   = brier_score_loss(y_test, y_prob)
ll   = log_loss(y_test, y_prob)
mcc  = matthews_corrcoef(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
sens = tp/(tp+fn);  spec = tn/(tn+fp)
ppv  = tp/(tp+fp);  npv  = tn/(tn+fn)
gini = 2*auc - 1
fpr, tpr, _ = roc_curve(y_test, y_prob)
ks_stat = float(np.max(tpr - fpr))
baseline_rate = float(y.mean())

print("=" * 55)
print("RANDOM FOREST — SUMMARY METRICS")
print("=" * 55)
for label, val in [
    ("Accuracy",    acc),  ("ROC-AUC",       auc),
    ("Avg Precision", ap), ("Brier Score",    bs),
    ("Log Loss",     ll),  ("MCC",            mcc),
    ("Gini",        gini), ("KS Statistic",   ks_stat),
    ("Sensitivity", sens), ("Specificity",    spec),
    ("PPV",         ppv),  ("NPV",            npv),
]:
    print(f"  {label:<22}: {val:.4f}")
print(f"\n  Confusion matrix  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

# ── 4. Cross-validation (5-fold) ─────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(rf, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
cv_f1  = cross_val_score(rf, X_train, y_train, cv=cv, scoring='f1',      n_jobs=-1)
print(f"\n  5-fold CV AUC : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print(f"  5-fold CV F1  : {cv_f1.mean():.4f}  ± {cv_f1.std():.4f}")

# ── 5. Feature importance (Gini) ─────────────────────────────
fi_df = pd.DataFrame({
    'feature':    feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)
fi_df['rank'] = fi_df.index + 1

print("\nTop 15 features:")
print(fi_df.head(15).to_string(index=False))

# ── 6. Decile table ───────────────────────────────────────────
df_out = df.copy()
df_out['pred_prob'] = y_prob_all
df_out['decile']    = pd.qcut(y_prob_all, q=10, labels=False, duplicates='drop') + 1

decile_lift = df_out.groupby('decile').agg(
    n=('credit_card_holder_flag', 'count'),
    events=('credit_card_holder_flag', 'sum'),
    mean_prob=('pred_prob', 'mean')
).reset_index()
decile_lift['event_rate'] = (decile_lift['events'] / decile_lift['n'] * 100).round(3)
decile_lift['lift']       = (decile_lift['event_rate'] / (baseline_rate * 100)).round(3)
dec_sorted = decile_lift.sort_values('decile', ascending=False).reset_index(drop=True)

# ── 7. Nine-panel diagnostic plots ───────────────────────────
fig = plt.figure(figsize=(21, 17))
fig.suptitle("Random Forest — Full Diagnostic Report\nTarget: credit_card_holder_flag",
             fontsize=15, fontweight='bold', y=0.99)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.46, wspace=0.36)

# Panel 1 — ROC curve
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(fpr, tpr, color='#1B6CA8', lw=2.5, label=f'AUC = {auc:.4f}')
ax1.plot([0,1],[0,1],'k--',lw=1,alpha=0.4,label='Random')
ks_idx = int(np.argmax(tpr - fpr))
ax1.axvline(fpr[ks_idx], color='#E05C2D', lw=1.2, linestyle=':', label=f'KS = {ks_stat:.4f}')
ax1.fill_between(fpr, tpr, alpha=0.08, color='#1B6CA8')
ax1.set(xlabel='False positive rate', ylabel='True positive rate', title='ROC curve')
ax1.legend(fontsize=8.5); ax1.grid(True, alpha=0.3)

# Panel 2 — Precision-recall curve
prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(rec_c, prec_c, color='#177A56', lw=2.5, label=f'AP = {ap:.4f}')
ax2.axhline(baseline_rate, color='k', linestyle='--', lw=1, alpha=0.4,
            label=f'Baseline = {baseline_rate:.2f}')
ax2.fill_between(rec_c, prec_c, alpha=0.08, color='#177A56')
ax2.set(xlabel='Recall', ylabel='Precision', title='Precision-recall curve', ylim=[0, 1.05])
ax2.legend(fontsize=8.5); ax2.grid(True, alpha=0.3)

# Panel 3 — Confusion matrix
ax3 = fig.add_subplot(gs[0, 2])
ConfusionMatrixDisplay(cm, display_labels=['Non-holder','Holder']).plot(
    ax=ax3, colorbar=False, cmap='Blues')
ax3.set_title('Confusion matrix (threshold = 0.5)')

# Panel 4 — Calibration plot
prob_true, prob_pred_cal = calibration_curve(y_test, y_prob, n_bins=10)
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(prob_pred_cal, prob_true, 's-', color='#B07A14', lw=2, ms=6, label='Model')
ax4.plot([0,1],[0,1],'k--',lw=1,alpha=0.4,label='Perfect calibration')
ax4.set(xlabel='Mean predicted probability', ylabel='Fraction of positives',
        title='Calibration plot')
ax4.legend(fontsize=8.5); ax4.grid(True, alpha=0.3)

# Panel 5 — Metrics vs threshold
thresholds_t = np.linspace(0.01, 0.99, 150)
precs_t, recs_t, f1s_t = [], [], []
for t in thresholds_t:
    yp = (y_prob >= t).astype(int)
    precs_t.append(precision_score(y_test, yp, zero_division=0))
    recs_t.append(recall_score(y_test, yp, zero_division=0))
    f1s_t.append(f1_score(y_test, yp, zero_division=0))
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(thresholds_t, precs_t, color='#1B6CA8', lw=2, label='Precision')
ax5.plot(thresholds_t, recs_t,  color='#177A56', lw=2, label='Recall')
ax5.plot(thresholds_t, f1s_t,   color='#B07A14', lw=2, label='F1')
ax5.axvline(0.5, color='gray', linestyle='--', lw=1.2, alpha=0.7, label='Default (0.5)')
ax5.set(xlabel='Threshold', ylabel='Score', title='Metrics vs threshold', ylim=[0, 1.05])
ax5.legend(fontsize=8.5); ax5.grid(True, alpha=0.3)

# Panel 6 — Top 20 feature importances
top20 = fi_df.head(20).sort_values('importance')
ax6 = fig.add_subplot(gs[1, 2])
ax6.barh(top20['feature'], top20['importance'], color='#1B6CA8', alpha=0.82)
ax6.set(xlabel='Gini importance', title='Top 20 feature importances')
ax6.tick_params(axis='y', labelsize=7.5); ax6.grid(True, alpha=0.3, axis='x')

# Panel 7 — Score distribution by class
ax7 = fig.add_subplot(gs[2, 0])
ax7.hist(y_prob[y_test==0], bins=60, alpha=0.6, color='#1B6CA8',
         label='Non-holder (0)', density=True)
ax7.hist(y_prob[y_test==1], bins=60, alpha=0.6, color='#E05C2D',
         label='Holder (1)', density=True)
ax7.set(xlabel='Predicted probability', ylabel='Density',
        title='Score distribution by class')
ax7.legend(fontsize=8.5); ax7.grid(True, alpha=0.3)

# Panel 8 — Decile lift chart
n_dec = len(dec_sorted)
x_pos = range(1, n_dec + 1)
bar_colors = ['#1B6CA8' if r > 1 else '#B07A14' for r in dec_sorted['lift']]
ax8 = fig.add_subplot(gs[2, 1])
bars8 = ax8.bar(x_pos, dec_sorted['event_rate'], color=bar_colors, alpha=0.85)
ax8.axhline(baseline_rate*100, color='#E05C2D', linestyle='--', lw=2,
            label=f'Avg rate {baseline_rate*100:.1f}%')
for bar, lift in zip(bars8, dec_sorted['lift']):
    if bar.get_height() > 0.1:
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{lift:.1f}x', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax8.set(xlabel='Decile (D10 = highest score)', ylabel='Event rate (%)',
        title='Decile lift chart', xticks=list(x_pos),
        xticklabels=[f'D{i}' for i in range(n_dec, 0, -1)])
ax8.legend(fontsize=8.5); ax8.grid(True, alpha=0.3, axis='y')

# Panel 9 — KS plot
ax9 = fig.add_subplot(gs[2, 2])
thresholds_ks = np.linspace(0, 1, 300)
pos_total = int((y_test==1).sum()); neg_total = int((y_test==0).sum())
cum_pos = np.array([(y_prob[y_test==1] >= t).sum()/pos_total for t in thresholds_ks])
cum_neg = np.array([(y_prob[y_test==0] >= t).sum()/neg_total for t in thresholds_ks])
ax9.plot(thresholds_ks, cum_pos, color='#177A56', lw=2.5, label='Positive class')
ax9.plot(thresholds_ks, cum_neg, color='#E05C2D', lw=2.5, label='Negative class')
ks_i = int(np.argmax(np.abs(cum_pos - cum_neg)))
ax9.fill_between(thresholds_ks, cum_pos, cum_neg, alpha=0.1, color='gray')
ax9.plot([thresholds_ks[ks_i]]*2, [cum_pos[ks_i], cum_neg[ks_i]],
         'k-', lw=2.5, label=f'KS = {ks_stat:.4f}')
ax9.set(xlabel='Score threshold', ylabel='Cumulative %', title='KS plot')
ax9.legend(fontsize=8.5); ax9.grid(True, alpha=0.3)

plt.savefig("random_forest_diagnostics.png", dpi=150, bbox_inches='tight')
plt.show()

# ── 8. Save all output files ──────────────────────────────────
df_out.to_csv("rf_predictions_with_deciles.csv", index=False)
fi_df.round(6).to_csv("rf_feature_importance.csv", index=False)
decile_lift.to_csv("rf_decile_lift_table.csv", index=False)
X.describe().T.round(4).to_csv("rf_summary_statistics.csv")

# Save trained model to disk (joblib)
joblib.dump(rf, "random_forest_model.pkl")
print("Model saved to random_forest_model.pkl")

print("\nAll files saved.")
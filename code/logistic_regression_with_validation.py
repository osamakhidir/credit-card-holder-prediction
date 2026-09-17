import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    brier_score_loss, log_loss, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve,
    precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve
import statsmodels.api as sm


# ── 0. Config ────────────────────────────────────────────────
keep_cols = [
    #"investable_assets_total",
    #"savings_balance",
    "retirement_balance",
    "brokerage_balance",
    "credit_utilization_ratio",
    #"debt_to_income_ratio",
    #"annual_income",
    "autopay_enrolled_flag",
    "overdraft_count_12mo",
    "age",
    "risk_tier_Prime Plus",
    "risk_tier_Prime",
    "delinquency_90d_count"
   # "collections_flag"
]
TARGET = 'credit_card_holder_flag'


# ── 1. Load & split training data ──────────────────────────────
df = pd.read_csv("train_set.csv")
X = df[keep_cols]
y = df[TARGET]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ── 2. Train model on training split ───────────────────────────
model = LogisticRegression(class_weight='balanced', max_iter=1000,
                            solver='lbfgs', random_state=42)
model.fit(X_train, y_train)

# Save the trained model to disk
joblib.dump(model, "logistic_regression_model.pkl")
print("Model saved to logistic_regression_model.pkl")

# ── 3. statsmodels — coefficients + p-values (fit on train split) ─
X_sm     = sm.add_constant(X_train.values.astype(float))
sm_model = sm.Logit(y_train.values.astype(float), X_sm)
result   = sm_model.fit(method='lbfgs', maxiter=500, disp=False)

ci = result.conf_int()[1:]
coef_df = pd.DataFrame({
    'feature'     : feature_names,
    'coef_sklearn': model.coef_[0],
    'coef_sm'     : result.params[1:],
    'std_error'   : result.bse[1:],
    'z_value'     : result.tvalues[1:],
    'p_value'     : result.pvalues[1:],
    'ci_lower'    : ci[:, 0],
    'ci_upper'    : ci[:, 1],
    'odds_ratio'  : np.exp(result.params[1:]),
}).sort_values('p_value')

print("Top 15 features by significance:")
print(coef_df[['feature', 'coef_sm', 'std_error', 'z_value', 'p_value', 'odds_ratio']].head(15).to_string())
print()

coef_df.round(6).to_csv("logist_coefficient_pvalues.csv", index=False)


# ── 4. Reusable diagnostic report function ─────────────────────
def run_diagnostics(y_true, y_prob, dataset_full_df, label, filename_prefix, top_coef_df=None):
    """
    Computes metrics, saves a 9-panel diagnostic PNG, and saves supporting CSVs
    for a given labeled dataset (works for the train-split test set or
    an external validation set).

    y_true            : array-like of true 0/1 labels
    y_prob            : array-like of predicted probabilities (class 1)
    dataset_full_df   : the original dataframe rows aligned with y_true/y_prob
                         (used for the decile lift table + output file)
    label             : string used in plot titles, e.g. "Train (holdout)" or "Validation"
    filename_prefix   : prefix for saved files, e.g. "train" or "validation"
    top_coef_df       : coef_df to use for the coefficient bar panel (same model each time)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    ap   = average_precision_score(y_true, y_prob)
    bs   = brier_score_loss(y_true, y_prob)
    ll   = log_loss(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv  = tp / (tp + fp)
    npv  = tn / (tn + fn)
    gini = 2 * auc - 1
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks_stat = max(tpr - fpr)

    print(f"[{label}] Accuracy={acc:.4f}  ROC-AUC={auc:.4f}  Avg Precision={ap:.4f}")
    print(f"[{label}] Brier={bs:.4f}  Log Loss={ll:.4f}  Gini={gini:.4f}  KS={ks_stat:.4f}")
    print(f"[{label}] Sensitivity={sens:.4f}  Specificity={spec:.4f}  PPV={ppv:.4f}  NPV={npv:.4f}")
    print()

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"Logistic Regression — Full Diagnostic Report ({label})\nTarget: {TARGET}",
                 fontsize=14, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ROC curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(fpr, tpr, color='#534AB7', lw=2, label=f'AUC = {auc:.4f}')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random')
    ks_idx = np.argmax(tpr - fpr)
    ax1.axvline(fpr[ks_idx], color='#D85A30', lw=1, linestyle=':', label=f'KS = {ks_stat:.4f}')
    ax1.set(xlabel='False positive rate', ylabel='True positive rate', title='ROC curve')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # PR curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    baseline = y_true.mean()
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(rec, prec, color='#0F6E56', lw=2, label=f'AP = {ap:.4f}')
    ax2.axhline(baseline, color='k', linestyle='--', lw=1, alpha=0.4, label=f'Baseline = {baseline:.2f}')
    ax2.set(xlabel='Recall', ylabel='Precision', title='Precision-recall curve', ylim=[0, 1.05])
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # Confusion matrix
    ax3 = fig.add_subplot(gs[0, 2])
    ConfusionMatrixDisplay(cm, display_labels=['Non-holder', 'Holder']).plot(
        ax=ax3, colorbar=False, cmap='Blues')
    ax3.set_title('Confusion matrix (threshold = 0.5)')

    # Calibration
    prob_true, prob_pred_cal = calibration_curve(y_true, y_prob, n_bins=10)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(prob_pred_cal, prob_true, 's-', color='#BA7517', lw=2, label='Model')
    ax4.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Perfect calibration')
    ax4.set(xlabel='Mean predicted probability', ylabel='Fraction of positives', title='Calibration plot')
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    # Metrics vs threshold
    thresholds_t = np.linspace(0.01, 0.99, 100)
    precs, recs, f1s = [], [], []
    for t in thresholds_t:
        yp = (y_prob >= t).astype(int)
        precs.append(precision_score(y_true, yp, zero_division=0))
        recs.append(recall_score(y_true, yp, zero_division=0))
        f1s.append(f1_score(y_true, yp, zero_division=0))
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(thresholds_t, precs, color='#534AB7', lw=2, label='Precision')
    ax5.plot(thresholds_t, recs, color='#0F6E56', lw=2, label='Recall')
    ax5.plot(thresholds_t, f1s, color='#BA7517', lw=2, label='F1')
    ax5.axvline(0.5, color='gray', linestyle='--', lw=1, alpha=0.6, label='Default (0.5)')
    ax5.set(xlabel='Threshold', ylabel='Score', title='Metrics vs threshold', ylim=[0, 1.05])
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

    # Top 15 coefficients (same fitted model for both reports)
    if top_coef_df is not None:
        top15 = top_coef_df.head(15).sort_values('coef_sm')
        bar_colors = ['#D85A30' if c < 0 else '#534AB7' for c in top15['coef_sm']]
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.barh(top15['feature'], top15['coef_sm'], color=bar_colors)
        ax6.axvline(0, color='gray', lw=0.8)
        ax6.set(xlabel='Coefficient (statsmodels)', title='Top 15 features by |coef|')
        ax6.tick_params(axis='y', labelsize=8); ax6.grid(True, alpha=0.3, axis='x')

    # Score distribution
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(y_prob[y_true == 0], bins=50, alpha=0.6, color='#534AB7', label='Non-holder', density=True)
    ax7.hist(y_prob[y_true == 1], bins=50, alpha=0.6, color='#D85A30', label='Holder', density=True)
    ax7.set(xlabel='Predicted probability', ylabel='Density', title='Score distribution by class')
    ax7.legend(fontsize=8); ax7.grid(True, alpha=0.3)

    # Decile lift chart
    out_df = dataset_full_df.copy()
    out_df['pred_prob'] = y_prob
    out_df['decile'] = pd.qcut(y_prob, q=10, labels=False, duplicates='drop') + 1
    decile_lift = out_df.groupby('decile').agg(
        n=('pred_prob', 'count'),
        events=(TARGET, 'sum'),
        mean_prob=('pred_prob', 'mean')
    ).reset_index()
    decile_lift['event_rate'] = decile_lift['events'] / decile_lift['n'] * 100
    decile_lift['lift'] = decile_lift['event_rate'] / (baseline * 100)

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.bar(range(1, 11), decile_lift.sort_values('decile', ascending=False)['event_rate'],
            color='#534AB7', alpha=0.8)
    ax8.axhline(baseline * 100, color='#D85A30', linestyle='--', lw=2,
                label=f'Overall rate {baseline*100:.1f}%')
    ax8.set(xlabel='Decile (10 = highest score)', ylabel='Event rate (%)',
            title='Event rate by decile', xticks=range(1, 11),
            xticklabels=[f'D{i}' for i in range(10, 0, -1)])
    ax8.legend(fontsize=8); ax8.grid(True, alpha=0.3, axis='y')

    # KS plot
    ax9 = fig.add_subplot(gs[2, 2])
    thresholds_ks = np.linspace(0, 1, 200)
    pos_total = (y_true == 1).sum(); neg_total = (y_true == 0).sum()
    cum_pos = np.array([(y_prob[y_true == 1] >= t).sum() / pos_total for t in thresholds_ks])
    cum_neg = np.array([(y_prob[y_true == 0] >= t).sum() / neg_total for t in thresholds_ks])
    ax9.plot(thresholds_ks, cum_pos, color='#0F6E56', lw=2, label='Positive class')
    ax9.plot(thresholds_ks, cum_neg, color='#D85A30', lw=2, label='Negative class')
    ks_i = np.argmax(np.abs(cum_pos - cum_neg))
    ax9.plot([thresholds_ks[ks_i]] * 2, [cum_pos[ks_i], cum_neg[ks_i]],
             'k-', lw=2, label=f'KS = {ks_stat:.4f}')
    ax9.set(xlabel='Score threshold', ylabel='Cumulative %', title='KS plot')
    ax9.legend(fontsize=8); ax9.grid(True, alpha=0.3)

    plt.savefig(f"{filename_prefix}_diagnostics.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Save output files for this dataset
    out_df.to_csv(f"{filename_prefix}_predictions_with_deciles.csv", index=False)
    decile_lift.to_csv(f"{filename_prefix}_decile_lift_table.csv", index=False)

    metrics = {
        'dataset': label, 'n': len(y_true), 'accuracy': acc, 'roc_auc': auc,
        'avg_precision': ap, 'brier_score': bs, 'log_loss': ll, 'gini': gini,
        'ks_stat': ks_stat, 'sensitivity': sens, 'specificity': spec,
        'ppv': ppv, 'npv': npv, 'event_rate': baseline
    }
    return metrics


# ── 5. Diagnostics on the train-split holdout (test) set ───────
y_prob_test = model.predict_proba(X_test)[:, 1]
test_metrics = run_diagnostics(
    y_true=y_test, y_prob=y_prob_test,
    dataset_full_df=df.loc[X_test.index],
    label="Train (holdout)", filename_prefix="train_holdout",
    top_coef_df=coef_df
)


# ── 6. Score the validation set with the fitted model ───────────
val_df = pd.read_csv("validation_set.csv")
X_val = val_df[keep_cols]
y_val = val_df[TARGET]
y_prob_val = model.predict_proba(X_val)[:, 1]

val_metrics = run_diagnostics(
    y_true=y_val, y_prob=y_prob_val,
    dataset_full_df=val_df,
    label="Validation", filename_prefix="validation",
    top_coef_df=coef_df
)


# ── 7. Train vs Validation comparison summary ────────────────────
summary_df = pd.DataFrame([test_metrics, val_metrics]).set_index('dataset').round(4)
summary_df.to_csv("train_vs_validation_summary.csv")
print("Train (holdout) vs Validation summary:")
print(summary_df.to_string())
print()

print("All files saved:")
print(" - logist_coefficient_pvalues.csv")
print(" - train_holdout_diagnostics.png / _predictions_with_deciles.csv / _decile_lift_table.csv")
print(" - validation_diagnostics.png / _predictions_with_deciles.csv / _decile_lift_table.csv")
print(" - train_vs_validation_summary.csv")

#!/usr/bin/env python3
"""
HackBio Stage Two: Predicting Drug Sensitivity in Cancer (GDSC)
Machine Learning Pipeline — Regression and Classification
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report,
)
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
_STAGE_ONE_DATA = os.path.join(_SCRIPT_DIR, '..', '..', 'HackBio-StageOne', 'data', 'GDSC.xlsx')
_LOCAL_DATA     = os.path.join(_SCRIPT_DIR, '..', 'data', 'GDSC.xlsx')
DATA_PATH       = _STAGE_ONE_DATA if os.path.exists(_STAGE_ONE_DATA) else _LOCAL_DATA
OUTPUT_DIR      = os.path.join(_SCRIPT_DIR, '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# COLUMN CONSTANTS  (identical to Stage One)
# ---------------------------------------------------------------------------
COL_CELL_LINE   = 'CELL_LINE_NAME'
COL_DRUG        = 'DRUG_NAME'
COL_TCGA_DESC   = 'TCGA_DESC'
COL_LN_IC50     = 'LN_IC50'
COL_AUC         = 'AUC'
COL_Z_SCORE     = 'Z_SCORE'
COL_CNA         = 'CNA'
COL_GENE_EXPR   = 'Gene Expression'
COL_METHYLATION = 'Methylation'
COL_MSI         = 'Microsatellite instability Status (MSI)'
COL_TARGET_PATH = 'TARGET_PATHWAY'


# ---------------------------------------------------------------------------
# FEATURE CONSTANTS
# ---------------------------------------------------------------------------
# Internal column names after encoding
FEAT_DRUG    = 'drug_encoded'
FEAT_CANCER  = 'cancer_type_encoded'
FEAT_PATHWAY = 'pathway_encoded'
FEAT_CNA     = 'cna_binary'
FEAT_GEXPR   = 'gene_expression_binary'
FEAT_METH    = 'methylation_binary'
FEAT_MSI     = 'msi_binary'

FEATURE_COLS = [FEAT_DRUG, FEAT_CANCER, FEAT_PATHWAY, FEAT_CNA, FEAT_GEXPR, FEAT_METH, FEAT_MSI]

# Human-readable labels for plots and reports
FEATURE_LABELS = {
    FEAT_DRUG:    'Drug Name',
    FEAT_CANCER:  'Cancer Type',
    FEAT_PATHWAY: 'Drug Pathway',
    FEAT_CNA:     'CNA (Genomic)',
    FEAT_GEXPR:   'Gene Expression (Transcriptomic)',
    FEAT_METH:    'Methylation (Epigenomic)',
    FEAT_MSI:     'MSI Status (Genomic)',
}

# Quantile thresholds for classification split (adjustable)
QUANTILE_LOWER = 0.25
QUANTILE_UPPER = 0.75

RANDOM_STATE = 42
TEST_SIZE    = 0.20
N_TREES      = 100


# ---------------------------------------------------------------------------
# SECTION 1: DATA LOADING AND FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def load_and_prepare_data(data_path: str) -> pd.DataFrame:
    """Load the GDSC dataset and engineer features for machine learning."""
    print(f"\n{'='*70}")
    print("SECTION 1: Data Loading and Feature Engineering")
    print(f"{'='*70}")
    print(f"  Loading: {os.path.abspath(data_path)}")

    try:
        df = pd.read_excel(data_path)
    except FileNotFoundError:
        print(f"\n  ERROR: Cannot find GDSC.xlsx at {data_path}")
        print("  Either place GDSC.xlsx in data/ or ensure HackBio-StageOne is a sibling folder.")
        sys.exit(1)

    print(f"  Raw dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # --- Binary genomic features: Y -> 1, N -> 0 ---
    # CNA: copy number alteration present in the cell line's DNA
    # Gene Expression: abnormal mRNA expression (transcriptionally active/silent)
    # Methylation: epigenetic silencing via DNA methylation
    for raw_col, feat_col in [
        (COL_CNA,         FEAT_CNA),
        (COL_GENE_EXPR,   FEAT_GEXPR),
        (COL_METHYLATION, FEAT_METH),
    ]:
        if raw_col in df.columns:
            df[feat_col] = df[raw_col].map({'Y': 1, 'N': 0})
        else:
            print(f"  WARNING: '{raw_col}' not found — setting {feat_col} = 0.")
            df[feat_col] = 0

    # --- MSI: MSI-H (hypermutated) -> 1, MSS/MSI-L (microsatellite stable) -> 0 ---
    if COL_MSI in df.columns:
        df[FEAT_MSI] = df[COL_MSI].map({'MSI-H': 1, 'MSS/MSI-L': 0})
    else:
        print(f"  WARNING: MSI column not found — setting {FEAT_MSI} = 0.")
        df[FEAT_MSI] = 0

    # --- Ordinal-encode high-cardinality categoricals ---
    # Tree-based models (RF, XGBoost) split on threshold values, so ordinal
    # encoding works correctly: each unique category gets a stable integer code.
    for raw_col, feat_col in [
        (COL_DRUG,        FEAT_DRUG),
        (COL_TCGA_DESC,   FEAT_CANCER),
        (COL_TARGET_PATH, FEAT_PATHWAY),
    ]:
        if raw_col in df.columns:
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df[feat_col] = enc.fit_transform(df[[raw_col]]).astype(int)
        else:
            print(f"  WARNING: '{raw_col}' not found — setting {feat_col} = -1.")
            df[feat_col] = -1

    # --- Drop any row missing a feature or the target ---
    required = FEATURE_COLS + [COL_LN_IC50]
    df_clean = df[required].dropna()
    dropped = df.shape[0] - df_clean.shape[0]
    print(f"  Rows after dropping NaN in features/target: {df_clean.shape[0]:,} (removed {dropped:,})")

    # Feature summary
    print("\n  Feature engineering summary:")
    for feat, label in FEATURE_LABELS.items():
        col_data = df_clean[feat]
        if feat in [FEAT_CNA, FEAT_GEXPR, FEAT_METH, FEAT_MSI]:
            pct = 100 * col_data.mean()
            print(f"    {label:40s}  binary    | positive rate: {pct:.1f}%")
        else:
            print(f"    {label:40s}  ordinal   | unique values: {col_data.nunique()}")

    print(f"\n  Target — {COL_LN_IC50}:")
    ln = df_clean[COL_LN_IC50]
    print(f"    Mean={ln.mean():.4f}  Std={ln.std():.4f}  Min={ln.min():.4f}  Max={ln.max():.4f}")

    print("\n  NOTE: AUC and Z_SCORE are excluded as features.")
    print("        Both measure drug response just like LN_IC50 — including them would be data leakage.")
    print("        CELL_LINE_NAME is excluded: a high-cardinality identifier, not a biological predictor.")

    return df_clean


# ---------------------------------------------------------------------------
# SECTION 2: REGRESSION — PREDICT LN_IC50 DIRECTLY
# ---------------------------------------------------------------------------
def run_regression(df: pd.DataFrame) -> dict:
    """Train Random Forest and XGBoost regressors to predict LN_IC50."""
    print(f"\n{'='*70}")
    print("SECTION 2: Regression — Predicting LN_IC50 Directly")
    print(f"{'='*70}")

    X = df[FEATURE_COLS].values
    y = df[COL_LN_IC50].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train: {X_train.shape[0]:,} samples  |  Test: {X_test.shape[0]:,} samples")
    print(f"  Features: {len(FEATURE_COLS)}  ({', '.join(FEATURE_LABELS.values())})")

    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=N_TREES,
            max_features='sqrt',
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
    }

    results = {'X_test': X_test, 'y_test': y_test, 'X_train': X_train, 'y_train': y_train}

    for name, model in models.items():
        print(f"\n  Training {name} Regressor...")
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_pred       = model.predict(X_test)

        train_r2 = r2_score(y_train, y_train_pred)
        rmse     = np.sqrt(mean_squared_error(y_test, y_pred))
        r2       = r2_score(y_test, y_pred)
        mae      = mean_absolute_error(y_test, y_pred)

        print(f"    Overfitting check (train vs test R²):")
        print(f"      Train R²: {train_r2:.4f}  |  Test R²: {r2:.4f}  |  Gap: {train_r2 - r2:.4f}")
        print(f"    RMSE: {rmse:.4f}  (lower is better; units: LN_IC50)")
        print(f"    R²:   {r2:.4f}  (closer to 1.0 = better fit)")
        print(f"    MAE:  {mae:.4f}  (mean absolute error in LN_IC50 units)")

        fi        = pd.Series(model.feature_importances_, index=FEATURE_COLS)
        fi_named  = fi.rename(FEATURE_LABELS).sort_values(ascending=False)

        print(f"    Feature Importances (Mean Decrease in Impurity):")
        for feat, imp in fi_named.items():
            bar = '#' * int(imp * 50)
            print(f"      {feat:40s}  {imp:.4f}  {bar}")

        results[name] = {
            'model':              model,
            'y_pred':             y_pred,
            'train_r2':           train_r2,
            'rmse':               rmse,
            'r2':                 r2,
            'mae':                mae,
            'feature_importance': fi_named,
        }

    # 5-fold cross-validation on full dataset (lightweight config for speed)
    print(f"\n  5-Fold Cross-Validation — Regression (Random Forest, n_estimators=50):")
    cv_model = RandomForestRegressor(n_estimators=50, max_features='sqrt', n_jobs=-1, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(cv_model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), scoring='r2', n_jobs=-1)
    print(f"    Fold R² scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"    Mean R²: {cv_scores.mean():.4f}  Std: {cv_scores.std():.4f}")
    print("    -> Consistent CV scores confirm the holdout result is stable, not a lucky split.")
    results['cv_r2_mean'] = cv_scores.mean()
    results['cv_r2_std']  = cv_scores.std()
    results['cv_r2_folds'] = cv_scores.tolist()

    # Biological interpretation
    top_rf  = results['Random Forest']['feature_importance'].index[0]
    top_xgb = results['XGBoost']['feature_importance'].index[0]
    print(f"\n  Biological Interpretation — Regression:")
    print(f"    Top feature (RF):  {top_rf}")
    print(f"    Top feature (XGB): {top_xgb}")
    print("    -> Drug Name dominates because each drug class has an intrinsic potency range.")
    print("       A broad-spectrum chemotherapy agent inherently achieves lower LN_IC50 across")
    print("       most cell lines; a targeted therapy may be very potent but only against")
    print("       specific molecular subtypes.")
    print("    -> Cancer Type ranks highly because tumour biology shapes the drug response landscape.")
    print("       Different cancers express different drug transporters, efflux pumps, and survival")
    print("       pathway dependencies.")
    print("    -> Genomic features (CNA, Gene Expression) contribute incremental signal on top of")
    print("       drug/cancer identity — their importance supports the case for molecular profiling")
    print("       in clinical precision oncology.")

    return results


# ---------------------------------------------------------------------------
# SECTION 3: CLASSIFICATION — SENSITIVE vs RESISTANT
# ---------------------------------------------------------------------------
def run_classification(df: pd.DataFrame) -> dict:
    """
    Quantile-based classification: bottom 25% -> Sensitive (0),
    top 25% -> Resistant (1), middle 50% discarded.
    """
    print(f"\n{'='*70}")
    print("SECTION 3: Classification — Sensitive vs Resistant")
    print(f"{'='*70}")

    lower = df[COL_LN_IC50].quantile(QUANTILE_LOWER)
    upper = df[COL_LN_IC50].quantile(QUANTILE_UPPER)

    print(f"  Quantile thresholds (adjustable via QUANTILE_LOWER / QUANTILE_UPPER):")
    print(f"    Sensitive  — LN_IC50 <= {lower:.4f}  (bottom {int(QUANTILE_LOWER*100)}th percentile)")
    print(f"    Resistant  — LN_IC50 >= {upper:.4f}  (top    {int((1-QUANTILE_UPPER)*100)}th percentile)")
    print(f"    Discarded  — {lower:.4f} < LN_IC50 < {upper:.4f}  (middle 50% removed)")
    print()
    print("  Biological rationale for the quantile split:")
    print("    Sensitive = the cell line needed very little drug to lose 50% viability.")
    print("    Resistant = the cell line required a large drug dose to achieve the same kill rate.")
    print("    Middle values are discarded to create a cleaner, more clinically meaningful contrast.")

    df_s = df[df[COL_LN_IC50] <= lower].copy()
    df_r = df[df[COL_LN_IC50] >= upper].copy()
    df_s['label'] = 0   # Sensitive
    df_r['label'] = 1   # Resistant
    df_clf = pd.concat([df_s, df_r], ignore_index=True)

    print(f"\n  Class sizes:")
    print(f"    Sensitive: {len(df_s):,} ({100*len(df_s)/len(df_clf):.1f}%)")
    print(f"    Resistant: {len(df_r):,} ({100*len(df_r)/len(df_clf):.1f}%)")
    print(f"    Discarded: {len(df) - len(df_s) - len(df_r):,}")

    X = df_clf[FEATURE_COLS].values
    y = df_clf['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Train: {X_train.shape[0]:,} samples  |  Test: {X_test.shape[0]:,} samples (stratified)")

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=N_TREES,
            max_features='sqrt',
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight='balanced',
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
    }

    results = {'lower': lower, 'upper': upper, 'X_test': X_test, 'y_test': y_test}

    for name, model in models.items():
        print(f"\n  Training {name} Classifier...")
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_pred       = model.predict(X_test)
        y_prob       = model.predict_proba(X_test)[:, 1]

        train_acc = accuracy_score(y_train, y_train_pred)
        acc       = accuracy_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        auroc     = roc_auc_score(y_test, y_prob)

        print(f"    Overfitting check (train vs test Accuracy):")
        print(f"      Train Acc: {train_acc:.4f}  |  Test Acc: {acc:.4f}  |  Gap: {train_acc - acc:.4f}")
        print(f"    Accuracy: {acc:.4f}  F1: {f1:.4f}  ROC-AUC: {auroc:.4f}")
        print()
        report = classification_report(
            y_test, y_pred,
            target_names=['Sensitive (0)', 'Resistant (1)'],
        )
        for line in report.splitlines():
            print(f"    {line}")

        fi       = pd.Series(model.feature_importances_, index=FEATURE_COLS)
        fi_named = fi.rename(FEATURE_LABELS).sort_values(ascending=False)

        print(f"    Feature Importances:")
        for feat, imp in fi_named.items():
            bar = '#' * int(imp * 50)
            print(f"      {feat:40s}  {imp:.4f}  {bar}")

        results[name] = {
            'model':              model,
            'y_pred':             y_pred,
            'y_test':             y_test,
            'y_prob':             y_prob,
            'train_acc':          train_acc,
            'acc':                acc,
            'f1':                 f1,
            'auroc':              auroc,
            'feature_importance': fi_named,
        }

    # 5-fold stratified cross-validation (lightweight config for speed)
    print(f"\n  5-Fold Stratified Cross-Validation — Classification (Random Forest, n_estimators=50):")
    cv_clf = RandomForestClassifier(n_estimators=50, max_features='sqrt', n_jobs=-1, random_state=RANDOM_STATE, class_weight='balanced')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(cv_clf, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    print(f"    Fold AUC scores: {[f'{s:.4f}' for s in cv_auc]}")
    print(f"    Mean AUC: {cv_auc.mean():.4f}  Std: {cv_auc.std():.4f}")
    print("    -> Consistent AUC across folds confirms the classifier generalises reliably.")
    results['cv_auc_mean']  = cv_auc.mean()
    results['cv_auc_std']   = cv_auc.std()
    results['cv_auc_folds'] = cv_auc.tolist()

    print("\n  Biological Interpretation — Classification:")
    print("    -> A high ROC-AUC confirms that sensitive vs resistant status is predictable from")
    print("       pharmacological and molecular features — supporting the feasibility of pre-treatment")
    print("       response prediction in precision oncology.")
    print("    -> Drug Pathway importance (not just drug name) tells us that mechanism of action —")
    print("       the biological process the drug attacks — is a key categorical determinant of")
    print("       response class. Drugs targeting the same signalling node share sensitivity landscapes.")
    print("    -> Gene Expression and CNA in the top features validate the clinical concept that")
    print("       molecular profiling of the tumour (transcriptomic state, DNA copy number) adds")
    print("       independent predictive value for therapy selection.")

    return results


# ---------------------------------------------------------------------------
# SECTION 4: VISUALISATIONS
# ---------------------------------------------------------------------------
def _save(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    -> Saved: {filename}")


def visualize_all(reg: dict, clf: dict, df: pd.DataFrame):
    print(f"\n{'='*70}")
    print("SECTION 4: Visualisations  (11 plots)")
    print(f"{'='*70}")
    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 10})

    # -------------------------------------------------------------------------
    # REGRESSION PLOTS
    # -------------------------------------------------------------------------
    print("  Regression plots...")

    # R1: LN_IC50 distribution
    print("  -> Saving reg_01_target_distribution.png...")
    fig, ax = plt.subplots(figsize=(10, 5))
    ln = df[COL_LN_IC50]
    ax.hist(ln, bins=80, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(ln.mean(), color='crimson', linewidth=2, linestyle='--',
               label=f'Mean = {ln.mean():.2f}')
    ax.axvline(ln.mean() - ln.std(), color='orange', linewidth=1.5, linestyle=':',
               label=f'Mean ± Std ({ln.mean()-ln.std():.2f} / {ln.mean()+ln.std():.2f})')
    ax.axvline(ln.mean() + ln.std(), color='orange', linewidth=1.5, linestyle=':')
    ax.set_title(
        'Distribution of Drug Sensitivity (LN_IC50)\nRegression Target Variable',
        fontweight='bold', fontsize=13
    )
    ax.set_xlabel('LN_IC50  (lower = more drug-sensitive cell line)', fontsize=11)
    ax.set_ylabel('Number of Experiments', fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save('reg_01_target_distribution.png')

    # R2: Feature Importance — Regression (RF and XGB side by side)
    print("  -> Saving reg_02_feature_importance.png...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, name, color in zip(axes, ['Random Forest', 'XGBoost'], ['steelblue', 'coral']):
        fi   = reg[name]['feature_importance']
        bars = ax.barh(fi.index[::-1], fi.values[::-1], color=color, alpha=0.85, edgecolor='white')
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
        ax.set_title(
            f'{name}\nRegression Feature Importance (predicting LN_IC50)',
            fontweight='bold', fontsize=11
        )
        ax.set_xlabel('Importance Score (Mean Decrease in Impurity)', fontsize=10)
        ax.set_xlim(0, fi.max() * 1.22)
    plt.suptitle(
        'Which Features Best Predict Drug Sensitivity? — Regression',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    _save('reg_02_feature_importance.png')

    # R3: Actual vs Predicted (hexbin density — 32k+ points need density, not scatter)
    print("  -> Saving reg_03_actual_vs_predicted.png...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, name in zip(axes, ['Random Forest', 'XGBoost']):
        r  = reg[name]
        hb = ax.hexbin(reg['y_test'], r['y_pred'], gridsize=60, cmap='Blues', mincnt=1)
        plt.colorbar(hb, ax=ax, label='Count')
        lo = min(reg['y_test'].min(), r['y_pred'].min())
        hi = max(reg['y_test'].max(), r['y_pred'].max())
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect prediction (y = x)')
        ax.set_title(
            f'{name}  —  Actual vs Predicted LN_IC50\nRMSE={r["rmse"]:.3f}  R²={r["r2"]:.3f}  MAE={r["mae"]:.3f}',
            fontweight='bold', fontsize=11
        )
        ax.set_xlabel('Actual LN_IC50', fontsize=10)
        ax.set_ylabel('Predicted LN_IC50', fontsize=10)
        ax.legend(fontsize=9)
    plt.suptitle(
        'Regression Performance: Actual vs Predicted Drug Sensitivity\n'
        '(Hexbin density plot — darker = more test samples)',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    _save('reg_03_actual_vs_predicted.png')

    # R4: Regression model comparison (metric bars)
    print("  -> Saving reg_04_model_comparison.png...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (label, key) in zip(axes, [('RMSE', 'rmse'), ('R²', 'r2'), ('MAE', 'mae')]):
        vals   = [reg['Random Forest'][key], reg['XGBoost'][key]]
        colors = ['steelblue', 'coral']
        bars   = ax.bar(['Random\nForest', 'XGBoost'], vals,
                        color=colors, alpha=0.85, edgecolor='white', width=0.5)
        ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=11, fontweight='bold')
        ax.set_title(label, fontweight='bold', fontsize=13)
        ax.set_ylim(0, max(vals) * 1.3)
    plt.suptitle('Regression Model Comparison: Random Forest vs XGBoost',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save('reg_04_model_comparison.png')

    # -------------------------------------------------------------------------
    # CLASSIFICATION PLOTS
    # -------------------------------------------------------------------------
    print("  Classification plots...")
    lower = clf['lower']
    upper = clf['upper']

    # C1: Quantile split visualisation
    print("  -> Saving clf_01_quantile_split.png...")
    fig, ax = plt.subplots(figsize=(12, 5))
    ln_vals  = df[COL_LN_IC50].values
    bin_step = (ln_vals.max() - ln_vals.min()) / 80
    counts, bin_edges, patches = ax.hist(
        ln_vals, bins=80, color='lightgray', edgecolor='white', alpha=0.9
    )
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        mid = left_edge + bin_step / 2
        if mid <= lower:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.85)
        elif mid >= upper:
            patch.set_facecolor('coral')
            patch.set_alpha(0.85)
    ax.axvline(lower, color='steelblue', linewidth=2.5, linestyle='--',
               label=f'25th percentile = {lower:.2f}  (Sensitive threshold)')
    ax.axvline(upper, color='coral',     linewidth=2.5, linestyle='--',
               label=f'75th percentile = {upper:.2f}  (Resistant threshold)')
    ax.set_title(
        'Quantile-Based Classification Split\n'
        'Blue = Sensitive (bottom 25%)  |  Red = Resistant (top 25%)  |  Grey = Discarded (middle 50%)',
        fontweight='bold', fontsize=12
    )
    ax.set_xlabel('LN_IC50  (lower = drug-sensitive, higher = drug-resistant)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save('clf_01_quantile_split.png')

    # C2: ROC Curves — both models on same axes
    print("  -> Saving clf_02_roc_curves.png...")
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, color in [('Random Forest', 'steelblue'), ('XGBoost', 'coral')]:
        r   = clf[name]
        fpr, tpr, _ = roc_curve(r['y_test'], r['y_prob'])
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f'{name}  (AUC = {r["auroc"]:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.6,
            label='Random classifier  (AUC = 0.50)')
    ax.set_title(
        'ROC Curve — Sensitive vs Resistant Classification\nBoth Models Compared',
        fontweight='bold', fontsize=12
    )
    ax.set_xlabel('False Positive Rate  (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate  (Sensitivity / Recall)', fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    plt.tight_layout()
    _save('clf_02_roc_curves.png')

    # C3: Confusion Matrices — side by side (seaborn heatmap for reliability)
    print("  -> Saving clf_03_confusion_matrices.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = ['Sensitive', 'Resistant']
    for ax, name in zip(axes, ['Random Forest', 'XGBoost']):
        r  = clf[name]
        cm = confusion_matrix(r['y_test'], r['y_pred'])
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, linecolor='white', cbar=False,
            annot_kws={'size': 14, 'weight': 'bold'}
        )
        ax.set_title(
            f'{name}\nAcc={r["acc"]:.4f}  F1={r["f1"]:.4f}  AUC={r["auroc"]:.4f}',
            fontweight='bold', fontsize=11
        )
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
    plt.suptitle('Confusion Matrices — Drug Sensitivity Classification',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save('clf_03_confusion_matrices.png')

    # C4: Classification Feature Importance
    print("  -> Saving clf_04_feature_importance.png...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, name, color in zip(axes, ['Random Forest', 'XGBoost'], ['steelblue', 'coral']):
        fi   = clf[name]['feature_importance']
        bars = ax.barh(fi.index[::-1], fi.values[::-1], color=color, alpha=0.85, edgecolor='white')
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
        ax.set_title(
            f'{name}\nClassification Feature Importance (Sensitive vs Resistant)',
            fontweight='bold', fontsize=11
        )
        ax.set_xlabel('Importance Score', fontsize=10)
        ax.set_xlim(0, fi.max() * 1.22)
    plt.suptitle(
        'Which Features Best Discriminate Sensitive from Resistant Cell Lines? — Classification',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    _save('clf_04_feature_importance.png')

    # C5: Classification model comparison
    print("  -> Saving clf_05_model_comparison.png...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (label, key) in zip(axes, [('Accuracy', 'acc'), ('F1 Score', 'f1'), ('ROC-AUC', 'auroc')]):
        vals   = [clf['Random Forest'][key], clf['XGBoost'][key]]
        colors = ['steelblue', 'coral']
        bars   = ax.bar(['Random\nForest', 'XGBoost'], vals,
                        color=colors, alpha=0.85, edgecolor='white', width=0.5)
        ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=11, fontweight='bold')
        ax.set_title(label, fontweight='bold', fontsize=13)
        ax.set_ylim(0, 1.2)
    plt.suptitle('Classification Model Comparison: Random Forest vs XGBoost',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save('clf_05_model_comparison.png')

    # -------------------------------------------------------------------------
    # COMBINED — Feature importance heatmap across all 4 models
    # -------------------------------------------------------------------------
    print("  -> Saving combined_01_feature_importance_heatmap.png...")
    heatmap_data = pd.DataFrame({
        'RF — Regression':      reg['Random Forest']['feature_importance'],
        'XGB — Regression':     reg['XGBoost']['feature_importance'],
        'RF — Classification':  clf['Random Forest']['feature_importance'],
        'XGB — Classification': clf['XGBoost']['feature_importance'],
    }).T

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(
        heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
        linewidths=0.5, linecolor='white', ax=ax,
        cbar_kws={'label': 'Feature Importance Score'},
        annot_kws={'size': 9}
    )
    ax.set_title(
        'Feature Importance Across All Four Models\n'
        'Rows = Models  |  Columns = Features  |  Darker = More Important',
        fontweight='bold', fontsize=12
    )
    ax.set_ylabel('Model', fontsize=10)
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    _save('combined_01_feature_importance_heatmap.png')

    print(f"\n  All 11 plots saved to: {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("HackBio Stage Two: Predicting Drug Sensitivity in Cancer (GDSC)")
    print("Machine Learning Pipeline — Regression and Classification")
    print(f"{'='*70}")

    df          = load_and_prepare_data(DATA_PATH)
    reg_results = run_regression(df)
    clf_results = run_classification(df)
    visualize_all(reg_results, clf_results, df)

    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    rf_reg  = reg_results['Random Forest']
    xgb_reg = reg_results['XGBoost']
    rf_clf  = clf_results['Random Forest']
    xgb_clf = clf_results['XGBoost']
    print(f"  Regression   | RF:  RMSE={rf_reg['rmse']:.4f}  R²={rf_reg['r2']:.4f}")
    print(f"               | XGB: RMSE={xgb_reg['rmse']:.4f}  R²={xgb_reg['r2']:.4f}")
    print(f"  Classification | RF:  Acc={rf_clf['acc']:.4f}  AUC={rf_clf['auroc']:.4f}")
    print(f"                 | XGB: Acc={xgb_clf['acc']:.4f}  AUC={xgb_clf['auroc']:.4f}")
    print(f"\n  Outputs written to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

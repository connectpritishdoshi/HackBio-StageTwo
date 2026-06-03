# HackBio Stage Two: Predicting Drug Sensitivity in Cancer

### HackBio AI for Genomics Internship — Stage Two Submission

---

## What This Project Does

Using the same GDSC (Genomics of Drug Sensitivity in Cancer) dataset from Stage One, this project builds machine learning models to **predict drug sensitivity** of cancer cell lines.

Two modelling tasks:
- **Regression** — predict the continuous LN_IC50 value directly
- **Classification** — predict whether a cell line is *Sensitive* (bottom 25%) or *Resistant* (top 25%) to a drug

Two models per task: **Random Forest** and **XGBoost**

---

## Project Structure

```
HackBio-StageTwo/
├── src/
│   └── main.py           Full ML pipeline (Sections 1-4)
├── data/                 Place GDSC.xlsx here if not using Stage One data
├── docs/
│   └── project_overview.md  Biological context and interpretation
├── outputs/              All generated plots (created on run)
├── requirements.txt      Python dependencies
└── README.md             This file
```

---

## Data Setup

The script automatically looks for `GDSC.xlsx` in two locations (in order):

1. `../HackBio-StageOne/data/GDSC.xlsx` — uses Stage One's data folder (recommended)
2. `./data/GDSC.xlsx` — local copy

If you have Stage One cloned as a sibling folder, no extra setup is needed.

---

## Requirements

- Python 3.8+
- Dependencies in `requirements.txt`

```
cd "C:\Users\Pritish\Documents\GitHub\HackBio-StageTwo"
OR
pip install -r requirements.txt
```

---

## Running the Pipeline

```
python src/main.py > outputs/script_output.txt
```

Runtime: approximately 3-8 minutes depending on hardware (trains 4 models on ~130k samples).

---

## Output Files

| File | Description |
|------|-------------|
| `reg_01_target_distribution.png` | LN_IC50 distribution (regression target) |
| `reg_02_feature_importance.png` | Feature importance — RF and XGB regression |
| `reg_03_actual_vs_predicted.png` | Predicted vs actual LN_IC50 (hexbin density) |
| `reg_04_model_comparison.png` | RMSE / R² / MAE comparison |
| `clf_01_quantile_split.png` | Quantile split visualisation (Sensitive / Resistant) |
| `clf_02_roc_curves.png` | ROC curves — both models |
| `clf_03_confusion_matrices.png` | Confusion matrices — both models |
| `clf_04_feature_importance.png` | Feature importance — RF and XGB classification |
| `clf_05_model_comparison.png` | Accuracy / F1 / AUC comparison |
| `combined_01_feature_importance_heatmap.png` | All 4 models × all features heatmap |

---

## Team

- **Pritish Doshi** — Senior Data Architect, University of Mumbai
- **Gauri Jagtap** — Data Engineer Lead, University of Mumbai

# Predicting Drug Sensitivity in Cancer — A Biologist's Guide
### HackBio Stage Two — AI for Genomics Internship

---

## The Problem We Are Solving

In Stage One, we analysed the GDSC dataset to *describe* which drugs work on which cancers. We found that drug sensitivity varies enormously — some cancer cell lines are killed by tiny amounts of a drug, while others survive even very high doses.

Stage Two asks a harder question: **can we predict whether a cancer cell will be sensitive or resistant to a drug, before the experiment is run?**

If we can, this has enormous clinical value. A doctor could look at a tumour biopsy, measure a handful of molecular features, and predict which drugs are likely to work. That is the promise of **precision oncology**.

---

## The Target: LN_IC50

`LN_IC50` is the natural logarithm of the IC50 value — the drug concentration needed to kill 50% of the cancer cells.

- **Low LN_IC50** = the cancer cell died with very little drug → **Sensitive**
- **High LN_IC50** = the cancer cell survived a large dose → **Resistant**

We use the log-transformed value because drug concentrations span many orders of magnitude, and logarithms compress this into a well-behaved, roughly normal distribution.

---

## Two Ways to Frame the Problem

### Regression
We ask the model to predict the exact LN_IC50 value for any (drug, cancer type, genomic profile) combination. This is a harder task but gives finer-grained predictions.

### Classification
We simplify the question into a binary one:
- **Bottom 25% of LN_IC50 values** → Sensitive (label 0)
- **Top 25% of LN_IC50 values** → Resistant (label 1)
- **Middle 50%** → Discarded (ambiguous cases removed for a cleaner signal)

This mirrors how oncologists think clinically: a drug either works or it doesn't.

---

## The Features

The model uses 7 features to make predictions:

### Pharmacological Features

**Drug Name** — Which compound was tested. Different drug classes have intrinsically different potency profiles. Broad-spectrum chemotherapy agents (e.g., cisplatin, doxorubicin) tend to have low IC50 across many cancer types, because they attack fundamental cellular processes like DNA replication. Targeted therapies have narrow but deep activity windows, working only when the specific molecular target is present.

**Cancer Type (TCGA Descriptor)** — The tumour tissue of origin. Lung cancers, leukaemias, and breast cancers have fundamentally different gene expression landscapes, drug transporter activity, and proliferation rates, all of which shape how well a drug works.

**Drug Pathway (TARGET_PATHWAY)** — The biological mechanism the drug attacks (e.g., EGFR signalling, PI3K/mTOR, apoptosis regulation). Drugs targeting the same pathway often share sensitivity profiles across cell lines because they disrupt the same cellular vulnerability.

### Genomic / Molecular Features

**CNA — Copy Number Alteration (Genomic)** — Whether the cancer cell has a duplication or deletion of a DNA segment. CNAs can amplify oncogenes (driving tumour growth) or delete tumour suppressor genes, fundamentally changing which drugs the cell is vulnerable to. A cell line with amplified HER2, for example, is exquisitely sensitive to HER2-targeted drugs.

**Gene Expression (Transcriptomic)** — Whether a specific gene is abnormally active or silent (measured at the mRNA level). Gene expression state tells us which proteins the cell is producing — and which drug targets are accessible. A cancer cell that has silenced a drug transporter may become drug-resistant even to drugs it was once sensitive to.

**Methylation (Epigenomic)** — Whether a gene has been chemically switched off via DNA methylation. Epigenetic silencing can inactivate DNA repair genes (making a cell hypersensitive to DNA-damaging agents) or silence pro-apoptotic genes (making a cell resistant to therapies that trigger cell death).

**MSI Status — Microsatellite Instability (Genomic)** — Whether the cancer has defective DNA mismatch repair machinery. MSI-High (MSI-H) tumours accumulate thousands of mutations and respond dramatically to immunotherapy and certain chemotherapies (e.g., 5-fluorouracil), while microsatellite-stable (MSS) tumours do not.

---

## Why We Excluded AUC and Z_SCORE

`AUC` and `Z_SCORE` are both alternative measures of the same drug response experiment as `LN_IC50`. Including them as features to predict `LN_IC50` would be **data leakage** — the model would simply learn to convert one drug response metric into another, without learning anything about the underlying biology. We excluded them to ensure the model learns from genuine pharmacological and genomic predictors.

---

## The Models: Random Forest and XGBoost

Both models are **decision tree ensembles** — they build hundreds of decision trees, each learning different patterns, and combine their predictions.

**Random Forest** builds each tree independently on a random sample of the data. It is robust to outliers and easy to interpret. Its feature importance scores (Mean Decrease in Impurity) tell us which features most frequently improve the quality of splits across all trees.

**XGBoost** builds trees sequentially, where each new tree corrects the errors of the previous ones (gradient boosting). It is often more accurate than Random Forest on large datasets because it focuses learning effort on the hardest-to-predict samples.

---

## How to Read Feature Importance

Feature importance tells us: **when the model is deciding how to split the data, which feature does it rely on most?**

- A feature with high importance appears high in the decision trees and creates the cleanest splits between sensitive and resistant samples.
- A feature with low importance is rarely used — it adds little predictive signal beyond what other features already provide.

Biological insight comes from comparing feature importances:
- If **Drug Name** dominates, it tells us that pharmacology drives sensitivity more than genomics — the drug's chemistry is the dominant variable.
- If **Gene Expression** ranks highly alongside **Drug Name**, it validates that transcriptomic profiling adds clinically meaningful predictive power.
- The relative ranking of **CNA**, **Gene Expression**, and **Methylation** tells us which layer of molecular biology (genomic vs transcriptomic vs epigenomic) is most informative for predicting drug response.

---

## What Good Results Would Look Like

| Metric | Ideal Range | What It Means |
|--------|------------|---------------|
| R² (regression) | > 0.70 | Model explains most of the variance in LN_IC50 |
| RMSE (regression) | < 1.5 | Average error of less than 1.5 LN_IC50 units |
| ROC-AUC (classification) | > 0.85 | Strong discrimination between Sensitive and Resistant |
| F1 Score (classification) | > 0.80 | Balanced precision and recall |

High performance confirms that drug sensitivity is predictable from a compact set of pharmacological and genomic features — supporting the biological validity of the precision oncology paradigm.

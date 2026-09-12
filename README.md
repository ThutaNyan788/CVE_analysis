# NVD CVE Dashboard

A Streamlit viewer over the results of `final-01.ipynb` (IS-212 project:
*Data Mining and Analysis of Cybersecurity Vulnerabilities Using NVD CVE
Data*). It does **not** re-run the notebook's pipeline — TF-IDF, POS
lemmatization, UMAP and Apriori all take minutes on 100,000 records, which
would make a live app unusable. Instead it reads the CSVs the notebook
already saves with `.to_csv(...)`, so every page opens instantly.

## 1. Install dependencies

```bash
cd cve_dashboard
pip install -r requirements.txt
```

## 2. Add your data

Run the project notebook end-to-end, then copy these files (produced by the
cells noted) into `data/`:

| File | Produced by | Used by |
|---|---|---|
| `nvd_cve_with_description_clusters.csv` | section 7.11 | EDA, Clustering, model training |
| `description_cluster_summary.csv` | section 7.11 | Clustering Explorer |
| `nvd_cve_with_joint_clusters.csv` | section 9.4 | Clustering Explorer (joint mode) |
| `joint_cluster_summary.csv` | section 9.5 | Clustering Explorer (joint mode) |
| `cve_association_rules.csv` | section 11.7 | Association Rules |
| `cve_frequent_itemsets.csv` | section 11.3 | Association Rules (itemsets tab) |
| `severity_model_comparison.csv` | section 12.4 | Severity Predictor |
| `severity_test_predictions.csv` | section 12.6 | Severity Predictor |
| `severity_model_comparison_all_representations.csv` | section 13.3 | Severity Predictor (optional) |

The Home page shows a checklist of which of these it can currently find, so
you don't have to guess — run the app and look at the bottom of the page.

Only the first two files are required to open the EDA and Clustering pages;
everything else degrades gracefully with an on-page message telling you
which file to add.

## 3. Train the live severity predictor (optional but recommended)

The notebook never saves a fitted model — only score tables — so the
"type a description, get a prediction" feature needs one extra step:

```bash
python train_severity_model.py
```

This reproduces sections 12.1-12.3 (temporal split, TF-IDF, Logistic
Regression — chosen for `predict_proba`, same reasoning as the notebook's
own section 12.7 demo) using `data/nvd_cve_with_description_clusters.csv`,
and saves the fitted vectorizer + model to
`models/severity_predictor.joblib`. Point it at the joint-clusters file
instead with `--input data/nvd_cve_with_joint_clusters.csv` if you prefer —
both contain the same `description`/`desc_lemma`/`base_severity` columns.

Without this step, every other page still works; the Severity Predictor page
just shows the static model-comparison results and skips the live text box.

## 4. Run the app

```bash
streamlit run app.py
```

## Project structure

```
cve_dashboard/
├── app.py                          # Home page: overview + setup checklist
├── pages/
│   ├── 1_EDA_Dashboard.py          # Chapter 2 figures, interactive
│   ├── 2_Clustering_Explorer.py    # Sections 7 & 9, text-only vs. joint
│   ├── 3_Association_Rules.py      # Section 11, filterable rule table
│   └── 4_Severity_Predictor.py     # Sections 12-13, live demo + results
├── utils/
│   ├── data_loader.py              # Cached CSV/model loaders, file-presence checks
│   ├── style.py                    # Shared severity colors/order
│   └── text_preprocessing.py       # clean_text + predict_severity (mirrors section 7/12.7)
├── train_severity_model.py         # One-off script -> models/severity_predictor.joblib
├── data/                           # Put the notebook's CSV outputs here
├── models/                         # train_severity_model.py writes here
└── requirements.txt
```

## Notes on fidelity to the notebook

- Chart logic (column choices, vendor/product pipe-splitting, the 12 `is_*`
  vulnerability-type flags, numeric columns for the correlation heatmap) is
  copied from the exact cells that build each figure in the notebook, not
  reconstructed from the PDF's descriptions.
- The Clustering Explorer's "sample descriptions" are a random sample from
  each cluster, not the centroid-nearest example the notebook shows in
  section 7.8 — that requires the fitted TF-IDF/LSA objects, which aren't
  saved to disk. The 2D UMAP/t-SNE projection (section 10.1) is likewise not
  reproduced here for the same reason.
- External-validation NMI/ARI scores (section 7.10 / 9.6) are recomputed
  live from the cluster-assignment CSVs rather than read from a saved table,
  since the notebook only prints those, it doesn't save them.

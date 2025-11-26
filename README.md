# CAPYBARA 🐾  
**C**ross-study **A**daptive **P**redictions **Y**ielding **B**ayesian **A**ggregation with **R**ecursive **A**nalysis  

<<<<<<< HEAD
> *Imputing missing measurements **and** their uncertainty by learning across
> multiple antibody studies.*
=======
> *Imputing values and their uncertainty using multiple training datasets*
>>>>>>> 8d086c97608022ad57843aed3f0c7f46c0b4852b

---

## Table of Contents
1. [Why CAPYBARA?](#1-why-capybara)  
2. [Installation](#2-installation)  
3. [Input file format](#3-input-file-format)  
4. [Quick-start (five lines)](#4-quick-start-five-lines)  
5. [Notebook walk-through](#5-full-notebook-walk-through)  
   * 5.1 [Pre-processing](#51-pre-processing)  
   * 5.2 [Per-dataset feature selection (RFM)](#52-feature-selection--laplacerfm)  
   * 5.3 [Ridge models & transferability](#53-ridge-models--transferability)  
   * 5.4 [Bayesian combination & diagnostic plots](#54-bayesian-combination--plots)  
   * 5.5 [Adding your own dataset](#55-adding-your-own-dataset)  
   * 5.6 [Re-creating all figures 2-5](#56-re-creating-all-paper-figures)  
6. [Project layout](#6-project-layout)  
7. [Advanced configuration](#7-advanced-configuration)  
=======
- [CAPYBARA 🐾](#capybara-)
  - [Table of Contents](#table-of-contents)
  - [1  Why CAPYBARA?](#1--why-capybara)
  - [2 Setup \& Installation](#2-setup--installation)
  - [3 Input file format](#3-input-file-format)
  - [4  Quick-start (five lines)](#4--quick-start-five-lines)
  - [5  Full notebook walk-through](#5--full-notebook-walk-through)
    - [5.1  Pre-processing](#51--pre-processing)
    - [5.2  Feature learning/selection (LaplaceRFM)](#52--feature-learningselection-laplacerfm)
    - [5.3  Feature learning/selection when target virus is completely left out](#53--feature-learningselection-when-target-virus-is-completely-left-out)
    - [5.4  Get predictions and transferability across datasets for each virus, each dataset pair](#54--get-predictions-and-transferability-across-datasets-for-each-virus-each-dataset-pair)
    - [5.5  Bayesian combination \& plots](#55--bayesian-combination--plots)
    - [5.5  Adding your own dataset](#55--adding-your-own-dataset)
    - [5.6  Re-creating all paper figures](#56--re-creating-all-paper-figures)
  - [6  Project layout](#6--project-layout)
  - [7  Advanced configuration](#7--advanced-configuration)

---

## 1  Why CAPYBARA?
<<<<<<< HEADTraditional single-study models **over-fit** and provide no principled way to
transfer knowledge to a new cohort.  
CAPYBARA solves this by:

* Recursive Feature Machines → sparse but biologically meaningful
  feature sets  
* Ridge regression with *leave-one-overlap-out* (LOO) validation to
  quantify transferability (**σ<sub>Predict</sub>**)  
* Bayesian inverse-variance weighting to fuse predictions from many
  source studies  
* Ready-made plotting helpers (HAI, fold-error, …)  

Although the code was built around influenza HAI titres, **any numeric
endpoint** (neutralisation IC<sub>50</sub>, ELISA OD, …) works—just change
=======
Traditional models were trained on one dataset and tested on another, yet we are now approaching the regime where we have **many datasets** for training, some of which are far more informative than others. CAPYBARA provides an efficient, principled way to identify the most informative studies and combine their predictions using:

* Recursive Feature Machines to identify the most predictive feature set
* Ridge regression with error quantification (by predicting left-out data to quantify dataset transferability **σ<sub>Predict</sub>**)  
* Bayesian inverse-variance weighting to combine predictions from multiple studies  
* Helper functions for the specific context of influenza antibody responses we examine (HAI, fold-error, …)

Although the code was built around influenza HAI titres, **any numeric
endpoint** (neutralisation IC<sub>50</sub>, ELISA OD, …) will work seamlessly — just change
`response_col` & `response_transform`.

---
## 2 Setup & Installation
First, clone the repository:

```bash
git clone https://github.com/TalEinav/CAPYBARA.git
cd CAPYBARA
```

Option 1: Conda (recommended)
```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate capybara_env

# Install CAPYBARA (and the local RFM) in editable mode
pip install -e .
```
Option 2: Virtualenv + pip

```bash
# Create and activate a virtualenv (example for Unix-like systems)
python -m venv capybara_env
source capybara_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install Capybara (and the local RFM copy) in editable mode
pip install -e .
```

>>>>>>> 8d086c97608022ad57843aed3f0c7f46c0b4852b
GPU is optional; CPU is fine for the H3N2 case-study.

### Local Recursive Feature Machine (RFM)

CAPYBARA includes a local copy of the Recursive Feature Machine in:

```text
CAPYBARA/local_rfm/rfm/
    ├── __init__.py
    ├── kernel.py
    └── recursive_feature_machine.py
```
---

## 3 Input file format

| Column    | Example value           | Required? | Notes                                   |
| --------- | ----------------------- | --------- | --------------------------------------- |
| `Dataset` | `2021 UGA`              | ✅         | Study / Cohort ID                       |
| `Subject` | `Subject_001`              | ✅      | Subject ID                   |
| `Time`    | `Day21`                 | ✅         | Sample collection time-point              |
| `Virus`   | `H3N2 A/Darwin/9/2021` | ✅         | Feature column that will become a pivot column       |
| `HAI` \*  | `160`                   | ✅\*       | Numeric measurement (rename via response_col if it’s not “HAI”) |

*The name “HAI” is historical; you can pass response_col="Neutralization" (or anything else).

---

## 4  Quick-start (five lines)

```python
from capybara.preprocess import DataPreprocessor
from capybara.pipeline   import LaplaceRFMAnalyzer, RidgeTrainer

data = DataPreprocessor(paths=["data/master.csv"]).run()[1]    # dataset_dict
groups, imp   = LaplaceRFMAnalyzer(data).run_analysis()
ridge_res     = RidgeTrainer(groups, imp, data).run()
```

You now have:

* `groups` – chosen feature sets per (dataset, virus)
* `ridge_res` – σ<sub>Actual</sub>, σ<sub>Predict</sub>, R², per
  train→test pair

Jump to the notebook section below for the full publication-grade workflow.

---

## 5  Full notebook walk-through

**File:** `notebooks/CAPYBARA_Full_Pipeline.ipynb`
Run the cells top-to-bottom or use them as templates.

### 5.1  Pre-processing

```python
from capybara.preprocess import DataPreprocessor
pre = DataPreprocessor(
    paths=["Data/Dataset Master File.csv"],
    response_col="HAI",
    response_transform=lambda x: np.log2(x/5),  # default HAI transform
    viruses_to_keep=["H3N2"]                    # OR [] for all viruses
)
filtered_df, dataset_dict, *_ = pre.run()
```

The pre-processor:


1. merges multiple CSVs
2. drops duplicate subjects across studies
3. fixes *Egg-grown* vs *Cell-grown* naming clashes
4. pivots to `dataset_dict = {dataset → wide DataFrame}`
5. imputes missing cells with row/column means
=======
1. Merges multiple CSVs
2. Drops duplicate subjects across studies
3. Fixes *Egg-grown* vs *Cell-grown* naming clashes
4. Pivots to `dataset_dict = {dataset → wide DataFrame}`
5. Imputes missing cells with row/column means

### 5.2  Feature learning/selection (LaplaceRFM)

```python
from capybara.pipeline import LaplaceRFMAnalyzer
groups_dict, importance_dict = LaplaceRFMAnalyzer(dataset_dict).run_analysis()
```

Outputs (per dataset):
`target_virus → [selected_viruses]` + per-link importance.

### 5.3  Feature learning/selection when target virus is completely left out
Each dataset pair ran on overlapping viruses

```python
from capybara.pipeline import RFMGroupAnalysis, TransferabilityAnalysis

# leave-one-overlap-out Laplace-RFM per (train,test) pair
RFMGroupAnalysis(results_dir="results/leave_one_out_RFM").run(dataset_dict)

```
### 5.4  Get predictions and transferability across datasets for each virus, each dataset pair

```python
from capybara.pipeline import TransferabilityAnalysis

# σPredict via ODR + upper-bound line
TransferabilityAnalysis().run_transferability_analysis(
    dataset_dict,
    combined_virus_groups_dict_path="results/virus_groups_all_datasets.json",
    loo_folder="results/leave_one_out_RFM/groups",
    performance_folder="results/transferability",
    n_splits=5
)
```

### 5.5  Bayesian combination & plots

```python
from capybara.combining_predictions import (
    DataSetNameParser, TrainTestIndexBuilder, PredictionCombiner, PredictionPlotter
)

idx  = TrainTestIndexBuilder(DataSetNameParser()).build_train_test_index(
          "results/transferability", list(dataset_dict.keys()))
files = PredictionCombiner().filter_files_for_test_and_train(
          idx, test_dataset_name="2017 UGA", chosen_train_datasets=["2016 UGA", "2007-2011 Fonv Infect"])
combined = PredictionCombiner().combine_subset_predictions(files)
PredictionPlotter(jitter_strength=0.05, dot_color="plum").plot_combined_predictions(
          combined, dataset_name="2017 UGA", train_datasets_used=["2016 UGA", "2007-2011 Fonv Infect"])
```

Produces **Fig 2A** in the manuscript.

### 5.5  Adding your own dataset

```python
from capybara.preprocess import DataPreprocessor
from capybara.pipeline   import RFMGroupAnalysis, TransferabilityAnalysis
from capybara.combining_predictions import PredictionCombiner

# read only your cohort:
df_all  = pd.read_csv("Dataset File.csv")
my_ds   = "2024_NewStudy"
df_all[df_all["Dataset"] == my_ds].to_csv("tmp_only.csv", index=False)

# reuse the same pipeline
_, new_dict, *_ = DataPreprocessor(paths=["tmp_only.csv"]).run()
dataset_dict[my_ds] = new_dict[my_ds]

<<<<<<< HEAD
# run LOO RFM only for (train, my_dastaset) pairs that share ≥3 viruses
=======
# run leave-one-out RFM only for (train, my_dastaset) pairs that share ≥3 viruses
>>>>>>> 8d086c97608022ad57843aed3f0c7f46c0b4852b
RFMGroupAnalysis("results/leave_one_out_RFM").run(dataset_dict)

TransferabilityAnalysis().run_transferability_analysis(
    dataset_dict,
    "results/virus_groups_all_datasets.json",
    "results/leave_one_out_RFM/groups",
    "results/transferability",
    n_splits=5
)

# combine & plot
idx   = TrainTestIndexBuilder(DataSetNameParser()).build_train_test_index(
          "results/transferability", list(dataset_dict.keys()))
files = PredictionCombiner().filter_files_for_test_and_train(idx, my_ds,
          chosen_train_datasets=[ds for ds in dataset_dict if ds != my_ds])
combined = PredictionCombiner().combine_subset_predictions(files)
PredictionPlotter(dot_color="teal").plot_combined_predictions(
          combined, dataset_name=my_ds, train_datasets_used="all")
```
### 5.6  Re-creating all paper figures

The notebook cells titled **“Figure 2”, “Figure 3”, “Figure 4”, “Figure 5”**
emit PDF/PNG panels into `Figures/Panels/`.
Run them unchanged to reproduce the manuscript.

---

## 6  Project layout

```
CAPYBARA/
├── Capybara/                 # core modules
│   ├── __init__.py           # way to load all from pipeline and preprocess quickly
│   ├── preprocess.py         # long → wide
│   ├── pipeline.py           # RFM, Ridge, transferability, combining predictions
│   ├── performance_analyzer.py # graphs to see general reliability of capybara model
│   ├── ridge_equations.py    # interpretable linear formulas
│   └── utils.py
├── CAPYBARA_Full_Pipeline.ipynb
├── Data/                     # place raw CSVs here
├── Results/                  # models, JSONs, etc. (auto-generated)
├── Figures/                  # publication figures (auto-generated)
├── requirements.txt          # necessary packages
└── README.md
```

---

## 7  Advanced configuration

| Parameter            | Where                            | Default                  | Meaning                                                   |
| -------------------- | -------------------------------- | ------------------------ | --------------------------------------------------------- |
| `response_col`       | `DataPreprocessor`               | `"HAI"`                  | Column holding your measurement                           |
| `response_transform` | `DataPreprocessor`               | `lambda x: np.log2(x/5)` | Callable applied **per value**; set `lambda x: x` to skip |
| `viruses_to_keep`    | `DataPreprocessor`               | `[]`                     | Regex list—empty = keep all                               |
| `Config.BANDWIDTH`   | `capybara/config.py`             | `20`                   | Laplace-RFM kernel width                                  |
| `Config.REG`         | idem                             | `1e-6`                   | Ridge/Laplace regularisation                              |
| `min_overlap`        | `RidgeTrainer` / `OverlapFinder` | `3`                      | Minimum shared viruses to compare datasets                |

---

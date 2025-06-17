# CAPYBARA
**C**ross-study **A**daptive **P**redictions **Y**ielding **B**ayesian **A**ggregation with **R**ecursive **A**nalysis

Imputing values and their uncertainty using multiple training datasets

## 1  What is CAPYBARA?

CAPYBARA is a modular pipeline that

1. loads long-format antibody (or **any** numeric) response tables  
2. identifies informative features with a Laplace **R**ecursive **F**eature
  **M**achine (RFM)  
3. fits Ridge-regression models, including *leave-one-overlap-out*
  transferability analysis  
4. aggregates predictions via Bayesian inverse-variance weighting  
5. produces publication-quality plots of **σ<sub>Predict</sub>**
  vs **σ<sub>Actual</sub>**

Originally developed for influenza HAI titres, CAPYBARA is now *measurement
agnostic*: plug in neutralisation data, ELISA signals, or any other numeric
endpoint.

## 2 Setup & Installation

### Prerequisites
- Python ≥ 3.8
- Conda environment (optional, recommended)

### Installation

```bash
git clone https://github.com/TalEinav/CAPYBARA.git
cd CAPYBARA

# Create and activate a Conda environment
conda create -n capybara python=3.10
conda activate capybara

# Install requirements
pip install -r requirements.txt

## 3 Input Format

| Column    | Example value           | Required? | Notes                                   |
| --------- | ----------------------- | --------- | --------------------------------------- |
| `Dataset` | `2021 UGA`              | ✅         | Study / Cohort ID                       |
| `Subject` | `Subject_001`              | ✅      | Subject ID                   |
| `Time`    | `Day21`                 | ✅         | Sample collection time-point              |
| `Virus`   | `H3N2 A/Darwin/9/2021` | ✅         | Feature column that will become a pivot column       |
| `HAI` \*  | `160`                   | ✅\*       | Numeric measurement (rename via response_col if it’s not “HAI”) |

*The name “HAI” is historical; you can pass response_col="Neutralization" (or anything else).

## 4 Use your own data

from capybara.preprocess import DataPreprocessor

pre = DataPreprocessor(
    paths=["data/my_study.csv"],
    response_col="Neutralisation",            # <- your measurement column
    response_transform=lambda x: np.log2(x/5), # <- any transform you like
    viruses_to_keep=[],                       # <- keep every feature col
)
filtered_df, dataset_dict, *_ = pre.run()

If your data are already pivoted (rows = subjects, cols = features), just build dataset_dict yourself and skip the pre-processor.
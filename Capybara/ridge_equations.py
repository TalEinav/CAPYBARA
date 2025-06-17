# ridge_equations.py

import numpy as np
import pandas as pd
import re
from sklearn.linear_model import Ridge

class ThresholdEquationBuilder:
    """
    Builds a table of equations for each (Dataset, Target Virus) based on:
      - A 'combined_virus_groups_dict' that tells which viruses are 'selected' a priori.
      - A fallback threshold logic to prune Ridge coefficients.
      - Optionally, building both 'No Refit' and 'Refit' equations.

    Attributes
    ----------
    dataset_dict : dict of pd.DataFrame
        Maps dataset_name -> DataFrame with columns = viruses.
    combined_virus_groups_dict : dict
        Maps dataset_name -> { target_virus: [list_of_viruses] }.
    desired_thresh : float
        The starting absolute value threshold for including a coefficient.
    fallback_step : float
        Decrement the threshold by this amount until at least one coefficient is included.
        If threshold < 0, include all features.
    store_refit_only : bool
        If True, only store the final 'Refit' approach in the DataFrame.
        If False, store both 'No Refit' and 'Refit'.
    """

    def __init__(
        self,
        dataset_dict,
        combined_virus_groups_dict,
        desired_thresh=0.2,
        fallback_step=0.05,
        store_refit_only=False
    ):
        self.dataset_dict = dataset_dict
        self.combined_virus_groups_dict = combined_virus_groups_dict
        self.desired_thresh = desired_thresh
        self.fallback_step = fallback_step
        self.store_refit_only = store_refit_only

    def build_equations_dataframe(self):
        """
        Iterates over dataset_dict and combined_virus_groups_dict, trains
        a Ridge model for each (dataset, target_virus), applies threshold logic,
        and optionally collects 'No Refit' & 'Refit' equations.

        Returns
        -------
        results_df : pd.DataFrame
            Columns:
              - Dataset
              - Target Virus
              - Approach (No Refit or Refit)
              - Effective Threshold
              - Intercept
              - Features
              - Coefficients
              - Equation (string form)
              ...
        """
        results_records = []

        for dataset_name, df in self.dataset_dict.items():
            if dataset_name not in self.combined_virus_groups_dict:
                continue

            virus_groups = self.combined_virus_groups_dict[dataset_name]

            for target_virus, selected_viruses in virus_groups.items():
                # Basic checks
                if not selected_viruses or (target_virus not in df.columns):
                    continue

                # Ensure the selected viruses are present in the DF
                available_features = [v for v in selected_viruses if v in df.columns]
                if not available_features:
                    continue

                X = df[available_features].values
                y = df[target_virus].values

                # Train on all available features
                ridge_model = Ridge(alpha=1.0)
                ridge_model.fit(X, y)

                intercept_full = ridge_model.intercept_
                coeffs_full = ridge_model.coef_

                # -----------------------------
                # 1) Fallback threshold logic
                # -----------------------------
                current_thresh = self.desired_thresh
                selected_indices = self._select_features_with_fallback(
                    coeffs_full, current_thresh
                )
                # If still no features, skip or store empty
                if len(selected_indices) == 0:
                    # Optionally store something or skip
                    results_records.append({
                        'Dataset': dataset_name,
                        'Target Virus': target_virus,
                        'Approach': 'No Refit',
                        'Effective Threshold': 0,
                        'Intercept': intercept_full,
                        'Features': [],
                        'Coefficients': [],
                        'Equation': f"{target_virus} = {intercept_full:.2f}"
                    })
                    if not self.store_refit_only:
                        results_records.append({
                            'Dataset': dataset_name,
                            'Target Virus': target_virus,
                            'Approach': 'Refit',
                            'Effective Threshold': 0,
                            'Intercept': intercept_full,
                            'Features': [],
                            'Coefficients': [],
                            'Equation': f"{target_virus} = {intercept_full:.2f}"
                        })
                    continue

                # Build "No Refit" approach
                if not self.store_refit_only:
                    eq_str_no_refit = self._build_equation_string(
                        target_virus,
                        intercept_full,
                        coeffs_full[selected_indices],
                        [available_features[idx] for idx in selected_indices]
                    )
                    results_records.append({
                        'Dataset': dataset_name,
                        'Target Virus': target_virus,
                        'Approach': 'No Refit',
                        'Effective Threshold': current_thresh,
                        'Intercept': intercept_full,
                        'Features': [available_features[idx] for idx in selected_indices],
                        'Coefficients': coeffs_full[selected_indices].tolist(),
                        'Equation': eq_str_no_refit
                    })

                # Build "Refit" approach
                X_subset = X[:, selected_indices]
                ridge_refit = Ridge(alpha=1.0)
                ridge_refit.fit(X_subset, y)

                intercept_refit = ridge_refit.intercept_
                coeffs_refit = ridge_refit.coef_
                feat_refit = [available_features[idx] for idx in selected_indices]

                eq_str_refit = self._build_equation_string(
                    target_virus,
                    intercept_refit,
                    coeffs_refit,
                    feat_refit
                )

                results_records.append({
                    'Dataset': dataset_name,
                    'Target Virus': target_virus,
                    'Approach': 'Refit',
                    'Effective Threshold': current_thresh,
                    'Intercept': intercept_refit,
                    'Features': feat_refit,
                    'Coefficients': coeffs_refit.tolist(),
                    'Equation': eq_str_refit
                })

        results_df = pd.DataFrame(results_records)
        return results_df

    def _select_features_with_fallback(self, coeffs, current_thresh):
        """
        Iteratively lower the threshold until at least one coefficient is selected,
        or threshold goes <= 0, at which point we select all features.

        Returns
        -------
        selected_indices : list of int
            The feature indices that met the final threshold.
        """
        selected_indices = np.where(np.abs(coeffs) >= current_thresh)[0]
        while len(selected_indices) == 0:
            current_thresh -= self.fallback_step
            if current_thresh <= 0:
                # select all
                selected_indices = list(range(len(coeffs)))
                break
            selected_indices = np.where(np.abs(coeffs) >= current_thresh)[0]
        return selected_indices

    def _build_equation_string(self, target_virus, intercept, coeffs, features):
        """
        Constructs a string:  target_virus = intercept + sum(coeff * feature)
        """
        terms = []
        for coef, feat in zip(coeffs, features):
            terms.append(f"({coef:.2f})·{feat}")
        if terms:
            return f"{target_virus} = {intercept:.2f} + " + " + ".join(terms)
        else:
            # no features => just intercept
            return f"{target_virus} = {intercept:.2f}"


import numpy as np
import pandas as pd
import re
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Set
from typing import Union

# ──────────────────────────────────────────────────────────────────────────────
# 1) Evaluate every equation on arbitrary test sets
# ──────────────────────────────────────────────────────────────────────────────
class EquationEvaluator:
    """
    Scores each equation produced by ThresholdEquationBuilder on a collection
    of external datasets.
    """
    def __init__(self,
                 dataset_dict: Dict[str, pd.DataFrame],
                 skip_self: bool = True) -> None:
        self.dataset_dict = dataset_dict
        self.skip_self    = skip_self           # don’t test on training DS

    # -- private helper --------------------------------------------------------
    @staticmethod
    def _predict(df: pd.DataFrame,
                 target: str,
                 feats: Sequence[str],
                 coeffs: Sequence[float],
                 intercept: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (y_true, y_pred) *in log space* -– assumes cols are present."""
        y_true = df[target].values
        y_pred = intercept + np.sum(np.array(coeffs)[None, :] *
                                    df[feats].values, axis=1)
        return y_true, y_pred

    # -- public ----------------------------------------------------------------
    def score(self, equations_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each row in `equations_df` evaluate RMSE & R² on every dataset in
        `dataset_dict` (except its own if skip_self=True).
        """
        records = []
        all_ds  = list(self.dataset_dict.keys())

        for i, row in equations_df.iterrows():
            train_ds  : str          = row["Dataset"]
            tgt       : str          = row["Target Virus"]
            feats     : List[str]    = row["Features"]
            coeffs    : List[float]  = row["Coefficients"]
            intercept : float        = row["Intercept"]

            for test_ds in all_ds:
                if self.skip_self and test_ds == train_ds:
                    continue

                df_test = self.dataset_dict[test_ds]
                if tgt not in df_test.columns or any(f not in df_test.columns
                                                     for f in feats):
                    continue  # columns missing

                y_true, y_pred = self._predict(df_test, tgt, feats, coeffs,
                                               intercept)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2   = r2_score(y_true, y_pred)
                records.append({
                    "Equation Index": i,
                    "Train Dataset":  train_ds,
                    "Test Dataset":   test_ds,
                    "Target Virus":   tgt,
                    "RMSE":           rmse,
                    "R2":             r2,
                })

        return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# 2) Combine identical feature-sets → one averaged equation
# ──────────────────────────────────────────────────────────────────────────────
class EquationAverager:
    """
    Groups equations by (Target Virus, *unordered* feature set) and returns one
    averaged intercept / coefficient vector per group.
    """
    def __init__(self):
        pass

    @staticmethod
    def _sorted_tuple(lst: Sequence[str]) -> Tuple[str, ...]:
        return tuple(sorted(lst))

    def average(self, equations_df: pd.DataFrame) -> pd.DataFrame:
        eq_df = equations_df.copy()
        eq_df["Feature Tuple"] = eq_df["Features"].apply(self._sorted_tuple)

        rows = []
        for (virus, feat_tup), g in eq_df.groupby(["Target Virus",
                                                   "Feature Tuple"]):
            intercept_avg = g["Intercept"].mean()
            coeffs_avg    = np.vstack(g["Coefficients"]).mean(axis=0)
            rows.append({
                "Target Virus":     virus,
                "Feature Tuple":    feat_tup,
                "Intercept Avg":    intercept_avg,
                "Coefficients Avg": coeffs_avg,
                "Source Datasets":  set(g["Dataset"])
            })

        avg_df = pd.DataFrame(rows)
        avg_df["Num_Features"] = avg_df["Feature Tuple"].apply(len)
        avg_df["EquationStr"]  = avg_df.apply(self._eq_to_str, axis=1)
        return avg_df

    # -------------------------------------------------------------------------
    @staticmethod
    def _eq_to_str(row) -> str:
        intercept = row["Intercept Avg"]
        feats, coefs = row["Feature Tuple"], row["Coefficients Avg"]
        terms = [f"({c:+.2f})·{f}" for c, f in zip(coefs, feats)]
        return f"{intercept:+.2f} " + "+ ".join(terms) if terms else f"{intercept:+.2f}"


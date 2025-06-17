from __future__ import annotations
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset
from rfm import LaplaceRFM 
import os
import pickle
from typing import Optional
from typing import Dict, List
#  Our helpers
from .utils  import (
    set_global_seed, split_xy, rmse_log2, extract_groups,
    write_groups_csv, write_importance_csv, dump_json
)
from .config import Config



class LaplaceRFMAnalyzer:
    """
    • Fits LaplaceRFM to each target virus in every dataset.
    • Stores the *raw* group structure (list-of-lists),
      the *combined* group (flattened set) and diagonal importance.
    """

    def __init__(self, dataset_dict: Dict[str, "pd.DataFrame"]):
        self.dataset_dict = dataset_dict

        # outputs
        self.virus_groups_raw: Dict[str, List[List[List[str]]]] = {}
        self.virus_groups_combined: Dict[str, Dict[str, List[str]]] = {}
        self.importance_dict_combined: Dict[str, Dict[str, Dict[str, float]]] = {}

    # ------------------------------------------------------------------
    def _fit_one_target(self, X: np.ndarray, y: np.ndarray):
        """Return LaplaceRFM model trained on (X,y)."""
        Xtr, Xvl, ytr, yvl = split_xy(X, y)
        model = LaplaceRFM(
            bandwidth=Config.BANDWIDTH,
            reg=Config.REG,
            device="cpu",
        )
        model.fit(
            (torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)),
            (torch.tensor(Xvl, dtype=torch.float32), torch.tensor(yvl, dtype=torch.float32)),
            iters=5,
            loader=False,
            #iters=5,
            classif=False,
            return_mse=False,
        )
        return model

    # ------------------------------------------------------------------
    def run_analysis(self):
        """
        Returns
        -------
        combined_groups : dict
            { dataset → { target_virus → [selected_viruses] } }
        importance_dict : dict
            { dataset → { target_virus → {virus: importance} } }
        """
        for dset, df in self.dataset_dict.items():
            print(f"Laplace RFM on {dset}")

            raw_groups_per_target = []          # [[['v1'],['v2',…]], …]
            combined_groups = {}                # {target: [v1,v2,…]}
            importance_per_target = {}          # {target: {virus:score}}

            for tgt in df.columns:
                other = [c for c in df.columns if c != tgt]
                X = df[other].values.astype(np.float32)
                y = df[tgt].values.astype(np.float32).reshape(-1, 1)

                model = self._fit_one_target(X, y)
                groups, _, imp = extract_groups(model.M, other)

                raw_groups_per_target.append(groups)           # keep full structure
                combined_groups[tgt] = list({v for g in groups for v in g})
                importance_per_target[tgt] = imp

            # cache per-dataset results
            self.virus_groups_raw[dset]       = raw_groups_per_target
            self.virus_groups_combined[dset]  = combined_groups
            self.importance_dict_combined[dset] = importance_per_target

        return self.virus_groups_combined, self.importance_dict_combined

class RidgeTrainer:
    def __init__(self, groups_dict, importance_dict, dataset_dict, *, min_overlap=3):
        self.groups_dict     = groups_dict
        self.importance_dict = importance_dict
        self.dataset_dict    = dataset_dict
        self.min_overlap     = min_overlap
        self.results         = {}

    def _test_sets(self, train_ds):
        train_vs = set(self.dataset_dict[train_ds].columns)
        return [
            t for t, df in self.dataset_dict.items()
            if t != train_ds and len(train_vs & set(df.columns)) >= self.min_overlap
        ]

    def run(self):
        for tr_ds, grp_map in self.groups_dict.items():
            self.results[tr_ds] = {}
            df_tr = self.dataset_dict[tr_ds]

            for tgt, sel in grp_map.items():
                self.results[tr_ds][tgt] = []

                for te_ds in self._test_sets(tr_ds):
                    df_te = self.dataset_dict[te_ds]
                    overlap = list(set(sel) & set(df_te.columns))
                    if not overlap or tgt not in df_te.columns:
                        continue

                    X_tr,  y_tr  = df_tr[overlap].values, df_tr[tgt].values
                    X_test, y_te = df_te[overlap].values, df_te[tgt].values

                    rms_int, rms_ext, r2s = [], [], []
                    for i in range(Config.N_SPLITS):
                        Xtr, Xvl, ytr, yvl = split_xy(X_tr, y_tr, seed=i)
                        model = Ridge(alpha=1.0).fit(Xtr, ytr)
                        rms_int.append(rmse_log2(yvl, model.predict(Xvl)))
                        rms_ext.append(rmse_log2(y_te, model.predict(X_test)))
                        r2s.append(r2_score(y_te, model.predict(X_test)))

                    self.results[tr_ds][tgt].append({
                        "Test Dataset"      : te_ds,
                        "Sigma Internal"    : float(np.mean(rms_int)),
                        "Sigma Actual"      : float(np.mean(rms_ext)),
                        "R2 Score"          : float(np.mean(r2s)),
                        "Actual"            : y_te.tolist(),
                        "Predicted"         : model.predict(X_test).tolist(),
                        "Selected Viruses"  : overlap,
                        "Importance"        : [self.importance_dict[tr_ds][tgt].get(v) for v in overlap],
                    })

        dump_json("results/prediction_performance.json", self.results)
        return self.results

# -------------------------------------------------------------------------
# Main helper
# -------------------------------------------------------------------------
import os, gc, json
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from rfm import LaplaceRFM

from .config import Config
from .utils  import (
    extract_groups, split_xy,
    write_groups_csv, write_importance_csv,
    dump_json, load_json
)

# --------------------------------------------------------------------- #
#  Constants controlling chunking
# --------------------------------------------------------------------- #
_BIG_PAIR_THRESH = 25     # #overlap viruses that triggers batching
_BATCH_SIZE      = 20     # #left-out viruses per temporary chunk


class RFMGroupAnalysis:
    """
    Leave-one-overlap-out (LOO) Laplace-RFM analysis **per train→test pair**.
    Big overlaps are processed in chunks to cap memory;
    results are merged into two CSV files per pair:
        results_dir/
            groups/      <train>_<test>.csv
            importance/  <train>_<test>.csv
    """

    # -----------------------------------------------------------------
    def __init__(self, *, results_dir: str = "results/loo_rfm"):
        self.base = Path(results_dir)
        (self.base / "groups").mkdir(parents=True, exist_ok=True)
        (self.base / "importance").mkdir(exist_ok=True)

        # caches
        self._overlap_cache: Dict[Tuple[str, str], List[str]] = {}
        self._split_idx: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _overlap(self, tr: str, te: str, data: Dict[str, "pd.DataFrame"]) -> List[str]:
        key = (tr, te)
        if key not in self._overlap_cache:
            self._overlap_cache[key] = list(
                set(data[tr].columns).intersection(data[te].columns)
            )
        return self._overlap_cache[key]

    def _splits_for(self, name: str, n_rows: int):
        """Cache one 80/20 split per dataset (re-used for every target)."""
        if name not in self._split_idx:
            idx = np.arange(n_rows)
            self._split_idx[name] = train_test_split(
                idx, test_size=0.2, random_state=Config.RNG_SEED
            )
        return self._split_idx[name]

    # -----------------------------------------------------------------
    def _run_single_target(
        self,
        df_tr,                        # training DF, overlap-minus-V0
        tr_idx, vl_idx,               # row indices for split
        target: str,
        other: List[str],
    ):
        """Fit LaplaceRFM for one target and return (combined_group, importance)."""
        X = df_tr[other].to_numpy(np.float32)
        y = df_tr[target].to_numpy(np.float32).reshape(-1, 1)

        Xtr, Xvl = X[tr_idx], X[vl_idx]
        ytr, yvl = y[tr_idx], y[vl_idx]

        with torch.no_grad():
            mod = LaplaceRFM(
                bandwidth=Config.BANDWIDTH, reg=Config.REG, device="cpu"
            )
            mod.fit(
                (torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                (torch.from_numpy(Xvl), torch.from_numpy(yvl)),
                loader=False, iters=5, classif=False, return_mse=False, verbose=False,
            )

        groups, _, imp = extract_groups(mod.M, other)
        combined = list({v for g in groups for v in g})
        return combined, imp

    def _write_final_pair(self, tr: str, te: str, comb: dict, imp: dict):
        """Flush *final* JSONs for a train→test pair."""
        g_path = self.base / "groups"     / f"{tr}_{te}_groups.json"
        i_path = self.base / "importance" / f"{tr}_{te}_importance.json"
        dump_json(g_path, comb)
        dump_json(i_path, imp)
        print(f"saved {tr} → {te} (JSON)")


    # -----------------------------------------------------------------
    def run(self, data: Dict[str, "pd.DataFrame"], *, overwrite: bool = False):
        """
        Parameters
        ----------
        data : dict
            { dataset_name : pd.DataFrame (already log₂(HAI/5)) }
        overwrite : bool
            If False (default) skip pairs whose CSVs already exist.
        """
        names = list(data.keys())
        self._split_idx.clear()   # reset cache between calls

        for tr in names:
            tr_idx, vl_idx = self._splits_for(tr, len(data[tr]))

            for te in names:
                if te == tr:
                    continue

                # g_out = self.base / "groups"      / f"{tr}_{te}.csv"
                # i_out = self.base / "importance"  / f"{tr}_{te}.csv"
                # if not overwrite and g_out.exists() and i_out.exists():
                #     print(f"⏭{tr} → {te} already done")
                #     continue
                g_out = self.base/"groups"/f"{tr}_{te}_groups.json"
                i_out = self.base/"importance"/f"{tr}_{te}_importance.json"
                if not overwrite and g_out.exists() and i_out.exists():
                    print(f"{tr} → {te} already done")
                    continue


                overlap = self._overlap(tr, te, data)
                if len(overlap) < 3:
                    continue

                batching   = len(overlap) >= _BIG_PAIR_THRESH
                chunk_idx  = 0
                chunk_comb, chunk_imp = {}, {}

                df_tr_full = data[tr]
                df_te_full = data[te]

                # ------------- LOO loop on each v0 -------------------
                for v0 in overlap:
                    overlap_m1 = [v for v in overlap if v != v0]
                    if len(overlap_m1) < 2:
                        continue

                    df_tr = df_tr_full[overlap_m1]   # static view – cols only
                    # (df_te currently unused; keep if you extend logic)

                    comb_v0, imp_v0 = {}, {}
                    for tgt in overlap_m1:
                        other = [c for c in overlap_m1 if c != tgt]
                        cg, im = self._run_single_target(df_tr, tr_idx, vl_idx, tgt, other)
                        if len(cg) >= 1:
                            comb_v0[tgt] = cg
                            imp_v0[tgt]  = im


                    chunk_comb[v0] = comb_v0
                    chunk_imp [v0] = imp_v0

                    # ---- dump chunk if reached batch size ------------
                    if batching and len(chunk_comb) >= _BATCH_SIZE:
                        base = f"{tr}_{te}_chunk{chunk_idx}"
                        dump_json(self.base / f"{base}_groups.json", chunk_comb)
                        dump_json(self.base / f"{base}_imp.json",    chunk_imp)
                        print(f"tmp chunk {chunk_idx}  ({tr}→{te})")
                        chunk_idx += 1
                        chunk_comb.clear(); chunk_imp.clear(); gc.collect()

                # flush *last* chunk (if batching)
                if batching and chunk_comb:
                    base = f"{tr}_{te}_chunk{chunk_idx}"
                    dump_json(self.base / f"{base}_groups.json", chunk_comb)
                    dump_json(self.base / f"{base}_imp.json",    chunk_imp)
                    chunk_comb.clear(); chunk_imp.clear()

                # ---------------- merge or direct write -------------
                if batching:
                    merged_comb, merged_imp = {}, {}
                    for p in self.base.glob(f"{tr}_{te}_chunk*_groups.json"):
                        merged_comb.update(load_json(p)); p.unlink()
                    for p in self.base.glob(f"{tr}_{te}_chunk*_imp.json"):
                        merged_imp.update(load_json(p));  p.unlink()
                    self._write_final_pair(tr, te, merged_comb, merged_imp)
                else:
                    self._write_final_pair(tr, te, chunk_comb, chunk_imp)

        print("RFMGroupAnalysis completed")

class OverlapFinder:
    """Helper class to find test datasets with enough overlap in viruses."""

    def find_test_datasets_with_overlap(self, train_dataset_name, dataset_dict, min_overlap=4):
        """
        Identify test datasets whose viruses overlap with those in the training dataset
        by at least 'min_overlap' viruses.

        Args:
            train_dataset_name (str): Name of the training dataset.
            dataset_dict (dict of pd.DataFrame): 
                Key = dataset_name, Value = DataFrame with viruses as columns.
            min_overlap (int): Minimum number of overlapping viruses required.

        Returns:
            suitable_test_datasets (list of str): Dataset names that satisfy the overlap criteria.
        """
        train_viruses = set(dataset_dict[train_dataset_name].columns)
        suitable_test_datasets = []

        for test_dataset_name, df_test in dataset_dict.items():
            if test_dataset_name == train_dataset_name:
                continue
            test_viruses = set(df_test.columns)
            overlap = len(train_viruses.intersection(test_viruses))
            if overlap >= min_overlap:
                suitable_test_datasets.append(test_dataset_name)

        return suitable_test_datasets


import os
import gc
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

##############################################################
# 1. Tools for Perpendicular (TLS) line fitting + upper bound
##############################################################

class TransferabilityTools:
    """
    Provides methods for:
        - Fitting a perpendicular (TLS) line with slope bounds
        - Calculating upper bound lines
        - Parsing dataset names from filenames
    """

    def fit_perpendicular_line_with_bounds(self, x_data, y_data, min_slope=1.0, max_slope=10.0):
        """
        Fits a perpendicular regression line (TLS) to (x_data, y_data),
        enforcing min_slope <= slope <= max_slope.

        Returns:
            slope (float), intercept (float)
        """
        n = len(x_data)
        xAvg = np.mean(x_data)
        yAvg = np.mean(y_data)

        # Check for degenerate data
        if np.allclose(x_data, x_data[0]) and np.allclose(y_data, y_data[0]):
            raise ValueError("All data points are identical; cannot fit line.")

        # 1) Unconstrained slope
        numerator = (np.sum(y_data**2) - n * yAvg**2) - (np.sum(x_data**2) - n * xAvg**2)
        denominator = n * xAvg * yAvg - np.sum(x_data * y_data)

        if np.isclose(denominator, 0.0):
            # near-infinite slope
            unconstrained_slope = np.inf
        else:
            B = 0.5 * (numerator / denominator)
            unconstrained_slope = -B + np.sqrt(B**2 + 1)

        unconstrained_intercept = yAvg - unconstrained_slope * xAvg

        # 2) Check bounds
        if min_slope <= unconstrained_slope <= max_slope and not np.isinf(unconstrained_slope):
            slope = unconstrained_slope
            intercept = unconstrained_intercept
        else:
            # 3) Constrained optimization
            def tls_objective(params, x, y):
                m, b = params
                residuals = y - (m * x + b)
                return np.sum(residuals**2 / (m**2 + 1))

            bounds = [(min_slope, max_slope), (None, None)]
            init_m = (
                np.clip(unconstrained_slope, min_slope, max_slope)
                if not np.isinf(unconstrained_slope) else min_slope
            )
            init_b = unconstrained_intercept if not np.isnan(unconstrained_intercept) else 0.0

            res = minimize(
                tls_objective,
                x0=[init_m, init_b],
                args=(x_data, y_data),
                method='L-BFGS-B',
                bounds=bounds
            )

            if res.success:
                slope, intercept = res.x
            else:
                slope = min_slope
                intercept = yAvg - slope * xAvg

        return slope, intercept

    def calculate_upper_bound(self, x_data, slope, intercept, f_RMSE):
        """
        Build the line y = slope*x + intercept, then create an upper bound line
        by adding f_RMSE.

        Returns:
            x_values, fitted_upper_bound, raw_line
        """
        x_min = 0.0
        x_max = 12.0  # can be adjusted to suit your scale

        pad = 0.1 * (x_max - x_min) if (x_max - x_min) > 0 else 1.0
        x_values = np.linspace(x_min - pad, x_max + pad, 200)

        raw_line = slope * x_values + intercept
        fitted_upper_bound = raw_line + f_RMSE

        return x_values, fitted_upper_bound, raw_line

    def parse_dataset_names_from_filename(self, filename, dataset_names):
        """
        Now accepts filenames like '1997 Fonv_2009-2011 Hay Infect_groups.json'
        or '1997 Fonv_2009-2011 Hay Infect_importance.json', strips off the suffix,
        and returns (train_dataset_name, test_dataset_name).
        """
        # 1) Only accept JSON files
        if not filename.endswith(".json"):
            return None, None

        # 2) Remove either "_groups.json" or "_importance.json"
        if filename.endswith("_groups.json"):
            base_name = filename[: -len("_groups.json")]
        elif filename.endswith("_importance.json"):
            base_name = filename[: -len("_importance.json")]
        else:
            # anything else (e.g. "foo_something.json") we skip
            return None, None

        # 3) Now base_name should look like "<train>_<test>"
        #    We try to match known dataset_names against the end of base_name:
        for test_candidate in dataset_names:
            if base_name.endswith(test_candidate):
                train_part = base_name[: -len(test_candidate)]
                # if there's a trailing underscore, drop it
                if train_part.endswith("_"):
                    train_part = train_part[:-1]
                train_part = train_part.strip()

                if train_part in dataset_names:
                    return train_part, test_candidate

        # 4) If no match found, return (None, None)
        return None, None

##############################################################
# 2. Class for performing LOO transferability analysis
##############################################################

class TransferabilityAnalysis:
    """
    Provides:
      - loo_transferability_analysis_for_pair(...) method
      - (Optional) run_transferability_analysis(...) for a full directory of LOO results
    """

    def __init__(self):
        # You could store default parameters, or tool objects, etc.
        self.tools = TransferabilityTools()
    
    @staticmethod
    def load_pairwise_dicts(combined_dir, dataset_names):
        """
        Loads saved {target_virus: selected_viruses} pairwise files into
        a nested dict: train_dataset → test_dataset → target_virus → selected_viruses
        """
        from pathlib import Path
        import pickle
        from collections import defaultdict
        tools = TransferabilityTools()

        nested_dict = defaultdict(lambda: defaultdict(dict))

        for fpath in Path(combined_dir).glob("*.json"):
            train, test = tools.parse_dataset_names_from_filename(fpath.name, dataset_names)
            if train is None:
                print(f"Could not parse: {fpath.name}")
                continue
            pair_dict = load_json(fpath)  # or load_pickle if you're still using .pkl

            # with open(fpath, "rb") as f:
            #     pair_dict = pickle.load(f)

            nested_dict[train][test] = pair_dict

        return nested_dict
        # ------------------------------------------------------------------ #
    def _load_perf_file(path: str):
        """Return performance dict from either JSON or PKL (legacy)."""
        if path.endswith(".json"):
            return load_json(path)
        # ↓ delete once every old .pkl is gone
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


    def _save_perf_file(self, path: str, obj: dict, *, indent: int = 2):
        """Save a performance dict to JSON."""
        dump_json(path, obj, indent=indent)

    # inside class TransferabilityAnalysis
    def loo_transferability_analysis_for_pair(
        self,
        dataset_dict,
        combined_virus_groups_dict,
        combined_virus_groups_LOO_pair,
        train_dataset_name,
        test_dataset_name,
        n_splits: int = 5,
    ):
        """
        Leave-one-overlap-out transferability on a single (train, test) pair.

        Returns
        -------
        prediction_performance_dict : dict
            {train_dataset → {left_out_virus → [entry, …]}}
            Each *entry* now contains BOTH
            • 5-split diagnostics            (…Split keys)
            • refit-on-all prediction/σActual (default keys)
        """
        # --------------------------------------------------
        # quick sanity guards
        # --------------------------------------------------
        if (
            train_dataset_name not in dataset_dict
            or test_dataset_name not in dataset_dict
            or train_dataset_name not in combined_virus_groups_dict
        ):
            return {}

        df_train_full = dataset_dict[train_dataset_name]
        df_test_full  = dataset_dict[test_dataset_name]

        prediction_performance_dict: dict = {}

        # --------------------------------------------------
        # iterate over every virus we leave out
        # --------------------------------------------------
        for v0, combined_virus_groups_LOO in combined_virus_groups_LOO_pair.items():
            print(f"[LOO] {train_dataset_name} → {test_dataset_name}   left-out = {v0}")

            if v0 not in df_train_full.columns or v0 not in df_test_full.columns:
                continue

            overlap_all = list(set(df_train_full.columns) & set(df_test_full.columns))
            if len(overlap_all) < 3:
                continue

            predictors_v0 = list(
                set(combined_virus_groups_dict[train_dataset_name].get(v0, []))
                & set(overlap_all)
            )
            if len(predictors_v0) < 1:
                continue

            # --------------- data for the left-out virus v0 ----------------
            X_train_full_v0 = df_train_full[predictors_v0]
            y_train_full_v0 = df_train_full[v0]
            X_test_v0       = df_test_full[predictors_v0]
            y_test_v0       = df_test_full[v0]

            # Keep one copy of the actual-test vector (length = n_subjects)
            y_actual_test_v0 = y_test_v0.values

            rmse_int_list, rmse_ext_list = [], []
            y_pred_test_v0_split         = []

            # ---------------- 5 CV splits on the *training* dataset --------
            for i in range(n_splits):
                Xtr, Xvl, ytr, yvl = train_test_split(
                    X_train_full_v0, y_train_full_v0,
                    test_size=0.2, random_state=i
                )
                model = Ridge(alpha=1.0).fit(Xtr, ytr)

                # internal (train-dataset) RMSE
                y_pred_vl = np.maximum(model.predict(Xvl), 0)
                rmse_int_list.append(np.sqrt(mean_squared_error(yvl, y_pred_vl)))

                # external (test-dataset) RMSE
                y_pred_te = np.maximum(model.predict(X_test_v0), 0)
                rmse_ext_list.append(np.sqrt(mean_squared_error(y_test_v0, y_pred_te)))

                y_pred_test_v0_split.extend(y_pred_te.tolist())

            rmse_int_avg = float(np.mean(rmse_int_list))
            rmse_ext_avg = float(np.mean(rmse_ext_list))

            # -------------------- build transferability line --------------
            intra_vals, cross_vals = [], []
            for tgt, predictors_LOO in combined_virus_groups_LOO.items():
                if tgt == v0 or tgt not in df_train_full.columns or tgt not in df_test_full.columns:
                    continue

                preds_tgt = list(set(predictors_LOO) & set(overlap_all))
                if tgt in preds_tgt:
                    preds_tgt.remove(tgt)
                if len(preds_tgt) < 1:
                    continue

                X_train_tgt = df_train_full[preds_tgt]
                y_train_tgt = df_train_full[tgt]
                X_test_tgt  = df_test_full[preds_tgt]
                y_test_tgt  = df_test_full[tgt]

                # 5-split RMSEs for the helper viruses
                int_, ext_ = [], []
                for i in range(n_splits):
                    Xtr, Xvl, ytr, yvl = train_test_split(
                        X_train_tgt, y_train_tgt, test_size=0.2, random_state=i
                    )
                    m = Ridge(alpha=1.0).fit(Xtr, ytr)
                    int_.append(np.sqrt(mean_squared_error(yvl, m.predict(Xvl))))
                    ext_.append(np.sqrt(mean_squared_error(y_test_tgt, m.predict(X_test_tgt))))
                intra_vals.extend(int_)
                cross_vals.extend(ext_)

            # need ≥2 points to fit the TLS line
            if len(intra_vals) < 2:
                continue

            slope, intercept = self.tools.fit_perpendicular_line_with_bounds(
                np.asarray(intra_vals), np.asarray(cross_vals)
            )
            residuals = cross_vals - (slope * np.asarray(intra_vals) + intercept)
            f_RMSE    = float(np.sqrt(np.mean(residuals ** 2)))

            x_vals, upper_bound, raw_line = self.tools.calculate_upper_bound(
                np.asarray(intra_vals), slope, intercept, f_RMSE
            )

            # ------------------ σPredict for the left-out virus -----------
            divergence = None
            if intercept < 0 and not np.isclose(slope, 1.0):
                """ 
                x at which line + f_RMSE crosses y=x
                solve slope*x + intercept + f_RMSE = x
                => x*(1 - slope) = -intercept - f_RMSE
                => x = (-intercept - f_RMSE) / (1 - slope)
                (assuming slope != 1) """
                divergence = (-(intercept + f_RMSE)) / (1.0 - slope)

            if divergence is not None and rmse_int_avg < divergence:
                sigma_predict = rmse_int_avg
            else:
                sigma_predict = slope * rmse_int_avg + intercept + f_RMSE

            # ------------------ REFIT on *all* training rows --------------
            model_full         = Ridge(alpha=1.0).fit(X_train_full_v0, y_train_full_v0)
            y_pred_test_full = np.maximum(model_full.predict(X_test_v0), 0)
            rmse_ext_full    = float(np.sqrt(mean_squared_error(y_actual_test_v0, y_pred_test_full)))

            # ------------------ stash results -----------------------------
            perf_dict = prediction_performance_dict.setdefault(train_dataset_name, {})
            perf_dict.setdefault(v0, []).append({
                # --- transferability context ---
                "Test Dataset"            : test_dataset_name,
                "Intra RMSE Values"       : intra_vals,
                "Cross RMSE Values"       : cross_vals,
                "ODR Params"              : (slope, intercept),
                "Upper Bound Line"        : upper_bound.tolist(),
                "Raw Fitted Line"         : raw_line.tolist(),
                "x_values"                : x_vals.tolist(),
                "Divergence Point"        : divergence,
                "Sigma Predict"           : float(sigma_predict),

                # --- CV diagnostics (5 splits) ---
                "Sigma Internal V₀"       : rmse_int_avg,
                "Sigma Actual V₀ Split"   : rmse_ext_avg,
                "Predicted Left-Out Split": y_pred_test_v0_split,  # length 5×N

                # --- final refit on ALL rows ---
                "Actual Left-Out"         : y_actual_test_v0.tolist(),
                "Predicted Left-Out"      : y_pred_test_full.tolist(),
                "Sigma Actual V₀"         : rmse_ext_full,         # used for plots/combiner

                # bookkeeping
                "Subject IDs"             : df_test_full.index.tolist(),
                "Transferability Params"  : (slope, intercept, f_RMSE),
            })

        return prediction_performance_dict



    def run_transferability_analysis(self, dataset_dict, combined_virus_groups_dict_path,
                                     loo_folder, performance_folder, n_splits=5, all_files_override=None):
        """
        Convenience method that loops over LOO files in `loo_folder`,
        runs the LOO transferability analysis, and writes results to `performance_folder`.

        Args:
            dataset_dict (dict): your dataset dictionary { dataset_name: DataFrame }
            combined_virus_groups_dict_path (str): path to 'combined_virus_groups_dict.pkl'
            loo_folder (str): folder containing the LOO .pkl files
            performance_folder (str): folder to write performance .pkl files
            n_splits (int): how many train/val splits for each virus
        """
        os.makedirs(performance_folder, exist_ok=True)
        
        # ── pick which files to process ──────────────────────────────
        if all_files_override is not None:
            all_files = list(all_files_override)     
        else:
            all_files = [f for f in os.listdir(loo_folder) if f.endswith(".json")]
        combined_virus_groups_dict = load_json(combined_virus_groups_dict_path)

        dataset_names = list(dataset_dict.keys())

        # For each file in the LOO folder
        for filename in all_files:
            train_dataset_name, test_dataset_name = self.tools.parse_dataset_names_from_filename(filename, dataset_names)
            if train_dataset_name is None or test_dataset_name is None:
                print(f"Unable to parse dataset names from '{filename}'. Skipping.")
                continue

            combined_virus_groups_LOO_pair = load_json(
                os.path.join(loo_folder, filename)
            )
            # Run the analysis
            prediction_performance_dict = self.loo_transferability_analysis_for_pair(
                dataset_dict,
                combined_virus_groups_dict,
                combined_virus_groups_LOO_pair,
                train_dataset_name,
                test_dataset_name,
                n_splits=n_splits
            )

            # # Save results
            output_filename = f"{train_dataset_name}_{test_dataset_name}_perf.json"
            self._save_perf_file(
                os.path.join(performance_folder, output_filename),
                prediction_performance_dict,
            )
            print(f"Saved performance results for {train_dataset_name} -> {test_dataset_name}.")

            # Clean up
            del combined_virus_groups_LOO_pair
            del prediction_performance_dict
            gc.collect()
            

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from .utils import load_json 

"""
One‑stop helpers for

  • Parsing performance‑file names  (<Train>_<Test>_perf.{pkl|json})
  • Building an index   (train, test) -> [paths]
  • Combining many individual performance files with Bayesian weighting
  • Plotting (predicted vs. measured) in raw‑HAI space

Supports **both** legacy Pickle and new JSON outputs with *zero behaviour change*
inside the scientific pipeline.
"""

import os
import json
import pickle
import gc
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
"""
Helpers for

  • Parsing performance‑file names  (<Train>_<Test>_perf.{pkl|json})
  • Building an index   (train, test) -> [paths]
  • Combining many individual performance files with Bayesian weighting
  • Plotting (predicted vs. measured) in raw‑HAI space

Supports **both** legacy Pickle and new JSON outputs with zero behaviour change
"""

###############################################################################
# 0)  Small utility: load either *.json or *.pkl transparently
###############################################################################

def load_perf(path: str | Path):
    """Return a performance‑dict from either a JSON or Pickle file."""
    path = str(path)
    if path.endswith(".json"):
        with open(path, "r") as fh:
            return json.load(fh)
    elif path.endswith(".pkl"):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    else:
        raise ValueError(f"Unsupported extension in '{path}'.")

###############################################################################
# A)  Parse    '<Train>_<Test>_perf.{pkl|json}'   →   (train, test)
###############################################################################

class DataSetNameParser:
    """Encapsulates filename‑parsing logic for both PKL **and** JSON outputs."""

    def parse_dataset_names_from_filename(
        self, filename: str, dataset_names: List[str]
    ) -> Tuple[str | None, str | None]:
        # Accept either suffix
        if filename.endswith("_perf.pkl"):
            base = filename[:-9]   # strip '_perf.pkl'
        elif filename.endswith("_perf.json"):
            base = filename[:-10]  # strip '_perf.json'
        else:
            return None, None

        # Attempt to detect which piece at the end is the *test* dataset
        for test_ds in dataset_names:
            if base.endswith(test_ds):
                train_part = base[:-len(test_ds)]
                if train_part.endswith("_"):
                    train_part = train_part[:-1]
                train_part = train_part.strip()
                if train_part in dataset_names:
                    return train_part, test_ds
        return None, None

###############################################################################
# B)  Build index     (train, test) -> [file paths]
###############################################################################

class TrainTestIndexBuilder:
    """Scans a folder of *_perf.* files and groups them by (train, test)."""

    def __init__(self, parser: DataSetNameParser):
        self.parser = parser

    def build_train_test_index(
        self, performance_folder: str | Path, dataset_names: List[str]
    ) -> Dict[Tuple[str, str], List[str]]:
        performance_folder = Path(performance_folder)
        index: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        for fname in os.listdir(performance_folder):
            if not (fname.endswith("_perf.pkl") or fname.endswith("_perf.json")):
                continue
            train_ds, test_ds = self.parser.parse_dataset_names_from_filename(
                fname, dataset_names
            )
            if train_ds and test_ds:
                index[(train_ds, test_ds)].append(str(performance_folder / fname))
            else:
                print(f" Skipping '{fname}' (unparseable).")
        return index

###############################################################################
# C)  Combine predictions across many files with Bayesian weighting
###############################################################################

class PredictionCombiner:
    """Aggregates multiple perf‑files → combined per‑virus predictions."""

    # ------------------------------------------------------------------
    # 1) Locator helpers
    # ------------------------------------------------------------------
    def filter_files_for_test_and_train(
        self,
        index: Dict[Tuple[str, str], List[str]],
        test_dataset_name: str,
        chosen_train_datasets: List[str],
    ) -> List[str]:
        """Return all files whose *test* == `test_dataset_name` and *train* in list."""
        hits: List[str] = []
        for (train_ds, test_ds), files in index.items():
            if test_ds == test_dataset_name and train_ds in chosen_train_datasets:
                hits.extend(files)
        return hits

    # ------------------------------------------------------------------
    # 2) Core combiner
    # ------------------------------------------------------------------
    def combine_subset_predictions(
        self,
        relevant_files: List[str],
        *,
        n_splits: int = 1,
    ) -> List[Dict]:
        """Bayesian‑weight the individual predictions.

        Expects *each* perf‑file to follow the schema you posted earlier. Works
        identically for JSON/PKL because `load_perf` unifies them."""

        predictions_by_target: Dict[Tuple[str, str], Dict] = defaultdict(
            lambda: {"mu_list": [], "sigma_list": [], "actual_values": []}
        )

        tot_valid = tot_neg_sigma = tot_bad = 0

        for fpath in relevant_files:
            try:
                perf_dict = load_perf(fpath)
            except Exception as exc:
                print(f" {fpath}: {exc}")
                tot_bad += 1
                continue

            # ----------------------------------------------------------
            # Walk nested structure:  train → virus → [entry, entry, …]
            # ----------------------------------------------------------
            for train_ds, virus_map in perf_dict.items():
                for virus, entries in virus_map.items():
                    for entry in entries:
                        test_ds        = entry.get("Test Dataset")
                        pred_left_out  = entry.get("Predicted Left-Out")
                        act_left_out   = entry.get("Actual Left-Out")
                        sigma_predict  = entry.get("Sigma Predict")

                        if None in (test_ds, pred_left_out, act_left_out, sigma_predict):
                            tot_bad += 1
                            continue
                        if sigma_predict < 0:
                            tot_neg_sigma += 1
                            continue

                        tot_valid += 1

                        # ------------- reshape (splits, samples) -------------
                        samples_per_split = len(act_left_out) // n_splits
                        if samples_per_split == 0:
                            tot_bad += 1
                            continue
                        try:
                            pred_arr = np.array(pred_left_out).reshape(
                                n_splits, samples_per_split
                            )
                            act_arr  = np.array(act_left_out).reshape(
                                n_splits, samples_per_split
                            )
                        except ValueError:
                            tot_bad += 1
                            continue

                        mu_j = pred_arr.mean(axis=0)  # average split‑wise means
                        key  = (test_ds, virus)
                        bucket = predictions_by_target[key]
                        bucket["mu_list"].append(mu_j)
                        bucket["sigma_list"].append(sigma_predict)
                        if len(bucket["actual_values"]) == 0:
                            bucket["actual_values"] = act_arr[0]  # first split fine

            del perf_dict
            gc.collect()

        print(f"processed={tot_valid}  negativeσ={tot_neg_sigma}  skipped={tot_bad}")

        # --------------------------------------------------------------
        # Bayesian combination per (test_ds, virus)
        # --------------------------------------------------------------
        combined_results: List[Dict] = []
        eps = 1e-6
        for (test_ds, virus), data in predictions_by_target.items():
            mu_lists   = data["mu_list"]
            sigmas     = np.array(data["sigma_list"])
            actual_val = np.array(data["actual_values"])

            if len(mu_lists) == 0:
                continue

            mu_lists = np.stack(mu_lists)        # shape (J , N)
            sigmas   = np.clip(sigmas, eps, 5.0) # shape (J , )
            weights  = 1.0 / sigmas**2           # inverse‑variance

            # pooled mean & σ per sample
            w_sum      = weights.sum()
            mu_comb    = (mu_lists.T @ weights) / w_sum
            sigma_comb = 1.0 / np.sqrt(w_sum)

            rmse_actual = np.sqrt(mean_squared_error(actual_val, mu_comb))

            combined_results.append({
                "Test Dataset"    : test_ds,
                "Virus"           : virus,
                "Actual Values"   : actual_val,
                "Predicted Values": mu_comb,
                "Sigma Combined"  : np.full_like(mu_comb, sigma_comb),
                "Sigma Actual"    : rmse_actual,
            })
        return combined_results

    # ------------------------------------------------------------------
    # 3) Convenience: pool σ_actual per dataset
    # ------------------------------------------------------------------
    @staticmethod
    def pool_sigma_actual(combined_results: List[Dict]) -> Dict[str, float]:
        buckets: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"act": [], "pred": []})
        for rec in combined_results:
            ds = rec["Test Dataset"]
            buckets[ds]["act"].extend(rec["Actual Values"])
            buckets[ds]["pred"].extend(rec["Predicted Values"])
        return {
            ds: np.sqrt(mean_squared_error(v["act"], v["pred"]))
            for ds, v in buckets.items()
        }

class PredictionPlotter:
    """
    Provides a method to plot combined predictions on a log2 scale,
    with configurable parameters.
    """
    def __init__(self, dot_color='purple', jitter_strength=0.03, max_points=5000,
             tick_font_size=40, tick_font_weight='normal', annotation_font_size=55,
             spine_linewidth=3, line_width_diag=3.0, fill_alpha=0.5,
             base_folder="Figures\Panels", dot_size=200, dot_alpha=0.35):
        self.dot_color = dot_color
        self.jitter_strength = jitter_strength
        self.max_points = max_points
        self.tick_font_size = tick_font_size
        self.tick_font_weight = tick_font_weight
        self.annotation_font_size = annotation_font_size
        self.spine_linewidth = spine_linewidth
        self.line_width_diag = line_width_diag
        self.fill_alpha = fill_alpha
        self.base_folder = base_folder
        self.dot_size = dot_size
        self.dot_alpha = dot_alpha   


    def plot_combined_predictions(self, combined_results, dataset_name, train_datasets_used=None, all_train_datasets=None):
        """
        Plots predicted vs. actual data (log2 scale, then exponentiated to HAI),
        with configurable dot color and jitter.
        """
        dataset_folder = os.path.join(self.base_folder, dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)
        plt.rcParams["font.family"] = "Times New Roman"
        # -------------------------------------------------
        # Determine the training-datasets string
        # -------------------------------------------------
        if train_datasets_used:
            # Were we given the full universe to compare against?
            if all_train_datasets is not None and \
            set(train_datasets_used) == set(all_train_datasets):
                train_datasets_str = "all_datasets"
            else:
                # sort so "train1_train2_train3" is deterministic
                train_datasets_str = "_".join(sorted(train_datasets_used))
        else:
            train_datasets_str = "unspecified_train"

        save_path = os.path.join(dataset_folder, f"train_{train_datasets_str}_test_{dataset_name}.png")

        # Filter combined_results for the chosen dataset
        dataset_entries = [res for res in combined_results if res['Test Dataset'] == dataset_name]
        print(f"Number of dataset entries for '{dataset_name}': {len(dataset_entries)}")
        if not dataset_entries:
            print(f"No data available for dataset '{dataset_name}'.")
            return

        all_actual_log2 = []
        all_pred_log2   = []
        all_sigmas      = []

        for entry in dataset_entries:
            actual_vals    = entry['Actual Values']       # log2 domain (HAI/5)
            predicted_vals = entry['Predicted Values']
            sigma_combined = entry['Sigma Combined']      # in log2 domain

            all_actual_log2.extend(actual_vals)
            all_pred_log2.extend(predicted_vals)
            all_sigmas.extend(sigma_combined)

        all_actual_log2 = np.array(all_actual_log2)
        all_pred_log2   = np.array(all_pred_log2)
        all_sigmas      = np.array(all_sigmas)

        N = len(all_actual_log2)
        if N == 0:
            print("No actual or predicted values to plot.")
            return

        # Subsample if too many points
        if N > self.max_points:
            indices = np.random.choice(N, self.max_points, replace=False)
            all_actual_log2 = all_actual_log2[indices]
            all_pred_log2   = all_pred_log2[indices]
            all_sigmas      = all_sigmas[indices]

        # Convert log2(HAI/5) -> raw HAI
        all_actual_HAI = 5.0 * (2 ** all_actual_log2)
        all_pred_HAI   = 5.0 * (2 ** all_pred_log2)

        # rmse_logged      = np.sqrt(mean_squared_error(all_actual_log2, all_pred_log2))
        # rmse_actual_raw  = 2 ** rmse_logged
        # use the entries we already filtered for this dataset
        rmse_log2 = np.nanmean([e["Sigma Actual"] for e in dataset_entries])
        rmse_actual_raw = 2 ** rmse_log2

        sigma_pred_logged = np.mean(all_sigmas)
        sigma_predicted_raw = 2 ** sigma_pred_logged

        #print(f"rmse_logged = {rmse_logged:.3f}; => ~{rmse_actual_raw:.3f}× in raw domain.")
        print(f"sigma_pred_logged (mean) = {sigma_pred_logged:.3f}; => ~{sigma_predicted_raw:.3f}× in raw domain.")

        # Apply jitter
        eps = 1e-8
        all_actual_HAI = np.clip(all_actual_HAI, eps, None)
        all_pred_HAI   = np.clip(all_pred_HAI,   eps, None)
        jitter_x = np.random.normal(0, self.jitter_strength * all_actual_HAI)
        jitter_y = np.random.normal(0, self.jitter_strength * all_pred_HAI)
        all_actual_jittered = all_actual_HAI + jitter_x
        all_pred_jittered   = all_pred_HAI   + jitter_y

        plt.figure(figsize=(12, 12))
        plt.scatter(all_actual_jittered, all_pred_jittered, c=self.dot_color, edgecolors=getattr(self, "dot_edgecolor", 'black'),
         s=self.dot_size, alpha=self.dot_alpha, linewidth=1.5)

        plt.xscale('log', base=2)
        plt.yscale('log', base=2)
        plt.xlim(5, 5120)
        plt.ylim(5, 5120)

        x_line = np.geomspace(5, 5120, 100)
        plt.plot(x_line, x_line, 'k--', linewidth=self.line_width_diag, label="y = x")

        lower_bound = x_line / sigma_predicted_raw
        upper_bound = x_line * sigma_predicted_raw
        plt.fill_between(x_line, lower_bound, upper_bound, color='lightgray',
                         alpha=self.fill_alpha, label=f'± Mean σPredict = {sigma_predicted_raw:.2f}×')

        ticks = [5 * (2 ** i) for i in range(0, 11, 2)]
        plt.xticks(ticks, [str(t) for t in ticks], fontsize=self.tick_font_size, fontweight=self.tick_font_weight)
        plt.yticks(ticks, [str(t) for t in ticks], fontsize=self.tick_font_size, fontweight=self.tick_font_weight)

        annotation_text = (r'$\sigma_{\mathrm{Predict}}$ = ' + f'{sigma_predicted_raw:.1f}x\n' +
                        r'$\sigma_{\mathrm{Actual}}$ = ' + f'{rmse_actual_raw:.1f}x\n' +
                        r'$N$ = ' + f'{N}')

        plt.text(7, 4000, annotation_text, fontsize=self.annotation_font_size,
                 verticalalignment='top', horizontalalignment='left', linespacing=1.5, fontfamily='Times New Roman')

        ax = plt.gca()
        # Make ticks thicker and point inward
        ax.tick_params(axis='both', direction='in', width=4, length=10)  # Adjust width and length as desired

        for spine in ax.spines.values():
            spine.set_linewidth(self.spine_linewidth)

        # plt.savefig(save_path, dpi=300, format='png')
        plt.show()
        print(f"Figure {save_path}")
        # print(f"Figure saved to {save_path}")




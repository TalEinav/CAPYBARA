# Capybara/utils.py
import numpy as np, torch, json, csv, pathlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from .config import Config
from typing import Optional
import pickle

# ------------------------------------------------------------------ #
#  A)  Seeds & simple helpers
# ------------------------------------------------------------------ #
def set_global_seed(seed: int = Config.RNG_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

set_global_seed()      # run at import time


def split_xy(X, y, *, frac: Optional[float] = None, seed: Optional[int] = None):
    """Thin wrapper around train_test_split using global defaults."""
    return train_test_split(
        X, y,
        test_size=frac if frac is not None else Config.SPLIT_FRAC,
        random_state=seed if seed is not None else Config.RNG_SEED,
    )


def rmse_log2(a_log2, p_log2) -> float:
    """Root-MSE in log₂ domain (data already in log₂(HAI/5))."""
    return float(np.sqrt(mean_squared_error(a_log2, p_log2)))


def extract_groups(M, cols, diag_t: float = Config.DIAG_T):
    """
    Extract single-virus groups whose self-similarity (diag) > diag_t.
    Returns (groups, indices, importance_dict).
    """
    M = M / M.max()
    diag = M.diag()
    keep = [i for i, v in enumerate(diag) if v > diag_t]
    return (
        [[cols[i]] for i in keep],            # groups
        keep,                                 # indices
        {cols[i]: diag[i].item() for i in keep},  # importance
    )

# ------------------------------------------------------------------ #
#  B)  Portable I/O  – CSV for tidy tables, JSON for nested dicts
# ------------------------------------------------------------------ #
def write_groups_csv(path: str, data: dict, *, key_name="dataset") -> None:
    """
    Flatten   {dataset → {target → [sel,…]}}
    to CSV:   dataset,target,selected_viruses
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([key_name, "target", "selected_viruses"])
        for dataset, sub in data.items():
            for tgt, sel in sub.items():
                w.writerow([dataset, tgt, ";".join(sel)])



def write_importance_csv(path: str, data: dict, *, key_name="dataset") -> None:
    """
    Flatten   {dataset → {target → {related: score}}}
    to CSV:   dataset,target,related,importance
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([key_name, "target", "related", "importance"])
        for dataset, sub in data.items():
            for tgt, imp_map in sub.items():
                for rel, score in imp_map.items():
                    w.writerow([dataset, tgt, rel, score])


class _NumEncoder(json.JSONEncoder):
    """Make numpy scalars & arrays JSON-serialisable."""
    def default(self, o):
        if isinstance(o, (np.integer,)):   return int(o)
        if isinstance(o, (np.floating,)):  return float(o)
        if isinstance(o, (np.ndarray,)):   return o.tolist()
        return super().default(o)


def dump_json(path: str, obj: dict, *, indent: int = 2) -> None:
    pathlib.Path(path).write_text(json.dumps(obj, indent=indent, cls=_NumEncoder))


def load_json(path: str) -> dict:
    return json.loads(pathlib.Path(path).read_text())

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
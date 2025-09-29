"""
CAPYBARA – Cross-study Adaptive Predictions Yielding Bayesian Recursive Analysis
-------------------------------------------------------------------------------

Top-level API:

    >>> from Capybara import (
    ...     DataPreprocessor, MultiDS, easy_predict,
    ...     LaplaceRFMAnalyzer, RidgeTrainer,
    ...     RFMGroupAnalysis, TransferabilityAnalysis,DataSetNameParser,
    ...     TrainTestIndexBuilder,
    ...     PredictionCombiner,
    ...     PredictionPlotter,
    ...     load_perf
    ... )

Power-user helpers live in sub-packages:
    Capybara.preprocess
    Capybara.pipeline
    Capybara.ridge_equations
    Capybara.utils
"""

from __future__ import annotations
from importlib import metadata as _metadata
import warnings
import logging
from pathlib import Path

# ---------------------------------------------------------------------
# ▸ Version
# ---------------------------------------------------------------------
#__version__: str = _metadata.version("Capybara") if _metadata else "0+unknown"

# ---------------------------------------------------------------------
# ▸ Public symbols (re-export the *most* useful classes)
#   – keep this list short so import time stays low.
# ---------------------------------------------------------------------
from .preprocess import DataPreprocessor, MultiDS        # noqa: E402
from .pipeline   import (                                # noqa: E402
    LaplaceRFMAnalyzer,
    RidgeTrainer,
    RFMGroupAnalysis,
    TransferabilityAnalysis, DataSetNameParser,
    TrainTestIndexBuilder,
    PredictionCombiner,
    PredictionPlotter,
    load_perf
)

__all__ = [
    "DataPreprocessor", "MultiDS",
    "LaplaceRFMAnalyzer", "RidgeTrainer",
    "RFMGroupAnalysis", "TransferabilityAnalysis",
    "easy_predict", "DataSetNameParser", "TrainTestIndexBuilder",
    "PredictionCombiner", "PredictionPlotter",
    "load_perf",
    "__version__", 
]

# ---------------------------------------------------------------------
# ▸ Optional: silence tqdm bars globally unless env-var keeps them.
#   (Matches your notebook patching but works for CLI / scripts too.)
# ---------------------------------------------------------------------
import os
if os.getenv("CAPYBARA_PROGRESS", "off").lower() in {"off", "0", "false"}:
    try:
        from unittest.mock import patch
        import tqdm
        patch("tqdm.tqdm", lambda x, *a, **k: x)
        patch("tqdm.contrib.tenumerate", lambda it, *a, **k: enumerate(it))
    except ModuleNotFoundError:
        pass  # tqdm not installed; ignore

# ---------------------------------------------------------------------
# ▸ Logging defaults – user can override with logging.basicConfig()
# ---------------------------------------------------------------------
_log = logging.getLogger("capybara")
if not _log.handlers:
    handler = logging.StreamHandler()
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt, "%H:%M:%S"))
    _log.addHandler(handler)
    _log.setLevel(logging.INFO)

# ---------------------------------------------------------------------
# ▸ One-shot convenience:  easy_predict
# ---------------------------------------------------------------------
from .pipeline import (            # noqa: E402  keep import *inside* to avoid heavy deps at startup
    OverlapFinder, PredictionCombiner, PredictionPlotter, RFMGroupAnalysis,
    LaplaceRFMAnalyzer
)
from .utils import dump_json                       # noqa: E402
import pandas as pd
import numpy as np

def easy_predict(
    csv_or_df: str | Path | pd.DataFrame,
    *,
    response_col: str = "HAI",
    response_transform = lambda x: np.log2(x / 5),
    viruses: list[str] | None = None,
    min_overlap: int = 3,
    return_format: str = "dataframe",
    workdir: str | Path = "easy_results",
):
    """
    End-to-end CAPYBARA in **one call**.

    Parameters
    ----------
    csv_or_df        : Path to CSV **or** in-memory DataFrame.
    response_col     : Name of the measurement column in your table.
    response_transform : Function applied to that column (log2 by default).
    viruses          : Optional list of target viruses to keep in the output.
    min_overlap      : Minimum shared viruses required to use a training study.
    return_format    : 'dataframe' | 'xlsx' | 'json'
    workdir          : Folder to cache intermediate files.

    Returns
    -------
    pandas.DataFrame **or** writes a file and returns its path.
    """
    workdir = Path(workdir); workdir.mkdir(exist_ok=True)

    # ── 1)  Pre-process the new dataset
    pre = DataPreprocessor(
        paths=[csv_or_df] if isinstance(csv_or_df, (str, Path)) else [],
        response_col=response_col,
        response_transform=response_transform,
        viruses_to_keep=[],          # keep all
    )
    _, new_dict, *_ = pre.run()
    new_name = list(new_dict)[0]

    # ── 2)  Load canonical training studies (tiny helper shipped in utils)
    from .utils import default_dataset_dict          # a lightweight HDF5 stub
    dataset_dict = default_dataset_dict()
    dataset_dict[new_name] = new_dict[new_name]

    # ── 3)  Pick trains with enough overlap
    finder = OverlapFinder()
    train_candidates = [
        ds for ds in dataset_dict if ds != new_name and
        len(finder.find_test_datasets_with_overlap(
            ds, dataset_dict, min_overlap=min_overlap
        )) >= min_overlap
    ]
    if not train_candidates:
        raise RuntimeError("No training datasets share ≥"
                           f"{min_overlap} viruses with your data.")

    # ── 4)  Run leave-one-out RFM chunks only for (train ▸ new)
    loo_dir = workdir / "loo"
    rfm     = RFMGroupAnalysis(results_dir=loo_dir)
    for tr in train_candidates:
        rfm.run({tr: dataset_dict[tr], new_name: new_dict[new_name]}, overwrite=False)

    # ── 5)  Transferability → JSON perf files
    transf = TransferabilityAnalysis()
    transf.run_transferability_analysis(
        dataset_dict,
        combined_virus_groups_dict_path="results/virus_groups_all_datasets.json",
        loo_folder=loo_dir / "groups",
        performance_folder=workdir / "perf",
        n_splits=1
    )

    # ── 6)  Combine predictions across trains
    combiner = PredictionCombiner()
    parser   = DataSetNameParser(); builder = TrainTestIndexBuilder(parser)
    idx = builder.build_train_test_index(workdir / "perf", list(dataset_dict))
    files = combiner.filter_files_for_test_and_train(idx, new_name, train_candidates)
    combined = combiner.combine_subset_predictions(files, n_splits=1)

    if viruses:
        combined = [r for r in combined if r["Virus"] in viruses]

    df = pd.DataFrame(combined)

    if return_format == "dataframe":
        return df

    out_path = workdir / f"predictions.{return_format}"
    if return_format == "json":
        df.to_json(out_path, orient="records", indent=2)
    elif return_format in {"xlsx", "xls"}:
        df.to_excel(out_path, index=False)
    else:
        raise ValueError("return_format must be dataframe|json|xlsx")

    return out_path

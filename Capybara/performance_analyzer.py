# performance_analyzer.py
# --------------------------------------------------------------
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import mean_squared_error
from matplotlib.colors import LinearSegmentedColormap, Normalize

# ----  bring in existing helper classes  ------------------
from .pipeline import (
    DataSetNameParser,
    TrainTestIndexBuilder,
    PredictionCombiner,
    PredictionPlotter,
)

class PerformanceAnalyzer:
    """
    One-stop helper that:
      1. builds (train, test) → file index
      2. combines predictions with Bayesian weighting
      3. exposes   .rmse_df_log2    and   .rmse_df   (×-fold)
      4. plots: heat-map  |  scatter (pred vs meas)
    """

    # -----------------------------
    def __init__(
        self,
        performance_folder: str,
        dataset_dict: Dict[str, pd.DataFrame],
        n_splits: int = 5,
    ):
        self.performance_folder = Path(performance_folder)
        self.dataset_names      = list(dataset_dict.keys())
        self.n_splits           = n_splits

        # core helper objects
        self.parser   = DataSetNameParser()
        self.builder  = TrainTestIndexBuilder(self.parser)
        self.combiner = PredictionCombiner()

        # filled later
        self.index            = None
        self.rmse_df_log2     = None   # log2 σActual
        self.rmse_df          = None   # un-logged ×-fold
        self.combined_results = []     # flat list (all test sets)

    # -----------------------------
    def build_index(self):
        self.index = self.builder.build_train_test_index(
            self.performance_folder, self.dataset_names
        )

    # -----------------------------
    def _mean_sigma_actual(self, combined_results):
        if not combined_results:
            return np.nan
        return np.nanmean([e["Sigma Actual"] for e in combined_results])
   

    # -----------------------------
    def compute_rmse_matrix(self, keep_all_row: bool = True):
        if self.index is None:
            self.build_index()

        rmse_mat = defaultdict(dict)

        for test_ds in sorted({t for _, t in self.index.keys()}):

            train_pool = sorted({tr for (tr, ts) in self.index if ts == test_ds})
            if not train_pool:
                continue

            # -- All-datasets row
            if keep_all_row:
                files = self.combiner.filter_files_for_test_and_train(
                    self.index, test_dataset_name=test_ds,
                    chosen_train_datasets=train_pool
                )
                comb = self.combiner.combine_subset_predictions(files, n_splits=self.n_splits)
                rmse_mat["All Datasets"][test_ds] = self._mean_sigma_actual(comb)

            # -- individual train → test
            for tr in train_pool:
                files = self.combiner.filter_files_for_test_and_train(
                    self.index, test_dataset_name=test_ds,
                    chosen_train_datasets=[tr]
                )
                comb = self.combiner.combine_subset_predictions(files, n_splits=self.n_splits)
                rmse_mat[tr][test_ds] = self._mean_sigma_actual(comb)

        self.rmse_df_log2 = pd.DataFrame(rmse_mat).T.sort_index()
        self.rmse_df      = 2 ** self.rmse_df_log2    # ×-fold
        return self.rmse_df

    # -----------------------------
    def gather_all_combined_predictions(self):
        """Populate self.combined_results (flat list)."""
        if self.index is None:
            self.build_index()
        results = []
        for test_ds in sorted({t for _, t in self.index.keys()}):
            train_pool = sorted({tr for (tr, ts) in self.index if ts == test_ds})
            if not train_pool:
                continue
            files = self.combiner.filter_files_for_test_and_train(
                self.index, test_dataset_name=test_ds,
                chosen_train_datasets=train_pool
            )
            comb = self.combiner.combine_subset_predictions(files, n_splits=self.n_splits)
            results.extend(comb)
        self.combined_results = results
        return results


    def plot_heatmap(
        self,
        row_order: Optional[List[str]] = None,
        col_order: Optional[List[str]] = None,
        pretty_names: Optional[Dict[str, str]] = None,
        midpoint: float = 4.0,
        save_path: Optional[str] = None,
        max_color: str = "tomato",  
    ):
        """
        Plots a heatmap of RMSE values using fixed midpoint coloring.
        """

        if self.rmse_df is None:
            self.compute_rmse_matrix()

        # Copy and optionally reindex
        df = self.rmse_df.copy()

        if row_order:
            df = df.reindex(index=[r for r in row_order if r in df.index])
        if col_order:
            df = df.reindex(columns=[c for c in col_order if c in df.columns])

        # Rename for pretty display
        if pretty_names:
            df = df.rename(index=pretty_names, columns=pretty_names)

        # Setup plotting
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        plt.figure(figsize=(18, 16))

        # TwoSlopeNorm for centered coloring
        # vmin = df.min().min()
        # vmax = df.max().max()
        # norm = TwoSlopeNorm(vmin=vmin, vcenter=midpoint, vmax=vmax)
        

        # Create a custom colormap: white to red
        red_cmap = LinearSegmentedColormap.from_list("custom_red", ["white", max_color])
        # Pick whatever fill you like for missing entries
        red_cmap.set_bad(color="linen")      # or "#dddddd", "black", etc.

        # Make a mask so NaNs are actually treated as missing
        mask = df.isna()
        # Clip everything above 5 to red
        norm = Normalize(vmin=df.min().min(), vmax=5)
        ax = sns.heatmap(
            df,
            mask=mask,  
            annot=True,
            fmt=".1f",
            cmap=red_cmap,
            norm=norm,
            cbar=True,
            cbar_kws={"shrink": 0.6, "aspect": 30},
            annot_kws={"fontsize": 22},
            linewidths=0.7,
            linecolor="black",
            square=True,
            xticklabels=False,
            yticklabels=False,
        )
        formatted_label = pretty_names.get("All Datasets", "All Datasets")
        if formatted_label in df.index:
            row_idx = df.index.get_loc(formatted_label)
            ax.add_patch(plt.Rectangle(
                (0, row_idx),
                df.shape[1],
                1,
                fill=False,
                edgecolor="black",
                linewidth=5
            ))


        # # Colorbar tweaks
        # cbar = ax.collections[0].colorbar
        # cbar.ax.tick_params(labelsize=20)
        # cbar.set_label("RMSE", size=26)
        # ❺ Colorbar tweaks
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=22)
        # Add box to colorbar
        cbar.outline.set_edgecolor("black")    # Border color
        cbar.outline.set_linewidth(1.5)        # Border thickness
        #cbar.set_label("RMSE", size=26)
        cbar.set_label("")#r"$\sigma_{\mathrm{Actual}}$", size=30)

        # Change the last colorbar tick label to '>5.0'
        tick_locs = cbar.get_ticks()
        tick_labels = [f"{t:.1f}" for t in tick_locs]
        tick_labels[-1] = ">5.0"
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(tick_labels)

        # Manual ticks & labels
        ax.set_xticks(np.arange(df.shape[1]) + 0.5)
        ax.set_yticks(np.arange(df.shape[0]) + 0.5)
        ax.set_xticklabels(df.columns, rotation=90, ha='center', fontsize=18)
        ax.set_yticklabels(df.index, rotation=0, ha='right', fontsize=18)

        # Keep gridlines & spines
        n_rows, n_cols = df.shape
        ax.set_xlim(0, n_cols)
        ax.set_ylim(n_rows, 0)  # Flip y-axis

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)

        ax.xaxis.set_tick_params(rotation=90, labelsize=16, pad=1)
        ax.yaxis.set_tick_params(rotation=0, labelsize=16)

        # Final layout & save
        plt.xlabel("")  # could set "Test Dataset"
        plt.ylabel("")  # could set "Train Dataset"
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_pred_vs_meas(
        self,
        ordered_datasets: List[str],
        color_map: Dict[str, tuple],
        pretty_names: Dict[str, str],
        jitter_amount: float = 0.05,
        alpha_points: float = 0.25,
        figsize=(15, 15),
        font_family: str = "Times New Roman",
        tick_fontsize: int = 24,
        annot_spacing: int = 2,
        label_fontsize: int = 24,
        legend_fontsize: int = 18,
        legend_title_fontsize: int = 20,
        sigma_box_fontsize: int = 28,
        markerscale: float = 2.0,
        legend_groups=None,
        legend_titles=None,
        xlabel: str = "Measured HAI",
        ylabel: str = "Predicted HAI",
        save_path: Optional[str] = None,
    ):
        """
        Plots predicted vs measured HAI with optional aesthetic customizations.
        """
        if not self.combined_results:
            self.gather_all_combined_predictions()

        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import mean_squared_error

        plt.rcParams["font.family"] = font_family
        fig, ax = plt.subplots(figsize=figsize)

        all_actual_log2, all_pred_log2 = [], []

        for entry in self.combined_results:
            test_ds = entry["Test Dataset"]
            a_log2 = np.array(entry["Actual Values"])
            p_log2 = np.array(entry["Predicted Values"])

            all_actual_log2.extend(a_log2)
            all_pred_log2.extend(p_log2)

            a_raw = 5.0 * (2 ** a_log2)
            p_raw = 5.0 * (2 ** p_log2)

            a_raw_jit = a_raw + np.random.normal(0, jitter_amount * a_raw)
            p_raw_jit = p_raw + np.random.normal(0, jitter_amount * p_raw)

            ax.scatter(a_raw_jit, p_raw_jit, alpha=alpha_points,
                    color=color_map.get(test_ds, "gray"))

        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlim(5, 5120)
        ax.set_ylim(5, 5120)
        diag = np.geomspace(5, 5120, 100)
        ax.plot(diag, diag, "--", color="black", linewidth=2)

        ticks = [5 * (2 ** i) for i in range(0, 11, 2)]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

        rmse_log2 = np.sqrt(mean_squared_error(all_actual_log2, all_pred_log2))
        rmse_fold = 2 ** rmse_log2
        n_pts = len(all_actual_log2)

        ax.text(
            0.03, 0.97,
            fr"$\sigma_{{\rm Actual}}$ = {rmse_fold:.1f}x" + f"\n$N$ = {n_pts}",
            fontsize=sigma_box_fontsize,
            transform=ax.transAxes, va="top", linespacing=annot_spacing,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )


        # ax.legend(
        #     title="Test Dataset", bbox_to_anchor=(1.05, 1.02),
        #     loc="upper left", fontsize=legend_fontsize,
        #     title_fontsize=legend_title_fontsize,
        #     markerscale=markerscale
        # )

        ax.set_xlabel("") #xlabel, fontsize=label_fontsize, fontweight="bold")
        ax.set_ylabel("") #ylabel, fontsize=label_fontsize, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(
            axis="both",         # x and y axes
            direction="in",      # put ticks inside
            length=10,            # length of ticks
            width=2.5,           # thickness of ticks
            labelsize=tick_fontsize  # size of tick labels (already parameterized)
        )
        # Thicken plot border
        for spine in ax.spines.values():
            spine.set_linewidth(2)  # or any thickness you prefer
            spine.set_edgecolor("black")  # optional: make sure it’s black



        # --------- legends -------------------------------------------
        if legend_groups is None:
            # fallback: one big legend exactly as you had it
            for ds in ordered_datasets:
                ax.scatter([], [], color=color_map[ds],
                        label=pretty_names.get(ds, ds))

            ax.legend(title="Test Dataset",
                    bbox_to_anchor=(1.05, 1.02),
                    loc="upper left",
                    fontsize=legend_fontsize,
                    title_fontsize=legend_title_fontsize,
                    markerscale=markerscale)

        else:
            for i, group in enumerate(legend_groups):
                handles = [plt.Line2D([0], [0], ls="", marker="o",
                                    color=color_map[ds], markersize=10,
                                    label=pretty_names.get(ds, ds))
                        for ds in group]
                y0 = 1.02 - i * 0.30
                leg = ax.legend(handles=handles,
                                title=(legend_titles[i] if legend_titles
                                    and i < len(legend_titles) else None),
                                bbox_to_anchor=(1.05, y0),
                                loc="upper left",
                                fontsize=legend_fontsize,
                                title_fontsize=legend_title_fontsize,
                                markerscale=markerscale)
                ax.add_artist(leg)


        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def _gather_pairwise_predictions(self):
        """
        Returns
        -------
        pred_dict : dict
            {train_ds : {test_ds : list[result_dict]}}
        flat_list : list
            Same results flattened (useful for global RMSE).
        """
        if self.index is None:
            self.build_index()

        pred_dict = defaultdict(lambda: defaultdict(list))
        flat_list = []

        # self.index : { (train_ds, test_ds) : [file, file, …] }
        for (train_ds, test_ds), files in self.index.items():
            comb = self.combiner.combine_subset_predictions(
                files, n_splits=self.n_splits
            )
            for rec in comb:
                rec["Train Dataset"] = train_ds
                rec["Test Dataset"]  = test_ds
                pred_dict[train_ds][test_ds].append(rec)
                flat_list.append(rec)

        return pred_dict, flat_list

    # ---------------------------------------------------------
    #  REWRITE: collect_pairwise_predictions  -----------------
    # ---------------------------------------------------------
    def collect_pairwise_predictions(self):
        """
        Build {train → test → [result_dicts]} with *no* mixing of train sets.
        Also return the overall RMSE (log₂ domain).
        """
        pred_dict, flat = self._gather_pairwise_predictions()

        all_act  = np.concatenate([r["Actual"]    for r in flat])
        all_pred = np.concatenate([r["Predicted"] for r in flat])

        rmse_log2 = np.sqrt(mean_squared_error(all_act, all_pred))
        return pred_dict, rmse_log2


    def plot_pairwise_pred_vs_meas(
        self,
        prediction_performance_dict,
        avg_rmse_log2,
        ordered_datasets,
        color_map,
        pretty_names,
        jitter_amount: float = 0.05,
        alpha_points: float = 0.25,
        figsize=(15, 15),
        xlabel: str = "Measured HAI",
        ylabel: str = "Predicted HAI",
        title: Optional[str] = None,
        xlabel_fontsize: int = 24,
        ylabel_fontsize: int = 24,
        title_fontsize: int = 26,
        annot_spacing: int = 2.0,
        tick_fontsize: int = 22,
        tick_length: int = 10,
        tick_width: float = 2.5,
        legend_fontsize: int = 16,
        legend_title_fontsize: int = 18,
        markerscale: float = 2.0,
        max_points: Optional[int] = None,  # <-- NEW
        save_path: Optional[str] = None,
    ):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import mean_squared_error

        plt.rcParams["font.family"] = "Times New Roman"
        fig, ax = plt.subplots(figsize=figsize)

        total_pts = 0
        all_actual_log2, all_pred_log2 = [], []

        # ------------------------ Plot points ------------------------
        for train_ds, test_dict in prediction_performance_dict.items():
            for test_ds, lst in test_dict.items():
                for res in lst:
                    a = np.array(res["Actual"])
                    p = np.array(res["Predicted"])
                    if len(a) == 0:
                        continue

                    # Optional subsampling
                    if max_points is not None and total_pts >= max_points:
                        continue

                    if max_points is not None:
                        remaining = max_points - total_pts
                        if len(a) > remaining:
                            idx = np.random.choice(len(a), remaining, replace=False)
                            a = a[idx]
                            p = p[idx]

                    # accumulate for RMSE stats
                    all_actual_log2.extend(a)
                    all_pred_log2.extend(p)

                    # jitter + unlog
                    a_raw = 5.0 * (2 ** (a + np.random.normal(0, jitter_amount, size=a.shape)))
                    p_raw = 5.0 * (2 ** (p + np.random.normal(0, jitter_amount, size=p.shape)))

                    ax.scatter(a_raw, p_raw, alpha=alpha_points, color=color_map.get(test_ds, "gray"))
                    total_pts += len(a)

        # ------------------------ Axes ------------------------
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlim(5, 5120)
        ax.set_ylim(5, 5120)

        diag = np.geomspace(5, 5120, 100)
        ax.plot(diag, diag, "--", color="black", linewidth=2)

        ticks = [5 * (2 ** i) for i in range(0, 11, 2)]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

        # ------------------------ RMSE Text ------------------------
        rmse_fold = 2 ** avg_rmse_log2
        ax.text(
            0.03, 0.97,
            fr"$\sigma_{{\rm Actual}}$ = {rmse_fold:.1f}x" + f"\n$N$ = {total_pts}",
            transform=ax.transAxes, va="top",
            fontsize=title_fontsize, linespacing=annot_spacing,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )

        # ------------------------ Legend ------------------------
        for ds in ordered_datasets:
            ax.scatter([], [], color=color_map[ds], label=pretty_names.get(ds, ds))

        ax.legend(
            title="Test Dataset", bbox_to_anchor=(1.05, 1.02),
            loc="upper left", fontsize=legend_fontsize,
            title_fontsize=legend_title_fontsize, markerscale=markerscale
        )

        ax.set_xlabel("")#xlabel, fontsize=xlabel_fontsize, weight="bold")
        ax.set_ylabel("")#ylabel, fontsize=ylabel_fontsize, weight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(
            axis="both", direction="in",
            length=tick_length, width=tick_width,
            labelsize=tick_fontsize
        )
        for spine in ax.spines.values():
            spine.set_linewidth(2)  # or any thickness you prefer
            spine.set_edgecolor("black")  # optional: make sure it’s black
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


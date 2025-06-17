from __future__ import annotations 
import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset
from rfm import LaplaceRFM 
from typing import Callable, Optional, List
class DataPreprocessor:
    """
    Loads multiple CSVs, merges them, removes duplicates, filters out undesired rows,
    and returns a dictionary of pivoted DataFrames (one per dataset).
    """
    def __init__(
        self,
        paths: list[str],
        *,
        response_col: str = "HAI",
        response_transform: Optional[Callable] = None,
        viruses_to_keep: Optional[List[str]]   = None,
        exclude_datasets: Optional[List[str]]  = None,
    ):
        self.paths              = paths
        self.response_col       = response_col
        self.transformed_col    = f"Transformed_{response_col}"
        self.response_transform = (
            response_transform if response_transform is not None
            else (lambda x: np.log2(x / 5))          # original HAI logic
        )
        self.viruses_to_keep    = viruses_to_keep or []
        self.exclude_datasets   = exclude_datasets or []

    def load_and_combine_csv(self):
        data_frames = [pd.read_csv(path, low_memory=False) for path in self.paths]
        data = pd.concat(data_frames, ignore_index=True)
        return data

    def preprocess_data(self, data):
        # 1. Create a unique subject key
        data["Unique_Subject"] = data["Subject"] + "_" + data["Time"]

        # 2. Build a table to see how many subjects appear across datasets
        # hai_titer_distribution = data.groupby(["Dataset", "Unique_Subject"])["HAI"].apply(tuple).reset_index()
        # common_subjects = hai_titer_distribution.groupby("HAI").filter(lambda x: x["Dataset"].nunique() > 1)
        prof_col = self.response_col
        profile_tbl = (
            data.groupby(["Dataset", "Unique_Subject"])[prof_col]
                .apply(tuple).reset_index()
        )
        common_subjects = profile_tbl.groupby(prof_col).filter(
           lambda x: x["Dataset"].nunique() > 1
        )
        # 3. Find pairs of datasets with common subjects
        subject_pairs = []
        #for hai_profile, group in common_subjects.groupby("HAI"):
        for hai_profile, group in common_subjects.groupby(prof_col):
            subjects = group[["Dataset", "Unique_Subject"]].values
            for i, (dataset1, subject1) in enumerate(subjects):
                for dataset2, subject2 in subjects[i+1:]:
                    if dataset1 != dataset2:
                        subject_pairs.append((hai_profile, dataset1, subject1, dataset2, subject2))
        subject_pairs_df = pd.DataFrame(subject_pairs, columns=["HAI_Profile", "Dataset1", "Subject1", "Dataset2", "Subject2"])
        
        # 4. Identify repeated subjects specifically between 2014 Hin_V and 2015 Hin_V
        repeated_subjects_2014_2015 = subject_pairs_df[
            ((subject_pairs_df["Dataset1"] == "2014 Hin_V") & (subject_pairs_df["Dataset2"] == "2015 Hin_V")) |
            ((subject_pairs_df["Dataset1"] == "2015 Hin_V") & (subject_pairs_df["Dataset2"] == "2014 Hin_V"))
        ]
        subjects_to_remove = repeated_subjects_2014_2015[repeated_subjects_2014_2015["Dataset1"] == "2014 Hin_V"]["Subject1"].tolist()
        
        # 5. Remove repeated subjects from 2014 Hin_V
        filtered_data = data[
            ~((data["Unique_Subject"].isin(subjects_to_remove)) & (data["Dataset"] == "2014 Hin_V"))
        ]
        
        # 6. Exclude certain datasets
        filtered_data = filtered_data[~filtered_data["Dataset"].isin(self.exclude_datasets)]
        
        # 7. Remove day>=180 for "UGA" data
        filtered_data["Day_Number"] = filtered_data["Time"].str.extract(r"(\d+)").astype(float)
        filtered_data = filtered_data[~(
            filtered_data["Dataset"].str.contains("UGA") &
            (filtered_data["Day_Number"] >= 180)
        )]
        filtered_data.drop(columns=["Day_Number"], inplace=True)
        
        # 8. Filter out only viruses that contain certain patterns (e.g. "H3N2")
        # pattern = "|".join(self.viruses_to_keep)
        # filtered_data = filtered_data[filtered_data["Virus"].str.contains(pattern, na=False)]
        pattern = "|".join(self.viruses_to_keep) if self.viruses_to_keep else ".*"
        filtered_data = filtered_data[filtered_data["Virus"].str.contains(pattern, na=False)]

        # 9. Log2 transform HAI (HAI -> numeric, then log2(HAI/5))
        # filtered_data["HAI"] = pd.to_numeric(filtered_data["HAI"], errors="coerce")
        # filtered_data["Transformed_HAI"] = np.log2(filtered_data["HAI"] / 5)
        
        # 9. Numeric-cast + user-defined transform
        filtered_data[self.response_col] = pd.to_numeric(
            filtered_data[self.response_col], errors="coerce"
        )
        filtered_data[self.transformed_col] = self.response_transform(
            filtered_data[self.response_col]
        )
        # 10. Create a "Modified_Subject" (unique ID per subject/dataset/time)
        filtered_data["Modified_Subject"] = (
            filtered_data["Subject"] + "_" +
            filtered_data["Dataset"].str.replace(" ", "") + "_" +
            filtered_data["Time"].astype(str)
        )
        # Reconcile _Egg viruses before filtering or transforming
        filtered_data = self.reconcile_egg_viruses_longform(filtered_data)
        return filtered_data

    def build_dataset_dict(self, filtered_data):
        """
        Returns a dictionary { dataset_name: pivoted_df }.
        pivoted_df has rows=Modified_Subject, columns=Virus, and values=Transformed response,
        with missing values imputed by row_mean/col_mean average.
        """
        dataset_dict = {}
        unique_datasets = filtered_data["Dataset"].unique()
        
        for dataset_name in unique_datasets:
            # Subset
            subset_df = filtered_data[filtered_data["Dataset"] == dataset_name]
            
            # Pivot
            pivot_df = subset_df.pivot_table(index="Modified_Subject",
                                             columns="Virus",
                                             values=self.transformed_col)#values="Transformed_HAI")
            # Calculate row & column means for imputing NaNs
            row_means = pivot_df.apply(lambda row: row.mean(), axis=1)
            col_means = pivot_df.apply(lambda col: col.mean(), axis=0)
            
            for row_idx in pivot_df.index:
                for col_idx in pivot_df.columns:
                    if pd.isna(pivot_df.loc[row_idx, col_idx]):
                        row_mean = row_means[row_idx]
                        col_mean = col_means[col_idx]
                        pivot_df.loc[row_idx, col_idx] = (row_mean + col_mean) / 2
            
            dataset_dict[dataset_name] = pivot_df
        
        return dataset_dict

    def reconcile_egg_viruses_longform(self, df):
        """
        Reconciles _Egg viruses in longform (before pivot):
        - If both Virus and Virus_Egg exist for the same Dataset & Subject, keep Egg and drop the other.
        - If only Egg exists, rename it to non-Egg name.
        """
        # Step 1: Make a clean copy
        df = df.copy()

        # Step 2: Normalize Virus names
        df["Normalized_Virus"] = df["Virus"].str.replace("_Egg", "", regex=False)

        # Step 3: Mark _Egg rows for priority
        df["Egg_Flag"] = df["Virus"].str.endswith("_Egg")

        # Step 4: Sort to keep Egg version last (will overwrite non-Egg)
        df.sort_values(["Dataset", "Modified_Subject", "Normalized_Virus", "Egg_Flag"], inplace=True)

        df_deduped = df.drop_duplicates(subset=["Dataset", "Modified_Subject", "Normalized_Virus"], keep="last").copy()
        df_deduped.loc[:, "Virus"] = df_deduped["Normalized_Virus"]
        df_deduped = df_deduped.drop(columns=["Normalized_Virus", "Egg_Flag"])

        return df_deduped


    #  Handle `_Egg` Viruse
    def reconcile_egg_viruses(self, dataset_dict):
        """
        Processes `_Egg` viruses:
        - If both `Virus` and `Virus_Egg` exist, keep `Virus_Egg` data and remove `Virus`.
        - If only `Virus_Egg` exists, rename it to `Virus`.
        """
        for dataset_name, pivot_df in dataset_dict.items():
            egg_cols = [col for col in pivot_df.columns if col.endswith('_Egg')]

            for egg_col in egg_cols:
                normal_col = egg_col[:-4]  # Remove "_Egg"

                if normal_col in pivot_df.columns:
                    # If both exist, replace normal_col with egg_col data
                    pivot_df[normal_col] = pivot_df[egg_col]
                    pivot_df.drop(columns=[egg_col], inplace=True)
                else:
                    # If only egg_col exists, rename it
                    pivot_df.rename(columns={egg_col: normal_col}, inplace=True)

        return dataset_dict

    def build_day_dicts(self, filtered_data):
        """
        Returns two dictionaries:
          1) dataset_dict_day0
          2) dataset_dict_day21_40
        Each pivoted only for the relevant timepoints.
        """
        dataset_dict_day0 = {}
        dataset_dict_day21_40 = {}

        # Ensure 'Day_Number' column is present (for the day21_40 filter)
        filtered_data["Day_Number"] = filtered_data["Time"].str.extract(r"(\d+)").astype(float)

        unique_datasets = filtered_data["Dataset"].unique()
        for dataset_name in unique_datasets:
            subset_df = filtered_data[filtered_data["Dataset"] == dataset_name]

            # --- Day 0 ---
            day0_df = subset_df[subset_df["Time"] == "Day0"]
            if not day0_df.empty:
                pivot_day0 = day0_df.pivot_table(
                    index="Modified_Subject",
                    columns="Virus",
                    values="Transformed_HAI"
                )
                # Impute missing
                row_means = pivot_day0.apply(lambda row: row.mean(), axis=1)
                col_means = pivot_day0.apply(lambda col: col.mean(), axis=0)
                for row in pivot_day0.index:
                    for col in pivot_day0.columns:
                        if pd.isna(pivot_day0.loc[row, col]):
                            pivot_day0.loc[row, col] = (row_means[row] + col_means[col]) / 2

                dataset_dict_day0[dataset_name] = pivot_day0

            # --- Day 21–40 ---
            day21_40_df = subset_df[(subset_df["Day_Number"] >= 21) & (subset_df["Day_Number"] <= 40)]
            if not day21_40_df.empty:
                pivot_21_40 = day21_40_df.pivot_table(
                    index="Modified_Subject",
                    columns="Virus",
                    values="Transformed_HAI"
                )
                row_means = pivot_21_40.apply(lambda row: row.mean(), axis=1)
                col_means = pivot_21_40.apply(lambda col: col.mean(), axis=0)
                for row in pivot_21_40.index:
                    for col in pivot_21_40.columns:
                        if pd.isna(pivot_21_40.loc[row, col]):
                            pivot_21_40.loc[row, col] = (row_means[row] + col_means[col]) / 2

                dataset_dict_day21_40[dataset_name] = pivot_21_40

        # Drop Day_Number if you want
        if "Day_Number" in filtered_data.columns:
            filtered_data.drop(columns=["Day_Number"], inplace=True)

        return dataset_dict_day0, dataset_dict_day21_40

    def run(self):
        """
        Returns:
          filtered_data
          dataset_dict (all days)
          dataset_dict_day0
          dataset_dict_day21_40
        """
        data = self.load_and_combine_csv()
        filtered_data = self.preprocess_data(data)
         # 1) Make a “raw” pivot (no imputation) for reference
        #dataset_dict_raw = self.build_dataset_dict_raw(filtered_data)
        # Build pivot for ALL days
        dataset_dict = self.build_dataset_dict(filtered_data)
        dataset_dict = self.reconcile_egg_viruses(dataset_dict)

        # Build pivot for Day 0 only, Day 21–40 only
        dataset_dict_day0, dataset_dict_day21_40 = self.build_day_dicts(filtered_data)

        # Also apply reconcile_egg_viruses if you have any _Egg viruses in day0 or day21_40
        dataset_dict_day0 = self.reconcile_egg_viruses(dataset_dict_day0)
        dataset_dict_day21_40 = self.reconcile_egg_viruses(dataset_dict_day21_40)

        return filtered_data, dataset_dict, dataset_dict_day0, dataset_dict_day21_40

class MultiDS:
    """
    A wrapper around multiple pivoted DataFrames, each in a dictionary.
    Allows easy slicing, membership checks, and so on.
    """
    def __init__(self, dataset_dict):
        self.dataset_dict = dataset_dict
    
    def datasets(self):
        """
        Returns a list of available dataset names.
        """
        return list(self.dataset_dict.keys())
    
    def columns(self, dataset_name):
        """
        Returns a list of virus columns for the given dataset.
        """
        return self.dataset_dict[dataset_name].columns.tolist()

    def subjects(self, dataset_name):
        """
        Returns a list of subjects (row index) for the given dataset.
        """
        return self.dataset_dict[dataset_name].index.tolist()

    def get_dataset(self, dataset_name):
        """
        Returns the actual pivoted DataFrame for direct manipulation if desired.
        """
        return self.dataset_dict[dataset_name]

    def filter(self, dataset_name, viruses=None, subjects=None):
        """
        Returns a subset of the pivoted DataFrame from a single dataset:
         - viruses: list of viruses to keep (columns)
         - subjects: list of subjects to keep (index)
        """
        df = self.dataset_dict[dataset_name]

        # Filter by viruses if provided
        if viruses is not None:
            existing_viruses = [v for v in viruses if v in df.columns]
            df = df[existing_viruses]
        
        # Filter by subjects if provided
        if subjects is not None:
            df = df.loc[df.index.intersection(subjects)]
        
        return df
    def virus_counts(self, dataset_name):
        """
        Returns how many subjects (rows) have non-null data for each virus (column)
        in the specified dataset.

        Returns a Pandas Series, where the index = virus names,
        and the values = non-null counts.
        """
        df = self.dataset_dict[dataset_name]
        return df.notna().sum()
    def unique_viruses_across_datasets(self):
        """
        Gathers a set of all virus columns from all DataFrames in dataset_dict
        and returns it as a Python set.
        """
        virus_set = set()
        for name, df in self.dataset_dict.items():
            virus_set.update(df.columns)
        return virus_set
    



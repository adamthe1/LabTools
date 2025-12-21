import os
import pandas as pd
import numpy as np
import re
from .load_feature_df import get_body_system_column_names, load_feature_target_systems_as_df
from .temp_feature import create_merged_df_bundle
from ..utils.categorical_utils import CategoricalUtils


class PreprocessFeatures:
    def __init__(self, feature_systems: str, target_systems: str, confounders: list[str] = ['age', 'gender', 'bmi'], 
                 drop_threshold: float = 1, merge_closest_research_stage: bool = False, 
                 reference_system_dict: dict = None, use_cache: bool = True):
        """
        Initialize the PreprocessFeatures class
        :param feature_systems: name of feature system to preprocess (e.g. 'gait')
        :param target_systems: name of target system to preprocess (e.g. 'dxa')
        :param confounders: list of confounder column names (e.g. ['age', 'gender', 'bmi'])
        :param drop_threshold: threshold for dropping low variance columns
        :param merge_closest_research_stage: if True, match on closest research_stage when exact match unavailable
        :param reference_system_dict: dictionary of reference system to use if its a baseline preprocess
        :param use_cache: if True, use file-based caching; if False, load directly into memory
        """
        if feature_systems is None or target_systems is None:
            raise ValueError("features_dict and targets_dict must be provided")
        
        self.feature_system = feature_systems
        self.target_system = target_systems
        self.confounders = confounders
        self.drop_threshold = drop_threshold
        self.use_cache = use_cache
        
        # Get feature and target column names
        feature_columns = get_body_system_column_names(feature_systems)
        target_columns = get_body_system_column_names(target_systems)
        if confounders:
            # Add confounders to features if not already present
            feature_columns.extend([c for c in confounders if c not in feature_columns])
        
        self.features = feature_columns
        self.targets = target_columns
        
        if use_cache:
            # Use file-based caching via data_bundle
            self.data_bundle = create_merged_df_bundle(self.feature_system, self.target_system, self.confounders,
                                                       merge_closest_research_stage=merge_closest_research_stage)
            self.merged_df = None
        else:
            # Load directly into memory, no file caching
            self.merged_df = load_feature_target_systems_as_df(
                self.feature_system, self.target_system, self.confounders,
                merge_closest=merge_closest_research_stage
            )
            self.data_bundle = None
        

    

    def preprocess(self, target: str, filter_index: pd.Index = None):
        """
        Prepare x and y for the target.
        
        Args:
            target: The target column name to predict.
            filter_index: Optional index to filter data to. Use this to ensure
                          baseline runs use the same subjects as the full model.
        
        Returns:
            x: Feature DataFrame with MultiIndex (RegistrationCode, research_stage)
            y: Target DataFrame
        """
        # Get merged_df from cache file or in-memory
        if self.use_cache:
            filename = self.data_bundle['filename']
            if filename.endswith('.parquet'):
                merged_df = pd.read_parquet(filename)
            else:
                merged_df = pd.read_csv(filename, index_col=[0, 1])
        else:
            merged_df = self.merged_df
        
        # split dataframe into x and y
        # First extract y (the specific target being predicted)
        y = merged_df[[target]]
        # Then keep only feature columns in x (features + confounders, excluding all targets)
        # This prevents data leakage from the target system
        features_to_keep = [col for col in self.features if col in merged_df.columns and col != target]
        x = merged_df[features_to_keep]

        # replace infinite values with NaN
        x.replace([np.inf, -np.inf], np.nan, inplace=True)

        # drop low variance columns
        x = self.drop_low_variance_columns(x, max_proportion=self.drop_threshold)

        x = self.handle_confounders(x, target)

        # Drop rows where target has NaN
        valid_mask = y[target].notna()
        x = x.loc[valid_mask]
        y = y.loc[valid_mask]

        # Filter to specified index (for baseline runs to match full model subjects)
        if filter_index is not None:
            x, y = PreprocessFeatures.filter_index(x, y, filter_index)
            
        x = self.clean_column_names(x)
        y = self.clean_column_names(y)

        return x, y

    @staticmethod
    def filter_index(x, y, filter_index: pd.Index) -> pd.Index:
        # First: exact match on full MultiIndex (RegistrationCode, research_stage)
        common_index = x.index.intersection(filter_index)
        x_matched = x.loc[common_index]
        y_matched = y.loc[common_index]
        
        # Second: for unmatched RegistrationCodes, take last available research_stage
        matched_reg_codes = set(common_index.get_level_values(0))
        filter_reg_codes = set(filter_index.get_level_values(0))
        unmatched_reg_codes = filter_reg_codes - matched_reg_codes
        
        if unmatched_reg_codes:
            # Get last row per unmatched RegistrationCode (single filter + dedupe)
            mask = x.index.get_level_values(0).isin(unmatched_reg_codes)
            x_unmatched = x.loc[mask]
            # Keep last per subject using index reset trick (faster than groupby)
            keep_idx = ~x_unmatched.index.get_level_values(0).duplicated(keep='last')
            x_extra = x_unmatched.loc[keep_idx]
            y_extra = y.loc[x_extra.index]
            # Concatenate matched + extra
            x = pd.concat([x_matched, x_extra])
            y = pd.concat([y_matched, y_extra])
        else:
            x = x_matched
            y = y_matched
    
        return x, y


    @staticmethod
    def create_dummies(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert categorical/string columns to dummy variables.
        Uses CategoricalUtils to detect categorical columns, then applies pd.get_dummies.
        """
        cat_cols = CategoricalUtils.get_categorical_columns(df)
        # Filter to only object/string dtype columns (avoid dummifying numeric categoricals)
        string_cat_cols = [col for col in cat_cols if pd.api.types.is_object_dtype(df[col])]
        
        if not string_cat_cols:
            return df
        
        return pd.get_dummies(df, columns=string_cat_cols, drop_first=False, dtype=float)

    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the column names of a dataframe 
        Remove special characters and replace them with an underscore
        for lgbm compatibility
        """
        new_names = {col: re.sub(r"[^A-Za-z0-9_]+", "", col) for col in df.columns}

        # Handle duplicates by appending index
        name_list = list(new_names.values())
        final_names = {
            old: f"{new}_{name_list[:i].count(new)}" if name_list[:i].count(new) > 0 else new
            for i, (old, new) in enumerate(new_names.items())
        }

        return df.rename(columns=final_names)

    def handle_confounders(self, x: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Remove target from features if it's a confounder (prevents data leakage).
        Handles case-insensitive matching and cleaned column names.
        """
        target_lower = target.lower()
        target_cleaned = re.sub(r"[^A-Za-z0-9_]+", "", target).lower()
        
        cols_to_keep = []
        for col in x.columns:
            col_lower = col.lower()
            col_cleaned = re.sub(r"[^A-Za-z0-9_]+", "", col).lower()
            # Remove if exact match or cleaned match with target
            if col_lower == target_lower or col_cleaned == target_cleaned:
                continue
            cols_to_keep.append(col)
        
        return x[cols_to_keep]

    @staticmethod
    def drop_low_variance_columns(df: pd.DataFrame, max_proportion: float = 0.9) -> pd.DataFrame:
        """
        Drops columns from a DataFrame where the single most frequent value
        accounts for *at least* 'max_proportion' of the rows.

        Args:
            df: The input pandas DataFrame.
            max_proportion: The proportion threshold (e.g., 0.9 for 90%).
                            If the most frequent value's proportion is >= this
                            value, the column is dropped. A value of 1.0
                            will drop only columns with a single constant value.

        Returns:
            A new DataFrame with the low-variability columns removed.
        """
        if not (0 < max_proportion <= 1):
                raise ValueError("max_proportion must be between 0 (exclusive) and 1 (inclusive)")

        n_rows = len(df)
        if n_rows == 0:
                return df.copy()  # Return a copy if DataFrame is empty

        # List of columns to keep
        cols_to_keep = []

        for col in df.columns:
                # Calculate the count of the most frequent value.
                # We use dropna=False to treat NaN as a distinct value.
                counts = df[col].value_counts(dropna=False)

                # Get the count of the single most frequent value
                max_freq_count = counts.max()

                # Calculate its proportion
                proportion = max_freq_count / n_rows

                # We *keep* the column only if its max proportion is *less than* the threshold.
                # This means we *drop* it if proportion >= max_proportion.
                if proportion < max_proportion:
                        cols_to_keep.append(col)

        # Return a new DataFrame containing only the columns we want to keep
        return df[cols_to_keep]

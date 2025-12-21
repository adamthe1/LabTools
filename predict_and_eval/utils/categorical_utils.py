import json
import pandas as pd
from typing import List, Tuple
import os
from ..loading_features.load_feature_df import load_system_description_json

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from .env_loader import *  # Load .env with absolute path

# Cache for column descriptions JSON to avoid repeated file reads
_COLUMN_TYPES_CACHE = None

def _load_column_descriptions() -> dict:
    """Load column descriptions from JSON file (cached).
    Returns a dict of {column_name: "type"}
    Options are "regression", "categorical", "ordinal"
    """
    global _COLUMN_TYPES_CACHE
    if _COLUMN_TYPES_CACHE is not None:
        return _COLUMN_TYPES_CACHE
    
    systems_dict = load_system_description_json()
    column_types = {}
    for system, data in systems_dict.items():
        for column, info in data['columns'].items():
            column_types[column] = info['type']
    _COLUMN_TYPES_CACHE = column_types
    return column_types


class CategoricalUtils:
    """Utility functions for detecting and handling categorical columns."""
    
    @staticmethod
    def get_label_type(column: pd.Series, column_name: str = None, system_name: str = None) -> Literal["regression", "categorical", "ordinal"]:
        """
        Determine the prediction type for a target column.
        
        Checks JSON config first for explicit type, falls back to is_categorical() heuristic.
        
        Args:
            column: The target column data
            column_name: Column name to look up in JSON config (defaults to column.name)
            system_name: Body system name for JSON lookup (optional, searches all if None)
            
        Returns:
            "ordinal" if explicitly marked in JSON config
            "categorical" if is_categorical() returns True
            "regression" otherwise
        """
        col_name = column_name or column.name
        
        # Check JSON config for explicit type, fallback to heuristic if not found
        col_types = _load_column_descriptions()
        if col_types and col_name in col_types:
            label_type = col_types[col_name]
            # Validate the type from JSON
            if label_type in ("regression", "categorical", "ordinal"):
                return label_type
            # Invalid type in JSON, fall back to heuristic
            print(f"Warning: Invalid label_type '{label_type}' for column '{col_name}' in JSON, using heuristic")

        return CategoricalUtils.get_type(column)
   
    @staticmethod
    def is_categorical(column: pd.Series, unique_threshold: int = 10, ratio_threshold: float = 0.05) -> bool:
        """
        Determine if a column should be treated as categorical. or ordinal.
        Returns True if: object/category/bool dtype OR numeric with ≤unique_threshold unique values and ≤ratio_threshold uniqueness ratio.
        
        Args:
            column: The column to check
            unique_threshold: The maximum number of unique values for numeric columns to be considered categorical
            ratio_threshold: The maximum ratio of unique values to total non-null values for numeric columns
            
        Returns:
            True if the column should be treated as categorical, False otherwise
        """
        if len(column) == 0:
            return False
        
        dtype = column.dtype

        # Explicitly categorical types
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            return True
        
        # Numeric types with few unique values
        if pd.api.types.is_numeric_dtype(dtype):
            non_null = column.dropna()
            if len(non_null) == 0:
                return False
            n_unique = non_null.nunique()
            return n_unique <= unique_threshold and (n_unique / len(non_null)) <= ratio_threshold
        
        return False
    
    @staticmethod
    def get_type(column: pd.Series, unique_threshold: int = 10, ratio_threshold: float = 0.05) -> Literal["regression", "categorical", "ordinal"]:
        """
        Infer column type via heuristic (use get_label_type() if JSON config available).
        Ordinal: numeric with few unique sequential integers (e.g., 1,2,3,4,5).
        Categorical: non-numeric or numeric with non-sequential/non-integer values.
        """
        if not CategoricalUtils.is_categorical(column, unique_threshold, ratio_threshold):
            return "regression"
        
        # Distinguish ordinal from categorical
        non_null = column.dropna()
        if len(non_null) == 0:
            return "categorical"
        
        # Ordinal heuristic: integers that form a near-consecutive sequence (like Likert scales)
        if pd.api.types.is_numeric_dtype(column.dtype):
            unique_vals = sorted(non_null.unique())
            # Check if all values are integers (or float representations of integers)
            if all(float(v).is_integer() for v in unique_vals) and len(unique_vals) > 2:
                int_vals = [int(v) for v in unique_vals]
                # Sequential if range equals count (e.g., [1,2,3,4] or [0,1,2,3])
                if max(int_vals) - min(int_vals) + 1 == len(int_vals):
                    return "ordinal"
        
        return "categorical"
    
    @staticmethod
    def get_categorical_columns(df: pd.DataFrame, unique_threshold: int = 10, ratio_threshold: float = 0.05) -> List[str]:
        """
        Get list of categorical column names in a DataFrame.
        
        Args:
            df: The input pandas DataFrame
            unique_threshold: Maximum number of unique values for numeric columns to be considered categorical
            ratio_threshold: Maximum ratio of unique values to total non-null values for numeric columns
            
        Returns:
            List of column names that are categorical
        """
        return [col for col in df.columns if CategoricalUtils.get_type(df[col], unique_threshold, ratio_threshold) == "categorical"]
    
    @staticmethod
    def detect_categorical_and_numeric_columns(df: pd.DataFrame, unique_threshold: int = 10, ratio_threshold: float = 0.05) -> Tuple[List[int], List[int]]:
        """
        Detect categorical and numeric column indices in a DataFrame.
        
        Args:
            df: Input DataFrame
            unique_threshold: Max unique values for numeric columns to be considered categorical
            ratio_threshold: Max ratio of unique/total values for numeric columns to be categorical
        
        Returns:
            Tuple of (categorical_indices, numeric_indices) for use with numpy arrays
        """
        categorical_idx = []
        numeric_idx = []
        
        for i, col in enumerate(df.columns):
            if CategoricalUtils.get_type(df[col], unique_threshold, ratio_threshold) == "categorical" and df[col].nunique() > 2:
                categorical_idx.append(i)
            else:
                numeric_idx.append(i)
        
        return categorical_idx, numeric_idx


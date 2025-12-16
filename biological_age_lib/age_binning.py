"""
Age Binning Analysis

Functions for stratifying subjects by age bins and identifying extreme percentiles
within each bin for biological age analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class AgeBinAnalyzer:
    """
    Analyzer for age-stratified prediction analysis.
    
    Groups subjects into age bins and identifies top/bottom percentiles
    based on prediction residuals within each bin.
    """
    
    def __init__(self, 
                 min_age: int = 40,
                 max_age: int = 72,
                 bin_width: int = 2,
                 percentile: float = 0.25):
        """
        Initialize the analyzer.
        
        Args:
            min_age: Minimum age for binning
            max_age: Maximum age for binning
            bin_width: Width of each age bin in years
            percentile: Fraction for top/bottom percentile (e.g., 0.25 for 25%)
        """
        self.min_age = min_age
        self.max_age = max_age
        self.bin_width = bin_width
        self.percentile = percentile
    
    def analyze(self, predictions_df: pd. DataFrame) -> Dict:
        """
        Analyze predictions by age bins.
        
        Args:
            predictions_df: DataFrame with columns 'true_values', 'predictions',
                        'subject_number', 'visit_number' (optional), 'visit_priority' (optional)
        
        Returns:
            Dictionary with:
                - 'age_bins_dict': {bin_name: [indices]}
                - 'top_by_bin': {bin_name: [top percentile indices]}
                - 'bottom_by_bin': {bin_name: [bottom percentile indices]}
                - 'aggregated_top': all top percentile indices
                - 'aggregated_bottom': all bottom percentile indices
                - 'filtered_df': cleaned DataFrame used for analysis
        """
        # Clean and filter data
        df = self._prepare_data(predictions_df)
        
        # Initialize dictionaries
        age_bins_dict = {}
        top_by_bin = {}
        bottom_by_bin = {}
        aggregated_top = []
        aggregated_bottom = []
        
        # Create bin labels
        bin_labels = []
        for bin_start in range(self.min_age, self.max_age, self.bin_width):
            bin_end = bin_start + self. bin_width - 1
            bin_name = f"{bin_start}-{bin_end}"
            bin_labels.append(bin_name)
            age_bins_dict[bin_name] = []
            top_by_bin[bin_name] = []
            bottom_by_bin[bin_name] = []
        
        # Assign rows to bins based on true age
        for idx, row in df. iterrows():
            age = row['true_values']
            
            # properly calculate bin_start by flooring to nearest bin
            bin_start = self. min_age + ((int(age) - self.min_age) // self.bin_width) * self.bin_width
            bin_end = bin_start + self.bin_width - 1
            bin_name = f"{bin_start}-{bin_end}"
            
            if bin_name in age_bins_dict:
                age_bins_dict[bin_name].append(idx)
            else:
                # Age is outside the defined range (below min_age or >= max_age)
                print(f"Age {age} outside range [{self.min_age}, {self. max_age}), bin '{bin_name}' not found")
        
        # Compute top and bottom percentile per bin
        for bin_name in bin_labels:
            bin_indices = age_bins_dict[bin_name]
            if len(bin_indices) == 0:
                continue
            
            bin_df = df.loc[bin_indices]
            
            if len(bin_df) < 5:
                # Too few samples for percentiles - take min/max
                min_idx = bin_df['predictions'].idxmin()
                max_idx = bin_df['predictions'].idxmax()
                bottom_by_bin[bin_name] = [min_idx]
                top_by_bin[bin_name] = [max_idx]
            else:
                bottom_thr = bin_df['predictions'].quantile(self.percentile)
                top_thr = bin_df['predictions'].quantile(1 - self.percentile)
                bottom_ids = bin_df[bin_df['predictions'] <= bottom_thr].index.tolist()
                top_ids = bin_df[bin_df['predictions'] >= top_thr].index.tolist()
                bottom_by_bin[bin_name] = bottom_ids
                top_by_bin[bin_name] = top_ids
            
            aggregated_top.extend(top_by_bin[bin_name])
            aggregated_bottom.extend(bottom_by_bin[bin_name])
        
        return {
            'age_bins_dict': age_bins_dict,
            'top_by_bin': top_by_bin,
            'bottom_by_bin': bottom_by_bin,
            'aggregated_top': aggregated_top,
            'aggregated_bottom': aggregated_bottom,
            'filtered_df': df
        }
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate required columns exist."""
        required_cols = ['true_values', 'predictions']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        return df.dropna(subset=required_cols)
    
    def print_summary(self, results: Dict) -> None:
        """Print summary of age bin analysis."""
        age_bins = results['age_bins_dict']
        top_by_bin = results['top_by_bin']
        bottom_by_bin = results['bottom_by_bin']
        
        n_bins = len([b for b in age_bins.keys() if age_bins[b]])
        print(f"\nNumber of age bins: {n_bins}")
        print(f"\nDistribution across age bins:")
        
        percent = self.percentile * 100
        for bin_name in sorted(age_bins.keys()):
            total = len(age_bins[bin_name])
            top_n = len(top_by_bin[bin_name])
            bottom_n = len(bottom_by_bin[bin_name])
            if total > 0:
                print(f"  {bin_name}: {total} participants, "
                      f"Top {percent}%: {top_n}, Bottom {percent}%: {bottom_n}")
        
        print(f"\nTotal in aggregated top {percent}%: {len(results['aggregated_top'])}")
        print(f"Total in aggregated bottom {percent}%: {len(results['aggregated_bottom'])}")
    


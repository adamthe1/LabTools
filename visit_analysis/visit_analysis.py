"""
Visit Analysis

Longitudinal analysis of predictions and biomarkers across visits.
Computes visit differences and correlations between changes.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from typing import List, Tuple, Optional


class VisitAnalyzer:
    """
    Analyzer for longitudinal visit data.
    
    Computes differences between visits and correlations between
    prediction changes and biomarker changes.
    """
    
    DEFAULT_VISIT_PAIRS = [
        ('02_00_visit', 'baseline'),
        ('04_00_visit', 'baseline'),
        ('04_00_visit', '02_00_visit')
    ]
    
    def __init__(self,
                 subject_col: str = 'subject_number',
                 visit_col: str = 'visit_number',
                 visit_pairs: List[Tuple[str, str]] = None):
        """
        Initialize the analyzer.
        
        Args:
            subject_col: Column name for subject identifier
            visit_col: Column name for visit identifier
            visit_pairs: List of (later_visit, earlier_visit) pairs to compare
        """
        self.subject_col = subject_col
        self.visit_col = visit_col
        self.visit_pairs = visit_pairs or self.DEFAULT_VISIT_PAIRS
    
    def compute_differences(self,
                            df: pd.DataFrame,
                            features: List[str]) -> pd.DataFrame:
        """
        Compute differences between visits for specified features.
        
        Args:
            df: DataFrame with visit data
            features: List of feature columns to compute differences for
        
        Returns:
            Long-format DataFrame with computed differences
        """
        # Validate features
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"Features not found: {missing}")
        
        results = []
        
        for later_visit, earlier_visit in self.visit_pairs:
            later_data = df[df[self.visit_col] == later_visit].set_index(self.subject_col)[features]
            earlier_data = df[df[self.visit_col] == earlier_visit].set_index(self.subject_col)[features]
            
            # Find common subjects
            common = later_data.index.intersection(earlier_data.index)
            
            if len(common) == 0:
                print(f"Warning: No common subjects for {later_visit} vs {earlier_visit}")
                continue
            
            # Compute differences
            diff_data = later_data.loc[common] - earlier_data.loc[common]
            diff_data['comparison'] = f"{later_visit}_minus_{earlier_visit}"
            diff_data['n_subjects'] = len(common)
            diff_data = diff_data.reset_index()
            
            results.append(diff_data)
        
        if not results:
            return pd.DataFrame()
        
        return pd.concat(results, ignore_index=True)
    
    def compute_differences_wide(self,
                                  df: pd.DataFrame,
                                  features: List[str]) -> pd.DataFrame:
        """
        Compute differences in wide format (one row per subject).
        
        Args:
            df: DataFrame with visit data
            features: List of feature columns
        
        Returns:
            Wide-format DataFrame with columns for each feature-comparison combination
        """
        all_subjects = df[self.subject_col].unique()
        result_df = pd.DataFrame({self.subject_col: all_subjects})
        result_df = result_df.set_index(self.subject_col)
        
        for later_visit, earlier_visit in self.visit_pairs:
            later_data = df[df[self.visit_col] == later_visit].set_index(self.subject_col)[features]
            earlier_data = df[df[self.visit_col] == earlier_visit].set_index(self.subject_col)[features]
            
            diff_data = later_data - earlier_data
            
            # Rename columns with comparison suffix
            comparison_name = f"{later_visit}_minus_{earlier_visit}"
            diff_data.columns = [f"{col}_{comparison_name}" for col in diff_data.columns]
            
            result_df = result_df.join(diff_data, how='left')
        
        return result_df.reset_index()
    
    def compute_correlations(self,
                             df: pd.DataFrame,
                             target_feature: str,
                             method: str = 'pearson',
                             min_observations: int = 10,
                             filter_pattern: str = None) -> pd.DataFrame:
        """
        Compute correlation of one feature to all others.
        
        Args:
            df: DataFrame with features
            target_feature: Feature to correlate against all others
            method: 'pearson' or 'spearman'
            min_observations: Minimum non-NaN pairs required
            filter_pattern: Optional pattern to filter feature names
        
        Returns:
            DataFrame with correlation results sorted by absolute correlation
        """
        if target_feature not in df.columns:
            raise ValueError(f"Target feature '{target_feature}' not found")
        
        # Extract pattern from target if not provided
        if filter_pattern is None:
            filter_pattern = self._extract_pattern(target_feature)
        
        results = []
        target_series = df[target_feature]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter columns by pattern
        if filter_pattern:
            filtered_cols = [c for c in numeric_cols if filter_pattern in c]
        else:
            filtered_cols = numeric_cols
        
        for feature in filtered_cols:
            if feature == target_feature:
                continue
            
            feature_series = df[feature]
            
            # Find valid pairs
            valid_mask = ~(pd.isna(target_series) | pd.isna(feature_series))
            valid_target = target_series[valid_mask]
            valid_feature = feature_series[valid_mask]
            
            n_obs = len(valid_target)
            
            if n_obs < min_observations:
                correlation, p_value = np.nan, np.nan
            else:
                if method.lower() == 'pearson':
                    correlation, p_value = pearsonr(valid_target, valid_feature)
                else:
                    correlation, p_value = spearmanr(valid_target, valid_feature)
            
            results.append({
                'feature': feature,
                'correlation': correlation,
                'p_value': p_value,
                'n_observations': n_obs
            })
        
        results_df = pd.DataFrame(results)
        results_df['abs_correlation'] = abs(results_df['correlation'])
        results_df = results_df.sort_values('abs_correlation', ascending=False)
        results_df = results_df.drop('abs_correlation', axis=1)
        
        return results_df.reset_index(drop=True)
    
    def _extract_pattern(self, feature_name: str) -> Optional[str]:
        """Extract visit comparison pattern from feature name."""
        parts = feature_name.split('_')
        
        # Look for pattern like "XX_XX_visit_minus_baseline"
        for i in range(len(parts) - 3):
            if parts[i+2] == 'visit' and parts[i+3] == 'minus' and len(parts) > i+4:
                return '_'.join(parts[i:])
        
        # Fallback: take last 4 parts
        if len(parts) >= 4:
            return '_'.join(parts[-4:])
        
        return None
    
    def get_significant_changes(self,
                                 differences_df: pd.DataFrame,
                                 features: List[str],
                                 threshold_percentile: float = 0.9) -> pd.DataFrame:
        """
        Identify subjects with significant changes.
        
        Args:
            differences_df: Output from compute_differences()
            features: Features to analyze
            threshold_percentile: Percentile threshold for "significant" change
        
        Returns:
            DataFrame with subjects showing large changes
        """
        significant_changes = []
        
        for comparison in differences_df['comparison'].unique():
            comp_data = differences_df[differences_df['comparison'] == comparison]
            
            for feature in features:
                if feature not in comp_data.columns:
                    continue
                
                abs_changes = comp_data[feature].abs()
                threshold = abs_changes.quantile(threshold_percentile)
                significant_subjects = comp_data[abs_changes > threshold][self.subject_col].tolist()
                
                if significant_subjects:
                    significant_changes.append({
                        'comparison': comparison,
                        'feature': feature,
                        'subjects_with_large_changes': significant_subjects,
                        'n_subjects': len(significant_subjects),
                        'threshold': threshold
                    })
        
        return pd.DataFrame(significant_changes)


def compute_visit_differences(df: pd.DataFrame,
                               features: List[str],
                               visit_pairs: List[Tuple[str, str]] = None,
                               subject_col: str = 'subject_number',
                               visit_col: str = 'visit_number') -> pd.DataFrame:
    """
    Convenience function to compute visit differences.
    
    Args:
        df: DataFrame with visit data
        features: List of feature columns
        visit_pairs: List of (later, earlier) visit pairs
        subject_col: Subject column name
        visit_col: Visit column name
    
    Returns:
        Long-format DataFrame with differences
    """
    analyzer = VisitAnalyzer(
        subject_col=subject_col,
        visit_col=visit_col,
        visit_pairs=visit_pairs
    )
    return analyzer.compute_differences(df, features)


def compute_visit_differences_wide(df: pd.DataFrame,
                                    features: List[str],
                                    visit_pairs: List[Tuple[str, str]] = None,
                                    subject_col: str = 'subject_number',
                                    visit_col: str = 'visit_number') -> pd.DataFrame:
    """
    Convenience function to compute visit differences in wide format.
    
    Args:
        df: DataFrame with visit data
        features: List of feature columns
        visit_pairs: List of (later, earlier) visit pairs
        subject_col: Subject column name
        visit_col: Visit column name
    
    Returns:
        Wide-format DataFrame with differences
    """
    analyzer = VisitAnalyzer(
        subject_col=subject_col,
        visit_col=visit_col,
        visit_pairs=visit_pairs
    )
    return analyzer.compute_differences_wide(df, features)


def compute_feature_correlations(df: pd.DataFrame,
                                  target_feature: str,
                                  method: str = 'pearson',
                                  min_observations: int = 10,
                                  filter_pattern: str = None) -> pd.DataFrame:
    """
    Convenience function to compute feature correlations.
    
    Args:
        df: DataFrame with features
        target_feature: Feature to correlate against
        method: 'pearson' or 'spearman'
        min_observations: Minimum observations required
        filter_pattern: Optional pattern to filter features
    
    Returns:
        DataFrame with correlation results
    """
    analyzer = VisitAnalyzer()
    return analyzer.compute_correlations(
        df, target_feature, method=method,
        min_observations=min_observations,
        filter_pattern=filter_pattern
    )


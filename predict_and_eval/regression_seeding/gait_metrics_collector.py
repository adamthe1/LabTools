"""
GaitMetricsCollector class for evaluating gait predictions across multiple analysis subsets.

Splits raw predictions into:
- per activity: filtered by activity, averaged by (RegistrationCode, research_stage)  
- ensemble: all data averaged by (RegistrationCode, research_stage) (saved last)
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from ..utils.evaluate_predictions import (
    evaluate_regression, evaluate_regression_with_gender_split,
    evaluate_classification, evaluate_classification_with_gender_split,
    evaluate_ordinal, evaluate_ordinal_with_gender_split
)


class GaitMetricsCollector:
    """
    Collects raw predictions and evaluates across multiple analysis subsets.
    
    Each subset produces its own predictions.csv and metrics.csv.
    """
    
    def __init__(self, label_type: str, seeds: range, model_key: str = None):
        """
        Args:
            label_type: 'regression', 'categorical', or 'ordinal'
            seeds: Range of seeds for metrics collection
            model_key: Name of the model used (e.g., 'LGBM_regression')
        """
        self.label_type = label_type
        self.seeds = list(seeds)
        self.model_key = model_key
        self.raw_predictions = []  # List of DataFrames, one per seed
    
    def add_seed_predictions(self, seed: int, x: pd.DataFrame, y: pd.DataFrame, 
                             predictions: np.ndarray) -> None:
        """
        Add raw predictions from a single seed.
        
        Args:
            seed: Seed index
            x: Features DataFrame with activity, seq_idx, gender columns
            y: Target DataFrame
            predictions: Raw predictions array (before averaging)
        """
        raw_df = pd.DataFrame({
            'y_true': y.values.flatten(),
            'y_pred': predictions if predictions.ndim == 1 else predictions[:, 1],
            'activity': x['activity'].values if 'activity' in x.columns else 'unknown',
            'seq_idx': x['seq_idx'].values if 'seq_idx' in x.columns else 0,
            'gender': x['gender'].values if 'gender' in x.columns else np.nan,
            'seed': seed
        }, index=x.index)
        
        # Store full proba for classification if 2D
        if predictions.ndim == 2:
            raw_df['y_pred_proba'] = list(predictions)
        
        self.raw_predictions.append(raw_df)
    
    def evaluate_all_subsets(self, gender_split: bool = True, include_terciles: bool = False) -> Dict[str, Dict]:
        """
        Evaluate all analysis subsets.
        
        Args:
            gender_split: Whether to evaluate separately by gender
            include_terciles: Whether to include sequence tercile splits (first33/mid33/last33)
        
        Returns:
            Dict of {subset_name: {'predictions': df, 'metrics': df}}
        """
        if not self.raw_predictions:
            raise ValueError("No predictions added yet")
        
        combined = pd.concat(self.raw_predictions, ignore_index=False)
        results = {}
        
        # 1. Per activity
        activities = combined['activity'].unique()
        for activity in activities:
            subset = combined[combined['activity'] == activity]
            if len(subset) > 0:
                results[f'activity_{activity}'] = self._evaluate_subset(subset, gender_split)
                
                # Optional: per activity + sequence tercile
                if include_terciles:
                    subset_with_terciles = self._add_seq_terciles_subjectwise_nan(subset)
                    for tercile in ['first33', 'mid33', 'last33']:
                        sub_tercile = subset_with_terciles[subset_with_terciles['seq_tercile'] == tercile]
                        if len(sub_tercile) > 0:
                            name = f'activity_{activity}_{tercile}'
                            results[name] = self._evaluate_subset(sub_tercile, gender_split)
        
        # 2. Ensemble (all data) - saved last
        results['ensemble'] = self._evaluate_subset(combined, gender_split)
        
        return results
    
    def _add_seq_terciles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign first/mid/last 33% per subject based on seq_idx."""
        df = df.copy()
        
        def assign_tercile(s):
            if len(s) < 3:
                # Not enough sequences, assign all to 'mid33'
                return pd.Series(['mid33'] * len(s), index=s.index)
            try:
                return pd.qcut(s.rank(method='first'), 3, labels=['first33', 'mid33', 'last33'])
            except ValueError:
                # If qcut fails (e.g., too few unique values), fall back
                return pd.Series(['mid33'] * len(s), index=s.index)
        
        df['seq_tercile'] = df.groupby(level=0)['seq_idx'].transform(assign_tercile)
        return df
    
    def _add_seq_terciles_subjectwise_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign terciles per subject from unique seq_idx.
        Subjects with <3 unique seq_idx -> tercile = NaN (for all their rows).
        Works when RegistrationCode is only in the index (e.g., MultiIndex).
        Restores the original index before returning.
        """
        df_in = df.copy()
        if 'seq_idx' not in df_in.columns:
            df_in['seq_idx'] = 0

        # Original index info to restore later
        original_index_names = df_in.index.names
        subject_name = original_index_names[0] or 'RegistrationCode'

        # Work in a flat table with subject as a column
        flat = df_in.reset_index()  # brings index levels to columns
        flat['seq_idx'] = pd.to_numeric(flat['seq_idx'], errors='coerce')

        # Unique (subject, seq_idx) (dedup across seeds/rows)
        uniq = (
            flat[[subject_name, 'seq_idx']]
            .dropna()
            .drop_duplicates([subject_name, 'seq_idx'])
            .sort_values([subject_name, 'seq_idx'])
            .reset_index(drop=True)
        )

        # Assign terciles per subject on unique rows
        def assign_per_subject(g: pd.DataFrame) -> pd.DataFrame:
            if g['seq_idx'].nunique() < 3:
                g = g.copy()
                g['tercile'] = np.nan
                return g
            r = g['seq_idx'].rank(method='first')
            try:
                g = g.copy()
                g['tercile'] = pd.qcut(r, 3, labels=['first33', 'mid33', 'last33'])
            except ValueError:
                g['tercile'] = np.nan
            return g

        map_df = (
            uniq.groupby(subject_name, group_keys=False)
                .apply(assign_per_subject)
                .reset_index(drop=True)
        )

        # Merge terciles back to the full flat table
        merged = flat.merge(
            map_df[[subject_name, 'seq_idx', 'tercile']],
            on=[subject_name, 'seq_idx'],
            how='left'
        )

        # Rename to target column name and restore original index
        merged = merged.rename(columns={'tercile': 'seq_tercile'})
        df_out = merged.set_index(original_index_names)  # exact same index as input
        # Keep original column order + new column at the end
        if 'seq_tercile' not in df_in.columns:
            return df_out
        # If the column existed, overwrite in-place to preserve column order
        df_in['seq_tercile'] = df_out['seq_tercile']
        return df_in
    
    def _evaluate_subset(self, df: pd.DataFrame, gender_split: bool) -> Dict:
        """
        Average by (RegistrationCode, research_stage) per seed, then evaluate each seed.
        
        Returns:
            {'predictions': DataFrame, 'metrics': DataFrame}
        """
        # Group by (RegistrationCode, research_stage, seed), average predictions
        grouped = df.groupby([df.index.get_level_values(0), df.index.get_level_values(1), 'seed']).agg({
            'y_true': 'mean',
            'y_pred': 'mean',
            'gender': 'first'
        }).reset_index()
        grouped.columns = ['RegistrationCode', 'research_stage', 'seed', 'y_true', 'y_pred', 'gender']
        
        # Evaluate per seed and collect metrics
        all_metrics = []
        for seed in self.seeds:
            seed_data = grouped[grouped['seed'] == seed]
            if len(seed_data) == 0:
                continue
            
            y_true = seed_data['y_true'].values
            y_pred = seed_data['y_pred'].values
            gender = seed_data['gender'].values
            # Count unique (RegistrationCode, research_stage) pairs
            n_subjects = len(seed_data[['RegistrationCode', 'research_stage']].drop_duplicates())
            
            metrics = self._compute_metrics(y_true, y_pred, gender, gender_split)
            metrics['n_subjects'] = n_subjects
            metrics['seed'] = seed
            metrics['model_key'] = self.model_key
            all_metrics.append(metrics)
        
        # Reorder columns to match MetricsCollector format exactly:
        # seed, model_key, main_metrics, n_subjects, male_metrics, female_metrics
        metrics_df = pd.DataFrame(all_metrics)
        main_cols = [c for c in metrics_df.columns if not c.startswith(('male_', 'female_')) and c not in ['seed', 'model_key', 'n_subjects']]
        male_cols = [c for c in metrics_df.columns if c.startswith('male_')]
        female_cols = [c for c in metrics_df.columns if c.startswith('female_')]
        cols = ['seed', 'model_key'] + main_cols + ['n_subjects'] + male_cols + female_cols
        metrics_df = metrics_df[[c for c in cols if c in metrics_df.columns]]
        
        return {
            'predictions': grouped,
            'metrics': metrics_df
        }
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         gender: np.ndarray, gender_split: bool) -> Dict[str, Any]:
        """Call appropriate evaluate function based on label_type."""
        if self.label_type == 'regression':
            if gender_split and not np.all(np.isnan(gender)):
                return evaluate_regression_with_gender_split(gender, y_true, y_pred)
            return evaluate_regression(y_true, y_pred)
        
        elif self.label_type == 'categorical':
            # For classification, y_pred is probability, need to threshold
            y_pred_class = (y_pred > 0.5).astype(int)
            if gender_split and not np.all(np.isnan(gender)):
                return evaluate_classification_with_gender_split(gender, y_true, y_pred_class, y_pred)
            return evaluate_classification(y_true, y_pred_class, y_pred)
        
        elif self.label_type == 'ordinal':
            if gender_split and not np.all(np.isnan(gender)):
                return evaluate_ordinal_with_gender_split(gender, y_true, y_pred)
            return evaluate_ordinal(y_true, y_pred)
        
        else:
            raise ValueError(f"Unknown label_type: {self.label_type}")
    
    def save_all(self, base_dir: str, original_save_dir: str = None, folds: list = None) -> None:
        """
        Save all subset results to separate directories.
        
        Args:
            base_dir: Parent directory for all subset directories
            original_save_dir: Original save directory (used to derive {feature}_main folder name)
            folds: Optional folds to save (only saved in ensemble dir)
        """
        from ..utils.ids_folds import save_folds
        
        results = self.evaluate_all_subsets(gender_split=True)
        
        for subset_name, subset_results in results.items():
            subset_dir = os.path.join(base_dir, subset_name)
            os.makedirs(subset_dir, exist_ok=True)
            
            # Save predictions with no index, columns: RegistrationCode, research_stage, seed, y_true, y_pred, gender
            subset_results['predictions'].to_csv(os.path.join(subset_dir, 'predictions.csv'), index=False)
            subset_results['metrics'].to_csv(os.path.join(subset_dir, 'metrics.csv'))
            
            # Save folds only in ensemble directory
            if subset_name == 'ensemble' and folds is not None:
                save_folds(folds, subset_dir)
        
        # Save raw predictions as csv for later analysis
        combined = pd.concat(self.raw_predictions, ignore_index=False)
        combined.to_csv(os.path.join(base_dir, 'raw_predictions.csv'))
        
        # Save raw (no averaging) metrics to {feature}_main subfolder
        if original_save_dir is not None:
            feature_name = os.path.basename(original_save_dir.rstrip('/'))
            main_dir = os.path.join(base_dir, f'{feature_name}_main')
            os.makedirs(main_dir, exist_ok=True)
            raw_results = self._evaluate_raw_no_averaging()
            raw_results['predictions'].to_csv(os.path.join(main_dir, 'predictions.csv'))
            raw_results['metrics'].to_csv(os.path.join(main_dir, 'metrics.csv'))
            if folds is not None:
                save_folds(folds, main_dir)
    
    def _evaluate_raw_no_averaging(self) -> Dict:
        """
        Evaluate raw predictions without averaging by subject.
        
        Returns predictions in same format as MetricsCollector:
        - predictions.csv: id_research_pairs + seed_0, seed_1, ... columns
        - metrics.csv: seed, model_key, + all metrics
        """
        combined = pd.concat(self.raw_predictions, ignore_index=False)
        
        # Build predictions DataFrame in MetricsCollector format
        # Pivot: each seed becomes a column (seed_0, seed_1, ...)
        first_seed_data = combined[combined['seed'] == self.seeds[0]]
        # Extract RegistrationCode and research_stage from MultiIndex
        predictions_df = pd.DataFrame({
            'RegistrationCode': first_seed_data.index.get_level_values(0),
            'research_stage': first_seed_data.index.get_level_values(1)
        })
        
        for seed in self.seeds:
            seed_data = combined[combined['seed'] == seed]
            predictions_df[f'seed_{seed}'] = seed_data['y_pred'].values
        
        # Build metrics DataFrame
        all_metrics = []
        for seed in self.seeds:
            seed_data = combined[combined['seed'] == seed]
            if len(seed_data) == 0:
                continue
            
            y_true = seed_data['y_true'].values
            y_pred = seed_data['y_pred'].values
            gender = seed_data['gender'].values
            n_subjects = seed_data.index.get_level_values(0).nunique()
            
            metrics = self._compute_metrics(y_true, y_pred, gender, gender_split=True)
            metrics['n_subjects'] = n_subjects
            metrics['seed'] = seed
            metrics['model_key'] = self.model_key
            all_metrics.append(metrics)
        
        # Reorder columns to match MetricsCollector format exactly
        metrics_df = pd.DataFrame(all_metrics)
        main_cols = [c for c in metrics_df.columns if not c.startswith(('male_', 'female_')) and c not in ['seed', 'model_key', 'n_subjects']]
        male_cols = [c for c in metrics_df.columns if c.startswith('male_')]
        female_cols = [c for c in metrics_df.columns if c.startswith('female_')]
        cols = ['seed', 'model_key'] + main_cols + ['n_subjects'] + male_cols + female_cols
        metrics_df = metrics_df[[c for c in cols if c in metrics_df.columns]]
        
        return {
            'predictions': predictions_df,
            'metrics': metrics_df
        }


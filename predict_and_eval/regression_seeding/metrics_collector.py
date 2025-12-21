"""
MetricsCollector class for organizing and collecting metrics across multiple seeds.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any


class MetricsCollector:
    """
    Collects and organizes predictions and metrics across multiple CV seeds.
    
    Usage:
        collector = MetricsCollector(seeds=range(10))
        
        for seed in seeds:
            # ... run cross-validation ...
            collector.add_seed_results(
                seed=seed,
                predictions=predictions,
                id_research_pairs=id_research_pairs,
                metrics=metrics_dict,
                score=score,
                p_value=p_value
            )
        
        results = collector.get_results()
    """
    
    def __init__(self, seeds: range, model_key: str = None):
        """
        Initialize the metrics collector.
        
        Args:
            seeds: Range of seeds to collect metrics for
            model_key: Name of the model used (e.g., 'LGBM_regression')
        """
        self.seeds = list(seeds)
        self.model_key = model_key
        self.all_predictions = {"id_research_pairs": None}
        self.all_metrics = {}
        self._first_seed = True

        
    def add_seed_results(self,
                        seed: int,
                        predictions: np.ndarray,
                        id_research_pairs: pd.MultiIndex,
                        metrics: Dict[str, float],
                        metrics_male: Dict[str, float] = None,
                        metrics_female: Dict[str, float] = None,
                        true_values: np.ndarray = None) -> None:
        """
        Add results from a single seed iteration.
        
        Args:
            seed: Seed number
            predictions: Prediction array
            id_research_pairs: MultiIndex of (RegistrationCode, research_stage) pairs
            metrics: General metrics dictionary
            metrics_male: Male-specific metrics dictionary (optional)
            metrics_female: Female-specific metrics dictionary (optional)
            true_values: True target values (possibly averaged by subject)
        """
        # Handle None values
        if metrics_male is None:
            metrics_male = {}
        if metrics_female is None:
            metrics_female = {}
        # Store predictions
        self.all_predictions[f"seed_{seed}"] = predictions
        
        # Store id_research_pairs (should be same across seeds)
        if self.all_predictions["id_research_pairs"] is None:
            self.all_predictions["id_research_pairs"] = id_research_pairs
        else:
            # Verify consistency
            if not np.all(self.all_predictions["id_research_pairs"] == id_research_pairs):
                raise ValueError(f"Seed {seed}: id_research_pairs are not consistent across seeds")
        
        # Store true_values (should be same across seeds, only store once)
        if true_values is not None and "true_values" not in self.all_predictions:
            self.all_predictions["true_values"] = true_values
        
        # Store metrics
        if self._first_seed:
            # First seed: initialize metrics as lists
            self.all_metrics = {
                **{key: [value] for key, value in metrics.items()},
                **{key: [value] for key, value in metrics_male.items()},
                **{key: [value] for key, value in metrics_female.items()}
            }
            self._first_seed = False
        else:
            # Subsequent seeds: append to existing lists
            for key, value in metrics.items():
                if key not in self.all_metrics:
                    raise KeyError(f"Seed {seed}: Metric '{key}' not found in first seed results")
                self.all_metrics[key].append(value)
            
            for key, value in metrics_male.items():
                if key not in self.all_metrics:
                    raise KeyError(f"Seed {seed}: Metric '{key}' not found in first seed results")
                self.all_metrics[key].append(value)
            
            for key, value in metrics_female.items():
                if key not in self.all_metrics:
                    raise KeyError(f"Seed {seed}: Metric '{key}' not found in first seed results")
                self.all_metrics[key].append(value)

    
    def get_results(self) -> Dict[str, pd.DataFrame]:
        """
        Get collected results as DataFrames.
        
        Returns:
            Dictionary with:
                - 'all_predictions': DataFrame with predictions for each seed + id_research_pairs
                - 'scores_pvalues': DataFrame with scores, pvalues, and all metrics per seed
        """
        if self._first_seed:
            raise ValueError("No results have been added yet")
        
        # Flatten 2D predictions (classification proba) to 1D (positive class proba)
        predictions_flat = {}
        id_research_pairs = None
        for key, val in self.all_predictions.items():
            if key == 'id_research_pairs':
                id_research_pairs = val  # Handle separately
                continue
            if val is None:
                predictions_flat[key] = val
            elif isinstance(val, np.ndarray) and val.ndim == 2:
                # For binary classification, store probability of positive class
                predictions_flat[key] = val[:, 1] if val.shape[1] == 2 else val[:, 0]
            else:
                predictions_flat[key] = val
        
        # Create predictions DataFrame
        all_predictions_df = pd.DataFrame(predictions_flat)
        
        # Extract RegistrationCode and research_stage from MultiIndex
        if id_research_pairs is not None and hasattr(id_research_pairs, 'get_level_values'):
            all_predictions_df.insert(0, 'RegistrationCode', id_research_pairs.get_level_values(0))
            all_predictions_df.insert(1, 'research_stage', id_research_pairs.get_level_values(1))
        
        # Create scores/pvalues/metrics DataFrame
        scores_pvalues_df = pd.DataFrame({
            'seed': self.seeds,
            'model_key': [self.model_key] * len(self.seeds),
            **self.all_metrics
        })
        
        return {
            'predictions': all_predictions_df,
            'metrics': scores_pvalues_df
        }
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics (mean, std, min, max) for all metrics.
        
        Returns:
            Dictionary with summary statistics for each metric
        """
        if self._first_seed:
            raise ValueError("No results have been added yet")
        
        summary = {}
        
        # Add score statistics
        summary['score'] = {
            'mean': np.mean(self.all_scores),
            'std': np.std(self.all_scores),
            'min': np.min(self.all_scores),
            'max': np.max(self.all_scores)
        }
        
        # Add p-value statistics
        summary['pvalue'] = {
            'mean': np.mean(self.all_pvalues),
            'std': np.std(self.all_pvalues),
            'min': np.min(self.all_pvalues),
            'max': np.max(self.all_pvalues)
        }
        
        # Add statistics for all other metrics
        for metric_name, metric_values in self.all_metrics.items():
            summary[metric_name] = {
                'mean': np.mean(metric_values),
                'std': np.std(metric_values),
                'min': np.min(metric_values),
                'max': np.max(metric_values)
            }
        
        return summary
    
    def reset(self) -> None:
        """Reset the collector to initial state."""
        self.all_predictions = {"id_research_pairs": None}
        self.all_metrics = {}
        self._first_seed = True


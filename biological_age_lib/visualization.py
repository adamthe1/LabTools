"""
Visualization Module

Scatter plots, gradient plots, residuals, and metrics display for age prediction analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from typing import Dict, Tuple, Optional


class AgeVisualization:
    """
    Visualization tools for age prediction analysis.
    
    Creates scatter plots, residual plots, error histograms,
    and gradient-colored visualizations.
    """
    
    def __init__(self, style: str = 'default'):
        """
        Initialize visualization settings.
        
        Args:
            style: Matplotlib style to use
        """
        self.style = style
    
    def analyze_predictions(self,
                            df: pd.DataFrame,
                            drop_outliers: bool = False,
                            age_range: Tuple[int, int] = (40, 70)) -> Tuple[pd.DataFrame, Dict]:
        """
        Calculate performance metrics for predictions.
        
        Args:
            df: DataFrame with 'true_values' and 'predictions' columns
            drop_outliers: If True, filter to age_range
            age_range: (min_age, max_age) for filtering
        
        Returns:
            Tuple of (filtered_df, metrics_dict)
        """
        df_filtered = df.copy()
        
        if drop_outliers:
            mask = (df_filtered['true_values'] >= age_range[0]) & \
                   (df_filtered['true_values'] <= age_range[1])
            df_filtered = df_filtered[mask]
        
        true_values = df_filtered['true_values']
        predictions = df_filtered['predictions']
        
        mae = mean_absolute_error(true_values, predictions)
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        r2 = r2_score(true_values, predictions)
        correlation, _ = pearsonr(true_values, predictions)
        
        mean_error = np.mean(predictions - true_values)
        std_error = np.std(predictions - true_values)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'correlation': correlation,
            'mean_error': mean_error,
            'std_error': std_error,
            'n_samples': len(df_filtered),
            'n_unique_subjects': df_filtered['subject_number'].nunique() if 'subject_number' in df_filtered.columns else len(df_filtered)
        }
        
        return df_filtered, metrics
    
    def analyze_by_fold(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze performance metrics by cross-validation fold.
        
        Args:
            df: DataFrame with 'true_values', 'predictions', 'fold' columns
        
        Returns:
            DataFrame with metrics per fold
        """
        if 'fold' not in df.columns:
            return pd.DataFrame()
        
        fold_metrics = []
        
        for fold in sorted(df['fold'].unique()):
            fold_data = df[df['fold'] == fold]
            true_vals = fold_data['true_values']
            pred_vals = fold_data['predictions']
            
            fold_mae = mean_absolute_error(true_vals, pred_vals)
            fold_rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
            fold_r2 = r2_score(true_vals, pred_vals)
            fold_corr, _ = pearsonr(true_vals, pred_vals)
            
            fold_metrics.append({
                'Fold': fold,
                'N': len(fold_data),
                'MAE': fold_mae,
                'RMSE': fold_rmse,
                'R2': fold_r2,
                'Correlation': fold_corr
            })
        
        return pd.DataFrame(fold_metrics)
    
    def create_scatter_plot(self,
                            df: pd.DataFrame,
                            metrics: Dict,
                            save_path: str = None,
                            figsize: Tuple[int, int] = (12, 10),
                            scatter_title: str = None,
                            residuals_title: str = None,
                            error_dist_title: str = None) -> plt.Figure:
        """
        Create scatter plot of predicted vs true ages.
        
        Args:
            df: DataFrame with 'true_values' and 'predictions'
            metrics: Dictionary of performance metrics
            save_path: Path to save figure
            figsize: Figure size
            scatter_title: Custom title for scatter plot
            residuals_title: Custom title for residuals plot
            error_dist_title: Custom title for error distribution
        
        Returns:
            matplotlib Figure
        """
        plt.style.use(self.style)
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[3, 1, 1],
                             hspace=0.3, wspace=0.3)
        
        # Main scatter plot
        ax_main = fig.add_subplot(gs[0, 0])
        ax_main.scatter(df['true_values'], df['predictions'],
                       c='#2196F3', alpha=0.6, s=40, edgecolors='white', linewidth=0.5,
                       label=f'All samples (n={len(df)})')
        
        # Perfect prediction line
        min_age = min(df['true_values'].min(), df['predictions'].min()) - 2
        max_age = max(df['true_values'].max(), df['predictions'].max()) + 2
        ax_main.plot([min_age, max_age], [min_age, max_age],
                    'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')
        
        # Trendline
        z = np.polyfit(df['true_values'], df['predictions'], 1)
        p = np.poly1d(z)
        ax_main.plot(df['true_values'].sort_values(), p(df['true_values'].sort_values()),
                    "darkblue", alpha=0.8, linewidth=2,
                    label=f'Trend (slope={z[0]:.2f})')
        
        ax_main.set_xlabel('True Age (years)', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('Predicted Age (years)', fontsize=12, fontweight='bold')
        ax_main.set_title(scatter_title or 'Predicted vs True Age', fontsize=14, fontweight='bold', pad=20)
        ax_main.legend(loc='upper left', framealpha=0.9)
        ax_main.grid(True, alpha=0.3)
        ax_main.set_xlim(min_age, max_age)
        ax_main.set_ylim(min_age, max_age)
        
        # Residuals plot
        ax_residuals = fig.add_subplot(gs[0, 1])
        residuals = df['predictions'] - df['true_values']
        ax_residuals.scatter(df['true_values'], residuals, alpha=0.6, s=20, c='#2196F3')
        ax_residuals.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax_residuals.set_xlabel('True Age (years)', fontsize=10)
        ax_residuals.set_ylabel('Residuals', fontsize=10)
        ax_residuals.set_title(residuals_title or 'Residuals Plot', fontsize=12, fontweight='bold')
        ax_residuals.grid(True, alpha=0.3)
        
        # Error histogram
        ax_hist = fig.add_subplot(gs[0, 2])
        errors = np.abs(df['predictions'] - df['true_values'])
        ax_hist.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax_hist.axvline(errors.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {errors.mean():.2f}')
        ax_hist.axvline(errors.median(), color='orange', linestyle='--', linewidth=2,
                       label=f'Median: {errors.median():.2f}')
        ax_hist.set_xlabel('Absolute Error (years)', fontsize=10)
        ax_hist.set_ylabel('Frequency', fontsize=10)
        ax_hist.set_title(error_dist_title or 'Error Distribution', fontsize=12, fontweight='bold')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # Metrics text
        ax_metrics = fig.add_subplot(gs[1, :])
        ax_metrics.axis('off')
        
        metrics_text = f"""
    Performance Metrics (n = {metrics['n_samples']:,} samples):
    
    Mean Absolute Error (MAE):     {metrics['mae']:.2f} years
    Root Mean Square Error (RMSE): {metrics['rmse']:.2f} years
    R² Score:                      {metrics['r2']:.3f}
    Correlation Coefficient:       {metrics['correlation']:.3f}
    Mean Error (Bias):             {metrics['mean_error']:.2f} years
    """
        
        ax_metrics.text(0.02, 0.98, metrics_text, transform=ax_metrics.transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        return fig
    
    def create_gradient_scatter_plot(self,
                                      df: pd.DataFrame,
                                      metrics: Dict,
                                      save_path: str = None,
                                      figsize: Tuple[int, int] = (12, 10),
                                      scatter_title: str = None,
                                      residuals_title: str = None,
                                      error_dist_title: str = None) -> Tuple[plt.Figure, plt.Figure, plt.Figure, pd.DataFrame]:
        """
        Create scatter plot with gradient coloring by within-age quartile.
        
        Red = overpredicted (older), Green = underpredicted (younger)
        
        Args:
            df: DataFrame with 'true_values' and 'predictions'
            metrics: Dictionary of performance metrics
            save_path: Path prefix for saving figures
            figsize: Figure size
            scatter_title: Custom title for scatter plot
            residuals_title: Custom title for residuals plot
            error_dist_title: Custom title for error distribution
        
        Returns:
            Tuple of (main_fig, residuals_fig, histogram_fig, df_with_colors)
        """
        plt.style.use(self.style)
        
        # Calculate color values based on within-age quartiles
        df_plot = df.copy()
        df_plot['age_year'] = df_plot['true_values'].round()
        df_plot['residual'] = df_plot['predictions'] - df_plot['true_values']
        
        color_values = []
        for idx, row in df_plot.iterrows():
            age_year = row['age_year']
            age_year_data = df_plot[df_plot['age_year'] == age_year]['residual']
            
            if len(age_year_data) < 4:
                color_values.append(0)
            else:
                q1 = age_year_data.quantile(0.25)
                q3 = age_year_data.quantile(0.75)
                residual = row['residual']
                
                if residual >= q3:
                    # Top quartile (overpredicted) - red
                    normalized = (residual - q3) / (age_year_data.max() - q3 + 1e-10)
                    color_values.append(0.5 + 0.5 * normalized)
                elif residual <= q1:
                    # Bottom quartile (underpredicted) - green
                    normalized = (residual - age_year_data.min()) / (q1 - age_year_data.min() + 1e-10)
                    color_values.append(-1.0 + 0.5 * normalized)
                else:
                    # Middle 50% - white zone
                    normalized = (residual - q1) / (q3 - q1 + 1e-10)
                    color_values.append(-0.15 + 0.3 * normalized)
        
        df_plot['color_value'] = color_values
        
        # Create colormap: green -> white -> red
        colors_list = [
            '#228B22', '#90EE90', '#E8F5E9', '#FFFFFF',
            '#FFFFFF', '#FFE8E8', '#FFA07A', '#DC143C'
        ]
        cmap = LinearSegmentedColormap.from_list('green_white_red', colors_list, N=256)
        
        min_age = min(df['true_values'].min(), df['predictions'].min()) - 2
        max_age = max(df['true_values'].max(), df['predictions'].max()) + 2
        
        # Count quartiles
        n_bottom = (df_plot['color_value'] <= -0.5).sum()
        n_top = (df_plot['color_value'] >= 0.5).sum()
        n_middle = len(df_plot) - n_bottom - n_top
        
        # Main scatter plot
        fig_main = plt.figure(figsize=figsize)
        ax_main = fig_main.add_subplot(111)
        
        scatter = ax_main.scatter(df_plot['true_values'], df_plot['predictions'],
                                 c=df_plot['color_value'], cmap=cmap,
                                 alpha=0.7, s=40, edgecolors='gray', linewidth=0.5,
                                 vmin=-1, vmax=1)
        
        cbar = plt.colorbar(scatter, ax=ax_main, pad=0.02)
        cbar.set_label('Prediction Quartile\n(within age year)', rotation=270, labelpad=25, fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_ticks([-0.75, 0, 0.75])
        cbar.set_ticklabels(['Bottom 25%\n(younger)', 'Middle 50%', 'Top 25%\n(older)'])
        
        # Trendline
        z = np.polyfit(df['true_values'], df['predictions'], 1)
        p = np.poly1d(z)
        ax_main.plot(df['true_values'].sort_values(), p(df['true_values'].sort_values()),
                    "darkblue", alpha=0.8, linewidth=2.5,
                    label=f'Trend (slope={z[0]:.2f})', zorder=5)
        
        ax_main.set_xlabel('True Age (years)', fontsize=16, fontweight='bold')
        ax_main.set_ylabel('Predicted Age (years)', fontsize=16, fontweight='bold')
        ax_main.set_title(scatter_title or 'Predicted vs True Age',
                         fontsize=18, fontweight='bold', pad=20)
        ax_main.tick_params(axis='both', which='major', labelsize=14)
        ax_main.legend(loc='upper right', framealpha=0.9, fontsize=12)
        ax_main.grid(True, alpha=0.3)
        ax_main.set_xlim(min_age, max_age)
        ax_main.set_ylim(min_age, max_age)
        
        # Metrics text box
        metrics_text = f"""Performance Metrics (n={metrics['n_samples']:,}):
MAE: {metrics['mae']:.2f} years
RMSE: {metrics['rmse']:.2f} years
R²: {metrics['r2']:.3f}
Correlation: {metrics['correlation']:.3f}

Quartile Distribution:
- Bottom 25% (green): {n_bottom} ({100*n_bottom/len(df_plot):.1f}%)
- Middle 50% (white): {n_middle} ({100*n_middle/len(df_plot):.1f}%)
- Top 25% (red): {n_top} ({100*n_top/len(df_plot):.1f}%)"""
        
        ax_main.text(0.02, 0.98, metrics_text, transform=ax_main.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Residuals plot
        fig_residuals = plt.figure(figsize=(5, 6))
        ax_residuals = fig_residuals.add_subplot(111)
        residuals = df['predictions'] - df['true_values']
        ax_residuals.scatter(df['true_values'], residuals, alpha=0.3, s=15, c='#2196F3')
        ax_residuals.axhline(y=0, color='r', linestyle='--', alpha=0.8, linewidth=2)
        ax_residuals.set_xlabel('True Age (years)', fontsize=12, fontweight='bold')
        ax_residuals.set_ylabel('Residuals (Pred - True)', fontsize=12, fontweight='bold')
        ax_residuals.set_title(residuals_title or 'Residuals Plot', fontsize=14, fontweight='bold')
        ax_residuals.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Histogram
        fig_hist = plt.figure(figsize=(5, 6))
        ax_hist = fig_hist.add_subplot(111)
        errors = np.abs(df['predictions'] - df['true_values'])
        ax_hist.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax_hist.axvline(errors.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {errors.mean():.2f}')
        ax_hist.axvline(errors.median(), color='orange', linestyle='--', linewidth=2,
                       label=f'Median: {errors.median():.2f}')
        ax_hist.set_xlabel('Absolute Error (years)', fontsize=12, fontweight='bold')
        ax_hist.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax_hist.set_title(error_dist_title or 'Error Distribution', fontsize=14, fontweight='bold')
        ax_hist.legend(fontsize=11)
        ax_hist.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figures
        if save_path:
            fig_main.savefig(f"{save_path}_main.png", dpi=300, bbox_inches='tight')
            fig_residuals.savefig(f"{save_path}_residuals.png", dpi=300, bbox_inches='tight')
            fig_hist.savefig(f"{save_path}_histogram.png", dpi=300, bbox_inches='tight')
            plt.close(fig_main)
            plt.close(fig_residuals)
            plt.close(fig_hist)
        
        return fig_main, fig_residuals, fig_hist, df_plot[['true_values', 'predictions', 'age_year', 'residual', 'color_value']]
    
    def print_metrics_summary(self, metrics: Dict) -> None:
        """Print formatted metrics summary."""
        print("\n" + "=" * 60)
        print("OVERALL PERFORMANCE METRICS")
        print("=" * 60)
        print(f"Dataset size: {metrics['n_samples']:,} samples")
        print(f"Mean Absolute Error (MAE): {metrics['mae']:.2f} years")
        print(f"Root Mean Square Error (RMSE): {metrics['rmse']:.2f} years")
        print(f"R² Score: {metrics['r2']:.3f}")
        print(f"Correlation: {metrics['correlation']:.3f}")
        print(f"Mean Error (Bias): {metrics['mean_error']:.2f} years")
        print(f"Std Error: {metrics['std_error']:.2f} years")
    
    def print_fold_summary(self, fold_df: pd.DataFrame) -> None:
        """Print formatted fold analysis summary."""
        if fold_df.empty:
            return
        
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION FOLD ANALYSIS")
        print("=" * 60)
        
        for _, row in fold_df.iterrows():
            print(f"Fold {row['Fold']} (n={row['N']:4d}): "
                  f"MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}, "
                  f"R²={row['R2']:.3f}, r={row['Correlation']:.3f}")
        
        print(f"\nCross-Validation Summary:")
        print(f"Mean MAE: {fold_df['MAE'].mean():.2f} ± {fold_df['MAE'].std():.2f}")
        print(f"Mean R²: {fold_df['R2'].mean():.3f} ± {fold_df['R2'].std():.3f}")


def create_gradient_scatter_plot(df: pd.DataFrame,
                                  metrics: Dict,
                                  save_path: str = None,
                                  figsize: Tuple[int, int] = (12, 10)) -> Tuple:
    """Convenience function to create gradient scatter plot."""
    viz = AgeVisualization()
    return viz.create_gradient_scatter_plot(df, metrics, save_path, figsize)


def analyze_predictions(df: pd.DataFrame,
                        drop_outliers: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """Convenience function to analyze predictions."""
    viz = AgeVisualization()
    return viz.analyze_predictions(df, drop_outliers)


def create_scatter_plot_by_gender(
    df: pd.DataFrame,
    gender_column: str = 'gender',
    save_path: str = None,
    figsize: Tuple[int, int] = (16, 7)
) -> plt.Figure:
    """
    Create side-by-side scatter plots for male and female subjects.
    
    Args:
        df: DataFrame with 'true_values', 'predictions', and gender column
        gender_column: Name of the gender column (1=male, 0=female)
        save_path: Path prefix for saving figures
        figsize: Figure size
    
    Returns:
        matplotlib Figure with two subplots
    """
    if gender_column not in df.columns:
        raise ValueError(f"Gender column '{gender_column}' not found in DataFrame")
    
    viz = AgeVisualization()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Colors for each gender
    colors = {1: '#3498db', 0: '#e74c3c'}  # Blue for male, red for female
    labels = {1: 'Male', 0: 'Female'}
    
    for ax, (gender_val, gender_name) in zip(axes, [(1, 'Male'), (0, 'Female')]):
        gender_df = df[df[gender_column] == gender_val]
        
        if len(gender_df) < 2:
            ax.text(0.5, 0.5, f'Not enough {gender_name} data', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Calculate metrics
        _, metrics = viz.analyze_predictions(gender_df)
        
        # Scatter plot
        ax.scatter(gender_df['true_values'], gender_df['predictions'],
                  c=colors[gender_val], alpha=0.6, s=40, edgecolors='white', linewidth=0.5)
        
        # Perfect prediction line
        min_age = min(gender_df['true_values'].min(), gender_df['predictions'].min()) - 2
        max_age = max(gender_df['true_values'].max(), gender_df['predictions'].max()) + 2
        ax.plot([min_age, max_age], [min_age, max_age], 'k--', linewidth=2, alpha=0.5)
        
        # Trendline
        z = np.polyfit(gender_df['true_values'], gender_df['predictions'], 1)
        p = np.poly1d(z)
        ax.plot(gender_df['true_values'].sort_values(), 
               p(gender_df['true_values'].sort_values()),
               color='darkblue', alpha=0.8, linewidth=2)
        
        ax.set_xlabel('True Age (years)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Age (years)', fontsize=12, fontweight='bold')
        ax.set_title(f'{gender_name} (n={len(gender_df)})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min_age, max_age)
        ax.set_ylim(min_age, max_age)
        
        # Metrics text
        metrics_text = f"MAE: {metrics['mae']:.2f}\nR²: {metrics['r2']:.3f}\nr: {metrics['correlation']:.3f}"
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def create_gender_comparison_summary(
    df: pd.DataFrame,
    gender_column: str = 'gender',
    save_path: str = None
) -> pd.DataFrame:
    """
    Create summary statistics comparing male and female predictions.
    
    Args:
        df: DataFrame with 'true_values', 'predictions', and gender column
        gender_column: Name of the gender column
        save_path: Path to save CSV summary
    
    Returns:
        DataFrame with metrics for each gender
    """
    viz = AgeVisualization()
    results = []
    
    for gender_val, gender_name in [(1, 'Male'), (0, 'Female'), (None, 'All')]:
        if gender_val is not None:
            gender_df = df[df[gender_column] == gender_val]
        else:
            gender_df = df
        
        if len(gender_df) < 2:
            continue
        
        _, metrics = viz.analyze_predictions(gender_df)
        metrics['gender'] = gender_name
        results.append(metrics)
    
    summary_df = pd.DataFrame(results)
    # Reorder columns
    cols = ['gender', 'n_samples', 'mae', 'rmse', 'r2', 'correlation', 'mean_error', 'std_error']
    summary_df = summary_df[[c for c in cols if c in summary_df.columns]]
    
    if save_path:
        summary_df.to_csv(save_path, index=False)
    
    return summary_df


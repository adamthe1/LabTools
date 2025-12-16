"""
Volcano Plot Analysis

Statistical comparison of biomarkers between top and bottom age-prediction groups.
Uses Mann-Whitney U tests with FDR-BH correction.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, List

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False


class VolcanoAnalyzer:
    """
    Analyzer for volcano plot comparisons between groups.
    
    Compares biomarker distributions between top and bottom percentile groups
    using Mann-Whitney U tests with FDR correction.
    """
    
    def __init__(self,
                 alpha: float = 0.05,
                 fc_threshold: float = 0.0,
                 summary_stat: str = 'mean',
                 standardize: bool = False,
                 pre_scaled: bool = False):
        """
        Initialize the analyzer.
        
        Args:
            alpha: Significance level for FDR correction
            fc_threshold: Fold change threshold for significance
            summary_stat: 'mean' or 'median' for effect size calculation
            standardize: If True, z-score features internally
            pre_scaled: If True, assume data is already z-scored
        """
        self.alpha = alpha
        self.fc_threshold = fc_threshold
        self.summary_stat = summary_stat
        self.standardize = standardize
        self.pre_scaled = pre_scaled
    
    def compare(self,
                table1: pd.DataFrame,
                table2: pd.DataFrame,
                labels: Tuple[str, str] = None,
                subset_cols: List[str] = None,
                apply_fdr: bool = True) -> pd.DataFrame:
        """
        Compare two tables and return statistical results.
        
        Args:
            table1: DataFrame for group 1 (e.g., bottom percentile)
            table2: DataFrame for group 2 (e.g., top percentile)
            labels: Tuple of (group1_label, group2_label)
            subset_cols: Optional list of columns to compare
            apply_fdr: If True, apply FDR correction and significance flags
        
        Returns:
            DataFrame with comparison results including p-values and effect sizes
        """
        if labels is None:
            labels = ('Bottom', 'Top')
        
        # Validate inputs
        if self.standardize and self.pre_scaled:
            raise ValueError("Cannot use both standardize and pre_scaled")
        
        common = table1.columns.intersection(table2.columns)
        if not len(common):
            raise ValueError("No shared features between tables")
        
        if subset_cols:
            common = common.intersection(subset_cols)
            if not len(common):
                raise ValueError("No shared features after subset filter")
        
        # Apply scaling if needed
        t1, t2 = table1.copy(), table2.copy()
        if self.standardize:
            scaler = StandardScaler()
            combined = pd.concat([t1[common], t2[common]])
            scaler.fit(combined)
            t1[common] = scaler.transform(t1[common])
            t2[common] = scaler.transform(t2[common])
        
        # Determine effect size type
        if self.standardize or self.pre_scaled:
            effect_key = "delta_z"
        else:
            effect_key = "log2_fold_change"
        
        # Run comparisons
        results = self._run_comparisons(t1, t2, common, labels, effect_key)
        
        if not results:
            raise ValueError("No valid comparisons")
        
        # Build results DataFrame
        res = pd.DataFrame(results)
        
        # Apply FDR correction if requested
        if apply_fdr:
            res = self.apply_fdr(res)
        
        return res
    
    def apply_fdr(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Apply FDR-BH correction and significance flags to results.
        
        Args:
            results: DataFrame with 'p_value' column from compare()
        
        Returns:
            DataFrame with adj_p_value, -log10_p, -log10_q, significant, regulation
        """
        # Determine effect key
        effect_key = "delta_z" if "delta_z" in results.columns else "log2_fold_change"
        
        res = self._apply_fdr_correction(results.copy())
        res = self._add_significance_flags(res, effect_key)
        return res
    
    def _run_comparisons(self, 
                          t1: pd.DataFrame, 
                          t2: pd.DataFrame,
                          features: pd.Index,
                          labels: Tuple[str, str],
                          effect_key: str) -> List[dict]:
        """Run Mann-Whitney U tests for all features."""
        results = []
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            
            for feat in features:
                g1, g2 = t1[feat].dropna(), t2[feat].dropna()
                
                if len(g1) < 2 or len(g2) < 2:
                    continue
                
                # Calculate summary statistics
                if self.summary_stat == "median":
                    stat1, stat2 = np.median(g1), np.median(g2)
                else:
                    stat1, stat2 = g1.mean(), g2.mean()
                
                # Calculate effect size
                if self.standardize or self.pre_scaled:
                    effect = stat2 - stat1  # Delta z-score
                else:
                    if stat1 <= 0 or stat2 <= 0:
                        effect = np.nan
                    else:
                        effect = np.log2(stat2 / stat1)
                
                # Mann-Whitney U test
                try:
                    p_val = mannwhitneyu(g1, g2, alternative='two-sided').pvalue
                    results.append({
                        "feature": feat,
                        effect_key: effect,
                        "p_value": p_val,
                        f"{self.summary_stat}_{labels[0]}": stat1,
                        f"{self.summary_stat}_{labels[1]}": stat2,
                        f"n_{labels[0]}": len(g1),
                        f"n_{labels[1]}": len(g2)
                    })
                except ValueError:
                    continue
        
        return results
    
    def _apply_fdr_correction(self, res: pd.DataFrame) -> pd.DataFrame:
        """Apply FDR-BH correction to p-values."""
        p_series = pd.to_numeric(res["p_value"], errors="coerce")
        mask = p_series.notna().values
        adj_p = np.ones(len(res), dtype=float)
        
        if mask.any():
            pvals = p_series.values[mask].astype(float)
            pvals[~np.isfinite(pvals)] = np.nan
            keep = ~np.isnan(pvals)
            
            if keep.any():
                qvals = np.full(pvals.shape, np.nan, dtype=float)
                _, qvals[keep], _, _ = multipletests(
                    pvals[keep],
                    alpha=float(self.alpha),
                    method="fdr_bh",
                    is_sorted=False,
                    returnsorted=False
                )
                adj_p[mask] = qvals
        
        res["adj_p_value"] = adj_p
        res["-log10_p"] = -np.log10(p_series)
        res["-log10_q"] = -np.log10(res["adj_p_value"])
        
        return res
    
    def _add_significance_flags(self, res: pd.DataFrame, effect_key: str) -> pd.DataFrame:
        """Add significance and regulation flags."""
        # Clean effect column
        col = pd.to_numeric(res[effect_key], errors="coerce")
        res[effect_key] = col
        
        # Significance check
        effect_abs = col.abs().values
        comp_effect = np.greater(effect_abs, self.fc_threshold)
        comp_effect = np.where(np.isnan(effect_abs), False, comp_effect)
        
        lhs = (res["adj_p_value"] < float(self.alpha)).values
        signif_vec = np.logical_and(lhs, comp_effect)
        res["significant"] = pd.Series(signif_vec, index=res.index)
        
        # Regulation labels
        res["regulation"] = np.select(
            [
                res["significant"] & (res[effect_key] > 0),
                res["significant"] & (res[effect_key] < 0),
            ],
            ["upregulated", "downregulated"],
            default="not_significant",
        )
        
        return res
    
    def plot(self,
             results: pd.DataFrame,
             labels: Tuple[str, str] = None,
             save_path: str = None,
             gender: str = None,
             figsize: Tuple[int, int] = (14, 10),
             upregulated_color: str = '#FF5252',
             downregulated_color: str = '#4CAF50',
             labels_fontsize: int = 8,
             title: str = None) -> plt.Figure:
        """
        Create volcano plot from comparison results.
        
        Args:
            results: DataFrame from compare()
            labels: Group labels for title
            save_path: Path to save figure (without extension)
            gender: Gender label for title
            figsize: Figure size
            upregulated_color: Color for upregulated points
            downregulated_color: Color for downregulated points
            labels_fontsize: Font size for point labels
        
        Returns:
            matplotlib Figure object
        """
        if labels is None:
            labels = ('Bottom', 'Top')
        
        # Determine effect key
        effect_key = "delta_z" if "delta_z" in results.columns else "log2_fold_change"
        
        # Setup plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.set_style("whitegrid")
        
        # Plot points by regulation status
        for reg, color, label in [
            ("not_significant", "grey", "Not significant"),
            ("upregulated", upregulated_color, f"Upregulated (n={sum(results['regulation']=='upregulated')})"),
            ("downregulated", downregulated_color, f"Downregulated (n={sum(results['regulation']=='downregulated')})")
        ]:
            sub = results[results["regulation"] == reg]
            if not sub.empty:
                ax.scatter(
                    sub[effect_key], sub["-log10_p"],
                    s=80 if "regulated" in reg else 50,
                    alpha=0.9,
                    color=color,
                    edgecolor='dimgrey' if reg == "not_significant" else 'darkred' if reg == "upregulated" else 'darkgreen',
                    linewidth=0.8,
                    label=label
                )
        
        # Label significant points
        if HAS_ADJUST_TEXT:
            texts = []
            for _, r in results[results["significant"]].iterrows():
                texts.append(ax.text(r[effect_key], r["-log10_p"], r["feature"],
                                    fontsize=labels_fontsize, weight="bold"))
            if texts:
                adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        else:
            for _, r in results[results["significant"]].iterrows():
                ax.annotate(r["feature"], (r[effect_key], r["-log10_p"]),
                           fontsize=labels_fontsize, weight="bold")
        
        # Add threshold lines
        ax.axhline(-np.log10(self.alpha), ls="--", c="dimgrey", lw=1)
        ax.axvline(self.fc_threshold, ls="--", c="dimgrey", lw=1)
        ax.axvline(-self.fc_threshold, ls="--", c="dimgrey", lw=1)
        
        # FDR threshold line
        p_bh_cut = self._compute_fdr_cutoff(results)
        if p_bh_cut is not None:
            ax.axhline(-np.log10(p_bh_cut), ls="-.", c="dimgrey", lw=1.2,
                      label=f"FDR-BH threshold (q<{self.alpha})")
        
        # Labels and title
        x_label = "$\\Delta$SD" if effect_key == "delta_z" else f"Log2 FC ({labels[1]} / {labels[0]})"
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(r"$-\log_{10}$(p)", fontsize=12)
        
        n1 = results[f"n_{labels[0]}"].iloc[0] if f"n_{labels[0]}" in results.columns else "?"
        n2 = results[f"n_{labels[1]}"].iloc[0] if f"n_{labels[1]}" in results.columns else "?"
        
        # Build title (use custom or default)
        if title:
            plot_title = title.format(gender=gender or '', n1=n1, n2=n2, label1=labels[0], label2=labels[1])
        else:
            title_extra = f" - {gender}" if gender else ""
            title_suffix = " (z-score)" if effect_key == "delta_z" else ""
            plot_title = f"Volcano: {labels[1]} (n={n2}) vs {labels[0]} (n={n1}){title_extra}{title_suffix}"
        ax.set_title(plot_title, weight="bold")
        
        # Extend y-axis to make room for legend
        y_max = results["-log10_p"].max()
        ax.set_ylim(bottom=ax.get_ylim()[0], top=y_max * 1.15 + 1)
        
        ax.legend(loc='upper left', frameon=True, framealpha=0.85)
        plt.tight_layout()
        
        # Save if path provided (PNG only)
        if save_path:
            fig.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        return fig
    
    def _compute_fdr_cutoff(self, results: pd.DataFrame) -> Optional[float]:
        """Compute the effective FDR-BH p-value cutoff."""
        p_vals = results['p_value'].dropna().values
        m = len(p_vals)
        
        if m == 0:
            return None
        
        p_sorted = np.sort(p_vals)
        bh_thresholds = (np.arange(1, m + 1) / m) * self.alpha
        passed = np.where(p_sorted <= bh_thresholds)[0]
        
        if passed.size > 0:
            k_star = int(passed.max() + 1)
            return (k_star / m) * self.alpha
        
        return None


def compare_tables_and_plot_volcano(table1: pd.DataFrame,
                                     table2: pd.DataFrame,
                                     labels: Tuple[str, str] = None,
                                     alpha: float = 0.05,
                                     fc_threshold: float = 0.0,
                                     save_path: str = None,
                                     gender: str = None,
                                     summary_stat: str = "mean",
                                     standardize: bool = False,
                                     pre_scaled: bool = False,
                                     subset_cols: List[str] = None,
                                     figsize: Tuple[int, int] = (14, 10),
                                     labels_fontsize: int = 8,
                                     percentile: float = 0.25,
                                     apply_fdr: bool = True,
                                     **kwargs) -> pd.DataFrame:
    """
    Convenience function to compare tables and create volcano plot.
    
    Args:
        table1: DataFrame for group 1 (bottom percentile)
        table2: DataFrame for group 2 (top percentile)
        labels: Tuple of group labels
        alpha: Significance level
        fc_threshold: Fold change threshold
        save_path: Path to save figure
        gender: Gender label for title
        summary_stat: 'mean' or 'median'
        standardize: Apply z-scoring internally
        pre_scaled: Data is already z-scored
        subset_cols: Columns to compare
        figsize: Figure size
        labels_fontsize: Label font size
        percentile: Percentile used (for label generation)
        apply_fdr: If True, apply FDR correction
    
    Returns:
        DataFrame with comparison results
    """
    if labels is None:
        percent = int(percentile * 100)
        labels = (f'Bottom {percent}%', f'Top {percent}%')
    
    analyzer = VolcanoAnalyzer(
        alpha=alpha,
        fc_threshold=fc_threshold,
        summary_stat=summary_stat,
        standardize=standardize,
        pre_scaled=pre_scaled
    )
    
    results = analyzer.compare(table1, table2, labels=labels, subset_cols=subset_cols, apply_fdr=apply_fdr)
    analyzer.plot(results, labels=labels, save_path=save_path, gender=gender,
                 figsize=figsize, labels_fontsize=labels_fontsize)
    
    return results


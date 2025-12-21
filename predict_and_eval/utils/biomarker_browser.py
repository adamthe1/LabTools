"""
Biomarker Browser - A tool class for AI agents to explore and select prediction targets.

This class provides methods to browse body systems and their columns/biomarkers
from the dataset_columns.json metadata file.

USAGE WITH AI AGENTS:
---------------------
1. Initialize the browser:
   browser = BiomarkerBrowser()

2. List available body systems:
   systems = browser.list_systems()
   # Returns: ['Age_Gender_BMI', 'blood_lipids', 'cardiovascular_system', ...]

3. Get detailed info about specific systems:
   info = browser.get_system_info(['frailty', 'cardiovascular_system'])
   # Returns dict with columns, types, descriptions, and statistics

4. Filter columns by type:
   regression_cols = browser.get_columns_by_type('proteomics', 'regression')
   classification_cols = browser.get_columns_by_type('medical_conditions', 'classification')

5. Search for specific biomarkers:
   results = browser.search_columns('diabetes')
   # Searches across all systems for matching column names/descriptions

6. Get summary statistics:
   summary = browser.get_system_summary()
   # Returns column counts per system by type

EXPECTED OUTPUT FORMAT (for AI selection):
------------------------------------------
{
    "frailty_select": ["grip_strength", "walk_speed", ...],
    "proteomics_select": ["IL6", "TNF_alpha", ...],
    "medical_conditions_select": ["diabetes", "hypertension", ...]
}
"""

import json
import os
from typing import Dict, List, Optional, Any, Union

# Default path to dataset columns JSON
DEFAULT_JSON_PATH = "/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/gil_link/Analyses/10K_Trajectories/body_systems/body_systems_description/dataset_columns.json"


class BiomarkerBrowser:
    """Browse and select biomarkers from the body systems dataset."""
    
    def __init__(self, json_path: str = DEFAULT_JSON_PATH):
        """
        Initialize the browser with the dataset columns JSON.
        
        Args:
            json_path: Path to dataset_columns.json file
        """
        self.json_path = json_path
        self._data = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load the JSON data from file."""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Dataset JSON not found: {self.json_path}")
        with open(self.json_path, 'r') as f:
            self._data = json.load(f)
    
    def list_systems(self) -> List[str]:
        """
        List all available body systems.
        
        Returns:
            List of system names (e.g., ['Age_Gender_BMI', 'blood_lipids', ...])
        """
        return list(self._data.keys())
    
    def get_system_summary(self) -> Dict[str, Dict[str, int]]:
        """
        Get summary statistics for all systems.
        
        Note: 'categorical' with n_unique=2 and n_positives is effectively binary classification.
        
        Returns:
            Dict mapping system names to column counts by type:
            {
                'system_name': {
                    'total': N,
                    'regression': N,
                    'binary_classification': N,  # categorical with n_unique=2
                    'multi_categorical': N       # categorical with n_unique>2
                }
            }
        """
        summary = {}
        for sys_name, sys_data in self._data.items():
            cols = sys_data.get('columns', {})
            counts = {'total': len(cols), 'regression': 0, 'binary_classification': 0, 'multi_categorical': 0}
            for col_info in cols.values():
                col_type = col_info.get('type', 'unknown')
                if col_type == 'regression':
                    counts['regression'] += 1
                elif col_type == 'categorical':
                    # Binary categoricals are classification targets
                    if col_info.get('n_unique', 0) == 2:
                        counts['binary_classification'] += 1
                    else:
                        counts['multi_categorical'] += 1
            summary[sys_name] = counts
        return summary
    
    def get_system_info(self, systems: Union[str, List[str]], 
                        include_stats: bool = True,
                        max_columns: Optional[int] = None) -> Dict[str, Any]:
        """
        Get detailed information about specific body systems.
        
        Args:
            systems: Single system name or list of system names
            include_stats: Include min/max/mean/std for regression columns
            max_columns: Limit number of columns returned per system (None = all)
        
        Returns:
            Dict with system info:
            {
                'system_name': {
                    'directory': 'path/to/csv',
                    'n_columns': N,
                    'columns': {
                        'col_name': {
                            'description': '...',
                            'type': 'regression|classification|categorical',
                            ...
                        }
                    }
                }
            }
        """
        if isinstance(systems, str):
            systems = [systems]
        
        result = {}
        for sys_name in systems:
            if sys_name not in self._data:
                result[sys_name] = {'error': f"System '{sys_name}' not found"}
                continue
            
            sys_data = self._data[sys_name]
            cols = sys_data.get('columns', {})
            
            # Optionally limit columns
            if max_columns and len(cols) > max_columns:
                cols = dict(list(cols.items())[:max_columns])
            
            # Format column info
            formatted_cols = {}
            for col_name, col_info in cols.items():
                formatted = {
                    'description': col_info.get('description', ''),
                    'type': col_info.get('type', 'unknown'),
                    'n_unique': col_info.get('n_unique', None)
                }
                # Add type-specific stats
                if include_stats:
                    if col_info.get('type') == 'regression':
                        for stat in ['min', 'max', 'mean', 'std']:
                            if stat in col_info:
                                formatted[stat] = col_info[stat]
                    elif col_info.get('type') in ['classification', 'categorical']:
                        if 'n_positives' in col_info:
                            formatted['n_positives'] = col_info['n_positives']
                
                formatted_cols[col_name] = formatted
            
            result[sys_name] = {
                'directory': sys_data.get('directory', ''),
                'n_columns': len(sys_data.get('columns', {})),
                'columns': formatted_cols
            }
        
        return result
    
    def get_columns_by_type(self, system: str, col_type: str) -> Dict[str, Dict]:
        """
        Get all columns of a specific type from a system.
        
        Args:
            system: System name
            col_type: 'regression', 'binary_classification', 'multi_categorical', or 'categorical'
                      - 'binary_classification': categorical with n_unique=2 (suitable for classification)
                      - 'multi_categorical': categorical with n_unique>2
                      - 'categorical': all categoricals (both binary and multi)
        
        Returns:
            Dict of column names to their info
        """
        if system not in self._data:
            return {'error': f"System '{system}' not found"}
        
        cols = self._data[system].get('columns', {})
        
        if col_type == 'binary_classification':
            return {
                name: info for name, info in cols.items() 
                if info.get('type') == 'categorical' and info.get('n_unique', 0) == 2
            }
        elif col_type == 'multi_categorical':
            return {
                name: info for name, info in cols.items() 
                if info.get('type') == 'categorical' and info.get('n_unique', 0) > 2
            }
        else:
            return {
                name: info for name, info in cols.items() 
                if info.get('type') == col_type
            }
    
    def search_columns(self, query: str, systems: Optional[List[str]] = None,
                       search_descriptions: bool = True) -> Dict[str, Dict[str, Dict]]:
        """
        Search for columns matching a query string.
        
        Args:
            query: Search term (case-insensitive)
            systems: Limit search to specific systems (None = all)
            search_descriptions: Also search in column descriptions
        
        Returns:
            Dict mapping system names to matching columns:
            {
                'system_name': {
                    'col_name': {'description': ..., 'type': ...}
                }
            }
        """
        query_lower = query.lower()
        results = {}
        
        search_systems = systems if systems else self._data.keys()
        
        for sys_name in search_systems:
            if sys_name not in self._data:
                continue
            
            matches = {}
            cols = self._data[sys_name].get('columns', {})
            
            for col_name, col_info in cols.items():
                # Search in column name
                if query_lower in col_name.lower():
                    matches[col_name] = {
                        'description': col_info.get('description', ''),
                        'type': col_info.get('type', 'unknown')
                    }
                # Search in description
                elif search_descriptions:
                    desc = col_info.get('description', '').lower()
                    if query_lower in desc:
                        matches[col_name] = {
                            'description': col_info.get('description', ''),
                            'type': col_info.get('type', 'unknown')
                        }
            
            if matches:
                results[sys_name] = matches
        
        return results
    
    def list_columns(self, system: str, with_descriptions: bool = False) -> Union[List[str], Dict[str, str]]:
        """
        List all column names in a system.
        
        Args:
            system: System name
            with_descriptions: If True, return dict with descriptions
        
        Returns:
            List of column names, or dict of {col_name: description}
        """
        if system not in self._data:
            return [] if not with_descriptions else {}
        
        cols = self._data[system].get('columns', {})
        
        if with_descriptions:
            return {name: info.get('description', '') for name, info in cols.items()}
        return list(cols.keys())
    
    def get_column_details(self, system: str, column: str) -> Dict[str, Any]:
        """
        Get full details for a specific column.
        
        Args:
            system: System name
            column: Column name
        
        Returns:
            Full column info dict
        """
        if system not in self._data:
            return {'error': f"System '{system}' not found"}
        
        cols = self._data[system].get('columns', {})
        if column not in cols:
            return {'error': f"Column '{column}' not found in {system}"}
        
        return cols[column]
    
    def get_classification_targets(self, min_positives: int = 100, 
                                   max_positives: Optional[int] = None) -> Dict[str, List[str]]:
        """
        Get binary classification targets with sufficient positive samples.
        
        Note: Binary categoricals (n_unique=2 with n_positives) are classification targets.
        
        Args:
            min_positives: Minimum number of positive samples required
            max_positives: Maximum number of positive samples (None = no limit)
        
        Returns:
            Dict mapping system names to list of suitable column names
        """
        results = {}
        for sys_name, sys_data in self._data.items():
            cols = sys_data.get('columns', {})
            suitable = []
            for col_name, col_info in cols.items():
                # Binary categorical = classification target
                if col_info.get('type') != 'categorical' or col_info.get('n_unique', 0) != 2:
                    continue
                n_pos = col_info.get('n_positives', 0)
                if n_pos >= min_positives:
                    if max_positives is None or n_pos <= max_positives:
                        suitable.append(col_name)
            if suitable:
                results[sys_name] = suitable
        return results
    
    def get_regression_targets(self, min_unique: int = 50,
                               require_std: bool = True) -> Dict[str, List[str]]:
        """
        Get regression targets with sufficient variability.
        
        Args:
            min_unique: Minimum number of unique values
            require_std: Require std > 0
        
        Returns:
            Dict mapping system names to list of suitable column names
        """
        results = {}
        for sys_name, sys_data in self._data.items():
            cols = sys_data.get('columns', {})
            suitable = []
            for col_name, col_info in cols.items():
                if col_info.get('type') != 'regression':
                    continue
                n_unique = col_info.get('n_unique', 0)
                std = col_info.get('std', 0)
                if n_unique >= min_unique:
                    if not require_std or std > 0:
                        suitable.append(col_name)
            if suitable:
                results[sys_name] = suitable
        return results


# Convenience functions for quick access
def browse_systems() -> List[str]:
    """Quick function to list all body systems."""
    return BiomarkerBrowser().list_systems()


def browse_columns(system: str) -> Dict[str, str]:
    """Quick function to get columns with descriptions for a system."""
    return BiomarkerBrowser().list_columns(system, with_descriptions=True)


def search_biomarkers(query: str) -> Dict[str, Dict[str, Dict]]:
    """Quick function to search for biomarkers across all systems."""
    return BiomarkerBrowser().search_columns(query)


if __name__ == "__main__":
    # Demo usage
    browser = BiomarkerBrowser()
    
    print("=== Available Body Systems ===")
    systems = browser.list_systems()
    print(f"Found {len(systems)} systems: {systems}\n")
    
    print("=== System Summary ===")
    summary = browser.get_system_summary()
    for sys_name, counts in summary.items():
        print(f"{sys_name}: {counts['total']} cols "
              f"(reg: {counts['regression']}, binary: {counts['binary_classification']}, multi_cat: {counts['multi_categorical']})")
    
    print("\n=== Sample: frailty system ===")
    frailty_info = browser.get_system_info('frailty', max_columns=5)
    print(json.dumps(frailty_info, indent=2, default=str)[:1000])
    
    print("\n=== Classification targets (min 100 positives) ===")
    class_targets = browser.get_classification_targets(min_positives=100)
    for sys, cols in class_targets.items():
        print(f"{sys}: {len(cols)} targets - {cols[:3]}...")
    
    print("\n=== Search: 'diabetes' ===")
    diabetes_results = browser.search_columns('diabetes')
    for sys, cols in diabetes_results.items():
        print(f"{sys}: {list(cols.keys())}")


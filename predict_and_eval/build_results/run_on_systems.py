"""
Build and run cross-validation results for multiple body systems.

This module provides the BuildResults class for evaluating feature systems
against target systems using cross-validation with multiple seeds.
"""
import os
import pandas as pd
from ..loading_features.load_feature_df import (
    add_body_system_csv, 
    create_body_system_from_other_systems_csv,
    clear_temp_systems,
    get_body_system_column_names,
    filter_existing_columns
)
from ..loading_features.preprocess_features import PreprocessFeatures
from ..regression_seeding.seeding import seeding
from ..utils.ids_folds import id_fold_with_stratified_threshold
from ..correct_and_collect_results.compare_results import compare_and_collect_results
from LabUtils.addloglevels import sethandlers

try:
    from LabQueue.qp import qp
except ImportError:
    qp = None  # LabQueue not available


def is_csv_file(path: str) -> bool:
    """Check if path is a valid CSV file."""
    return os.path.isfile(path) and path.endswith('.csv')


def _save_skipped_result(save_dir: str, reason: str):
    """Save skip marker files matching seeding output format."""
    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame().to_csv(os.path.join(save_dir, 'predictions.csv'))
    pd.DataFrame({'skipped': [True], 'reason': [reason]}).to_csv(os.path.join(save_dir, 'metrics.csv'))
    print(f"Skipped: {reason} -> {save_dir}")


def _run_seeding_job(x, y, save_dir: str, model_key, num_seeds: int, num_splits: int,
                     stratified_minority_threshold: float, testing: bool = False):
    """
    Standalone queue worker: runs seeding on preprocessed x/y data.
    Receives DataFrames directly - no preprocessing needed in worker.
    """
    # Check if enough unique subjects for k-fold CV
    n_subjects = x.index.get_level_values(0).nunique()
    if n_subjects < num_splits:
        reason = f"insufficient_subjects ({n_subjects} < {num_splits} folds)"
        _save_skipped_result(save_dir, reason)
        return {'predictions': pd.DataFrame(), 'metrics': pd.DataFrame(), 'skipped': True}
    
    folds = id_fold_with_stratified_threshold(
        y, seeds=range(num_seeds), n_splits=num_splits,
        stratified_threshold=stratified_minority_threshold
    )
    os.makedirs(save_dir, exist_ok=True)
    
    return seeding(
        x, y, folds, model_key=model_key, average_by_subject_id=True,
        gender_split_evaluation=True, save_dir=save_dir, testing=testing
    )


class BuildResults:
    """
    Build and run cross-validation results for multiple body systems.
    
    Example run_list format:
        [
            'sleep',  # body system name
            {'my_features': '/path/to/csv'},  # name -> csv path
            {'some_name': ['column1', 'column2']},  # name -> column list
        ]
    
    Example target_systems format:
        [
            'blood_lipids',  # body system (predicts all columns)
            {'glycemic_markers': ['glucose', 'hba1c']},  # specific columns
            {'custom': '/path/to/targets.csv'},  # custom CSV
        ]
    
    Example column_descriptions format:
        {"activity": "classification", "score": "ordinal"}
        Options: "regression", "classification", "ordinal"
    """
    def __init__(self):
        # Feature systems to evaluate (set via run_prediction.py)
        self.run_list = []
        self.run_column_descriptions = {}
        
        # Target systems to predict
        self.target_systems = []
        self.target_column_descriptions = {}
        
        # Baseline for comparison
        self.baseline = 'Age_Gender_BMI'
        self.confounders = ['age', 'gender', 'bmi']
        
        # Output directory (required - set before running)
        self.save_dir = None
        
        # Model configuration
        self.model_key = 'all'  # 'all', single model name, or list of models
        
        # Cross-validation parameters
        self.stratified_minority_threshold = 0.2
        self.num_seeds = 20
        self.num_splits = 5
        
        # Compute resources
        self.num_threads = 16
        
        # Data handling
        self.merge_closest_research_stage = True
        self.use_cache = False
        
        # Testing mode (uses preset params instead of tuning)
        self.testing = False

    def run(self, with_queue: bool = True):
        """Main entry point to run the full pipeline."""
        if self.save_dir is None:
            raise ValueError("save_dir must be set before running")
        if not self.run_list:
            raise ValueError("run_list must contain at least one feature system")
        if not self.target_systems:
            raise ValueError("target_systems must contain at least one target")
        
        clear_temp_systems()  # Reset temp systems at start of each run
        self.prepare_data()
        print("Prepared data for all body systems.")
        
        if with_queue and qp is None:
            raise ImportError("LabQueue is required to run with queue")
        
        all_dirs = []
        baseline_name = self._get_name(self.baseline)
        baseline_columns = self._get_columns(self.baseline)
        
        if with_queue:
            self._run_with_queue(baseline_name, baseline_columns, all_dirs)
        else:
            self._run_without_queue(baseline_name, baseline_columns, all_dirs)
        
        compare_and_collect_results(self.save_dir, baseline_name)
    
    def _run_with_queue(self, baseline_name: str, baseline_columns: list, all_dirs: list):
        """Run pipeline with LabQueue - preprocess locally, send x/y pairs to queue."""
        sethandlers()
        queue_log_dir = os.path.join(self.save_dir, 'queue_logs')
        os.makedirs(queue_log_dir, exist_ok=True)
        os.chdir(queue_log_dir)
        
        with qp(jobname="predict", max_u=100, _trds_def=self.num_threads, _mem_def='5G') as q:
            q.startpermanentrun()
            all_methods = []
            
            for target_system in self.target_systems:
                target_name = self._get_name(target_system)
                
                for feature_run in self.run_list:
                    feature_name = self._get_name(feature_run)
                    
                    # Create preprocessors
                    preprocessor = PreprocessFeatures(
                        feature_systems=feature_name,
                        target_systems=target_name,
                        confounders=self.confounders,
                        merge_closest_research_stage=self.merge_closest_research_stage,
                        use_cache=self.use_cache,
                    )
                    baseline_preprocessor = PreprocessFeatures(
                        feature_systems=baseline_name,
                        target_systems=target_name,
                        confounders=[],
                        merge_closest_research_stage=self.merge_closest_research_stage,
                        use_cache=self.use_cache,
                    )
                    
                    # Preprocess locally, queue each label as separate x/y jobs
                    for label_name in preprocessor.targets:
                        full_dir = os.path.join(self.save_dir, target_name, label_name, feature_name)
                        baseline_dir = os.path.join(self.save_dir, target_name, label_name, baseline_name)
                        
                        # Skip if already processed
                        has_predictions = any(
                            f.endswith('predictions.csv')
                            for root, _, files in os.walk(full_dir)
                            for f in files
                        ) if os.path.exists(full_dir) else False
                        if has_predictions:
                            print(f"Skipping {label_name} features: {feature_name}, already finished.")
                            continue
                        
                        # Preprocess full model locally
                        x, y = preprocessor.preprocess(label_name)
                        valid_index = x.index
                        
                        # Queue full model seeding job
                        method = q.method(_run_seeding_job, [
                            x, y, full_dir, self.model_key, self.num_seeds, self.num_splits,
                            self.stratified_minority_threshold, self.testing
                        ])
                        all_methods.append(method)
                        print(f"Queued full: {feature_name} -> {target_name}/{label_name}")
                        
                        # Preprocess baseline locally (filtered to same subjects)
                        x_base, y_base = baseline_preprocessor.preprocess(label_name, filter_index=valid_index)
                        
                        # Queue baseline seeding job
                        method = q.method(_run_seeding_job, [
                            x_base, y_base, baseline_dir, self.model_key, self.num_seeds, self.num_splits,
                            self.stratified_minority_threshold, self.testing
                        ])
                        all_methods.append(method)
                        print(f"Queued baseline: {baseline_name} -> {target_name}/{label_name}")
            
            q.wait(all_methods, False)
    
    def _run_without_queue(self, baseline_name: str, baseline_columns: list, all_dirs: list):
        """Run pipeline without queue (local execution)."""
        for target_system in self.target_systems:
            target_name = self._get_name(target_system)
            target_columns = self._get_columns(target_system)
                
            for feature_run in self.run_list:
                feature_name = self._get_name(feature_run)
                feature_columns = self._get_columns(feature_run)
            
                preprocessor = PreprocessFeatures(
                    feature_systems=feature_name,
                    target_systems=target_name,
                    confounders=self.confounders,
                    merge_closest_research_stage=self.merge_closest_research_stage,
                )
                baseline_preprocessor = PreprocessFeatures(
                    feature_systems=baseline_name,
                    target_systems=target_name,
                    confounders=[],
                    merge_closest_research_stage=self.merge_closest_research_stage,
                )
                
                print(f"Running: {feature_name} -> {target_name}")

                self._run_full_and_baseline_system(
                    preprocessor, baseline_preprocessor,
                    feature_name, baseline_name, target_name
                )
    
    def prepare_data(self):
        """Prepare body system data from run_list and target_systems (writes to temp config)."""
        for run in self.run_list:
            try:
                if isinstance(run, dict):
                    name = list(run.keys())[0]
                    value = list(run.values())[0]
                    if isinstance(value, str) and is_csv_file(value):
                        add_body_system_csv(value, name, temp=True, column_types=self.run_column_descriptions)
                    elif isinstance(value, list):
                        create_body_system_from_other_systems_csv(name, value)
                print(f"Prepared run system: {run}")
            except Exception as e:
                print(f"Failed to prepare run system {run}: {e}")

        for target_system in self.target_systems:
            try:
                if isinstance(target_system, dict):
                    name = list(target_system.keys())[0]
                    value = list(target_system.values())[0]
                    if isinstance(value, str) and is_csv_file(value):
                        add_body_system_csv(value, name, temp=True, column_types=self.target_column_descriptions)
                    elif isinstance(value, list):
                        create_body_system_from_other_systems_csv(name, value)
                print(f"Prepared target system: {target_system}")
            except Exception as e:
                print(f"Failed to prepare target system {target_system}: {e}")

        # Baseline: string (existing system) or dict {name: csv_path or columns_list}
        try:
            if isinstance(self.baseline, str):
                print(f"Using existing baseline system: {self.baseline}")
            elif isinstance(self.baseline, dict):
                baseline_name = list(self.baseline.keys())[0]
                baseline_value = list(self.baseline.values())[0]
                if isinstance(baseline_value, str) and is_csv_file(baseline_value):
                    add_body_system_csv(baseline_value, baseline_name, temp=True, column_types=None)
                elif isinstance(baseline_value, list):
                    create_body_system_from_other_systems_csv(baseline_name, baseline_value)
                print(f"Prepared baseline system: {self.baseline}")
        except Exception as e:
            print(f"Failed to prepare baseline system {self.baseline}: {e}")
    
    def _run_full_and_baseline_system(self, preprocessor: PreprocessFeatures,
                                       baseline_preprocessor: PreprocessFeatures,
                                       feature_name: str, baseline_name: str,
                                       target_name: str):
        """Run full + baseline for all labels in a target system."""
        if self.testing:
            print(f"[DEBUG] _run_full_and_baseline_system: {feature_name} -> {target_name}")
            print(f"[DEBUG] Labels to process: {preprocessor.targets}")
        
        for label_name in preprocessor.targets:
            full_dir = os.path.join(self.save_dir, target_name, label_name, feature_name)
            baseline_dir = os.path.join(self.save_dir, target_name, label_name, baseline_name)

            # Check if any predictions.csv exists (flat or nested structure)
            has_predictions = any(
                f.endswith('predictions.csv') 
                for root, _, files in os.walk(full_dir) 
                for f in files
            ) if os.path.exists(full_dir) else False
            if has_predictions:
                print(f"Skipping {label_name} features: {feature_name}, already finished.")
                continue
            
            if self.testing:
                print(f"[DEBUG] Processing label: {label_name}")
                print(f"[DEBUG] Full dir: {full_dir}")
            
            # Run full model and get valid index
            x, y = preprocessor.preprocess(label_name)
            valid_index = x.index
            
            if self.testing:
                print(f"[DEBUG] Full model - X shape: {x.shape}, y shape: {y.shape}")
            
            self._run_seeding(x, y, full_dir)
            
            # Run baseline filtered to same subjects
            x_base, y_base = baseline_preprocessor.preprocess(label_name, filter_index=valid_index)
            
            if self.testing:
                print(f"[DEBUG] Baseline model - X shape: {x_base.shape}, y shape: {y_base.shape}")
            
            self._run_seeding(x_base, y_base, baseline_dir)

    def _run_seeding(self, x, y, save_dir: str):
        """Run seeding with standard parameters. Handles insufficient subjects edge case."""
        # Check if enough unique subjects for k-fold CV
        n_subjects = x.index.get_level_values(0).nunique()
        if n_subjects < self.num_splits:
            reason = f"insufficient_subjects ({n_subjects} < {self.num_splits} folds)"
            _save_skipped_result(save_dir, reason)
            return {'predictions': pd.DataFrame(), 'metrics': pd.DataFrame(), 'skipped': True}
        
        folds = id_fold_with_stratified_threshold(
            y, 
            seeds=range(self.num_seeds), 
            n_splits=self.num_splits, 
            stratified_threshold=self.stratified_minority_threshold
        )
        os.makedirs(save_dir, exist_ok=True)
        
        return seeding(
            x, y, folds, 
            model_key=self.model_key, 
            average_by_subject_id=True, 
            gender_split_evaluation=True, 
            save_dir=save_dir,
            testing=self.testing
        )

    def get_names(self):
        """Get run names and target names from configuration."""
        run_names = []
        target_names = []
        for run in self.run_list:
            if isinstance(run, dict):
                run_names.append(list(run.keys())[0])
            else:
                run_names.append(run)
        for target_system in self.target_systems:
            if isinstance(target_system, dict):
                target_names.append(list(target_system.keys())[0])
            else:
                target_names.append(target_system)
        return run_names, target_names

    def _get_name(self, item) -> str:
        """Extract name from run_list or target_systems item."""
        if isinstance(item, dict):
            return list(item.keys())[0]
        return item
    
    def _get_columns(self, item, filter_missing: bool = True) -> list:
        """Extract column list from run_list or target_systems item.
        
        Args:
            item: Can be a string (body system name), dict with CSV path, or dict with column list.
            filter_missing: If True, filter out columns that don't exist in any body system.
        """
        if isinstance(item, dict):
            value = list(item.values())[0]
            if isinstance(value, list):
                columns = value
            else:
                # If it's a CSV path, get columns from the registered system
                columns = get_body_system_column_names(list(item.keys())[0])
        else:
            # String = body system name
            columns = get_body_system_column_names(item)
        
        if filter_missing:
            columns = filter_existing_columns(columns)
        
        return columns

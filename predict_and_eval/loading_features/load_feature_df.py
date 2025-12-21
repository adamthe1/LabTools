import os
import tempfile
import pandas as pd
import numpy as np
import json
import warnings
import time
from ..utils.env_loader import *  # Load .env with absolute path

# #region agent log
_DEBUG_LOG = "/home/adamgab/PycharmProjects/GaitPredict/.cursor/debug.log"
def _log(msg, data=None, hyp=None):
    with open(_DEBUG_LOG, "a") as f:
        f.write(json.dumps({"ts": time.time(), "msg": msg, "data": data, "hyp": hyp}) + "\n")
# #endregion


def _atomic_json_write(filepath: str, data: dict, indent: int = 2):
    """Write JSON atomically to prevent corruption on interruption."""
    dir_path = os.path.dirname(filepath)
    with tempfile.NamedTemporaryFile('w', dir=dir_path, delete=False, suffix='.tmp') as tmp:
        json.dump(data, tmp, indent=indent)
        tmp_path = tmp.name
    os.replace(tmp_path, filepath)  # Atomic on POSIX

BODY_SYSTEMS = os.getenv('BODY_SYSTEMS')
TEMP_SYSTEMS = os.getenv('TEMP_SYSTEMS', os.path.join(BODY_SYSTEMS, 'temp_systems'))
JAFAR_BASE = os.getenv('JAFAR_BASE')

# Paths for main and temp description files
_MAIN_DESC_PATH = os.path.join(BODY_SYSTEMS, "body_systems_description", 'dataset_columns.json')
_TEMP_DESC_PATH = os.path.join(BODY_SYSTEMS, "body_systems_description", 'temp_dataset_columns.json')


def load_dataset_filenames_dict() -> dict:
    """Load all body system CSV files."""
    return {
        'Age_Gender_BMI': os.path.join(BODY_SYSTEMS, 'Age_Gender_BMI.csv'),
        'blood_lipids': os.path.join(BODY_SYSTEMS, 'blood_lipids.csv'),\
        'blood_tests_lipids': os.path.join(BODY_SYSTEMS, 'blood_tests_lipids.csv'),
        'body_composition': os.path.join(BODY_SYSTEMS, 'body_composition.csv'),
        'bone_density': os.path.join(BODY_SYSTEMS, 'bone_density.csv'),
        'cardiovascular_system': os.path.join(BODY_SYSTEMS, 'cardiovascular_system.csv'),
        'diet': os.path.join(BODY_SYSTEMS, 'diet.csv'),
        'diet_questions': os.path.join(BODY_SYSTEMS, 'diet_questions.csv'),
        'exercise_logging': os.path.join(BODY_SYSTEMS, 'exercise_logging.csv'),
        'frailty': os.path.join(BODY_SYSTEMS, 'frailty.csv'),
        'gait': os.path.join(BODY_SYSTEMS, 'gait.csv'),
        'glycemic_status': os.path.join(BODY_SYSTEMS, 'glycemic_status.csv'),
        'hematopoietic': os.path.join(BODY_SYSTEMS, 'hematopoietic_system.csv'),
        'immune_system': os.path.join(BODY_SYSTEMS, 'immune_system.csv'),
        'lifestyle': os.path.join(BODY_SYSTEMS, 'lifestyle.csv'),
        'liver': os.path.join(BODY_SYSTEMS, 'liver.csv'),
        'medical_conditions': os.path.join(BODY_SYSTEMS, 'medical_conditions.csv'),
        'medications': os.path.join(BODY_SYSTEMS, 'medications.csv'),
        'mental': os.path.join(BODY_SYSTEMS, 'mental.csv'),
        'metabolites': os.path.join(BODY_SYSTEMS, 'metabolites.csv'),
        'microbiome': os.path.join(BODY_SYSTEMS, 'microbiome.csv'),
        'proteomics': os.path.join(BODY_SYSTEMS, 'proteomics.csv'),
        'rna': os.path.join(BODY_SYSTEMS, 'rna.csv'),
        'nightingale': os.path.join(BODY_SYSTEMS, 'nightingale.csv'),
        'renal_function': os.path.join(BODY_SYSTEMS, 'renal_function.csv'),
        'sleep': os.path.join(BODY_SYSTEMS, 'sleep.csv'),
    }


def prepare_column_json():
	"""
	Read every csv file in the BODY_SYSTEMS directory and create one large json file with the types of systems
	their directory name and column names and descriptions.
	"""
	systems = load_dataset_filenames_dict()
	column_descriptions = {}
	for system, filename in systems.items():
		df = pd.read_csv(filename, index_col=[0, 1])
		# first 2 columns are index and registration code, so turn them into index
		df = df.set_index(['index', 'RegistrationCode'])
		column_names = df.columns.tolist()
		column_descriptions[system]['column_names'] = column_names
		column_descriptions[system]['column_descriptions'] = ""
		column_descriptions[system]['directory'] = filename
	with open(os.path.join(BODY_SYSTEMS, "body_systems_description", 'column_descriptions.json'), 'w') as f:
		json.dump(column_descriptions, f)

def add_body_system_csv(csv_file: str, system: str, temp: bool = True, column_types: dict = None) -> None:
	"""
	Add a csv file to body systems description. Only reads header row for speed.
	Uses temp file by default to avoid polluting main config.
	Raises error if system name already exists in main config (prevents accidental override).
	Pulls type and description from main config if column exists there.
	"""
	# Check if system already exists in main config
	main_systems = _load_main_description_json()
	if system in main_systems:
		raise ValueError(f"System '{system}' already exists in main config. Use a different name or update main config directly.")
	
	# Build lookup of column info from all main systems
	main_column_info = {}
	for sys_data in main_systems.values():
		if sys_data and 'columns' in sys_data:
			for col_name, col_info in sys_data['columns'].items():
				main_column_info[col_name] = col_info
	
	# Read only header (nrows=0) to get column names - much faster than full read
	columns = pd.read_csv(csv_file, index_col=[0, 1], nrows=0).columns.tolist()
	columns_dict = {}
	for col in columns:
		# Pull type and description from main config if available
		if col in main_column_info:
			columns_dict[col] = main_column_info[col].copy()
		else:
			columns_dict[col] = {"description": "", "type": ""}

	# Allow overriding the column types for certain columns
	if column_types is not None:
		for col, col_type in column_types.items():
			if col in columns_dict:
				columns_dict[col]['type'] = col_type

	target_path = _TEMP_DESC_PATH if temp else _MAIN_DESC_PATH
	systems_dict = _load_temp_description_json() if temp else _load_main_description_json()
	systems_dict[system] = {'directory': csv_file, 'columns': columns_dict, 'temp': temp}
	_atomic_json_write(target_path, systems_dict)

def create_body_system_from_other_systems_csv(system: str, column_names: list[str]) -> pd.DataFrame:
	"""Create a temp body system CSV from columns in other systems.
	Raises error if system name already exists in main config.
	"""
	# #region agent log
	_log("create_body_system START", {"system": system, "columns": column_names}, "H1,H2")
	# #endregion
	# Check handled by add_body_system_csv
	df = load_columns_as_df(column_names)
	# #region agent log
	_log("create_body_system LOADED", {"system": system, "df_shape": list(df.shape), "df_cols": list(df.columns)}, "H1,H2")
	# #endregion
	df.to_csv(os.path.join(TEMP_SYSTEMS, f'{system}.csv'))
	add_body_system_csv(os.path.join(TEMP_SYSTEMS, f'{system}.csv'), system, temp=True)

def remove_body_system_csv(system: str) -> None:
	"""Remove a body system from temp descriptions (not main)."""
	temp_dict = _load_temp_description_json()
	if system in temp_dict:
		temp_dict.pop(system)
		_atomic_json_write(_TEMP_DESC_PATH, temp_dict)

def clear_temp_systems() -> None:
	"""Clear all temp systems - call at start of new run to reset."""
	_atomic_json_write(_TEMP_DESC_PATH, {})

def _load_main_description_json() -> dict:
	"""Load the main (permanent) body systems description."""
	with open(_MAIN_DESC_PATH, 'r') as f:
		return json.load(f)

def _load_temp_description_json() -> dict:
	"""Load temp body systems description, creating empty if doesn't exist."""
	if not os.path.exists(_TEMP_DESC_PATH):
		return {}
	with open(_TEMP_DESC_PATH, 'r') as f:
		return json.load(f)

def load_system_description_json() -> dict:
	"""Load merged main + temp body systems descriptions."""
	main = _load_main_description_json()
	temp = _load_temp_description_json()
	return {**main, **temp}  # Temp overrides main if same name

def get_body_system_column_names(system: str) -> list[str]:
	"""
	TOOL
	Get the column names for a given body system.
	"""
	systems_dict = load_system_description_json()
	# New format: columns is a dict with column names as keys
	return list(systems_dict[system]['columns'].keys())

def get_body_system_column_descriptions(system: str) -> dict:
	"""
	Get the column descriptions for a given body system.
	Returns dict of {column_name: {"description": str, "type": str}}
	"""
	systems_dict = load_system_description_json()
	return systems_dict[system]['columns']


def get_column_info(column_name: str) -> dict:
	"""
	Get description and type for a column by searching all body systems.
	Returns {'description': str, 'type': str} or empty strings if not found.
	"""
	systems_dict = load_system_description_json()
	for sys_data in systems_dict.values():
		if not sys_data:
			continue
		col_info = sys_data.get("columns", {}).get(column_name, {})
		if col_info:
			return {
				'description': col_info.get('description', ''),
				'type': col_info.get('type', '')
			}
	return {'description': '', 'type': ''}


def filter_existing_columns(columns: list[str]) -> list[str]:
	"""
	Filter a list of columns to only those that exist in any body system.
	Warns about columns that are not found.
	
	Args:
		columns: List of column names to check.
	
	Returns:
		List of column names that exist in at least one body system.
	"""
	systems_dict = load_system_description_json()
	# Build set of all available columns across all systems
	all_available_columns = set()
	for sys_data in systems_dict.values():
		if sys_data and 'columns' in sys_data:
			all_available_columns.update(sys_data['columns'].keys())
	
	existing = [col for col in columns if col in all_available_columns]
	missing = [col for col in columns if col not in all_available_columns]
	
	# #region agent log
	_log("filter_existing_columns", {"requested": columns, "existing": existing, "missing": missing}, "H1")
	# #endregion
	
	if missing:
		warnings.warn(f"Columns {missing} not found in any body system and will be skipped")
	
	return existing


def load_body_system_filename(system: str) -> str:
	"""
	Get the filename for a given body system.
	"""
	systems_dict = load_system_description_json()
	return systems_dict[system]['directory']


def is_temp_system(system: str) -> bool:
	"""Check if a system is a temp system."""
	systems_dict = load_system_description_json()
	return systems_dict.get(system, {}).get('temp', False)


def get_temp_systems_row_counts(feature_system: str, target_system: str) -> dict:
	"""
	Get row counts for temp systems (fast - reads only first column).
	Returns {system_name: row_count} for systems marked as temp.
	"""
	systems_dict = load_system_description_json()
	counts = {}
	for system in [feature_system, target_system]:
		sys_data = systems_dict.get(system, {})
		if sys_data.get('temp', False):
			path = sys_data.get('directory', '')
			if path and os.path.exists(path):
				# Read only first column, count rows (fast)
				counts[system] = len(pd.read_csv(path, usecols=[0]))
	return counts


def load_body_system_df(system: str, specific_columns: list[str] = None) -> pd.DataFrame:
	"""
	TOOL
	Load the body system dataframe.
	Can also load specific columns if desired.
	:param system: The body system to load.
	:param specific_columns: A list of columns to load. If None, all columns will be loaded. 
	:return: A dataframe with the body system data.
	"""
	filename = load_body_system_filename(system)
	df = pd.read_csv(filename, index_col=[0, 1])
	if specific_columns is not None:
		if not all(col in df.columns for col in specific_columns):
			raise ValueError(f"Columns {specific_columns} not found in {filename}")
		df = df[specific_columns]
	return df

def load_feature_target_systems_as_df(feature_system: str, target_system: str, confounders: list[str] = None,
                                      merge_closest: bool = False) -> pd.DataFrame:
	"""Load feature and target systems merged into one dataframe."""
	feature_df = load_body_system_df(feature_system)
	if confounders is not None and len(confounders) > 0:
		# Load confounder columns as DataFrame first
		confounders_df = load_columns_as_df(confounders, merge_closest_research_stage=merge_closest)
		if merge_closest:
			feature_df = _merge_closest_research_stage(feature_df, confounders_df)
		else:
			feature_df = pd.merge(feature_df, confounders_df, left_index=True, right_index=True, how='left')
	target_df = load_body_system_df(target_system)
	if merge_closest:
		return _merge_closest_research_stage(feature_df, target_df)
	return pd.merge(feature_df, target_df, left_index=True, right_index=True, how='left')


def _merge_closest_research_stage(left_df: pd.DataFrame, right_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Merge on RegistrationCode + research_stage, falling back to any available 
	research_stage when exact match unavailable.
	"""
	left = left_df.reset_index()
	right = right_df.reset_index()
	
	# Get new columns from right (exclude index cols and duplicates)
	index_cols = ['RegistrationCode', 'research_stage']
	new_cols = [c for c in right.columns if c not in index_cols and c not in left.columns]
	if not new_cols:
		return left_df
	right = right[index_cols + new_cols]
	
	# Exact merge first
	result = left.merge(right, on=index_cols, how='left')
	unmatched = result[new_cols[0]].isna()
	
	if unmatched.any():
		# Build fallback lookup: last row per subject
		fallback = right.groupby('RegistrationCode')[new_cols].last()
		# Fill unmatched rows via RegistrationCode lookup
		unmatched_codes = result.loc[unmatched, 'RegistrationCode']
		result.loc[unmatched, new_cols] = fallback.reindex(unmatched_codes).values
	
	return result.set_index(index_cols)


def load_columns_as_df(columns_to_load: list[str], anchor_columns: list[str] = None, 
                       merge_closest_research_stage: bool = False,
                       priority_temp_systems: bool = False) -> pd.DataFrame:
	"""
	Load columns from body systems and merge into one dataframe.
	
	Args:
		columns_to_load: Column names to load.
		anchor_columns: If provided, load these first to define subject set.
		merge_closest_research_stage: Fall back to any research_stage if exact match unavailable.
		priority_temp_systems: Load from temp systems before main systems.
	"""
	# Build system lookup (temp first if priority)
	if priority_temp_systems:
		temp, main = _load_temp_description_json(), _load_main_description_json()
		body_systems = {**temp, **{k: v for k, v in main.items() if k not in temp}}
	else:
		body_systems = load_system_description_json()
	
	# Build column -> system mapping
	col_to_system = {}
	for system, sdict in body_systems.items():
		for col in sdict['columns'].keys():
			if col not in col_to_system:  # First system wins
				col_to_system[col] = system
	
	# Order columns: anchor first (if specified), then rest
	ordered_cols = list(anchor_columns or []) + [c for c in columns_to_load if c not in (anchor_columns or [])]
	
	# Group columns by system for batch loading
	system_cols = {}
	missing = []
	for col in ordered_cols:
		if col in col_to_system:
			system_cols.setdefault(col_to_system[col], []).append(col)
		else:
			missing.append(col)
	
	if missing:
		warnings.warn(f"Columns {missing} not found in any body system")
	
	# Load and merge
	merge_fn = _merge_closest_research_stage if merge_closest_research_stage else \
	           lambda l, r: pd.merge(l, r, left_index=True, right_index=True, how='left')
	
	df = None
	for system, cols in system_cols.items():
		sys_df = load_body_system_df(system, cols)
		df = sys_df if df is None else merge_fn(df, sys_df)
	
	if df is None:
		raise ValueError("No columns found in any body system")
	return df

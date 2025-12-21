import os
import numpy as np
import pandas as pd
import json
import warnings
from collections import defaultdict
from .env_loader import *  # Load .env with absolute path
from .categorical_utils import CategoricalUtils

BODY_SYSTEMS = os.getenv('BODY_SYSTEMS')
TEMP_SYSTEMS = os.getenv('TEMP_SYSTEMS', os.path.join(BODY_SYSTEMS, 'temp_systems'))
JAFAR_BASE = os.getenv('JAFAR_BASE')
OUTPUT_DIR = os.path.join(BODY_SYSTEMS, "body_systems_description")
LLM_RESULTS_FILE = os.path.join(OUTPUT_DIR, 'column_descriptions_claude.json')
DATASET_COLUMNS_FILE = os.path.join(OUTPUT_DIR, 'dataset_columns.json')

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
        'high_level_diet': os.path.join(BODY_SYSTEMS, 'high_level_diet.csv'),
        'immune_system': os.path.join(BODY_SYSTEMS, 'immune_system.csv'),
        'lifestyle': os.path.join(BODY_SYSTEMS, 'lifestyle.csv'),
        'liver': os.path.join(BODY_SYSTEMS, 'liver.csv'),
        'medical_conditions': os.path.join(BODY_SYSTEMS, 'medical_conditions.csv'),
        'medications': os.path.join(BODY_SYSTEMS, 'medications.csv'),
        'mental': os.path.join(BODY_SYSTEMS, 'mental.csv'),
        'metabolites': os.path.join(BODY_SYSTEMS, 'metabolites.csv'),
        'microbiome': os.path.join(BODY_SYSTEMS, 'microbiome.csv'),
        'proteomics': os.path.join(BODY_SYSTEMS, 'proteomics.csv'),
        #'rna': os.path.join(BODY_SYSTEMS, 'rna.csv'),
        'nightingale': os.path.join(BODY_SYSTEMS, 'nightingale.csv'),
        'renal_function': os.path.join(BODY_SYSTEMS, 'renal_function.csv'),
        'sleep': os.path.join(BODY_SYSTEMS, 'sleep.csv'),
    }


def load_llm_results() -> dict:
	"""Load LLM-generated column descriptions and types."""
	if os.path.exists(LLM_RESULTS_FILE):
		with open(LLM_RESULTS_FILE, 'r') as f:
			return json.load(f)
	return {}


def load_existing_dataset_columns() -> dict:
	"""Load existing dataset_columns.json if it exists."""
	if os.path.exists(DATASET_COLUMNS_FILE):
		with open(DATASET_COLUMNS_FILE, 'r') as f:
			return json.load(f)
	return {}


def build_existing_descriptions_lookup(existing_data: dict) -> dict:
	"""Build a lookup of column_name -> description from existing dataset_columns.json."""
	lookup = {}
	for sys_data in existing_data.values():
		if sys_data and 'columns' in sys_data:
			for col_name, col_info in sys_data['columns'].items():
				if col_info.get('description') and col_name not in lookup:
					lookup[col_name] = col_info['description']
	return lookup


def prepare_column_json():
	"""
	Read every csv file in the BODY_SYSTEMS directory and create one large json file with the types of systems
	their directory name and column names and descriptions.
	Uses LLM-generated descriptions and types from column_descriptions_claude.json.
	
	Features:
	- Reuses existing descriptions by column name (checks existing dataset_columns.json)
	- Tracks duplicate columns across systems (adds duplicate_in_systems field)
	- Adds statistics: min/max/mean/std for regression, n_positives for categorical
	"""
	systems = load_dataset_filenames_dict()
	llm_results = load_llm_results()
	existing_data = load_existing_dataset_columns()
	existing_desc_lookup = build_existing_descriptions_lookup(existing_data)
	
	print(f"Found {len(existing_desc_lookup)} existing column descriptions to reuse")
	
	# First pass: build column -> systems mapping for duplicate detection
	column_to_systems = defaultdict(list)
	for system, filename in systems.items():
		if not os.path.exists(filename):
			continue
		df = pd.read_csv(filename, index_col=[0, 1], nrows=0)  # Just headers
		for col in df.columns:
			column_to_systems[col].append(system)
	
	# Count duplicates
	duplicates = {col: syss for col, syss in column_to_systems.items() if len(syss) > 1}
	print(f"Found {len(duplicates)} columns appearing in multiple systems")
	
	column_descriptions = {}
	
	for system, filename in systems.items():
		if not os.path.exists(filename):
			print(f"Skipping {system}: file not found")
			continue
		
		# Read full data for statistics
		df = pd.read_csv(filename, index_col=[0, 1])
		column_names = df.columns.tolist()
		
		# Get LLM data for this system if available
		llm_system_data = llm_results.get(system, {}).get("columns", {})
		
		column_descriptions[system] = {
			'directory': filename,
			'columns': {}
		}
		
		for col in column_names:
			llm_col_data = llm_system_data.get(col, {})
			col_data = df[col]
			
			# Get type: prefer LLM final_type, fallback to heuristic
			final_type = llm_col_data.get("final_type", "")
			if not final_type:
				is_cat = CategoricalUtils.is_categorical(col_data)
				final_type = "categorical" if is_cat else "regression"
			
			# Get description: first check existing, then LLM, then empty
			description = existing_desc_lookup.get(col, "") or llm_col_data.get("description", "")
			
			# Build column info
			col_info = {
				"description": description,
				"type": final_type,
				"n_unique": int(col_data.nunique()),
			}
			
			# Add statistics based on type
			if final_type == "categorical":
				# n_positives: count of non-zero/True values
				n_positives = int((col_data.fillna(0) != 0).sum())
				col_info["n_positives"] = n_positives
			else:
				# Regression: min, max, mean, std
				col_info["min"] = float(col_data.min()) if not pd.isna(col_data.min()) else None
				col_info["max"] = float(col_data.max()) if not pd.isna(col_data.max()) else None
				col_info["mean"] = float(col_data.mean()) if not pd.isna(col_data.mean()) else None
				col_info["std"] = float(col_data.std()) if not pd.isna(col_data.std()) else None
			
			# Add duplicate_in_systems if column appears in multiple systems
			if col in duplicates:
				col_info["duplicate_in_systems"] = duplicates[col]
			
			column_descriptions[system]['columns'][col] = col_info
		
		print(f"Processed {system}: {len(column_names)} columns")
	
	# Ensure output directory exists
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	
	output_file = os.path.join(OUTPUT_DIR, 'dataset_columns.json')
	with open(output_file, 'w') as f:
		json.dump(column_descriptions, f, indent=2)
	
	# Print summary
	total_cols = sum(len(s['columns']) for s in column_descriptions.values())
	type_counts = {'categorical': 0, 'regression': 0, 'ordinal': 0}
	for sys_data in column_descriptions.values():
		for col_data in sys_data['columns'].values():
			t = col_data['type']
			type_counts[t] = type_counts.get(t, 0) + 1
	
	print(f"\nSaved {total_cols} columns across {len(column_descriptions)} systems to {output_file}")
	print(f"Types: {type_counts}")
	print(f"Duplicate columns: {len(duplicates)}")
	return column_descriptions


def prepare_column_json_heuristic_only():
	"""
	Original version: uses only heuristic to determine types.
	"""
	systems = load_dataset_filenames_dict()
	column_descriptions = {}
	
	for system, filename in systems.items():
		if not os.path.exists(filename):
			print(f"Skipping {system}: file not found")
			continue
		
		df = pd.read_csv(filename, index_col=[0, 1])
		column_names = df.columns.tolist()
		
		column_descriptions[system] = {
			'directory': filename,
			'columns': {}
		}
		
		for col in column_names:
			is_cat = CategoricalUtils.is_categorical(df[col])
			column_descriptions[system]['columns'][col] = {
				"description": "",
				"type": "categorical" if is_cat else "regression"
			}
	
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	
	with open(os.path.join(OUTPUT_DIR, 'column_descriptions.json'), 'w') as f:
		json.dump(column_descriptions, f, indent=2)
	
	print(f"Saved descriptions for {len(column_descriptions)} systems")


if __name__ == "__main__":
	prepare_column_json()

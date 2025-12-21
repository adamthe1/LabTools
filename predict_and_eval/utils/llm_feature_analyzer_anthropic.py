"""
Anthropic Claude-powered pipeline to analyze CSV columns with DuckDuckGo web search.
Uses Claude with real web search to understand physiological features.

Requirements:
    pip install anthropic ddgs pandas python-dotenv
"""
import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional
import anthropic
import time
from .env_loader import *  # Load .env with absolute path
from .categorical_utils import CategoricalUtils

BODY_SYSTEMS = os.getenv('BODY_SYSTEMS')
OUTPUT_DIR = os.path.join(BODY_SYSTEMS, "body_systems_description")

# Try to import ddgs (formerly duckduckgo_search)
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    print("Warning: ddgs not installed. Run: pip install ddgs")


def get_anthropic_client() -> anthropic.Anthropic:
    """Initialize Anthropic client."""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    return anthropic.Anthropic(api_key=api_key)


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
        'rna': os.path.join(BODY_SYSTEMS, 'rna.csv'),
        'nightingale': os.path.join(BODY_SYSTEMS, 'nightingale.csv'),
        'renal_function': os.path.join(BODY_SYSTEMS, 'renal_function.csv'),
        'sleep': os.path.join(BODY_SYSTEMS, 'sleep.csv'),
    }


def get_column_sample_info(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Extract sample values and statistics for a column."""
    col_data = df[column]
    sample_values = col_data.dropna().head(6).tolist()
    return {
        "sample_values": sample_values,
        "dtype": str(col_data.dtype),
        "n_unique": int(col_data.nunique()),
        "n_null": int(col_data.isna().sum()),
        "n_total": len(col_data),
    }


def duckduckgo_search(query: str, max_results: int = 3) -> str:
    """
    Perform a DuckDuckGo web search and return formatted results.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
    
    Returns:
        Formatted string with search results
    """
    if not DDGS_AVAILABLE:
        return f"[Web search unavailable] Query: {query}"
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        
        if not results:
            return f"No results found for: {query}"
        
        # Format results
        formatted = []
        for r in results:
            formatted.append(f"Title: {r.get('title', 'N/A')}\nSnippet: {r.get('body', 'N/A')}")
        
        return "\n---\n".join(formatted)
    except Exception as e:
        return f"Search error: {e}"


def search_biomarker_info(column_name: str, system_context: str) -> str:
    """Search for medical information about a biomarker/feature."""
    # Clean column name for better search
    clean_name = column_name.replace('_', ' ').replace('.', ' ')
    query = f"{clean_name} biomarker medical meaning {system_context}"
    return duckduckgo_search(query, max_results=2)


def analyze_columns_with_claude_and_search(client: anthropic.Anthropic, system_name: str,
                                            columns_info: Dict[str, Dict],
                                            use_web_search: bool = True) -> Dict[str, Dict]:
    """
    Query Claude to analyze columns, optionally enriched with web search results.
    
    Args:
        client: Anthropic client
        system_name: Name of the body system being analyzed
        columns_info: Dict of column names to their sample info
        use_web_search: Whether to perform web searches for unknown terms
    """
    # First, identify columns that might need web search (unfamiliar terms)
    search_results = {}
    if use_web_search and DDGS_AVAILABLE:
        print(f"    Searching web for unfamiliar biomarkers...")
        for col_name in columns_info.keys():
            # Skip obvious columns
            if any(skip in col_name.lower() for skip in ['age', 'gender', 'sex', 'bmi', 'weight', 'height']):
                continue
            search_results[col_name] = search_biomarker_info(col_name, system_name)
            time.sleep(0.3)  # Rate limit DuckDuckGo
    
    # Build prompt with search context
    columns_text = ""
    for col_name, info in columns_info.items():
        columns_text += f"""
Column: {col_name}
  Data type: {info['dtype']}
  Unique values: {info['n_unique']} / {info['n_total']} total
  Sample values: {info['sample_values'][:5]}"""
        
        if col_name in search_results:
            columns_text += f"\n  Web search context: {search_results[col_name]}"
        columns_text += "\n"
    
    prompt = f"""You are a medical data scientist analyzing physiological biomarker data from a clinical study.

SYSTEM: {system_name}

For each column, I need you to:
1. Identify what this biomarker/feature measures (use your medical knowledge + provided web search context)
2. Classify it as "categorical" or "regression" for machine learning

Classification guidelines:
- CATEGORICAL: Binary flags, diagnoses (yes/no), medication use, discrete categories, ordinal scales with <10 levels
- REGRESSION: Continuous measurements (blood levels, concentrations, percentages, counts, ratios, physical measurements)

COLUMNS TO ANALYZE:
{columns_text}

Respond with a JSON object where each key is the column name:
{{
    "column_name": {{
        "description": "Medical description of what this measures",
        "type": "categorical" or "regression",
        "reasoning": "Brief explanation of classification"
    }}
}}

IMPORTANT: Output ONLY valid JSON, no other text."""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        # Handle markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        return json.loads(response_text.strip())
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        return {}
    except Exception as e:
        print(f"  Claude error for {system_name}: {e}")
        return {}


def analyze_single_system(client: anthropic.Anthropic, system_name: str, 
                          filepath: str, batch_size: int = 15,
                          use_web_search: bool = True,
                          existing_col_descriptions: Dict[str, Dict] = None) -> Dict[str, Any]:
    """Analyze all columns in a single CSV file.
    
    Args:
        existing_col_descriptions: Dict of column_name -> {description, type} from previous runs.
                                   Columns with existing descriptions will skip LLM.
    """
    print(f"\nAnalyzing: {system_name}")
    
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return None
    
    try:
        df = pd.read_csv(filepath, index_col=[0, 1], nrows=100)
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return None
    
    columns = [c for c in df.columns if c not in ['RegistrationCode', 'research_stage']]
    print(f"  Found {len(columns)} columns")
    
    # Collect column info
    columns_info = {col: get_column_sample_info(df, col) for col in columns}
    
    # Skip LLM for metabolites, microbiome, and rna - use default descriptions
    skip_llm_systems = {
        'metabolites': ('Metabolite biomarker', 'regression'),
        'microbiome': ('Microbiome species prevalence', 'regression'),
        'rna': ('RNA expression level', 'regression'),
    }
    
    all_results = {}
    existing_col_descriptions = existing_col_descriptions or {}
    
    if system_name in skip_llm_systems:
        desc, typ = skip_llm_systems[system_name]
        print(f"  Skipping LLM (using default: {typ})")
        for col in columns:
            all_results[col] = {
                "description": desc,
                "type": typ,
                "reasoning": f"Default assignment for {system_name}"
            }
    else:
        # Separate columns: those with existing descriptions vs those needing LLM
        cols_with_desc = []
        cols_need_llm = []
        for col in columns:
            if col in existing_col_descriptions and existing_col_descriptions[col].get("description"):
                cols_with_desc.append(col)
                all_results[col] = {
                    "description": existing_col_descriptions[col]["description"],
                    "type": existing_col_descriptions[col].get("type", "unknown"),
                    "reasoning": "Reused from existing description"
                }
            else:
                cols_need_llm.append(col)
        
        if cols_with_desc:
            print(f"  Reusing {len(cols_with_desc)} existing descriptions")
        
        # Process only columns needing LLM in batches
        if cols_need_llm:
            print(f"  Processing {len(cols_need_llm)} columns with LLM")
            for i in range(0, len(cols_need_llm), batch_size):
                batch_cols = cols_need_llm[i:i+batch_size]
                batch_info = {c: columns_info[c] for c in batch_cols}
                batch_num = i // batch_size + 1
                total_batches = (len(cols_need_llm) - 1) // batch_size + 1
                print(f"  Processing batch {batch_num}/{total_batches} ({len(batch_cols)} columns)")
                
                llm_results = analyze_columns_with_claude_and_search(
                    client, system_name, batch_info, use_web_search=use_web_search
                )
                all_results.update(llm_results)
                time.sleep(1)  # Rate limiting for Claude API
    
    # Build final result combining LLM and heuristic
    system_result = {
        "filepath": filepath,
        "n_columns": len(columns),
        "columns": {}
    }
    
    for col in columns:
        heuristic_is_cat = CategoricalUtils.is_categorical(df[col])
        llm_info = all_results.get(col, {})
        
        llm_type = llm_info.get("type", "unknown")
        heuristic_type = "categorical" if heuristic_is_cat else "regression"
        
        # Final decision: prefer LLM for medical knowledge
        if llm_type == "unknown":
            final_type = heuristic_type
        else:
            final_type = llm_type
        
        system_result["columns"][col] = {
            "description": llm_info.get("description", ""),
            "llm_type": llm_type,
            "llm_reasoning": llm_info.get("reasoning", ""),
            "heuristic_type": heuristic_type,
            "final_type": final_type,
            "conflict": llm_type != heuristic_type and llm_type != "unknown",
            "sample_values": columns_info[col]["sample_values"],
            "n_unique": columns_info[col]["n_unique"],
        }
    
    return system_result


def run_full_pipeline(output_file: str = None, systems_to_process: List[str] = None,
                      resume: bool = True, use_web_search: bool = True):
    """
    Main pipeline: analyze all CSVs and save results to JSON.
    
    Args:
        output_file: Path to save JSON results
        systems_to_process: List of system names to process (default: all)
        resume: Whether to reuse existing column descriptions (by column name, not by system)
        use_web_search: Whether to use DuckDuckGo web search for context
    """
    if output_file is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, 'column_descriptions_claude.json')
    
    client = get_anthropic_client()
    datasets = load_dataset_filenames_dict()
    
    if systems_to_process:
        datasets = {k: v for k, v in datasets.items() if k in systems_to_process}
    
    # Load existing results
    results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing systems from file")
    
    # Build lookup of existing column descriptions (by column name across all systems)
    # This allows reusing descriptions for columns that appear in multiple systems
    existing_col_descriptions = {}
    if resume:
        for sys_data in results.values():
            if sys_data and "columns" in sys_data:
                for col_name, col_data in sys_data["columns"].items():
                    # Only store if we have a description and don't already have one for this column
                    if col_data.get("description") and col_name not in existing_col_descriptions:
                        existing_col_descriptions[col_name] = {
                            "description": col_data["description"],
                            "type": col_data.get("final_type", col_data.get("type", "unknown"))
                        }
        print(f"Found {len(existing_col_descriptions)} existing column descriptions to reuse")
    
    print(f"Web search: {'enabled' if use_web_search and DDGS_AVAILABLE else 'disabled'}")
    
    # Process ALL systems - check for new columns in each
    for system_name, filepath in datasets.items():
        # Check if system has new columns that need processing
        if system_name in results and results[system_name]:
            # Get existing columns for this system
            existing_system_cols = set(results[system_name].get("columns", {}).keys())
            # Get current columns from CSV
            try:
                df_check = pd.read_csv(filepath, index_col=[0, 1], nrows=0)
                current_cols = set(df_check.columns)
                new_cols = current_cols - existing_system_cols
                
                if not new_cols:
                    print(f"\nSkipping {system_name}: no new columns (has {len(existing_system_cols)} columns)")
                    continue
                else:
                    print(f"\n{system_name}: Found {len(new_cols)} new columns (out of {len(current_cols)} total)")
            except Exception as e:
                print(f"\nError checking {system_name}: {e}, will reprocess")
        
        system_result = analyze_single_system(
            client, system_name, filepath, use_web_search=use_web_search,
            existing_col_descriptions=existing_col_descriptions
        )
        if system_result:
            # Merge with existing results if system was already processed
            if system_name in results and results[system_name]:
                # Keep existing columns, add/update new ones
                existing_columns = results[system_name].get("columns", {})
                new_columns = system_result["columns"]
                # Merge: existing columns + new columns (new ones overwrite if duplicate)
                merged_columns = {**existing_columns, **new_columns}
                system_result["columns"] = merged_columns
                system_result["n_columns"] = len(merged_columns)
            
            results[system_name] = system_result
            
            # Update existing_col_descriptions with newly processed columns
            for col_name, col_data in system_result["columns"].items():
                if col_data.get("description") and col_name not in existing_col_descriptions:
                    existing_col_descriptions[col_name] = {
                        "description": col_data["description"],
                        "type": col_data.get("final_type", "unknown")
                    }
            # Save after each system for fault tolerance
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  ✓ Saved {system_name}")
    
    print(f"\n=== Pipeline Complete ===")
    print(f"Results saved to: {output_file}")
    return results


def export_to_categorical_params(results: Dict, output_file: str = None):
    """Export final classifications to categorical_params.json format for CategoricalUtils."""
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, 'categorical_params_llm.json')
    
    params = {}
    for sys_data in results.values():
        if not sys_data:
            continue
        for col_name, col_data in sys_data["columns"].items():
            params[col_name] = "cat" if col_data["final_type"] == "categorical" else "reg"
    
    with open(output_file, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Exported {len(params)} column classifications to: {output_file}")
    return params


def print_conflict_summary(results: Dict):
    """Print summary of LLM vs heuristic conflicts for manual review."""
    print("\n=== Conflicts (LLM vs Heuristic) ===")
    conflicts = []
    for sys_name, sys_data in results.items():
        if not sys_data:
            continue
        for col_name, col_data in sys_data["columns"].items():
            if col_data.get("conflict"):
                conflicts.append({
                    "system": sys_name,
                    "column": col_name,
                    "llm": col_data["llm_type"],
                    "heuristic": col_data["heuristic_type"],
                    "reasoning": col_data["llm_reasoning"],
                    "samples": col_data["sample_values"][:3]
                })
    
    if not conflicts:
        print("No conflicts found!")
        return
    
    print(f"Found {len(conflicts)} conflicts:\n")
    for c in conflicts:
        print(f"{c['system']}.{c['column']}:")
        print(f"  LLM: {c['llm']} | Heuristic: {c['heuristic']}")
        print(f"  Reasoning: {c['reasoning']}")
        print(f"  Samples: {c['samples']}\n")


# === Interactive single-column analysis with web search ===
def analyze_single_column_interactive(column_name: str, sample_values: list, 
                                       system_context: str = "clinical biomarker") -> Dict:
    """
    Analyze a single column interactively with web search.
    Useful for testing or manual analysis.
    """
    print(f"Analyzing: {column_name}")
    print(f"Sample values: {sample_values[:5]}")
    
    # Web search
    search_result = search_biomarker_info(column_name, system_context)
    print(f"\nWeb search results:\n{search_result}\n")
    
    # LLM analysis
    client = get_anthropic_client()
    prompt = f"""Analyze this biomarker column:

Column: {column_name}
Sample values: {sample_values[:5]}
Web search context: {search_result}

Provide:
1. Medical description
2. Type: "categorical" or "regression"
3. Reasoning

Output as JSON."""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    print(f"LLM Analysis:\n{response.content[0].text}")
    return {"column": column_name, "analysis": response.content[0].text}


if __name__ == "__main__":
    # Run the full pipeline with web search enabled
    results = run_full_pipeline(use_web_search=True)
    print_conflict_summary(results)
    export_to_categorical_params(results)

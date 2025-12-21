"""
Interactive script to review LLM vs heuristic conflicts one by one.
Allows manual override of classifications.
"""
import os
import json
from .env_loader import *  # Load .env with absolute path

BODY_SYSTEMS = os.getenv('BODY_SYSTEMS')
OUTPUT_DIR = os.path.join(BODY_SYSTEMS, "body_systems_description")
RESULTS_FILE = os.path.join(OUTPUT_DIR, 'column_descriptions_claude.json')
OVERRIDES_FILE = os.path.join(OUTPUT_DIR, 'manual_overrides.json')


def load_results() -> dict:
    """Load the LLM analysis results."""
    with open(RESULTS_FILE, 'r') as f:
        return json.load(f)


def load_overrides() -> dict:
    """Load existing manual overrides."""
    if os.path.exists(OVERRIDES_FILE):
        with open(OVERRIDES_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_overrides(overrides: dict):
    """Save manual overrides."""
    with open(OVERRIDES_FILE, 'w') as f:
        json.dump(overrides, f, indent=2)


def get_conflicts(results: dict) -> list:
    """Extract all conflicts from results."""
    conflicts = []
    for sys_name, sys_data in results.items():
        if not sys_data:
            continue
        for col_name, col_data in sys_data.get("columns", {}).items():
            if col_data.get("conflict"):
                conflicts.append({
                    "system": sys_name,
                    "column": col_name,
                    "llm_type": col_data["llm_type"],
                    "heuristic_type": col_data["heuristic_type"],
                    "description": col_data.get("description", ""),
                    "reasoning": col_data.get("llm_reasoning", ""),
                    "samples": col_data.get("sample_values", [])[:5],
                    "n_unique": col_data.get("n_unique", "?"),
                    "final_type": col_data.get("final_type", ""),
                })
    return conflicts


def display_conflict(conflict: dict, idx: int, total: int):
    """Display a single conflict for review."""
    print("\n" + "=" * 70)
    print(f"CONFLICT {idx + 1}/{total}")
    print("=" * 70)
    print(f"System:      {conflict['system']}")
    print(f"Column:      {conflict['column']}")
    print(f"Description: {conflict['description']}")
    print("-" * 70)
    print(f"LLM says:       {conflict['llm_type'].upper()}")
    print(f"Heuristic says: {conflict['heuristic_type'].upper()}")
    print(f"Current final:  {conflict['final_type'].upper()}")
    print("-" * 70)
    print(f"LLM Reasoning: {conflict['reasoning']}")
    print(f"Unique values: {conflict['n_unique']}")
    print(f"Sample values: {conflict['samples']}")
    print("=" * 70)


def review_conflicts_interactive():
    """Interactive review of all conflicts."""
    results = load_results()
    overrides = load_overrides()
    conflicts = get_conflicts(results)
    
    if not conflicts:
        print("No conflicts found!")
        return
    
    print(f"\nFound {len(conflicts)} conflicts to review.")
    print("Commands: [c]ategorical, [r]egression, [l]lm (keep LLM), [h]euristic, [s]kip, [q]uit\n")
    
    reviewed = 0
    for idx, conflict in enumerate(conflicts):
        key = f"{conflict['system']}.{conflict['column']}"
        
        # Skip if already overridden
        if key in overrides:
            print(f"[{idx+1}/{len(conflicts)}] {key}: already overridden to '{overrides[key]}', skipping...")
            continue
        
        display_conflict(conflict, idx, len(conflicts))
        
        while True:
            choice = input("\nYour choice [c/r/l/h/s/q]: ").strip().lower()
            
            if choice == 'c':
                overrides[key] = 'categorical'
                print(f"  -> Set to CATEGORICAL")
                reviewed += 1
                break
            elif choice == 'r':
                overrides[key] = 'regression'
                print(f"  -> Set to REGRESSION")
                reviewed += 1
                break
            elif choice == 'l':
                overrides[key] = conflict['llm_type']
                print(f"  -> Keeping LLM choice: {conflict['llm_type'].upper()}")
                reviewed += 1
                break
            elif choice == 'h':
                overrides[key] = conflict['heuristic_type']
                print(f"  -> Using heuristic: {conflict['heuristic_type'].upper()}")
                reviewed += 1
                break
            elif choice == 's':
                print("  -> Skipped")
                break
            elif choice == 'q':
                print(f"\nQuitting. Reviewed {reviewed} conflicts.")
                save_overrides(overrides)
                print(f"Saved {len(overrides)} overrides to {OVERRIDES_FILE}")
                return
            else:
                print("Invalid choice. Use c/r/l/h/s/q")
    
    save_overrides(overrides)
    print(f"\n=== Review Complete ===")
    print(f"Reviewed {reviewed} new conflicts")
    print(f"Total overrides: {len(overrides)}")
    print(f"Saved to: {OVERRIDES_FILE}")


def apply_overrides_to_results():
    """Apply manual overrides back to the results file."""
    results = load_results()
    overrides = load_overrides()
    
    applied = 0
    for key, override_type in overrides.items():
        system, column = key.split('.', 1)
        if system in results and results[system]:
            if column in results[system].get("columns", {}):
                results[system]["columns"][column]["final_type"] = override_type
                results[system]["columns"][column]["manual_override"] = True
                applied += 1
    
    # Save updated results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Applied {applied} overrides to {RESULTS_FILE}")
    return results


def export_final_params(results: dict = None):
    """Export final classifications after overrides."""
    if results is None:
        results = load_results()
    
    output_file = os.path.join(OUTPUT_DIR, 'categorical_params_final.json')
    params = {}
    
    for sys_data in results.values():
        if not sys_data:
            continue
        for col_name, col_data in sys_data.get("columns", {}).items():
            params[col_name] = "cat" if col_data.get("final_type") == "categorical" else "reg"
    
    with open(output_file, 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"Exported {len(params)} final classifications to: {output_file}")
    return params


def show_summary():
    """Show summary of current state."""
    results = load_results()
    overrides = load_overrides()
    conflicts = get_conflicts(results)
    
    print("\n=== Summary ===")
    print(f"Total systems: {len(results)}")
    total_cols = sum(len(s.get('columns', {})) for s in results.values() if s)
    print(f"Total columns: {total_cols}")
    print(f"Total conflicts: {len(conflicts)}")
    print(f"Manual overrides: {len(overrides)}")
    
    # Count by type
    cat_count = 0
    reg_count = 0
    ord_count = 0
    for sys_data in results.values():
        if not sys_data:
            continue
        for col_data in sys_data.get("columns", {}).values():
            final_type = col_data.get("final_type", "")
            if final_type == "categorical":
                cat_count += 1
            elif final_type == "ordinal":
                ord_count += 1
            else:
                reg_count += 1
    
    print(f"Categorical: {cat_count}")
    print(f"Regression: {reg_count}")
    print(f"Ordinal: {ord_count}")


def set_system_to_ordinal(system_names: list):
    """Set all columns in specified systems to ordinal type."""
    results = load_results()
    
    changed = 0
    for sys_name in system_names:
        if sys_name not in results or not results[sys_name]:
            print(f"System '{sys_name}' not found, skipping...")
            continue
        
        for col_name in results[sys_name].get("columns", {}):
            results[sys_name]["columns"][col_name]["final_type"] = "ordinal"
            results[sys_name]["columns"][col_name]["manual_override"] = True
            changed += 1
        
        print(f"Set {len(results[sys_name].get('columns', {}))} columns in '{sys_name}' to ordinal")
    
    # Save updated results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTotal: {changed} columns set to ordinal")
    return results


def set_diet_to_ordinal():
    """Set diet_questions to ordinal regression."""
    return set_system_to_ordinal(['diet_questions'])


def restore_system_to_llm(system_names: list):
    """Restore columns in specified systems back to LLM's original classification."""
    results = load_results()
    
    changed = 0
    for sys_name in system_names:
        if sys_name not in results or not results[sys_name]:
            print(f"System '{sys_name}' not found, skipping...")
            continue
        
        for col_name, col_data in results[sys_name].get("columns", {}).items():
            llm_type = col_data.get("llm_type", "regression")
            if llm_type == "unknown":
                llm_type = col_data.get("heuristic_type", "regression")
            results[sys_name]["columns"][col_name]["final_type"] = llm_type
            # Remove manual override flag
            if "manual_override" in results[sys_name]["columns"][col_name]:
                del results[sys_name]["columns"][col_name]["manual_override"]
            changed += 1
        
        print(f"Restored {len(results[sys_name].get('columns', {}))} columns in '{sys_name}' to LLM type")
    
    # Save updated results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTotal: {changed} columns restored to LLM classification")
    return results


def list_conflicts():
    """Print a simple list of all conflicts: system.column"""
    results = load_results()
    conflicts = get_conflicts(results)
    
    if not conflicts:
        print("No conflicts found!")
        return
    
    print(f"=== {len(conflicts)} Conflicts ===\n")
    
    # Group by system
    by_system = {}
    for c in conflicts:
        sys_name = c['system']
        if sys_name not in by_system:
            by_system[sys_name] = []
        by_system[sys_name].append(c['column'])
    
    for sys_name, columns in sorted(by_system.items()):
        print(f"\n{sys_name} ({len(columns)}):")
        for col in columns:
            # Find the conflict to get types
            conflict = next(c for c in conflicts if c['system'] == sys_name and c['column'] == col)
            print(f"  - {col}  [LLM: {conflict['llm_type']}, Heur: {conflict['heuristic_type']}]")


def set_system_ordinal_by_unique(system_name: str, unique_values: list):
    """Set columns in a system to ordinal if n_unique is in the given list."""
    results = load_results()
    
    if system_name not in results or not results[system_name]:
        print(f"System '{system_name}' not found")
        return
    
    changed = 0
    for col_name, col_data in results[system_name].get("columns", {}).items():
        n_unique = col_data.get("n_unique", 0)
        if n_unique in unique_values:
            results[system_name]["columns"][col_name]["final_type"] = "ordinal"
            results[system_name]["columns"][col_name]["manual_override"] = True
            changed += 1
            print(f"  {col_name} (n_unique={n_unique}) -> ordinal")
    
    # Save updated results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSet {changed} columns in '{system_name}' to ordinal")
    return results


def remove_column_from_system(system_name: str, column_name: str):
    """Remove a column from a system."""
    results = load_results()
    
    if system_name not in results or not results[system_name]:
        print(f"System '{system_name}' not found")
        return
    
    if column_name not in results[system_name].get("columns", {}):
        print(f"Column '{column_name}' not found in '{system_name}'")
        return
    
    del results[system_name]["columns"][column_name]
    results[system_name]["n_columns"] = len(results[system_name]["columns"])
    
    # Save updated results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Removed '{column_name}' from '{system_name}'")


def add_system(system_name: str, filepath: str, columns: dict):
    """
    Add a new system with columns to the results.
    
    Args:
        system_name: Name of the system
        filepath: Path to the CSV file
        columns: Dict of column_name -> {"description": str, "type": str}
    """
    results = load_results()
    
    results[system_name] = {
        "filepath": filepath,
        "n_columns": len(columns),
        "columns": {}
    }
    
    for col_name, col_info in columns.items():
        results[system_name]["columns"][col_name] = {
            "description": col_info.get("description", ""),
            "llm_type": col_info.get("type", "regression"),
            "llm_reasoning": "Manually added",
            "heuristic_type": col_info.get("type", "regression"),
            "final_type": col_info.get("type", "regression"),
            "conflict": False,
            "manual_override": True,
            "sample_values": [],
            "n_unique": 0,
        }
    
    # Save updated results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Added system '{system_name}' with {len(columns)} columns")


def add_age_gender_bmi():
    """Add Age_Gender_BMI system with age, gender, bmi columns."""
    add_system(
        system_name="Age_Gender_BMI",
        filepath="/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/10K_Trajectories/body_systems/Age_Gender_BMI.csv",
        columns={
            "age": {"description": "Age of subject at time of measurement in years", "type": "regression"},
            "gender": {"description": "Gender of subject (1=male, 2=female)", "type": "categorical"},
            "bmi": {"description": "Body Mass Index (weight/height^2)", "type": "regression"},
        }
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "summary":
            show_summary()
        elif cmd == "apply":
            results = apply_overrides_to_results()
            export_final_params(results)
        elif cmd == "export":
            export_final_params()
        elif cmd == "diet-ordinal":
            set_diet_to_ordinal()
        elif cmd == "ordinal":
            # Set specific systems to ordinal: python review_conflicts.py ordinal system1 system2
            if len(sys.argv) > 2:
                set_system_to_ordinal(sys.argv[2:])
            else:
                print("Usage: python review_conflicts.py ordinal <system1> <system2> ...")
        elif cmd == "restore-llm":
            # Restore systems to LLM classification: python review_conflicts.py restore-llm system1 system2
            if len(sys.argv) > 2:
                restore_system_to_llm(sys.argv[2:])
            else:
                print("Usage: python review_conflicts.py restore-llm <system1> <system2> ...")
        elif cmd == "list-conflicts":
            list_conflicts()
        elif cmd == "mental-ordinal-4-5":
            set_system_ordinal_by_unique("mental", [4, 5])
        elif cmd == "remove-column":
            # Remove a column from a system: python review_conflicts.py remove-column system column
            if len(sys.argv) == 4:
                remove_column_from_system(sys.argv[2], sys.argv[3])
            else:
                print("Usage: python review_conflicts.py remove-column <system> <column>")
        elif cmd == "add-age-gender-bmi":
            add_age_gender_bmi()
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python review_conflicts.py [summary|apply|export|diet-ordinal|ordinal|restore-llm|list-conflicts|mental-ordinal-4-5|remove-column|add-age-gender-bmi]")
    else:
        # Default: interactive review
        review_conflicts_interactive()


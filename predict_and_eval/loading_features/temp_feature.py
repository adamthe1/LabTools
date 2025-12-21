import os
import json
import time
import hashlib
from .load_feature_df import load_feature_target_systems_as_df, get_body_system_column_names, get_temp_systems_row_counts
from ..utils.env_loader import *  # Load .env with absolute path

# #region agent log
_DEBUG_LOG = "/home/adamgab/PycharmProjects/GaitPredict/.cursor/debug.log"
def _log(msg, data=None, hyp=None):
    with open(_DEBUG_LOG, "a") as f:
        f.write(json.dumps({"ts": time.time(), "msg": msg, "data": data, "hyp": hyp}) + "\n")
# #endregion

# Cache configuration
TEMP_DIR = os.getenv('DATA_CACHE_DIR', '/tmp/data_cache')
DATA_DIR = os.path.join(TEMP_DIR, "data")
TRACK_FILE = os.path.join(TEMP_DIR, "data_track.json")
CACHE_MAX_AGE_DAYS = 30
STALE_THRESHOLD_HOURS = 48


def _cache_key(feature_system: str, target_system: str, 
               feature_columns: list, target_columns: list, merge_closest: bool) -> str:
    """Unique identifier for this bundle (system names + column names)."""
    key = f"{feature_system}|{target_system}|{sorted(feature_columns)}|{sorted(target_columns)}|{merge_closest}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _ensure_cache_dirs():
    """Ensure cache directories and tracking file exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(TRACK_FILE):
        with open(TRACK_FILE, "w") as f:
            json.dump([], f)


def _load_track() -> list:
    _ensure_cache_dirs()
    with open(TRACK_FILE, "r") as f:
        return json.load(f)


def _save_track(data_track: list):
    with open(TRACK_FILE, "w") as f:
        json.dump(data_track, f)


def _cleanup_old_cache(data_track: list) -> list:
    """Remove cache entries older than CACHE_MAX_AGE_DAYS."""
    cutoff = time.time() - (CACHE_MAX_AGE_DAYS * 24 * 60 * 60)
    cleaned = []
    for entry in data_track:
        file_path = os.path.join(DATA_DIR, entry['filename']) if 'filename' in entry else None
        if file_path and os.path.exists(file_path):
            if os.path.getmtime(file_path) < cutoff:
                os.remove(file_path)
                continue
        elif file_path is None:
            continue  # Skip malformed entries
        cleaned.append(entry)
    return cleaned


def _get_source_mtime(features: list[str], targets: list[str]) -> int:
    """Get max mtime of source CSV files (cheap OS call for cache invalidation)."""
    from .load_feature_df import load_system_description_json
    systems = load_system_description_json()
    all_cols = set(features + targets)
    max_mtime = 0
    for sys_data in systems.values():
        if not sys_data:
            continue
        if set(sys_data.get('columns', {}).keys()) & all_cols:
            path = sys_data.get('directory', '')
            if path and os.path.exists(path):
                max_mtime = max(max_mtime, int(os.path.getmtime(path)))
    return max_mtime


def _find_valid_cache(data_track: list, key: str, source_mtime: int, 
                      temp_row_counts: dict) -> tuple:
    """Return (valid_entry, cleaned_track). Removes stale/orphaned entries."""
    # #region agent log
    _log("_find_valid_cache", {"key": key, "temp_row_counts": temp_row_counts, "track_len": len(data_track)}, "CACHE")
    # #endregion
    cleaned = []
    for entry in data_track:
        if entry.get('cache_key') != key:
            cleaned.append(entry)
            continue
        
        # #region agent log
        _log("cache_key_match", {"key": key, "entry_file": entry.get('filename')}, "CACHE")
        # #endregion
        
        full_path = os.path.join(DATA_DIR, entry['filename'])
        if not os.path.exists(full_path):
            print(f"Cache ORPHAN: {entry['filename']}")
            # #region agent log
            _log("cache_orphan", {"file": entry['filename']}, "CACHE")
            # #endregion
            continue
        
        hours_stale = (source_mtime - entry.get('source_mtime', 0)) / 3600
        if hours_stale > STALE_THRESHOLD_HOURS:
            print(f"Cache STALE: {hours_stale:.0f}h old, regenerating")
            os.remove(full_path)
            continue
        
        # Validate temp systems row counts (detect different data with same schema)
        cached_counts = entry.get('temp_row_counts', {})
        if temp_row_counts != cached_counts:
            print(f"Cache STALE: temp system row counts changed {cached_counts} -> {temp_row_counts}")
            # #region agent log
            _log("cache_stale_counts", {"cached": cached_counts, "current": temp_row_counts}, "CACHE")
            # #endregion
            os.remove(full_path)
            continue
        
        # Valid hit
        print(f"Cache HIT: {entry['filename']} ({entry.get('n_rows', '?')} rows)")
        entry['filename'] = full_path
        cleaned.append(entry)
        return entry, cleaned
    
    # #region agent log
    _log("cache_no_match", {"key": key}, "CACHE")
    # #endregion
    return None, cleaned


def create_merged_df_bundle(feature_system: str, target_system: str, 
                            confounders: list[str] = ['age', 'gender', 'bmi'],
                            merge_closest_research_stage: bool = False) -> dict:
    """Get or create cached merged dataframe for given features and targets."""
    feature_columns = get_body_system_column_names(feature_system)
    target_columns = get_body_system_column_names(target_system)
    if confounders:
        feature_columns.extend([c for c in confounders if c not in feature_columns])
    
    key = _cache_key(feature_system, target_system, feature_columns, target_columns, merge_closest_research_stage)
    source_mtime = _get_source_mtime(feature_columns, target_columns)
    temp_row_counts = get_temp_systems_row_counts(feature_system, target_system)
    
    data_track = _cleanup_old_cache(_load_track())
    cached, data_track = _find_valid_cache(data_track, key, source_mtime, temp_row_counts)
    if cached:
        _save_track(data_track)
        return cached
    
    # Create new cache
    print(f"Cache MISS: creating bundle (feat: {feature_system}, tgt: {target_system})")
    merged_df = load_feature_target_systems_as_df(feature_system, target_system, confounders, merge_closest_research_stage)

    # Try parquet first, fall back to CSV if pyarrow not available
    try:
        filename = f"{key}.parquet"
        full_path = os.path.join(DATA_DIR, filename)
        merged_df.to_parquet(full_path)
    except ImportError:
        filename = f"{key}.csv"
        full_path = os.path.join(DATA_DIR, filename)
        merged_df.to_csv(full_path)
    
    entry = {
        'cache_key': key, 'filename': filename,
        'features': feature_columns, 'targets': target_columns,
        'n_rows': len(merged_df), 'source_mtime': source_mtime,
        'temp_row_counts': temp_row_counts,
        'feature_system': feature_system, 'target_system': target_system,
        'confounders': confounders, 'merge_closest_research_stage': merge_closest_research_stage,
    }
    data_track.append(entry)
    _save_track(data_track)
    
    entry['filename'] = full_path  # Return full path
    return entry

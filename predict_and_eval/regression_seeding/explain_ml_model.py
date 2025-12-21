import os
import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe on servers
import matplotlib.pyplot as plt
from ..utils.ids_folds import create_cv_folds
import joblib
import json
from ..utils.model_and_pipeline import ModelAndPipeline


import numpy as np
import pandas as pd
from scipy.stats import trim_mean  # optional; comment out if not available

def make_subject_pooled_bg(x_train: pd.DataFrame,
                           method: str = "median",
                           trim_p: float = 0.1) -> pd.DataFrame:
    """
    Build a background set with exactly one row per *train* subject by pooling that subject's rows.

    Args:
        x_train: DataFrame with MultiIndex where level 0 is subject id.
        method:  "median" (robust, default), "mean", or "trimmed_mean".
        trim_p:  Proportion to trim from each tail per feature when method='trimmed_mean' (e.g., 0.1).

    Returns:
        DataFrame with one pooled row per subject (keeps a 2-level index: (subject, 0)).
    """
    def pooled_row(df):
        if method == "median":
            # robust to outliers; ignores NaNs by default
            return df.median(axis=0, skipna=True)
        elif method == "mean":
            return df.mean(axis=0)
        elif method == "trimmed_mean":
            # apply per-column trimmed mean
            return df.apply(lambda col: trim_mean(col.dropna().values, proportiontocut=trim_p))
        else:
            raise ValueError(f"Unknown method: {method}")

    parts = []
    for subj, Xi in x_train.groupby(level=0):
        pooled = pooled_row(Xi).to_frame().T
        # keep MultiIndex compatibility
        pooled.index = pd.MultiIndex.from_product([[subj], [0]], names=x_train.index.names)
        parts.append(pooled)

    bg = pd.concat(parts, axis=0)
    return bg

def explain_model(model_path, model_key, fold, x, y, save_dir, target):
    """
    Explain model predictions using SHAP values.
    
    Args:
        model: Trained model
        model_key: Model type identifier
        x: Feature DataFrame
        y: Target values
        save_dir: Directory to save SHAP scores
        target: Target name for file naming
    """

    subject_ids = x.index.get_level_values(0).values
    sample_folds = create_cv_folds(fold, subject_ids)
    print(sample_folds)
    x_train = x.iloc[sample_folds[0][0]]
    x_test = x.iloc[sample_folds[0][1]]
    y_train = y.iloc[sample_folds[0][0]]
    y_test = y.iloc[sample_folds[0][1]]
    bg = make_subject_pooled_bg(x_train, method="median") 
    pipeline = ModelAndPipeline.initialize_model_and_pipeline(model_key, model_path=model_path)
    model = pipeline.fit(x_train, y_train)['model']
    if model_key == "LR_ridge":
        print_lr_ridge_coefs(model, x_train, save_dir=save_dir, target=target, model_key=model_key)
    if 'SV' in model_key:
        print(model_key)
        exp = shap.KernelExplainer(model.predict, bg)
    elif 'Logit' in model_key or 'Ordinal' in model_key or 'LR' in model_key:
        exp = shap.LinearExplainer(model, bg, feature_perturbation="interventional")
    else:
        print(model_key)
        exp = shap.TreeExplainer(model, data=bg)
    shap_values = exp.shap_values(x_test)
    shap_df = pd.DataFrame(shap_values, columns=x_test.columns, index=x_test.index)
    save_in = os.path.join(save_dir, 'shap_scores')
    os.makedirs(save_in, exist_ok=True)
    shap_df.to_csv(save_in + f'{target}_{model_key}.csv')
    png_path = save_shap_beeswarm(shap_values, x_test, save_in, target, model_key, topk=20)
    return shap_df

def print_lr_ridge_coefs(model, X, save_dir=None, target=None, model_key=None, top=25):
    """
    Print and (optionally) save Ridge (LR_ridge) coefficients with feature names.
    X: the training DataFrame used to fit (for column names).
    """
    if not hasattr(model, "coef_"):
        raise TypeError("Model has no coef_. Are you sure this is LR_ridge / linear?")

    coefs = np.asarray(model.coef_).ravel()
    coef_df = (
        pd.DataFrame({"feature": X.columns, "coef": coefs})
        .assign(abs_coef=lambda d: d["coef"].abs())
        .sort_values("abs_coef", ascending=False)
    )

    # Pretty print top positives/negatives
    print("\nTop + coefficients:")
    print(coef_df.sort_values("coef", ascending=False).head(top).to_string(index=False))
    print("\nTop − coefficients:")
    print(coef_df.sort_values("coef", ascending=True).head(top).to_string(index=False))

    # Optional: save to CSV
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        name = f"{target}_{model_key}_coefficients.csv" if target and model_key else "coefficients.csv"
        out = os.path.join(save_dir, name)
        coef_df.to_csv(out, index=False)
        print(f"\nSaved full coefficients table to: {out}")

    return coef_df


def save_shap_beeswarm(shap_values, x, save_dir, target, model_key, topk=10, class_index=None):
    """
    Save a single SHAP beeswarm plot (summary plot) with a color bar for feature values.

    Args:
        shap_values: np.ndarray [n_samples, n_features] or list of arrays (multiclass)
        x: pandas DataFrame of features used for SHAP
        save_dir: base directory
        target: target name (str)
        model_key: model id (str)
        topk: max number of features to display
        class_index: which class to plot if multiclass (default: auto-choose by total |SHAP|)
    Returns:
        str: path to the saved PNG
    """
    plots_dir = os.path.join(save_dir)
    os.makedirs(plots_dir, exist_ok=True)

    # Pick the right SHAP matrix for plotting
    if isinstance(shap_values, list):  # multiclass
        if class_index is None:
            # choose the class with largest total |SHAP|
            totals = [np.abs(sv).sum() for sv in shap_values]
            class_index = int(np.argmax(totals))
        sv = shap_values[class_index]
        suffix = f"_class{class_index}"
    else:
        sv = shap_values
        suffix = ""

    # Beeswarm
    plt.figure()
    shap.summary_plot(sv, x, max_display=topk, show=False)   # this creates the beeswarm w/ colorbar
    out_path = os.path.join(
        plots_dir, f"{target}_{model_key}{suffix}_shap{topk}.png".replace(os.sep, "_")
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

def _create_feature_df(prediction_files: dict, seed: int) -> pd.DataFrame:
    """
    Create a DataFrame where each subject/visit has features from every prediction file for every activity.
    
    Args:
        prediction_files: Dict mapping seq_type names to prediction file paths
        seed: The seed value to filter by
        
    Returns:
        DataFrame with index (RegistrationCode, research_stage) and columns for each 
        seq_type/activity combination (e.g., 'gait_walking_y_pred', 'gait_running_y_pred', ...)
    """
    feature_dfs = []
    
    for seq_type, prediction_file in prediction_files.items():
        # Read the CSV with RegistrationCode and research_stage as index
        predictions = pd.read_csv(prediction_file, index_col=[0, 1])
        
        # Filter by seed
        predictions = predictions[predictions['seed'] == seed]
        
        # Check if 'activity' column exists
        if 'activity' in predictions.columns:
            # Group by subject, research_stage, and activity, averaging across seq_idx
            grouped = predictions.groupby(
                [predictions. index.get_level_values(0),  # RegistrationCode
                predictions. index.get_level_values(1),  # research_stage
                'activity']
            )['y_pred'].mean()
            
            # Unstack activity to create one column per activity
            activity_features = grouped.unstack(level='activity')
            
            # Rename columns to include seq_type and activity
            activity_features.columns = [
                f"{seq_type}_{activity}_age_pred" for activity in activity_features.columns
            ]
        else:
            # No activity column - create single feature for this seq_type
            grouped = predictions.groupby(
                [predictions.index.get_level_values(0),  # RegistrationCode
                predictions.index.get_level_values(1)]  # research_stage
            )['y_pred'].mean()
            
            activity_features = grouped.to_frame(name=f"{seq_type}_age_pred")
        
        feature_dfs.append(activity_features)
    
    # Concatenate all feature DataFrames along columns
    ensemble_df = pd.concat(feature_dfs, axis=1)
    
    # Ensure index names are set correctly
    ensemble_df.index.names = ['RegistrationCode', 'research_stage']
    
    return ensemble_df

def explain_age_model() -> pd.DataFrame:
    """Predict age from each activity's embeddings."""
    label_cache_file = "/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/label_cache_new/label_cache_with_nan.csv"
    prediction_files = {"long_seq": "/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/results/biological_age_multiple/long_seq/raw_predictions.csv",
    "short_seq": "/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/results/biological_age_multiple/short_seq/raw_predictions.csv",
    "features": "/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/results/biological_age_multiple/features/features/raw_predictions.csv"}
    save_dir = "/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/results/biological_age_multiple/all_together/new/"
    os.makedirs(save_dir, exist_ok=True)
    # Load age from label cache once
    print("  Loading age data from cache...")
    age_df = pd.read_csv(label_cache_file, usecols=['RegistrationCode', 'research_stage', 'age'])
    gender_df = pd.read_csv(label_cache_file, usecols=['RegistrationCode', 'research_stage', 'gender'])
    age_df.set_index(['RegistrationCode', 'research_stage'], inplace=True)
    gender_df.set_index(['RegistrationCode', 'research_stage'], inplace=True)
    
    df = _create_feature_df(prediction_files, 0)
    # Separate features from confounders
    drop_cols = ['Unnamed: 0'] + ['age']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Merge age from cache
    df = df.join(age_df[['age']], how='inner')
    df = df.join(gender_df[['gender']], how='inner')
    df = df.dropna(subset=['age'])

    y = df[['age']]
    x = df.drop(columns=['age'])
    
    print(f"Samples: {len(x)}, Features: {x.columns}")
    
        # load folds from json file
    folds = json.load(open("/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/results/biological_age_multiple/long_seq/long_seq/folds.json"))
    fold = folds[0]
    # Evaluate predictions (use label_type from payload to ensure consistency)
    model_path = "/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/results/biological_age_multiple/all_together/example_model_seed_0.pkl"
    explain_model(model_path, "LR_ridge", fold, x, y, save_dir, "age")



def explain_single_model(activity_name, label = "age", save_dir = None, feature_file = None):
    """Predict age from each activity's embeddings."""
    label_cache_file = "/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/label_cache_new/label_cache_with_nan.csv"
    # short /net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/results/embeddings_11_30/qjo95s37
    if feature_file is None:
        feature_file = "/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/results/embeddings_11_30/qjo95s37/variant5_merged_groups_percentile/all.csv"
    feature_df = pd.read_csv(feature_file, index_col=[0, 1])
    feature_df = feature_df[feature_df['activity'] == activity_name]
    feature_df = feature_df.drop(columns=['activity'])
    if activity_name != "tm_3kmh":
        feature_df = feature_df.drop(columns=['seq_idx'])
    df = feature_df.copy()
    if save_dir is None:
        save_dir = f"/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/results/biological_age_multiple/short_seq/activity_{activity_name}/"
    # Load age from label cache once
    print("  Loading age data from cache...")
    age_df = pd.read_csv(label_cache_file, usecols=['RegistrationCode', 'research_stage', label])
    gender_df = pd.read_csv(label_cache_file, usecols=['RegistrationCode', 'research_stage', 'gender'])
    age_df.set_index(['RegistrationCode', 'research_stage'], inplace=True)
    gender_df.set_index(['RegistrationCode', 'research_stage'], inplace=True)
    
    # Separate features from confounders
    drop_cols = ['Unnamed: 0'] + [label]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Merge age from cache
    df = df.join(age_df[[label]], how='inner')
    if label != "gender":
        df = df.join(gender_df[['gender']], how='inner')
    df = df.dropna(subset=[label])

    y = df[[label]]
    x = df.drop(columns=[label])
    
    print(f"Samples: {len(x)}, Features: {x.columns}")
    
        # load folds from json file
    folds = json.load(open("/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/results/biological_age_multiple/long_seq/long_seq/folds.json"))
    fold = folds[0]
    # Evaluate predictions (use label_type from payload to ensure consistency)
    model_path = "/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/results/biological_age_multiple/long_seq/long_seq/example_model_seed_0.pkl"

    explain_model(model_path, "Logit", fold, x, y, save_dir, label)
   

if __name__ == "__main__":
    #explain_single_model("self_selected_gait_speed")
    activity_name = "tm_3kmh"
    explain_single_model("tm_3kmh", label='Depression', feature_file="/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/results/embeddings_11_30/lukd4yjy/variant5_merged_groups_percentile/all.csv",
    save_dir=f"/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/results/biological_age_multiple/long_seq/activity_{activity_name}/")
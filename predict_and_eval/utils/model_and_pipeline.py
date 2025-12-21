from sklearn.linear_model import Ridge, Lasso, LogisticRegression, ElasticNet
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from statsmodels.miscmodels.ordinal_model import OrderedModel
import numpy as np
import json
import os
from typing import Dict, Any, List
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from ..params.model_params import FIXED_PARAM_PRESETS, TUNABLE_PARAM_RANGES
from .env_loader import *  # Load .env with absolute path
import joblib


class OrdinalModelWrapper(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for statsmodels OrderedModel.
    
    Wraps statsmodels.miscmodels.ordinal_model.OrderedModel to provide
    fit/predict interface compatible with sklearn cross-validation.
    
    The model fits an ordered logit (proportional odds) model and predicts
    the expected ordinal value as a weighted sum of class probabilities.
    
    Args:
        distr: Distribution for the ordinal model ('logit' or 'probit')
        maxiter: Maximum iterations for optimizer (default 100)
    """
    def __init__(self, distr: str = 'logit', maxiter: int = 1000):
        self.distr = distr
        self.maxiter = maxiter
        self.model_ = None
        self.result_ = None
        self.classes_ = None
    
    def fit(self, X, y):
        """Fit the ordinal model."""
        self.classes_ = np.sort(np.unique(y))
        # hasconst=False: we don't include a constant column, statsmodels handles intercept
        self.model_ = OrderedModel(y, X, distr=self.distr, hasconst=False)
        self.result_ = self.model_.fit(method='lbfgs', disp=False, maxiter=self.maxiter)
        return self
    
    def predict(self, X):
        """Predict expected ordinal value (weighted sum of class probabilities)."""
        proba = self.predict_proba(X)
        # Expected value: sum of class_index * probability
        return np.sum(proba * self.classes_, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.result_.predict(X)
    
    def get_params(self, deep=True):
        return {'distr': self.distr, 'maxiter': self.maxiter}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

PARAMS_DIR = os.getenv('LINEAR_PARAM_DIR', './params/')
STATIC_PARAMS_FILE = os.path.join(PARAMS_DIR, 'static_params.json')
DYNAMIC_PARAMS_FILE = os.path.join(PARAMS_DIR, 'dynamic_params.json')


class ModelAndPipeline:
    @staticmethod
    def initialize_model(model_type: str, n_jobs: int = 8, params: Dict[str, Any] = None):
        model = None
        if model_type == 'LR_ridge':
            model = Ridge()
        elif model_type == 'LR_lasso':
            model = Lasso()
        elif model_type == 'LR_elastic':
            model = ElasticNet()
        elif model_type == 'LGBM_regression':
            model = lgb.LGBMRegressor(n_jobs=n_jobs, verbose=-1, objective="regression")
        elif model_type == 'XGB_regression':
            model = xgb.XGBRegressor(n_jobs=n_jobs, verbosity=0)
        elif model_type == 'SVM_regression':
            model = LinearSVR()
        elif model_type == 'XGB_classifier':
            model = xgb.XGBClassifier(n_jobs=n_jobs, verbosity=0)
        elif model_type == 'SVM_classifier':
            model = LinearSVC()
        elif model_type == 'Logit':
            model = LogisticRegression()
        elif model_type == 'LGBM_classifier':
            model = lgb.LGBMClassifier(n_jobs=n_jobs, verbose=-1, objective="binary")
        elif model_type == 'Ordinal_logit':
            model = OrdinalModelWrapper(distr='logit')

        if params is not None:
            model = model.set_params(**params)

        if model is None:
            raise ValueError(f"Model type '{model_type}' not found")
        return model
    
    @staticmethod
    def initialize_model_and_pipeline(model_type: str, n_jobs: int = 8, params: Dict[str, Any] = None, 
                                    categorical_cols: List[str] = None, numeric_cols: List[str] = None, 
                                    ordinal_cols: List[str] = None, model_path: str = None):
        """
        Initialize model with preprocessing pipeline.
        
        Tree models (LGBM, XGB) only get variance threshold.
        Linear models and Ordinal get imputation + scaling.
        
        Args:
            model_type: Type of model to initialize
            n_jobs: Number of parallel jobs
            params: Model hyperparameters
        
        Returns:
            Pipeline with preprocessing and model
        """
        if model_path is not None:
            model = joblib.load(model_path)
        else:
            model = ModelAndPipeline.initialize_model(model_type, n_jobs, params)
        
        # Tree models handle NaN and don't need scaling
        # Linear models and Ordinal need imputation + scaling
        # use label encoder on categorical columns if categorical_cols is not None
        # else detect automatically
        # Auto-detect columns by dtype if not provided
        cat_selector = categorical_cols 
        num_selector = numeric_cols 
        
        is_tree_model = 'LGBM' in model_type or 'XGB' in model_type
        
        if is_tree_model:
            if cat_selector is None:
                # make all columns act as numerical
                preprocessor = 'passthrough'
            else:
                preprocessor = ColumnTransformer([
                    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_selector),
                    ('num', 'passthrough', num_selector)
                ], remainder='drop')
                # Note: LGBM categorical_feature param doesn't work with sklearn Pipeline
                # OrdinalEncoder handles categoricals fine - LGBM treats them as numeric
                if "XGB" in model_type:
                    model.set_params(enable_categorical=True, tree_method='hist')
        else:
            if cat_selector is None:
                # make all columns act as numerical
                preprocessor = make_pipeline(KNNImputer(n_neighbors=5), StandardScaler())
            else:
                preprocessor = ColumnTransformer([
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_selector),
                    ('num', make_pipeline(KNNImputer(n_neighbors=5), StandardScaler()), num_selector)
                ], remainder='drop')
        
        # Drop constant columns after preprocessing (OneHotEncoder can create them)
        # Critical for Ordinal model which adds its own intercept
        return Pipeline([
            ('preprocessor', preprocessor),
            ('drop_constant', VarianceThreshold(threshold=0.0)),
            ('model', model)
        ])

    @staticmethod
    def get_model_types() -> List[str]:
        try:
            return list(ModelAndPipeline.load_hyperparams_from_json(STATIC_PARAMS_FILE).keys())
        except FileNotFoundError:
            return ['LR_ridge', 'LR_lasso', 'LR_elastic', 'LGBM_regression', 'LGBM_classifier',
                    'XGB_regression', 'XGB_classifier', 'SVM_regression', 'SVM_classifier', 'Logit']

    @staticmethod
    def load_hyperparams_from_json(json_path: str) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(json_path):
            if "static" in json_path:
                return ModelAndPipeline.load_hyperparams_from_file('static')
            elif "dynamic" in json_path:
                return ModelAndPipeline.load_hyperparams_from_file('dynamic')
            else:
                raise FileNotFoundError(f"Hyperparameter file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def load_hyperparams_from_file(type: str) -> Dict[str, Dict[str, Any]]:
        if type == 'static':
            return FIXED_PARAM_PRESETS  
        elif type == 'dynamic':
            return TUNABLE_PARAM_RANGES
        else:
            raise ValueError(f"Invalid type: {type}")
    
    @staticmethod
    def get_model_type_static_params(model_type: str, x) -> Dict[str, Any]:
        params = ModelAndPipeline.load_hyperparams_from_json(STATIC_PARAMS_FILE)
        if model_type not in params:
            raise KeyError(f"Model type '{model_type}' not found in static params")
        model_params = params[model_type]
        # Only Ridge/Lasso have solver param, not ElasticNet
        if len(x) > 50000 and model_type in ('LR_ridge', 'LR_lasso'):
            model_params['solver'] = 'lsqr'
        return model_params

    @staticmethod
    def get_model_type_dynamic_params(model_type: str, x) -> Dict[str, List[Any]]:
        params = ModelAndPipeline.load_hyperparams_from_json(DYNAMIC_PARAMS_FILE)  
        if model_type not in params:
            print(f"{model_type} model has no tunable hyperparameters, using preset params")
            raise KeyError(f"Model type '{model_type}' not found in dynamic params")
        model_params = params[model_type]
        # Only Ridge/Lasso have solver param, not ElasticNet
        if len(x) > 50000 and model_type in ('LR_ridge', 'LR_lasso'):
            model_params['solver'] = ['lsqr']
        return model_params

    @staticmethod
    def get_model_params_from_path(model_type: str, path: str) -> Dict[str, Any]:
        params = ModelAndPipeline.load_hyperparams_from_json(path)
        if model_type not in params:
            raise KeyError(f"Model type '{model_type}' not found in {path}")
        return params[model_type]

    @staticmethod
    def create_model_params_json(params: Dict[str, Dict[str, Any]], path: str) -> str:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(params, f, indent=4)
        return path
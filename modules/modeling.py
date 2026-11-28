"""Общие части моделирования: загрузка данных из ноутбука 01 и сборка пайплайнов."""
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from modules.config import CAT_FEATURES, TARGET, RANDOM_STATE, AREA_COLS, POLY_COLS   # константы общие для всех ноутбуков и inference
from modules.features import FeatureBuilder



def load_processed(processed_dir):
    """Данные после ноутбука 01: обучение без блока high_rent, validation целиком и маска «без блока»."""
    read = lambda name: pd.read_csv(Path(processed_dir) / name, index_col="row_id", dtype={c: str for c in CAT_FEATURES})
    train, valid = read("train_filled.csv"), read("valid_filled.csv")
    train_flags, valid_flags = read("train_price_flags.csv"), read("valid_price_flags.csv")
    if "use_in_fit" not in train_flags:   # запасной вариант, если маска ещё не создана в 01
        train_flags["use_in_fit"] = 1 - train_flags["high_rent_flag"]

    feature_cols = [c for c in train.columns if c != TARGET]
    fit_mask = train_flags["use_in_fit"] == 1
    clean_val = valid_flags["high_rent_flag"] == 0
    return (train.loc[fit_mask, feature_cols], train.loc[fit_mask, TARGET],
            valid[feature_cols], valid[TARGET], clean_val, valid_flags)


def numeric_columns(X, exclude=()):
    return [c for c in X.select_dtypes("number").columns if c not in exclude]


def categorical_columns(X):
    return list(X.select_dtypes(exclude="number").columns)


def _numeric_steps():
    return [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]


def make_preprocessor(pca=None, poly=False, n_components=3):
    """Кодирование и масштаб. pca: None | "area" (сжать 5 площадных признаков) | "all" (сжать все числа, 95 % дисперсии)."""
    if pca == "area":
        parts = [("area_pca", Pipeline(_numeric_steps() + [("pca", PCA(n_components=n_components, random_state=RANDOM_STATE))]), AREA_COLS),
                 ("num", Pipeline(_numeric_steps()), partial(numeric_columns, exclude=AREA_COLS))]
    elif pca == "all":
        parts = [("num_pca", Pipeline(_numeric_steps() + [("pca", PCA(n_components=0.95, random_state=RANDOM_STATE))]), numeric_columns)]
    else:
        parts = [("num", Pipeline(_numeric_steps()), numeric_columns)]
    if poly:   # попарные произведения ключевых признаков
        parts.append(("poly", Pipeline(_numeric_steps() + [("poly", PolynomialFeatures(2, interaction_only=True, include_bias=False))]), POLY_COLS))
    parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns))
    return ColumnTransformer(parts)


def build_pipeline(regressor, groups=None, pca=None, poly=False, n_components=3, log=True):
    """Pipeline: [FeatureBuilder] -> препроцессор -> регрессор (цель учится на log1p, прогноз возвращается в рублях)."""
    steps = [("fe", FeatureBuilder(tuple(groups)))] if groups else []
    steps.append(("prep", make_preprocessor(pca, poly, n_components)))
    if log:
        regressor = TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)
    steps.append(("model", regressor))
    return Pipeline(steps)

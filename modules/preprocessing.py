import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from difflib import SequenceMatcher

JUNK_TOKENS = {"", ".", "-", "?", "nan", "none", "n/a"}
# Латинские буквы, внешне неотличимые от кириллических
LATIN_TO_CYR = str.maketrans("aeopcxykmthb", "аеорсхукмтнв")


def normalize_text(s):
    """Единый вид строки: пробелы по краям убраны, нижний регистр, повторные пробелы схлопнуты."""
    return s.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def canon_key(s):
    """Ключ для сравнения написаний: ё=е, латинские двойники -> кириллица, без знаков и пробелов."""
    s = normalize_text(s).str.replace("ё", "е", regex=False)
    s = s.map(lambda v: v.translate(LATIN_TO_CYR) if isinstance(v, str) else v).astype("string")
    return s.str.replace(r"[^\w]|_", "", regex=True)

def key_of(text):
    """canon_key для одной строки."""
    return canon_key(pd.Series([text]))[0]


class NumericCleaner(BaseEstimator, TransformerMixin):
    """Исправляет единицы расстояния и заменяет недопустимые числа на NaN."""

    def __init__(
        self, num_cols, bounds=None, pairs=None,
        distance_threshold=50,  # порог после которого км -> метры
    ):
        self.num_cols = num_cols
        self.bounds = bounds
        self.pairs = pairs
        self.distance_threshold = distance_threshold

    def fit(self, X, y=None):
        # Правила заданы вручную: статистики по данным не вычисляем
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()  # исходную таблицу не изменяем

        # 1. Приводим числа к единому типу
        for c in self.num_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")

        # 2. Предполагаемые метры переводим в километры
        if "distance_to_center_km" in self.num_cols:
            mask = X["distance_to_center_km"] > self.distance_threshold
            X.loc[mask, "distance_to_center_km"] /= 1000

        # 3. Нулевая этажность означает неизвестное значение
        if "floors_total" in self.num_cols:
            X.loc[X["floors_total"] == 0, "floors_total"] = np.nan

        # 4. Значения вне допустимых границ заменяем на пропуски
        for c, (lo, hi) in (self.bounds or {}).items():
            bad = pd.Series(False, index=X.index)
            if lo is not None:
                bad |= X[c] < lo
            if hi is not None:
                bad |= X[c] > hi
            X.loc[bad, c] = np.nan

        # 5. Проверяем отношения между признаками
        for small, big in (self.pairs or []):
            bad = X[small] > X[big]
            X.loc[bad, small] = np.nan

            # каждая площадь допустима, но сумма больше общей
        bad_sum = (X["living_area"] + X["kitchen_area"] > X["total_area"])
        X.loc[bad_sum, ["living_area", "kitchen_area"]] = np.nan

        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class CategoryCleaner(BaseEstimator, TransformerMixin):
    """Сводит разные написания одной категории к одному и заменяет мусор на NaN.

    1. Написания с одним canon_key объединяются (fit запоминает самое частое в train).
    2. Пустой ключ и JUNK_TOKENS -> NaN.
    3. Если задан valid (допустимые категории из описания), написание приводится к допустимой:
       по ключу, по началу слова (косм. -> косметический) или по сходству (опечатки).
    4. fixes - ручные исправления для оставшихся случаев.
    """
    def __init__(self, cat_cols, fixes=None, valid=None, fuzzy_threshold=0.8, min_prefix=3):
        self.cat_cols, self.fixes, self.valid = cat_cols, fixes, valid
        self.fuzzy_threshold, self.min_prefix = fuzzy_threshold, min_prefix

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.canon_map_ = {}
        for c in self.cat_cols:
            pairs = pd.DataFrame({"norm": normalize_text(X[c]), "key": canon_key(X[c])}).dropna()
            pairs = pairs[pairs["key"] != ""]
            top = (pairs.groupby(["key", "norm"]).size().reset_index(name="n")
                   .sort_values(["key", "n"], ascending=[True, False]).drop_duplicates("key"))
            self.canon_map_[c] = dict(zip(top["key"], top["norm"]))
        self.valid_keys_ = {c: {key_of(v): v.lower() for v in vals} for c, vals in (self.valid or {}).items()}
        return self

    def _match_valid(self, value, c):
        vk = self.valid_keys_[c]
        k = key_of(value)
        if k in vk:
            return vk[k]
        if len(k) >= self.min_prefix:
            by_prefix = {v for key, v in vk.items() if key.startswith(k)}
            if len(by_prefix) == 1:
                return by_prefix.pop()
        scored = sorted(((SequenceMatcher(None, k, key).ratio(), v) for key, v in vk.items()), reverse=True)
        if scored and scored[0][0] >= self.fuzzy_threshold and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.05):
            return scored[0][1]
        return value

    def transform(self, X):
        X = X.copy()
        for c in self.cat_cols:
            norm, key = normalize_text(X[c]), canon_key(X[c])
            out = key.map(self.canon_map_[c]).astype("string")
            out = out.where(out.notna(), norm)
            out = out.mask(key.fillna("") == "").mask(norm.isin(JUNK_TOKENS))
            if c in self.valid_keys_:
                mapping = {u: self._match_valid(u, c) for u in out.dropna().unique()}
                out = out.map(lambda v: mapping.get(v, v) if isinstance(v, str) else v).astype("string")
            if self.fixes and self.fixes.get(c):
                out = out.replace(self.fixes[c])
            X[c] = out.astype(object).where(out.notna(), np.nan)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_
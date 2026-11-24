import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
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



class KitchenMissingFlag(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["kitchen_area_original_missing"] = X["kitchen_area"].isna().astype(int)
        return X


class AreaShareImputer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        total = X["total_area"].astype(float)
        living = X["living_area"].astype(float)
        kitchen = X["kitchen_area"].astype(float)
        original = X["kitchen_area_original_missing"].eq(1)

        # Не учим доли на известных противоречиях
        conflict = living.notna() & kitchen.notna() & (living + kitchen > total)
        valid = total.gt(0) & ~conflict

        living_ok = valid & living.gt(0) & living.lt(total)
        kitchen_ok = valid & ~original & kitchen.gt(0) & kitchen.lt(total)

        self.living_share_ = (living[living_ok] / total[living_ok]).median()
        self.kitchen_share_ = (kitchen[kitchen_ok] / total[kitchen_ok]).median()

        if pd.isna(self.living_share_) or pd.isna(self.kitchen_share_):
            raise ValueError("Недостаточно корректных площадей train для расчёта долей.")
        return self

    def transform(self, X):
        check_is_fitted(self, ["living_share_", "kitchen_share_"])
        X = X.copy()
        cols = ["total_area", "living_area", "kitchen_area"]
        X[cols] = X[cols].astype(float)

        original = X["kitchen_area_original_missing"].eq(1)
        living_missing = X["living_area"].isna()
        kitchen_missing = X["kitchen_area"].isna() & ~original

        # Флаги сохраняем до заполнения
        X["living_area_missing"] = living_missing.astype(int)
        X["kitchen_area_cleaning_missing"] = kitchen_missing.astype(int)

        # Технический ноль — только для исходного пропуска кухни
        X.loc[original, "kitchen_area"] = 0

        total = X["total_area"]
        total_ok = total.gt(0) & np.isfinite(total)
        fill_living = living_missing & total_ok
        fill_kitchen = kitchen_missing & total_ok

        X.loc[fill_living, "living_area"] = total[fill_living] * self.living_share_
        X.loc[fill_kitchen, "kitchen_area"] = total[fill_kitchen] * self.kitchen_share_

        # Ограничиваем только оценки, сохраняя известные площади
        estimated_living = X["living_area"].where(fill_living, 0)
        estimated_kitchen = X["kitchen_area"].where(fill_kitchen, 0)
        estimated_sum = estimated_living + estimated_kitchen

        known_living = X["living_area"].where(~fill_living, 0).fillna(0)
        known_kitchen = X["kitchen_area"].where(~fill_kitchen, 0).fillna(0)
        available = (total - known_living - known_kitchen).clip(lower=0)

        # 0.95 — выбранный запас, а не норматив площади
        scale = (0.95 * available / estimated_sum.where(estimated_sum > 0)).clip(upper=1)
        scale = scale.fillna(1)

        X.loc[fill_living, "living_area"] *= scale[fill_living]
        X.loc[fill_kitchen, "kitchen_area"] *= scale[fill_kitchen]

        # Если положительного остатка нет, не выдумываем площадь
        X.loc[fill_living & available.le(0), "living_area"] = np.nan
        X.loc[fill_kitchen & available.le(0), "kitchen_area"] = np.nan
        return X
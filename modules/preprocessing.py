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



class KitchenZeroFiller(TransformerMixin, BaseEstimator):
    """Заполняет пропуски площади кухни нулём только для студий (rooms == 0) и ставит флаг."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        mask = X["kitchen_area"].isna() & (X["rooms"] == 0)
        X["kitchen_area_original_missing"] = mask.astype(int)
        X.loc[mask, "kitchen_area"] = 0
        return X


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class AreaShareImputer(TransformerMixin, BaseEstimator):
    """Заполняет площади медианными долями от общей площади, рассчитанными на train."""

    def fit(self, X, y=None):
        total = X["total_area"].astype(float)
        living = X["living_area"].astype(float)
        kitchen = X["kitchen_area"].astype(float)

        # Исключаем известные противоречия между площадями
        conflict = living.notna() & kitchen.notna() & (living + kitchen > total)
        valid = total.gt(0) & ~conflict

        # Технические нули кухни не участвуют в расчёте её доли
        living_ok = valid & living.gt(0) & living.lt(total)
        kitchen_ok = valid & kitchen.gt(0) & kitchen.lt(total)

        self.living_share_ = (living[living_ok] / total[living_ok]).median()
        self.kitchen_share_ = (kitchen[kitchen_ok] / total[kitchen_ok]).median()
        return self

    def transform(self, X):
        X = X.copy()
        cols = ["total_area", "living_area", "kitchen_area"]
        X[cols] = X[cols].astype(float)

        # Сохраняем происхождение значений до заполнения
        living_missing = X["living_area"].isna()
        kitchen_missing = X["kitchen_area"].isna()
        X["living_area_missing"] = living_missing.astype(int)
        X["kitchen_area_missing"] = kitchen_missing.astype(int)

        total = X["total_area"]
        fill_living = living_missing & total.gt(0)
        fill_kitchen = kitchen_missing & total.gt(0)

        X.loc[fill_living, "living_area"] = total[fill_living] * self.living_share_
        X.loc[fill_kitchen, "kitchen_area"] = total[fill_kitchen] * self.kitchen_share_

        # Отделяем известные площади от подставленных оценок
        known_living = X["living_area"].where(~fill_living, 0).fillna(0)
        known_kitchen = X["kitchen_area"].where(~fill_kitchen, 0).fillna(0)
        available = (total - known_living - known_kitchen).clip(lower=0)

        estimated = X["living_area"].where(fill_living, 0) + X["kitchen_area"].where(fill_kitchen, 0)
        scale = (available / estimated.where(estimated.gt(0))).clip(upper=1).fillna(1)

        # При необходимости уменьшаем только расчётные значения
        X.loc[fill_living, "living_area"] *= scale[fill_living]
        X.loc[fill_kitchen, "kitchen_area"] *= scale[fill_kitchen]

        # Если свободного остатка нет, площадь не выдумываем
        X.loc[fill_living & available.le(0), "living_area"] = np.nan
        X.loc[fill_kitchen & available.le(0), "kitchen_area"] = np.nan
        return X


class TenantsFromAreaFiller(TransformerMixin, BaseEstimator):
    """Подбирает число жильцов по ближайшей средней площади с учётом комнат."""

    def fit(self, X, y=None):
        data = X[["rooms", "total_area", "max_tenants"]].astype(float)
        known = data.loc[data["max_tenants"].gt(0) & data["total_area"].gt(0)]

        # Общие средние — резерв для неизвестного количества комнат
        self.area_means_ = known.groupby("max_tenants")["total_area"].median()
        self.room_means_ = known.groupby(["rooms", "max_tenants"])["total_area"].median()
        self.fallback_ = data["max_tenants"].dropna().mode().iloc[0]
        return self

    def transform(self, X):
        X = X.copy()
        X["max_tenants"] = X["max_tenants"].astype(float)
        missing = X["max_tenants"].isna()
        X["max_tenants_missing"] = missing.astype(int)

        # Меняем только неизвестное число жильцов
        for idx in X.index[missing]:
            area = X.at[idx, "total_area"]
            rooms = X.at[idx, "rooms"]
            means = self.area_means_

            if pd.notna(rooms) and rooms in self.room_means_.index.get_level_values("rooms"):
                means = self.room_means_.xs(rooms, level="rooms")

            if pd.notna(area) and not means.empty:
                X.at[idx, "max_tenants"] = (means - area).abs().idxmin()
            else:
                X.at[idx, "max_tenants"] = self.fallback_

        return X
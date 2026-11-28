import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from difflib import SequenceMatcher
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler



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


def to_num(s):
    """Переводит текст в число: убирает пробелы, меняет запятую на точку."""
    # Если данные уже числовые — возвращаем как есть
    if pd.api.types.is_numeric_dtype(s):
        return s

    # Если это Series — обрабатываем как текст
    if isinstance(s, pd.Series):
        clean = s.astype(str).str.replace(r"[\s]", "", regex=True).str.replace(",", ".", regex=False)
        return pd.to_numeric(clean, errors="coerce")

    # Скалярное значение
    return pd.to_numeric(str(s).replace(" ", "").replace(",", "."), errors="coerce")



class NumericCleaner(BaseEstimator, TransformerMixin):
    """Исправляет единицы расстояния и заменяет недопустимые числа на NaN."""

    def __init__(
        self, num_cols, bounds=None, pairs=None,
        distance_threshold=50,  # порог после которого км -> метры
        add_flags=False,
    ):
        self.num_cols = num_cols
        self.bounds = bounds
        self.pairs = pairs
        self.distance_threshold = distance_threshold
        self.add_flags = add_flags

    def _clean(self, X):
        X = X.copy()  # исходную таблицу не изменяем

        # 1. Приводим числа к единому типу
        for c in self.num_cols:
            X[c] = to_num(X[c]).astype("float64")

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

    def _flags(self, X, cleaned):
        raw = X[self.num_cols].apply(pd.to_numeric, errors="coerce")
        flags = pd.DataFrame({c + "_invalid": (raw[c].notna() & cleaned[c].isna()).astype(int) for c in self.num_cols}, index=X.index)
        if "distance_to_center_km" in self.num_cols:
            flags["distance_to_center_km_fixed"] = (raw["distance_to_center_km"] > self.distance_threshold).astype(int)
        return flags

    def fit(self, X, y=None):
        # Правила заданы вручную: статистики по данным не вычисляем
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        if self.add_flags:
            flags = self._flags(X, self._clean(X))
            self.flag_cols_ = flags.columns[flags.any()].tolist()   # флаги только там, где очистка сработала на train
        return self

    def transform(self, X):
        cleaned = self._clean(X)
        if self.add_flags:
            cleaned = pd.concat([cleaned, self._flags(X, cleaned)[self.flag_cols_]], axis=1)
        return cleaned



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




class AreaShareImputer(TransformerMixin, BaseEstimator):
    """Жилая площадь — медианная доля от общей; кухня — медиана по числу комнат (оба правила считаются на train)."""

    def fit(self, X, y=None):
        total = X["total_area"].astype(float)
        living = X["living_area"].astype(float)
        kitchen = X["kitchen_area"].astype(float)

        # Исключаем известные противоречия между площадями
        conflict = living.notna() & kitchen.notna() & (living + kitchen > total)
        valid = total.gt(0) & ~conflict

        # Технические нули кухни не участвуют в расчёте
        living_ok = valid & living.gt(0) & living.lt(total)
        kitchen_ok = valid & kitchen.gt(0) & kitchen.lt(total)

        self.living_share_ = (living[living_ok] / total[living_ok]).median()
        self.kitchen_by_rooms_ = kitchen[kitchen_ok].groupby(X.loc[kitchen_ok, "rooms"].astype(float)).median()
        self.kitchen_global_ = kitchen[kitchen_ok].median()
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

        # Кухня: медиана для этого числа комнат, если группы нет — общая медиана
        kitchen_guess = X["rooms"].astype(float).map(self.kitchen_by_rooms_).fillna(self.kitchen_global_)
        X.loc[fill_living, "living_area"] = total[fill_living] * self.living_share_
        X.loc[fill_kitchen, "kitchen_area"] = kitchen_guess[fill_kitchen]

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


class TenantsFromRoomsFiller(TransformerMixin, BaseEstimator):
    """Жильцы: мода жильцов при данном числе комнат."""

    def fit(self, X, y=None):
        data = X[["rooms", "max_tenants"]].astype(float).dropna()
        self.tenants_by_rooms_ = data.groupby("rooms")["max_tenants"].agg(lambda s: s.mode().iloc[0])
        self.tenants_global_ = data["max_tenants"].mode().iloc[0]
        return self

    def transform(self, X):
        X = X.copy()
        rooms, tenants = X["rooms"].astype(float), X["max_tenants"].astype(float)
        X["max_tenants_missing"] = tenants.isna().astype(int)

        # Жильцы по комнатам; общая мода — запасной вариант
        tenants = tenants.fillna(rooms.map(self.tenants_by_rooms_)).fillna(self.tenants_global_)
        X["max_tenants"] = tenants
        return X




class GroupMedianFiller(TransformerMixin, BaseEstimator):
    """Заполняет числа медианой типа дома; если группы нет — общей медианой train."""

    def __init__(self, cols, group_col="house_type"):
        self.cols = cols
        self.group_col = group_col

    def fit(self, X, y=None):
        group = X[self.group_col].astype(object)
        self.medians_ = {c: X[c].astype(float).groupby(group).median() for c in self.cols}
        self.global_ = {c: X[c].astype(float).median() for c in self.cols}
        return self

    def transform(self, X):
        X = X.copy()
        group = X[self.group_col].astype(object)
        for c in self.cols:
            # Индикатор создаём до заполнения
            X[c + "_missing"] = X[c].isna().astype(int)
            guess = group.map(self.medians_[c]).astype(float).fillna(self.global_[c])
            X[c] = X[c].astype(float).fillna(guess)
        return X


class FloorsFiller(TransformerMixin, BaseEstimator):
    """Этажность: медиана по десятилетию постройки, не меньше этажа. Этаж: медиана для этой этажности, не выше неё."""

    def _decade(self, X):
        return X["year_built"].astype(float) // 10 * 10

    def fit(self, X, y=None):
        total, floor = X["floors_total"].astype(float), X["floor"].astype(float)
        self.total_by_decade_ = total.groupby(self._decade(X)).median()
        self.total_global_ = total.median()
        self.floor_by_total_ = floor.groupby(total).median()
        self.floor_global_ = floor.median()
        return self

    def transform(self, X):
        X = X.copy()
        total, floor = X["floors_total"].astype(float), X["floor"].astype(float)
        X["floors_total_missing"] = total.isna().astype(int)
        X["floor_missing"] = floor.isna().astype(int)

        # 1. Этажность: медиана по десятилетию (если год неизвестен, общая), не меньше известного этажа
        guess = self._decade(X).map(self.total_by_decade_).fillna(self.total_global_).round()
        total = total.fillna(np.fmax(guess, floor))

        # 2. Этаж: медиана для такой этажности, не выше этажности
        floor_guess = total.map(self.floor_by_total_).fillna(self.floor_global_).round()
        X["floors_total"] = total
        X["floor"] = np.minimum(floor.fillna(floor_guess), total)
        return X



class AreaPCA(TransformerMixin, BaseEstimator):
    """Добавляет главные компоненты коррелирующих столбцов; исходные столбцы остаются в таблице."""

    def __init__(self, cols, n_components=0.95, prefix="area_pc", random_state=42):
        self.cols = cols
        self.n_components = n_components
        self.prefix = prefix
        self.random_state = random_state

    def fit(self, X, y=None):
        # Масштаб и компоненты считаем только по обучающим данным
        self.scaler_ = StandardScaler().fit(X[self.cols].astype(float))
        self.pca_ = PCA(n_components=self.n_components, random_state=self.random_state).fit(self.scaler_.transform(X[self.cols].astype(float)))
        self.pc_cols_ = [f"{self.prefix}{i + 1}" for i in range(self.pca_.n_components_)]
        return self

    def transform(self, X):
        scores = self.pca_.transform(self.scaler_.transform(X[self.cols].astype(float)))
        return pd.concat([X, pd.DataFrame(scores, index=X.index, columns=self.pc_cols_)], axis=1)



class NumericKNNFiller(TransformerMixin, BaseEstimator):
    """Заполняет числа по k похожим объектам train: числа масштабируются, категории идут 0/1-столбцами с весом."""

    def __init__(self, cols, features, cat_cols=(), cat_weight=1.0, n_neighbors=10):
        self.cols = cols
        self.features = features
        self.cat_cols = cat_cols
        self.cat_weight = cat_weight
        self.n_neighbors = n_neighbors

    def _dummies(self, X):
        if not self.cat_cols:
            return pd.DataFrame(index=X.index)
        return pd.get_dummies(X[list(self.cat_cols)].astype(object)).reindex(columns=self.dummy_cols_, fill_value=0).astype(float) * self.cat_weight

    def _matrix(self, X):
        z = pd.DataFrame(self.scaler_.transform(X[self.features].astype(float)), index=X.index, columns=self.features)
        return pd.concat([z, self._dummies(X)], axis=1)

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X[self.features].astype(float))
        self.dummy_cols_ = pd.get_dummies(X[list(self.cat_cols)].astype(object)).columns if self.cat_cols else []
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors).fit(self._matrix(X))
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            X[c + "_missing"] = X[c].isna().astype(int)

        # Соседей ищем среди train; значения не округляем: это вход модели
        filled = self.imputer_.transform(self._matrix(X))[:, :len(self.features)]
        filled = pd.DataFrame(self.scaler_.inverse_transform(filled), index=X.index, columns=self.features)
        for c in self.cols:
            X[c] = X[c].astype(float).fillna(filled[c])
        return X



class UnknownCategoryFiller(TransformerMixin, BaseEstimator):
    """Пропуски категорий заменяет отдельным значением «неизвестно»."""

    def __init__(self, cols, value="неизвестно"):
        self.cols = cols
        self.value = value

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            X[c] = X[c].astype(object).where(X[c].notna(), self.value)
        return X


class LogicFlagger(TransformerMixin, BaseEstimator):
    """Флаги странных сочетаний значений. Значения не меняются, цена не используется (флаги считаются и на новых данных)."""

    def __init__(self, floors_limit=8, no_lift="нет", low_floors=(1, 2), metro_far_m=3000, center_near_km=3, kitchen_share_max=0.4):
        self.floors_limit = floors_limit
        self.no_lift = no_lift
        self.low_floors = low_floors
        self.metro_far_m = metro_far_m
        self.center_near_km = center_near_km
        self.kitchen_share_max = kitchen_share_max

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        # Высокий дом без лифта (квартиры на нижних этажах не считаем: лифт могли не указать)
        X["high_floor_no_lift_flag"] = ((X["elevator"] == self.no_lift) & (X["floors_total"] > self.floors_limit) & ~X["floor"].isin(self.low_floors)).astype(int)
        # Метро далеко, а центр близко
        X["metro_far_center_near_flag"] = ((X["metro_distance_m"] > self.metro_far_m) & (X["distance_to_center_km"] < self.center_near_km)).astype(int)
        # Кухня занимает слишком большую долю площади (только по известным значениям)
        known_kitchen = X["kitchen_area_missing"] == 0 if "kitchen_area_missing" in X.columns else True
        X["kitchen_big_share_flag"] = (known_kitchen & (X["kitchen_area"] / X["total_area"] > self.kitchen_share_max)).astype(int)
        return X
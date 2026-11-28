"""Конструирование признаков: все новые столбцы считаются только из других столбцов (цена не используется)."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

REFERENCE_YEAR = 2026   # год выгрузки: возраст дома считаем от него
RENOVATION_SCORE = {"без ремонта": 0, "косметический": 1, "евроремонт": 2, "дизайнерский": 3}   # порядок качества ремонта
NO_VALUES = ["нет", "0", "неизвестно"]                                                          # значения «этого нет»

# Группы новых признаков: имя группы -> столбцы, которые она добавляет
FEATURE_GROUPS = {
    "logs": ["log_total_area", "log_distance_to_center", "log_metro_distance"],
    "age": ["house_age", "is_new_house", "is_old_house"],
    "floors": ["floor_ratio", "is_first_floor", "is_last_floor"],
    "shares": ["area_per_room", "kitchen_share", "living_share", "tenants_per_room"],
    "tenants": ["area_per_tenant", "living_area_per_room"],
    "center": ["center_close", "metro_close"],
    "listing": ["photo_density", "missing_count"],
    "interactions": ["area_x_distance", "area_x_age", "area_x_metro"],
    "crosses": ["district_house_type", "district_renovation", "house_type_renovation"],
}


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Добавляет к таблице признаки выбранных групп; исходные столбцы остаются на месте."""

    def __init__(self, groups=tuple(FEATURE_GROUPS)):
        self.groups = groups

    def fit(self, X, y=None):
        unknown = set(self.groups) - set(FEATURE_GROUPS)
        if unknown:
            raise ValueError(f"Неизвестные группы признаков: {sorted(unknown)}")
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        groups = set(self.groups)
        area = X["total_area"].astype(float).clip(lower=1)
        rooms = X["rooms"].astype(float).clip(lower=1)           # у студий rooms == 0
        floors_total = X["floors_total"].astype(float).clip(lower=1)
        dist = X["distance_to_center_km"].astype(float)
        metro = X["metro_distance_m"].astype(float)
        age = REFERENCE_YEAR - X["year_built"].astype(float)

        if "logs" in groups:
            X["log_total_area"] = np.log(area)
            X["log_distance_to_center"] = np.log1p(dist)
            X["log_metro_distance"] = np.log1p(metro)
        if "age" in groups:
            X["house_age"] = age
            X["is_new_house"] = (age <= 5).astype(int)
            X["is_old_house"] = (age >= 50).astype(int)
        if "floors" in groups:
            X["floor_ratio"] = X["floor"].astype(float) / floors_total
            X["is_first_floor"] = (X["floor"] == 1).astype(int)
            X["is_last_floor"] = (X["floor"] == X["floors_total"]).astype(int)
        if "shares" in groups:
            X["area_per_room"] = area / rooms
            X["kitchen_share"] = X["kitchen_area"].astype(float) / area
            X["living_share"] = X["living_area"].astype(float) / area
            X["tenants_per_room"] = X["max_tenants"].astype(float) / rooms
        if "tenants" in groups:
            X["area_per_tenant"] = area / X["max_tenants"].astype(float).clip(lower=1)
            X["living_area_per_room"] = X["living_area"].astype(float) / rooms
        if "center" in groups:
            X["center_close"] = (dist < 3).astype(int)
            X["metro_close"] = (metro < 500).astype(int)
        if "listing" in groups:
            X["photo_density"] = X["photos_count"].astype(float) / area
            X["missing_count"] = X[[c for c in X.columns if c.endswith("_missing")]].sum(axis=1)   # сколько характеристик пришлось заполнить
        if "interactions" in groups:
            X["area_x_distance"] = np.log(area) * np.log1p(dist)
            X["area_x_age"] = np.log(area) * age / 100
            X["area_x_metro"] = np.log(area) * np.log1p(metro)
        if "crosses" in groups:   # категориальные пары: OneHotEncoder в препроцессоре подхватит их как обычные категории
            X["district_house_type"] = X["district"].astype(str) + "|" + X["house_type"].astype(str)
            X["district_renovation"] = X["district"].astype(str) + "|" + X["renovation"].astype(str)
            X["house_type_renovation"] = X["house_type"].astype(str) + "|" + X["renovation"].astype(str)
        return X

    def get_feature_names_out(self, input_features=None):
        new = [c for g in self.groups for c in FEATURE_GROUPS[g]]
        return np.array(list(self.feature_names_in_) + new, dtype=object)

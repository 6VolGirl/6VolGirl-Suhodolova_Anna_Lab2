#константы, общие для ноутбука и inference

TARGET = "rent_rub"
ID_COLS = ["listing_id"]
EXCLUDE = ["response_time_hours"]

# Списки колонок по типам (берутся из паспорта данных)
STR_COLS = ["listing_id", "publication_date"]
CAT_COLS = ["district", "house_type", "renovation", "elevator", "furniture",
            "balcony", "parking", "appliances", "pets_allowed", "seller_type"]
INT_COLS = ["rooms", "max_tenants", "floor", "floors_total", "year_built",
            "photos_count"]
FLOAT_COLS = ["total_area", "living_area", "kitchen_area",
              "distance_to_center_km", "metro_distance_m", "ceiling_height",
              TARGET]

NUM_FEATURES = [c for c in INT_COLS + FLOAT_COLS if c not in [TARGET] + EXCLUDE]
CAT_FEATURES = [c for c in CAT_COLS if c not in EXCLUDE]
FEATURES = NUM_FEATURES + CAT_FEATURES

# Параметры NumericCleaner
NUM_BOUNDS = {
    "rooms": (0, 10),
    "max_tenants": (1, 20),
    "floor": (1, 100),
    "floors_total": (1, 100),
    "year_built": (1900, 2026),
    "photos_count": (0, 100),
    "total_area": (10, 500),
    "living_area": (5, 400),
    "kitchen_area": (2, 100),
    "distance_to_center_km": (0, 50),
    "metro_distance_m": (0, 20000),
    "ceiling_height": (2.0, 4.5),
}

NUM_PAIRS = [
    ("floor", "floors_total", "floor <= floors_total"),
    ("living_area", "total_area", "living_area <= total_area"),
    ("kitchen_area", "total_area", "kitchen_area <= total_area"),
]

# Параметры CategoryCleaner
CATEGORY_FIXES = {
    "renovation": {"евро": "евроремонт", "евро-ремонт": "евроремонт"},
}

VALID_CATEGORIES = {
    "district": ["советский", "октябрьский", "ленинский", "кировский",
                 "орджоникидзевский", "мотовилихинский", "индустриальный",
                 "дзержинский", "пермский", "закамский", "другое"],
    "house_type": ["панельный", "кирпичный", "монолитный", "кирпично-монолитный",
                   "блочный", "другое"],
    "renovation": ["косметический", "евроремонт", "без ремонта", "дизайнерский"],
    "elevator": ["да", "нет"],
    "furniture": ["да", "нет"],
    "balcony": ["да", "нет"],
    "parking": ["да", "нет"],
    "appliances": ["да", "нет"],
    "pets_allowed": ["да", "нет"],
    "seller_type": ["собственник", "агент", "застройщик"],
}
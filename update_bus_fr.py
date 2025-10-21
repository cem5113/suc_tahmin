# update_bus.py
# AMAÇ: Otobüs duraklarına LAT/LON bazlı mesafe & buffer zone (300/600/900 m) özellikleri eklemek.
# NOT: Kesinlikle API'den indirme YOK. Hazır "sf_bus_stops_with_geoid.csv" dosyası doğrudan kullanılır.
# Girdi : crime_prediction_data/fr_crime_03.csv  (ENV: CRIME_INPUT_NAME ile değiştirilebilir)
# Çıktı : crime_prediction_data/fr_crime_04.csv  (ENV: CRIME_OUTPUT_NAME ile değiştirilebilir)

import os, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

# ========== küçük yardımcılar ==========
def log_shape(df: pd.DataFrame, label: str):
    r, c = df.shape
    print(f"📊 {label}: {r} satır × {c} sütun")

def log_delta(before_shape, after_shape, label: str):
    br, bc = before_shape
    ar, ac = after_shape
    print(f"🔗 {label}: {br}×{bc} → {ar}×{ac} (Δr={ar-br}, Δc={ac-bc})")

def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, path: str):
    ensure_parent(path)
    tmp = str(path) + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))
def normalize_geoid(s: pd.Series, target_len: int = DEFAULT_GEOID_LEN) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d+)", expand=False)
    L = int(target_len)
    return s.str[:L].str.zfill(L)

def extract_lat_lon(df: pd.DataFrame, out_lat="lat", out_lon="lon") -> pd.DataFrame:
    """
    Farklı şemalardan lat/lon çıkarır ve out_lat/out_lon olarak yazar.
    Denenen kolonlar:
      - (latitude, longitude) / (lat, long) / (y, x)
      - (stop_lat, stop_lon) / (stop_latitude, stop_longitude)
      - (position_latitude, position_longitude)
      - location (JSON) veya the_geom (GeoJSON)
    """
    df = df.copy()
    # doğrudan çiftler
    candidates = [
        ("latitude", "longitude"),
        ("lat", "long"),
        ("y", "x"),
        ("stop_lat", "stop_lon"),
        ("stop_latitude", "stop_longitude"),
        ("position_latitude", "position_longitude"),
    ]
    for la, lo in candidates:
        if la in df.columns and lo in df.columns:
            df[out_lat] = pd.to_numeric(df[la], errors="coerce")
            df[out_lon] = pd.to_numeric(df[lo], errors="coerce")
            return df

    # location: {"latitude":..,"longitude":..}
    if "location" in df.columns:
        def _g(o, k):
            if isinstance(o, dict): return o.get(k)
            if isinstance(o, str):
                try: j = json.loads(o); return j.get(k)
                except Exception: return None
            return None
        df[out_lat] = pd.to_numeric(df["location"].apply(lambda o: _g(o, "latitude")), errors="coerce")
        df[out_lon] = pd.to_numeric(df["location"].apply(lambda o: _g(o, "longitude")), errors="coerce")
        if df[out_lat].notna().any() and df[out_lon].notna().any():
            return df

    # the_geom: {"type":"Point","coordinates":[lon,lat]}
    if "the_geom" in df.columns:
        def _coords(o):
            if isinstance(o, dict) and "coordinates" in o and isinstance(o["coordinates"], (list, tuple)) and len(o["coordinates"]) >= 2:
                lon, lat = o["coordinates"][:2]; return lat, lon
            if isinstance(o, str):
                try:
                    j = json.loads(o)
                    if "coordinates" in j and len(j["coordinates"]) >= 2:
                        lon, lat = j["coordinates"][:2]; return lat, lon
                except Exception:
                    return None, None
            return None, None
        latlon = df["the_geom"].apply(_coords)
        df[out_lat] = pd.to_numeric(latlon.apply(lambda t: t[0]), errors="coerce")
        df[out_lon] = pd.to_numeric(latlon.apply(lambda t: t[1]), errors="coerce")
        return df

    df[out_lat] = np.nan
    df[out_lon] = np.nan
    return df

def build_kdtree_from_points_xy(xx: np.ndarray, yy: np.ndarray) -> cKDTree:
    coords = np.column_stack([xx, yy])
    return cKDTree(coords)

def query_buffer_counts(tree: cKDTree, points_xy: np.ndarray, radii_m: list[int]) -> dict:
    """
    KDTree üzerinden çoklu yarıçap için sayım döndürür.
    points_xy: (N,2)  EPSG:3857 x/y
    radii_m:   liste [300, 600, 900] gibi (metre)
    """
    out = {}
    for r in radii_m:
        idx_lists = tree.query_ball_point(points_xy, r)
        out[f"bus_within_{r}m"] = np.array([len(ix) for ix in idx_lists], dtype=np.int32)
    return out

# ========== yollar & ENV ==========
BASE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

CRIME_INPUT  = os.path.join(BASE_DIR, os.getenv("CRIME_INPUT_NAME", "fr_crime_03.csv"))
CRIME_OUTPUT = os.path.join(BASE_DIR, os.getenv("CRIME_OUTPUT_NAME", "fr_crime_04.csv"))

# ZORUNLU: Hazır hesaplanmış durak dosyası (API YOK)
BUS_CANON_RAW = os.path.join(BASE_DIR, os.getenv("BUS_CANON_RAW", "sf_bus_stops_with_geoid.csv"))

# Buffer yarıçapları (metre)
RADIUS_LIST = [int(x) for x in os.getenv("BUS_BUFFER_METERS", "300,600,900").split(",")]

# ========== 1) crime oku ==========
if not os.path.exists(CRIME_INPUT):
    raise FileNotFoundError(f"❌ Suç girdi dosyası yok: {CRIME_INPUT}")
crime = pd.read_csv(CRIME_INPUT, low_memory=False)
_before_shape = crime.shape
log_shape(crime, "CRIME (okundu)")

# LAT/LON çıkar (crime)
crime = extract_lat_lon(crime, out_lat="crime_lat", out_lon="crime_lon")
if not (crime["crime_lat"].notna().any() and crime["crime_lon"].notna().any()):
    raise RuntimeError("❌ CRIME verisinde latitude/longitude tespit edilemedi. (latitude/longitude|lat/long|y/x vb.)")

# ========== 2) bus stops: hazır dosyadan oku ==========
if not os.path.exists(BUS_CANON_RAW):
    raise FileNotFoundError(f"❌ Güncel durak dosyası yok: {BUS_CANON_RAW} (API kullanılmayacak).")
bus = pd.read_csv(BUS_CANON_RAW, low_memory=False)
log_shape(bus, "BUS (okundu)")

# LAT/LON çıkar (bus)
if not ({"stop_lat","stop_lon"} <= set(bus.columns)):
    bus = extract_lat_lon(bus, out_lat="stop_lat", out_lon="stop_lon")
bus = bus.dropna(subset=["stop_lat","stop_lon"]).copy()
log_shape(bus, "BUS (lat/lon temiz)")

# ========== 3) Projeksiyon: EPSG:4326 → 3857 (metre) ==========
gdf_crime = gpd.GeoDataFrame(
    crime,
    geometry=gpd.points_from_xy(crime["crime_lon"], crime["crime_lat"]),
    crs="EPSG:4326"
).to_crs(3857)

gdf_bus = gpd.GeoDataFrame(
    bus,
    geometry=gpd.points_from_xy(bus["stop_lon"], bus["stop_lat"]),
    crs="EPSG:4326"
).to_crs(3857)

# KDTree üzerine durak koordinatlarını hazırla
bus_x = gdf_bus.geometry.x.to_numpy()
bus_y = gdf_bus.geometry.y.to_numpy()
if bus_x.size == 0:
    raise RuntimeError("❌ Durak listesi boş görünüyor (stop_lat/stop_lon yok).")

tree = build_kdtree_from_points_xy(bus_x, bus_y)

# ========== 4) Nearest distance + buffer sayıları ==========
crime_xy = np.column_stack([gdf_crime.geometry.x.to_numpy(), gdf_crime.geometry.y.to_numpy()])
# En yakın mesafe (metre)
distances, _ = tree.query(crime_xy, k=1)
# Çoklu yarıçap sayımları
buf_counts = query_buffer_counts(tree, crime_xy, RADIUS_LIST)

# ========== 5) Özellik kolonlarını ekle ==========
crime["distance_to_bus_m"] = distances.astype(float)
for r, arr in buf_counts.items():
    crime[r] = arr

# İsteğe

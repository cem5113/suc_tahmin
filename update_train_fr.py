# update_train.py
# AMAÇ:
# - ZATEN HAZIR "sf_train_stops_with_geoid.csv" dosyasındaki (stop_lat, stop_lon) noktalarını KULLAN.
# - fr_crime_04 ile GEOID bazında zenginleştir:
#     * Blok centroid’ine göre tren durağına EN YAKIN MESAFE (metre)
#     * 300m / 600m / 900m buffer İÇİNDEKİ tren durağı sayıları (kümülatif)
#     * İsteğe bağlı: halka bazlı sayılar (0–300, 300–600, 600–900)
# - ÇIKTI: fr_crime_05.csv
#
# NOTLAR:
# - Mesafe hesapları için tüm geometri EPSG:3857’e projeksiyonlanır (metre).
# - GEOID evreni fr_crime_04’teki GEOID’lerle sınırlanır.
# - İndirme/GTFS yok. Sadece hazır CSV okunur.

import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

pd.options.mode.copy_on_write = True

# =========================
# Küçük yardımcılar
# =========================
def ensure_parent(path: str) -> None:
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, path: str) -> None:
    """Atomic yazım: tmp → replace; hata halinde .bak bırak."""
    ensure_parent(path)
    tmp = path + ".tmp"
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
        print(f"💾 Kaydedildi: {path}")
    except Exception as e:
        print(f"❌ Kaydetme hatası: {path}\n{e}")
        try:
            df.to_csv(path + ".bak", index=False)
            print(f"📁 Yedek oluşturuldu: {path}.bak")
        except Exception:
            pass

def log_shape(df: pd.DataFrame, label: str) -> None:
    r, c = df.shape
    print(f"📊 {label}: {r} satır × {c} sütun")

def log_delta(before_shape, after_shape, label: str) -> None:
    br, bc = before_shape
    ar, ac = after_shape
    print(f"🔗 {label}: {br}×{bc} → {ar}×{ac} (Δr={ar-br}, Δc={ac-bc})")

DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

def normalize_geoid(series: pd.Series, target_len: int = DEFAULT_GEOID_LEN) -> pd.Series:
    """Yalnızca rakamları al, SOL’dan target_len haneyi tut, zfill."""
    s = series.astype(str).str.extract(r"(\d+)", expand=False)
    L = int(target_len)
    return s.str[:L].str.zfill(L)

# =========================
# ENV / Yollar
# =========================
BASE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

CRIME_INPUT  = os.path.join(BASE_DIR, os.getenv("CRIME_INPUT_NAME", "fr_crime_04.csv"))
CRIME_OUTPUT = os.path.join(BASE_DIR, os.getenv("CRIME_OUTPUT_NAME", "fr_crime_05.csv"))

# Hazır tren durakları (lat/lon + opsiyonel GEOID) — ZATEN HAZIR
TRAIN_STOPS_WITH_GEOID = os.path.join(BASE_DIR, os.getenv("TRAIN_STOPS_NAME", "sf_train_stops_with_geoid.csv"))

# Census GeoJSON (blok geometrileri → centroid)
CENSUS_CANDIDATES = [
    os.path.join(BASE_DIR, "sf_census_blocks_with_population.geojson"),
    os.path.join(".",      "sf_census_blocks_with_population.geojson"),
]

# Buffer yarıçapları (metre)
BUF_RADII = tuple(int(x) for x in os.getenv("TRAIN_BUFFER_RADII", "300,600,900").split(","))  # 300,600,900

# =========================
# 1) Girdileri oku
# =========================
if not os.path.exists(CRIME_INPUT):
    raise FileNotFoundError(f"❌ Suç girdi dosyası yok: {CRIME_INPUT}")
crime = pd.read_csv(CRIME_INPUT, low_memory=False)
log_shape(crime, "CRIME (okundu)")
if "GEOID" not in crime.columns:
    raise KeyError("❌ Suç verisinde 'GEOID' kolonu yok.")
crime["GEOID"] = normalize_geoid(crime["GEOID"], DEFAULT_GEOID_LEN)
crime_geoids = pd.Series(crime["GEOID"].unique(), name="GEOID")
print(f"🧩 CRIME farklı GEOID sayısı: {crime_geoids.size}")

if not os.path.exists(TRAIN_STOPS_WITH_GEOID):
    raise FileNotFoundError(f"❌ Hazır tren durakları CSV bulunamadı: {TRAIN_STOPS_WITH_GEOID}")
stops = pd.read_csv(TRAIN_STOPS_WITH_GEOID, low_memory=False)
# Kolon isimlerini sağlamla
low = {c.lower(): c for c in stops.columns}
lat_col = low.get("stop_lat") or low.get("latitude") or low.get("lat")
lon_col = low.get("stop_lon") or low.get("longitude") or low.get("long") or low.get("lon")
if not lat_col or not lon_col:
    raise KeyError("❌ sf_train_stops_with_geoid.csv içinde stop_lat/stop_lon (veya latitude/longitude) bulunamadı.")
stops["stop_lat"] = pd.to_numeric(stops[lat_col], errors="coerce")
stops["stop_lon"] = pd.to_numeric(stops[lon_col], errors="coerce")
stops = stops.dropna(subset=["stop_lat", "stop_lon"]).copy()
log_shape(stops, "TRAIN stops (temiz)")

# =========================
# 2) Blok geometrileri (centroid) — GEOID evreni
# =========================
census_path = next((p for p in CENSUS_CANDIDATES if os.path.exists(p)), None)
if census_path is None:
    raise FileNotFoundError("❌ Nüfus blokları GeoJSON bulunamadı (sf_census_blocks_with_population.geojson).")

gdf_blocks = gpd.read_file(census_path)
# CRS → EPSG:4326
if gdf_blocks.crs is None:
    gdf_blocks.set_crs("EPSG:4326", inplace=True, allow_override=True)
else:
    try:
        epsg = gdf_blocks.crs.to_epsg()
    except Exception:
        epsg = None
    if epsg != 4326:
        gdf_blocks = gdf_blocks.to_crs(epsg=4326)

# GEOID sütunu bul/normalize et
gcol = "GEOID" if "GEOID" in gdf_blocks.columns else next(
    (c for c in gdf_blocks.columns if str(c).upper().startswith("GEOID")), None
)
if not gcol:
    raise KeyError("❌ GeoJSON'da GEOID benzeri sütun yok.")
gdf_blocks["GEOID"] = normalize_geoid(gdf_blocks[gcol], DEFAULT_GEOID_LEN)

# Sadece CRIME evrenindeki bloklar
gdf_blocks = gdf_blocks[gdf_blocks["GEOID"].isin(crime_geoids)].copy()
log_shape(gdf_blocks, "BLOCKS (crime evreni)")

# =========================
# 3) Geometriyi 3857’ye projekte et ve centroid al
# =========================
gdf_blocks_3857 = gdf_blocks[["GEOID", "geometry"]].to_crs(3857).copy()
gdf_blocks_3857["cx"] = gdf_blocks_3857.geometry.centroid.x
gdf_blocks_3857["cy"] = gdf_blocks_3857.geometry.centroid.y
blocks_xy = np.vstack([gdf_blocks_3857["cx"].to_numpy(), gdf_blocks_3857["cy"].to_numpy()]).T

gdf_stops_3857 = gpd.GeoDataFrame(
    stops[["stop_lat", "stop_lon"]].copy(),
    geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
    crs="EPSG:4326",
).to_crs(3857)
stops_xy = np.vstack([gdf_stops_3857.geometry.x.to_numpy(), gdf_stops_3857.geometry.y.to_numpy()]).T

# =========================
# 4) KD-Tree ile yakınlık ve buffer sayıları (300/600/900m)
# =========================
metrics = pd.DataFrame({"GEOID": gdf_blocks_3857["GEOID"].to_numpy()})
metrics["distance_to_train_m"] = np.nan
for r in BUF_RADII:
    metrics[f"train_stop_count_{r}m"] = 0

# Ayrıca halka sayıları (0–300, 300–600, 600–900)
for i, r in enumerate(BUF_RADII):
    lo = 0 if i == 0 else BUF_RADII[i - 1]
    metrics[f"train_stop_count_{lo}_{r}m"] = 0

if len(stops_xy) > 0 and len(blocks_xy) > 0:
    tree = cKDTree(stops_xy)
    # En yakın mesafe
    nearest_dist, _ = tree.query(blocks_xy, k=1)
    metrics["distance_to_train_m"] = nearest_dist.astype(float)

    # Kümülatif & halka sayıları
    prev_counts = None
    for i, r in enumerate(BUF_RADII):
        lists = tree.query_ball_point(blocks_xy, r=r)
        counts = np.array([len(x) for x in lists], dtype=int)
        metrics[f"train_stop_count_{r}m"] = counts
        if i == 0:
            metrics[f"train_stop_count_0_{r}m"] = counts
            prev_counts = counts
        else:
            ring = counts - prev_counts
            ring[ring < 0] = 0
            lo = BUF_RADII[i - 1]
            metrics[f"train_stop_count_{lo}_{r}m"] = ring
            prev_counts = counts
else:
    print("⚠️ Uyarı: Ya blok ya da durak listesi boş; metrikler NaN/0 ile dolduruldu.")

# Tip/na temizliği
metrics["GEOID"] = normalize_geoid(metrics["GEOID"], DEFAULT_GEOID_LEN)
for c in metrics.columns:
    if c.startswith("train_stop_count_"):
        metrics[c] = pd.to_numeric(metrics[c], errors="coerce").fillna(0).astype(int)
metrics["distance_to_train_m"] = pd.to_numeric(metrics["distance_to_train_m"], errors="coerce")

log_shape(metrics, "GEOID-bazlı tren metrikleri")

# =========================
# 5) CRIME ile merge
# =========================
_before = crime.shape
crime["GEOID"]  = normalize_geoid(crime["GEOID"], DEFAULT_GEOID_LEN)
metrics["GEOID"] = normalize_geoid(metrics["GEOID"], DEFAULT_GEOID_LEN)

crime_enriched = crime.merge(metrics, on="GEOID", how="left", validate="many_to_one")

log_delta(_before, crime_enriched.shape, "CRIME ⨯ TRAIN (buffer enrich)")
log_shape(crime_enriched, "CRIME (train buffer enrich sonrası)")

# =========================
# 6) Kaydet & önizleme
# =========================
safe_save_csv(crime_enriched, CRIME_OUTPUT)
print("📦 Eklenen başlıca sütunlar:", ["distance_to_train_m"] +
      [f"train_stop_count_{r}m" for r in BUF_RADII] +
      [f"train_stop_count_{0 if i==0 else BUF_RADII[i-1]}_{r}m" for i, r in enumerate(BUF_RADII)])
print(f"✅ Güncellenmiş veri kaydedildi → {CRIME_OUTPUT}")
try:
    print("Örnek ilk 5 satır:")
    with pd.option_context("display.max_columns", 60, "display.width", 2000):
        print(crime_enriched.head(5).to_string(index=False))
except Exception:
    pass

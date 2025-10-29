# Update_forecast_crime.py
# — sf_crime_grid_full_labeled.csv → GEOID-Y_label özeti (sf_crime_L.csv)
# — sf_crime_y.csv ile birleştirir, takvim bayraklarını ve tüm zenginleştirmeleri ekler
# — nihai çıktı: fr_crime_grid_full_labeled.csv
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np
import holidays

# =========================
# CONFIG
# =========================
DATA_DIR = os.getenv("CRIME_DATA_DIR", ".")  # gerekirse "/content/drive/MyDrive/crime_data"
GRID_FULL_PATH = os.path.join(DATA_DIR, "sf_crime_grid_full_labeled.csv")   # kaynak grid
Y_ONLY_PATH    = os.path.join(DATA_DIR, "sf_crime_y.csv")                   # en güncel olay tabanı

# Entegrasyon kaynakları (mevcutsa otomatik eklenir; yoksa atlanır)
SRC_911   = os.path.join(DATA_DIR, "sf_911_last_5_year.csv")
SRC_311   = os.path.join(DATA_DIR, "sf_311_last_5_years.csv")
SRC_WEATH = os.path.join(DATA_DIR, "sf_weather_5years.csv")
SRC_POP   = os.path.join(DATA_DIR, "sf_population.csv")
SRC_POI   = os.path.join(DATA_DIR, "sf_pois_with_risk_score.csv")
SRC_NEI   = os.path.join(DATA_DIR, "sf_crime_11.csv")               # neighbor_* özellikleri
SRC_GOV   = os.path.join(DATA_DIR, "sf_police_gov_crime.csv")       # distance_to_police, is_near_*
SRC_BUS   = os.path.join(DATA_DIR, "sf_crime_06.csv")               # bus_* (distance_to_bus, bus_stop_count)
SRC_TRAIN = os.path.join(DATA_DIR, "sf_crime_07.csv")               # train_* (distance_to_train, train_stop_count)

# Çıktılar
OUT_GEOID_L = os.path.join(DATA_DIR, "sf_crime_L.csv")              # yalnızca GEOID & Y_label
OUT_FINAL   = os.path.join(DATA_DIR, "fr_crime_grid_full_labeled.csv")

# İSTENMEYEN ALANLAR: oluşturma/kullanma
DENY_EXACT = {
    "911_request_count_hour_range_x",
    "911_request_count_daily_before_24_hours_x",
    "temp_range",
    "population_x",
}
# Güvenlik ağı: kazara kalan *_x'leri de sil
def is_denied(col: str) -> bool:
    if col in DENY_EXACT:
        return True
    if col.endswith("_x"):
        return True
    return False

# =========================
# HELPERS
# =========================
def normalize_geoid(series: pd.Series, target_len: int = 11) -> pd.Series:
    s = series.astype(str).str.extract(r"(\d+)", expand=False)
    return s.str[:target_len].str.zfill(target_len)

def load_df(path: str, dtype=None) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        print(f"ℹ️ Bulunamadı (atlandı): {p.name}")
        return None
    try:
        df = pd.read_csv(p, dtype=dtype)
        print(f"✓ Yüklendi: {p.name} → {df.shape[0]} satır, {df.shape[1]} sütun")
        return df
    except Exception as e:
        print(f"⚠️ Okuma hatası ({p.name}): {e}")
        return None

def ensure_datetime_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # date/time yoksa türet
    if "date" not in out.columns and "datetime" in out.columns:
        out["date"] = pd.to_datetime(out["datetime"], errors="coerce").dt.date
    if "time" not in out.columns:
        out["time"] = "00:00:00"
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["datetime"] = pd.to_datetime(out["date"].dt.strftime("%Y-%m-%d") + " " + out["time"].astype(str), errors="coerce")
    if "event_hour" not in out.columns:
        out["event_hour"] = out["datetime"].dt.hour
    out["day_of_week"] = out["datetime"].dt.weekday
    out["month"]       = out["datetime"].dt.month
    return out

def add_calendar_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dates_only = pd.to_datetime(out["datetime"], errors="coerce").dt.normalize()
    if dates_only.notna().any():
        min_year = int(dates_only.min().year)
        max_year = int(dates_only.max().year)
        us_hol = holidays.US(years=range(min_year, max_year + 1))
        hol_idx = pd.DatetimeIndex(pd.to_datetime(list(us_hol.keys()))).normalize()
        out["is_holiday"] = dates_only.isin(hol_idx).astype(int)
    else:
        out["is_holiday"] = 0
    out["is_weekend"]       = (out["day_of_week"] >= 5).astype(int)
    out["is_night"]         = ((out["event_hour"] >= 22) | (out["event_hour"] <= 5)).astype(int)
    out["is_school_hour"]   = out["event_hour"].between(8, 15).astype(int)
    out["is_business_hour"] = (out["event_hour"].between(9, 17) & (out["day_of_week"] < 5)).astype(int)
    season_map = {12:"Winter",1:"Winter",2:"Winter", 3:"Spring",4:"Spring",5:"Spring",
                  6:"Summer",7:"Summer",8:"Summer", 9:"Fall",10:"Fall",11:"Fall"}
    out["season"] = out["month"].map(season_map)
    out["hr_key"] = (out["event_hour"] // 3) * 3
    out["hour_range"] = out["hr_key"].map(lambda h: f"{h:02d}-{min(h+3,24):02d}")
    # SF iklim bayrakları
    out["sf_wet_season"] = out["season"].map({"Winter":1,"Spring":1,"Summer":0,"Fall":0}).fillna(0).astype(int)
    out["sf_dry_season"] = out["season"].map({"Winter":0,"Spring":0,"Summer":1,"Fall":1}).fillna(0).astype(int)
    out["sf_fog_season"] = out["season"].map({"Winter":0,"Spring":0,"Summer":1,"Fall":0}).fillna(0).astype(int)
    return out

def geoid_len_mode(s: pd.Series, default_len: int = 11) -> int:
    s = s.dropna().astype(str).str.len()
    return int(s.mode().iat[0]) if not s.empty else default_len

def _drop_overlaps_from_addon(main: pd.DataFrame, addon: pd.DataFrame) -> pd.DataFrame:
    """Ana DF'de zaten bulunan kolonları (GEOID hariç) ek veri setinden ayıkla ki *_x oluşmasın."""
    inter = addon.copy()
    overlap = set(inter.columns).intersection(set(main.columns))
    overlap.discard("GEOID")
    if overlap:
        inter = inter.drop(columns=list(overlap))
    return inter

def merge_on_geoid(main: pd.DataFrame, addon: pd.DataFrame, cols_keep: list[str], how: str = "left") -> pd.DataFrame:
    inter = addon.copy()
    for c in ("GEOID",):
        if c in inter.columns:
            inter[c] = normalize_geoid(inter[c], geoid_len_mode(inter[c]))
    # Sadece istenen kolonları ve GEOID'i al
    inter = inter[["GEOID"] + [c for c in cols_keep if c in inter.columns]].drop_duplicates("GEOID")
    # Ana DF'de zaten olan kolonları at (GEOID hariç)
    inter = _drop_overlaps_from_addon(main, inter)
    out = main.merge(inter, on="GEOID", how=how)
    return out

def merge_on_geoid_date_hour(main: pd.DataFrame, addon: pd.DataFrame, cols_keep: list[str], how: str = "left") -> pd.DataFrame:
    inter = addon.copy()
    if "datetime" not in inter.columns:
        if "date" in inter.columns:
            if "time" not in inter.columns:
                inter["time"] = "00:00:00"
            inter["date"] = pd.to_datetime(inter["date"], errors="coerce")
            inter["datetime"] = pd.to_datetime(inter["date"].dt.strftime("%Y-%m-%d") + " " + inter["time"].astype(str), errors="coerce")
    inter["event_hour"] = pd.to_datetime(inter["datetime"]).dt.hour
    inter["date"] = pd.to_datetime(inter["datetime"]).dt.date
    for c in ("GEOID",):
        if c in inter.columns:
            inter[c] = normalize_geoid(inter[c], geoid_len_mode(inter[c]))
    keys = ["GEOID", "date", "event_hour"]
    inter = inter[keys + [c for c in cols_keep if c in inter.columns]]
    # Ana tarafta da date olmalı (main["datetime"] var)
    m2 = main.copy()
    if "date" not in m2.columns:
        m2["date"] = pd.to_datetime(m2["datetime"]).dt.date
    # Üst üste binmeleri engelle: ek DF'den ana ile çakışan kolonları düş
    inter = _drop_overlaps_from_addon(m2, inter)
    out = m2.merge(inter, on=keys, how=how)
    return out

def final_prune_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if not is_denied(c)]
    return df[cols].copy()

# =========================
# 1) GRID_FULL → sf_crime_L.csv
# =========================
grid_full = load_df(GRID_FULL_PATH, dtype={"GEOID": str})
if grid_full is None:
    raise SystemExit("sf_crime_grid_full_labeled.csv bulunamadı.")

gl = geoid_len_mode(grid_full["GEOID"])
grid_full["GEOID"] = normalize_geoid(grid_full["GEOID"], gl)

if "Y_label" not in grid_full.columns:
    raise SystemExit("grid_full içinde 'Y_label' kolonu bulunamadı.")

sf_crime_L = (grid_full.groupby("GEOID")["Y_label"]
              .max()
              .reset_index()
              .astype({"Y_label":"int8"}))
sf_crime_L.to_csv(OUT_GEOID_L, index=False)
print(f"✅ Yazıldı: {Path(OUT_GEOID_L).name} — {sf_crime_L.shape[0]} GEOID")

# =========================
# 2) sf_crime_y.csv → normalize + calendar flags
# =========================
y_df = load_df(Y_ONLY_PATH, dtype={"GEOID": str})
if y_df is None:
    raise SystemExit("sf_crime_y.csv bulunamadı.")

y_gl = geoid_len_mode(y_df["GEOID"])
y_df["GEOID"] = normalize_geoid(y_df["GEOID"], y_gl)
y_df = ensure_datetime_cols(y_df)
y_df = add_calendar_flags(y_df)

# Merge GEOID→Y_label
merged = y_df.merge(sf_crime_L, on="GEOID", how="left")
merged["Y_label"] = merged["Y_label"].fillna(0).astype("int8")
if "crime_mix" not in merged.columns:
    merged["crime_mix"] = ""

# =========================
# 3) ENRICH: 911, 311, weather, population, poi, neighbor, gov, bus, train
# =========================
# --- 911: saatlik/GEOID bazlı yoğunluk (varsa) ---
df_911 = load_df(SRC_911)
if df_911 is not None:
    # varsa zaten saatlik özet kolonları olabilir. yoksa sayım üret.
    has_counts = any(k in df_911.columns for k in ["count", "911_request_count_hour_range", "request_count"])
    if not has_counts:
        # GEOID + date + event_hour sayım
        df_911 = ensure_datetime_cols(df_911)
        tmp = (df_911.groupby([normalize_geoid(df_911["GEOID"], geoid_len_mode(df_911["GEOID"])) ,
                               pd.to_datetime(df_911["datetime"]).dt.date,
                               df_911["event_hour"]])
               .size().reset_index())
        tmp.columns = ["GEOID", "date", "event_hour", "call_911_count"]
        merged = merge_on_geoid_date_hour(merged, tmp, ["call_911_count"])
    else:
        # Doğrudan uygun kolonları bırak (deny list dışı)
        keep_all = [c for c in df_911.columns if c.startswith("911_") or "call" in c.lower() or "request" in c.lower()]
        keep = [c for c in keep_all if not is_denied(c)]
        if keep:
            merged = merge_on_geoid(merged, df_911, keep)

# --- 311: günlük/GEOID bazlı ---
df_311 = load_df(SRC_311)
if df_311 is not None:
    has_counts = any(k in df_311.columns for k in ["count", "311_request_count", "request_count"])
    if not has_counts:
        df_311 = ensure_datetime_cols(df_311)
        tmp = (df_311.groupby([normalize_geoid(df_311["GEOID"], geoid_len_mode(df_311["GEOID"])) ,
                               pd.to_datetime(df_311["datetime"]).dt.date])
               .size().reset_index())
        tmp.columns = ["GEOID", "date", "req_311_count"]
        # Günlük bazlı merge (date)
        m2 = merged.copy()
        if "date" not in m2.columns:
            m2["date"] = pd.to_datetime(m2["datetime"]).dt.date
        # Çakışan kolonları önceden düşür
        tmp = _drop_overlaps_from_addon(m2, tmp)
        merged = m2.merge(tmp, on=["GEOID", "date"], how="left")
    else:
        keep = [c for c in df_311.columns if (c.startswith("311_") or "request" in c.lower()) and not is_denied(c)]
        if keep:
            merged = merge_on_geoid(merged, df_311, keep)

# --- Weather: günlük ---
df_w = load_df(SRC_WEATH)
if df_w is not None:
    # beklenen olası kolonlar: precipitation_mm, temp_min, temp_max, temp_range (temp_range yasak)
    # merge on date
    if "date" in df_w.columns:
        df_w["date"] = pd.to_datetime(df_w["date"], errors="coerce").dt.date
    elif "datetime" in df_w.columns:
        df_w["date"] = pd.to_datetime(df_w["datetime"], errors="coerce").dt.date
    keep_all = [c for c in df_w.columns if any(k in c.lower() for k in ["precip", "temp", "wind", "humidity", "weather"])]
    # temp_range'i özellikle çıkar
    keep = [c for c in keep_all if c != "temp_range" and not is_denied(c)]
    if keep:
        m2 = merged.copy()
        if "date" not in m2.columns:
            m2["date"] = pd.to_datetime(m2["datetime"]).dt.date
        m2["date"] = pd.to_datetime(m2["date"], errors="coerce")
        df_w["date"] = pd.to_datetime(df_w["date"], errors="coerce")
        # Çakışmaları önle
        dfw_trim = _drop_overlaps_from_addon(m2, df_w[["date"] + keep])
        merged = m2.merge(dfw_trim, on="date", how="left")

# --- Population: GEOID ---
df_pop = load_df(SRC_POP, dtype={"GEOID": str})
if df_pop is not None:
    df_pop["GEOID"] = normalize_geoid(df_pop["GEOID"], geoid_len_mode(df_pop["GEOID"]))
    keep_all = [c for c in df_pop.columns if c != "GEOID"]
    keep = [c for c in keep_all if not is_denied(c)]
    if keep:
        merged = merge_on_geoid(merged, df_pop, keep)

# --- POI risk: GEOID ---
df_poi = load_df(SRC_POI, dtype={"GEOID": str})
if df_poi is not None:
    df_poi["GEOID"] = normalize_geoid(df_poi["GEOID"], geoid_len_mode(df_poi["GEOID"]))
    keep = [c for c in df_poi.columns if c != "GEOID" and not is_denied(c)]
    if keep:
        merged = merge_on_geoid(merged, df_poi, keep)

# --- Neighbor features: GEOID (örn. neighbor_crime_7d, crime_last_7d) ---
df_nei = load_df(SRC_NEI, dtype={"GEOID": str})
if df_nei is not None:
    df_nei["GEOID"] = normalize_geoid(df_nei["GEOID"], geoid_len_mode(df_nei["GEOID"]))
    keep = [c for c in df_nei.columns
            if c != "GEOID" and any(k in c for k in ["neighbor", "neigh", "_7d", "_3d", "crime_last"]) and not is_denied(c)]
    if keep:
        merged = merge_on_geoid(merged, df_nei, keep)

# --- Government & Police: GEOID (distance_to_police, is_near_*) ---
df_gov = load_df(SRC_GOV, dtype={"GEOID": str})
if df_gov is not None:
    df_gov["GEOID"] = normalize_geoid(df_gov["GEOID"], geoid_len_mode(df_gov["GEOID"]))
    keep = [c for c in df_gov.columns if c != "GEOID" and not is_denied(c)]
    if keep:
        merged = merge_on_geoid(merged, df_gov, keep)

# --- Bus: GEOID (distance_to_bus, bus_stop_count, *_range) ---
df_bus = load_df(SRC_BUS, dtype={"GEOID": str})
if df_bus is not None:
    df_bus["GEOID"] = normalize_geoid(df_bus["GEOID"], geoid_len_mode(df_bus["GEOID"]))
    keep = [c for c in df_bus.columns if c != "GEOID" and not is_denied(c)]
    if keep:
        merged = merge_on_geoid(merged, df_bus, keep)

# --- Train: GEOID (distance_to_train, train_stop_count, *_range) ---
df_train = load_df(SRC_TRAIN, dtype={"GEOID": str})
if df_train is not None:
    df_train["GEOID"] = normalize_geoid(df_train["GEOID"], geoid_len_mode(df_train["GEOID"]))
    keep = [c for c in df_train.columns if c != "GEOID" and not is_denied(c)]
    if keep:
        merged = merge_on_geoid(merged, df_train, keep)

# =========================
# 4) Son temizlik ve yazım
# =========================
# tip düzeltmeleri
if "Y_label" in merged.columns:
    merged["Y_label"] = merged["Y_label"].fillna(0).astype("int8")

# İstenmeyen kolonları kesin olarak temizle
merged = final_prune_columns(merged)

# Kolon sıralama (önemliler öne)
cols_first = [
    "id","date","time","datetime","GEOID",
    "event_hour","day_of_week","month",
    "Y_label","crime_mix",
    "is_holiday","is_weekend","is_night","is_school_hour","is_business_hour",
    "season","hr_key","hour_range",
    "sf_wet_season","sf_dry_season","sf_fog_season"
]
final_cols = [c for c in cols_first if c in merged.columns] + [c for c in merged.columns if c not in cols_first]
merged[final_cols].to_csv(OUT_FINAL, index=False)

print(f"\n🎉 Bitti! Yazıldı: {Path(OUT_FINAL).name} — {merged.shape[0]} satır, {merged.shape[1]} sütun")
print(f"   Ayrıca GEOID etiket özeti: {Path(OUT_GEOID_L).name} — {sf_crime_L.shape[0]} satır")

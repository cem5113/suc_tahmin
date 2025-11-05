#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# update_911_daily.py — sf_crime.csv → (1) fr_crime_events_daily.csv, (2) fr_crime_grid_daily.csv
# - Artifact ZIP açılır (varsa), sonra çoklu konumda 911/census/crime aranır
# - ENV ile tek satır override: FR_911=/path/to/sf_911_last_5_year.csv python update_911_daily.py

from __future__ import annotations
import os, zipfile
from pathlib import Path
from typing import Optional
import pandas as pd
import geopandas as gpd

def log(msg: str): print(msg, flush=True)
def ensure_parent(path: str): Path(path).parent.mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, path: str):
    try:
        ensure_parent(path); tmp = path + ".tmp"
        df.to_csv(tmp, index=False); os.replace(tmp, path)
        log(f"💾 Kaydedildi: {path} (satır={len(df):,})")
    except Exception as e:
        b = path + ".bak"
        try: df.to_csv(b, index=False)
        except Exception: pass
        log(f"❌ Kaydetme hatası: {path} — Yedek: {b}\n{e}")

def to_date(s): return pd.to_datetime(s, errors="coerce").dt.date
def normalize_geoid(s: pd.Series, n: int) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d+)", expand=False); return s.str[:n].str.zfill(n)

def first_existing(paths) -> Optional[Path]:
    for p in paths:
        if p and Path(p).exists(): return Path(p)
    return None

# —— Ortak kökler
CRIME_DATA_DIR = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data")).resolve()
ARTIFACT_ZIP   = Path(os.getenv("ARTIFACT_ZIP", "artifact/sf-crime-pipeline-output.zip"))
ARTIFACT_DIR   = Path(os.getenv("ARTIFACT_DIR", "artifact_unzipped"))

# Giriş/çıkış isimleri
INPUT_CRIME_FILENAME = os.getenv("FR_CRIME_FILE", "sf_crime.csv")
OUTPUT_EVENTS_DAILY  = os.getenv("FR_EVENTS_DAILY_OUT", "fr_crime_events_daily.csv")
OUTPUT_GRID_DAILY    = os.getenv("FR_GRID_DAILY_OUT",   "fr_crime_grid_daily.csv")
OUTPUT_DIR           = Path(os.getenv("FR_OUTPUT_DIR", "crime_prediction_data"))

def build_candidates():
    return {
        "FR_911": [
            # artifact klasörü (uzantılı + uzantısız)
            ARTIFACT_DIR / "sf_911_last_5_year_y.csv",
            ARTIFACT_DIR / "sf_911_last_5_year.csv",
            ARTIFACT_DIR / "sf_911_last_5_year_y",
            ARTIFACT_DIR / "sf_911_last_5_year",

            # artifact alt zip açılmış klasör adıyla
            ARTIFACT_DIR / "sf-crime-pipeline-output" / "sf_911_last_5_year_y.csv",
            ARTIFACT_DIR / "sf-crime-pipeline-output" / "sf_911_last_5_year.csv",
            ARTIFACT_DIR / "sf-crime-pipeline-output" / "sf_911_last_5_year_y",
            ARTIFACT_DIR / "sf-crime-pipeline-output" / "sf_911_last_5_year",

            # repo kökü (mutlak CRIME_DATA_DIR ve göreli)
            CRIME_DATA_DIR / "sf_911_last_5_year_y.csv",
            CRIME_DATA_DIR / "sf_911_last_5_year.csv",
            CRIME_DATA_DIR / "sf_911_last_5_year_y",
            CRIME_DATA_DIR / "sf_911_last_5_year",
            Path("crime_prediction_data") / "sf_911_last_5_year_y.csv",
            Path("crime_prediction_data") / "sf_911_last_5_year.csv",
            Path("crime_prediction_data") / "sf_911_last_5_year_y",
            Path("crime_prediction_data") / "sf_911_last_5_year",

            # emniyet payı: bazı repo'larda ...years
            ARTIFACT_DIR / "sf_911_last_5_years.csv",
            ARTIFACT_DIR / "sf_911_last_5_years",
            CRIME_DATA_DIR / "sf_911_last_5_years.csv",
            CRIME_DATA_DIR / "sf_911_last_5_years",
            Path("crime_prediction_data") / "sf_911_last_5_years.csv",
            Path("crime_prediction_data") / "sf_911_last_5_years",
        ],
        "CENSUS": [
            ARTIFACT_DIR / "sf_census_blocks_with_population.geojson",
            ARTIFACT_DIR / "sf-crime-pipeline-output" / "sf_census_blocks_with_population.geojson",
            CRIME_DATA_DIR / "sf_census_blocks_with_population.geojson",
            Path("crime_prediction_data") / "sf_census_blocks_with_population.geojson",
            Path("./sf_census_blocks_with_population.geojson"),
        ],
        "CRIME": [
            ARTIFACT_DIR / INPUT_CRIME_FILENAME,
            ARTIFACT_DIR / "sf-crime-pipeline-output" / INPUT_CRIME_FILENAME,
            CRIME_DATA_DIR / INPUT_CRIME_FILENAME,
            Path("crime_prediction_data") / INPUT_CRIME_FILENAME,
            Path(INPUT_CRIME_FILENAME),
        ],
    }

def _is_within_directory(directory: Path, target: Path) -> bool:
    try:
        return str(target.resolve()).startswith(str(directory.resolve()))
    except Exception:
        return False

def safe_unzip(zip_path: Path, dest_dir: Path):
    if not zip_path.exists():
        log(f"ℹ️ Artifact ZIP bulunamadı: {zip_path} — klasörlerden denenecek."); return
    dest_dir.mkdir(parents=True, exist_ok=True)
    log(f"📦 ZIP açılıyor: {zip_path} → {dest_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for m in zf.infolist():
            out_path = dest_dir / m.filename
            if not _is_within_directory(dest_dir, out_path.parent):
                raise RuntimeError(f"Zip path outside target dir engellendi: {m.filename}")
            if m.is_dir(): out_path.mkdir(parents=True, exist_ok=True); continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m, 'r') as src, open(out_path, 'wb') as dst:
                dst.write(src.read())
    log("✅ ZIP çıkarma tamam.")

def _load_blocks(census_candidates) -> tuple[gpd.GeoDataFrame, int]:
    census_path = first_existing(census_candidates)
    if census_path is None:
        raise FileNotFoundError("❌ GEOID poligon dosyası yok: sf_census_blocks_with_population.geojson")
    gdf = gpd.read_file(census_path)
    if "GEOID" not in gdf.columns:
        cand = [c for c in gdf.columns if str(c).upper().startswith("GEOID")]
        if not cand: raise ValueError("GeoJSON içinde GEOID benzeri sütun yok.")
        gdf = gdf.rename(columns={cand[0]: "GEOID"})
    tlen = gdf["GEOID"].astype(str).str.len().mode().iat[0]
    gdf["GEOID"] = normalize_geoid(gdf["GEOID"], int(tlen))
    if gdf.crs is None: gdf.set_crs("EPSG:4326", inplace=True)
    elif gdf.crs.to_epsg() != 4326: gdf = gdf.to_crs(4326)
    return gdf, int(tlen)

def ensure_geoid_from_latlon(df: pd.DataFrame, census_candidates) -> pd.DataFrame:
    if "GEOID" in df.columns and df["GEOID"].notna().any(): return df
    lat_col = next((c for c in ["latitude","lat","y"] if c in df.columns), None)
    lon_col = next((c for c in ["longitude","lon","x"] if c in df.columns), None)
    if not lat_col or not lon_col: raise ValueError("❌ Veride GEOID yok ve lat/lon bulunamadı.")
    gdf_blocks, tlen = _load_blocks(census_candidates)
    tmp = df.copy()
    tmp[lat_col] = pd.to_numeric(tmp[lat_col], errors="coerce")
    tmp[lon_col] = pd.to_numeric(tmp[lon_col], errors="coerce")
    tmp = tmp.dropna(subset=[lat_col,lon_col]).copy()
    pts = gpd.GeoDataFrame(tmp, geometry=gpd.points_from_xy(tmp[lon_col], tmp[lat_col]), crs="EPSG:4326")
    joined = gpd.sjoin(pts, gdf_blocks[["GEOID","geometry"]], how="left", predicate="within")
    out = pd.DataFrame(joined.drop(columns=["geometry","index_right"], errors="ignore"))
    out["GEOID"] = normalize_geoid(out["GEOID"], tlen)
    return out

def read_911_daily(fr911_candidates, census_candidates) -> pd.DataFrame:
    # ENV ile doğrudan kaynak belirleme
    env_911 = os.getenv("FR_911", "").strip()
    if env_911 and Path(env_911).exists():
        fr911_candidates = [Path(env_911)] + list(fr911_candidates)

    src = first_existing(fr911_candidates)
    if src is None: raise FileNotFoundError("❌ 911 verisi bulunamadı (artifact/klasör).")
    log(f"📥 911 kaynağı yükleniyor: {src}")

    df = pd.read_csv(src, low_memory=False, dtype={"GEOID":"string"})
    ts_col = next((c for c in ["received_time","received_datetime","datetime","timestamp",
                               "call_received_datetime","date"] if c in df.columns), None)
    if ts_col is None: raise ValueError("911 verisinde zaman kolonu yok.")
    if "GEOID" not in df.columns or df["GEOID"].isna().all():
        log("ℹ️ 911: GEOID yok; lat/lon → GEOID hesaplanacak.")
        df = ensure_geoid_from_latlon(df, census_candidates)

    df["date"] = to_date(df[ts_col])
    df = df.dropna(subset=["GEOID","date"]).copy()

    day = (df.groupby(["GEOID","date"], as_index=False).size()
             .rename(columns={"size":"n_911_day"})
             .sort_values(["GEOID","date"])
             .reset_index(drop=True))

    day["n_911_last1d"] = day.groupby("GEOID")["n_911_day"].transform(lambda s: s.shift(1).fillna(0)).astype("float32")
    def roll_sum(s: pd.Series, W:int)->pd.Series: return s.shift(1).rolling(W, min_periods=1).sum()
    day["n_911_last3d"] = day.groupby("GEOID")["n_911_day"].transform(lambda s: roll_sum(s,3)).fillna(0).astype("float32")
    day["n_911_last7d"] = day.groupby("GEOID")["n_911_day"].transform(lambda s: roll_sum(s,7)).fillna(0).astype("float32")
    return day

def main():
    safe_unzip(ARTIFACT_ZIP, ARTIFACT_DIR)
    cands = build_candidates()

    print("== Candidates ==")
    for key, arr in cands.items():
        print(f"[{key}]")
        for p in arr:
            try:
                exists = Path(p).exists()
            except Exception:
                exists = False
            print("  -", p, "OK" if exists else "")

    for k, arr in cands.items():
        print(f"🔎 Candidates[{k}]:")
        for p in arr:
            print("   -", p, "EXISTS" if Path(p).exists() else "")

    # --- 911 kaynağını gerçekten var mı diye kontrol et
    src_911 = first_existing(cands["FR_911"])
    if src_911 is None:
        log("ℹ️ 911 kaynağı bulunamadı → bu adım NAZİKÇE atlanacak (hata yok).")
        # Boş bir çerçeve verelim ki join sonrası fillna(0) çalışsın
        fr911_daily = pd.DataFrame(columns=[
            "GEOID","date","n_911_day","n_911_last1d","n_911_last3d","n_911_last7d"
        ])
    else:
        fr911_daily = read_911_daily(cands["FR_911"], cands["CENSUS"])
        log(f"📊 911 günlük özet: {fr911_daily.shape[0]:,} satır × {fr911_daily.shape[1]} sütun")

    crime_path = first_existing(cands["CRIME"])
    if crime_path is None:
        raise FileNotFoundError("❌ sf_crime.csv bulunamadı (ENV=FR_CRIME_FILE ile override edebilirsin).")
    crime = pd.read_csv(crime_path, low_memory=False, dtype={"GEOID":"string"})
    log(f"📥 sf_crime.csv: {crime_path} — satır: {len(crime):,}")

    # GEOID normalizasyonu
    try:
        gdf_blocks, tlen = _load_blocks(cands["CENSUS"])
    except Exception:
        tlen = 11
    if "GEOID" in crime.columns and crime["GEOID"].notna().any():
        crime["GEOID"] = normalize_geoid(crime["GEOID"], tlen)
    else:
        log("ℹ️ sf_crime: GEOID yok; lat/lon → GEOID hesaplanacak.")
        crime = ensure_geoid_from_latlon(crime, cands["CENSUS"])
        crime["GEOID"] = normalize_geoid(crime["GEOID"], tlen)

    # DATE üret
    if "date" in crime.columns:
        crime["date"] = to_date(crime["date"])
    else:
        dt_col = next((c for c in ["datetime","event_datetime","occurred_at","timestamp"] if c in crime.columns), None)
        if dt_col is None:
            raise ValueError("❌ sf_crime.csv içinde 'date' veya 'datetime' benzeri bir kolon yok.")
        crime["date"] = to_date(crime[dt_col])

    # EVENTS
    keys = ["GEOID","date"]
    events_daily = crime.merge(fr911_daily, on=keys, how="left")
    for c in ["n_911_day","n_911_last1d","n_911_last3d","n_911_last7d"]:
        if c in events_daily.columns:
            events_daily[c] = pd.to_numeric(events_daily[c], errors="coerce").fillna(0)

    # GRID
    agg_crime = (events_daily.groupby(keys, as_index=False).size()
                 .rename(columns={"size":"crime_count_day"}))
    grid_daily = agg_crime.merge(fr911_daily, on=keys, how="left")
    grid_daily["crime_count_day"] = pd.to_numeric(grid_daily["crime_count_day"], errors="coerce").fillna(0).astype(int)
    grid_daily["Y_day"] = (grid_daily["crime_count_day"] > 0).astype("int8")
    for c in ["n_911_day","n_911_last1d","n_911_last3d","n_911_last7d"]:
        if c in grid_daily.columns:
            grid_daily[c] = pd.to_numeric(grid_daily[c], errors="coerce").fillna(0)

    # Yaz
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_save_csv(events_daily, str(OUTPUT_DIR / OUTPUT_EVENTS_DAILY))
    safe_save_csv(grid_daily,   str(OUTPUT_DIR / OUTPUT_GRID_DAILY))

    # Önizleme
    try:
        log("—— fr_crime_events_daily.csv — örnek —")
        cols = ["GEOID","date","n_911_day","n_911_last1d","n_911_last3d","n_911_last7d"]
        log(events_daily[[c for c in cols if c in events_daily.columns]].head(8).to_string(index=False))
    except Exception:
        pass
    try:
        log("—— fr_crime_grid_daily.csv — örnek —")
        cols = ["GEOID","date","crime_count_day","Y_day","n_911_day","n_911_last1d","n_911_last3d","n_911_last7d"]
        log(grid_daily[[c for c in cols if c in grid_daily.columns]].head(8).to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()

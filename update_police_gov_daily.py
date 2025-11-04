#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enrich_with_police_gov_daily.py
- Police & Government building noktalarını kullanarak:
  • EVENTS (olay satırları) için: olay lat/lon → en yakın mesafeler, 300 m bayraklar, range etiketleri
  • GRID (GEOID×date) için: GEOID centroid → en yakın mesafeler, 300 m bayraklar, range etiketleri
- Saatlik yok; tamamen günlük akış.

ENV (varsayılanlar):
  CRIME_DATA_DIR          (crime_prediction_data)
  FR_GRID_DAILY_IN        (fr_crime_grid_daily.csv)
  FR_GRID_DAILY_OUT       (fr_crime_grid_daily.csv)     # üzerine yazar
  FR_EVENTS_DAILY_IN      (fr_crime_events_daily.csv)
  FR_EVENTS_DAILY_OUT     (fr_crime_events_daily.csv)   # üzerine yazar

  POLICE_FILE             (sf_police_stations.csv)
  GOV_FILE                (sf_government_buildings.csv)
  BLOCKS_GEOJSON          (sf_census_blocks_with_population.geojson)

  NEAR_THRESHOLD_M        (300)     # yakın sayılacak mesafe eşiği
  ARTIFACT_ZIP            (artifact/sf-crime-pipeline-output.zip)
  ARTIFACT_DIR            (artifact_unzipped)
"""
from __future__ import annotations
import os, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

pd.options.mode.copy_on_write = True

# ---------------- Utils ----------------
def log(msg: str): print(msg, flush=True)

def _digits_only(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)", expand=False).fillna("")

def _geoid11(s: pd.Series) -> pd.Series:
    return _digits_only(s).str.replace(" ", "", regex=False).str.zfill(11).str[:11]

def _read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        log(f"ℹ️ Dosya yok: {p}"); return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    log(f"📖 Okundu: {p}  ({len(df):,}×{df.shape[1]})")
    return df

def _safe_write_csv(df: pd.DataFrame, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    # hafif downcast
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64","Int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(p)
    log(f"💾 Yazıldı: {p}  ({len(df):,}×{df.shape[1]})")

def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    else:
        for c in ("event_date","dt","day","incident_datetime","datetime","timestamp","created_datetime"):
            if c in out.columns:
                out["date"] = pd.to_datetime(out[c], errors="coerce").dt.date
                break
    return out

def _find_col(cols, cands):
    m = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in m: return m[c.lower()]
    return None

def _prep_points(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["latitude","longitude"])
    lat = _find_col(df.columns, ["latitude","lat","y"])
    lon = _find_col(df.columns, ["longitude","lon","x"])
    if not lat or not lon: return pd.DataFrame(columns=["latitude","longitude"])
    out = df.rename(columns={lat:"latitude", lon:"longitude"})[["latitude","longitude"]].copy()
    out["latitude"]  = pd.to_numeric(out["latitude"],  errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out.dropna(subset=["latitude","longitude"])
    return out

def _quantile_ranges(series: pd.Series, max_bins=5, fallback="Unknown") -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf,-np.inf], np.nan)
    mask = s.notna(); s_valid = s[mask]
    if s_valid.nunique() <= 1 or len(s_valid) < 2:
        return pd.Series([fallback]*len(series), index=series.index, dtype="object")
    q = min(max_bins, max(3, s_valid.nunique()))
    try:
        _, edges = pd.qcut(s_valid, q=q, retbins=True, duplicates="drop")
    except Exception:
        return pd.Series([fallback]*len(series), index=series.index, dtype="object")
    if len(edges) < 3:
        return pd.Series([fallback]*len(series), index=series.index, dtype="object")
    labs = []
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        labs.append(f"Q{i+1} (≤{hi:.1f})" if i==0 else f"Q{i+1} ({lo:.1f}-{hi:.1f})")
    out = pd.Series(fallback, index=series.index, dtype="object")
    out.loc[mask] = pd.cut(s_valid, bins=edges, labels=labs, include_lowest=True).astype(str)
    return out

def _centroids_from_blocks(blocks_csv: pd.DataFrame) -> pd.DataFrame:
    # beklenen kolonlar: GEOID, centroid_lat, centroid_lon (önceden hazırlanmış)
    df = blocks_csv.copy()
    need = [_find_col(df.columns, ["GEOID","geoid","geoid10"]), 
            _find_col(df.columns, ["centroid_lat","cent_lat","y"]),
            _find_col(df.columns, ["centroid_lon","cent_lon","x"])]
    if any(v is None for v in need):
        raise ValueError("Blocks centroid tablosunda GEOID/centroid_lat/centroid_lon yok.")
    gcol, clat, clon = need
    out = df[[gcol, clat, clon]].rename(columns={gcol:"GEOID", clat:"centroid_lat", clon:"centroid_lon"}).copy()
    out["GEOID"] = _geoid11(out["GEOID"])
    out["centroid_lat"]  = pd.to_numeric(out["centroid_lat"], errors="coerce")
    out["centroid_lon"]  = pd.to_numeric(out["centroid_lon"], errors="coerce")
    out = out.dropna(subset=["centroid_lat","centroid_lon"])
    return out

def _haversine_query(src_latlon: np.ndarray, dst_latlon: np.ndarray) -> np.ndarray:
    """src: N×2 [lat,lon], dst: M×2 [lat,lon] → en yakın mesafe (metre), N×1."""
    if len(dst_latlon) == 0:
        return np.full((len(src_latlon),), np.nan, dtype="float64")
    EARTH_R = 6_371_000.0
    src_rad = np.radians(src_latlon.astype(float))
    dst_rad = np.radians(dst_latlon.astype(float))
    tree = BallTree(dst_rad, metric="haversine")
    dist_rad, _ = tree.query(src_rad, k=1)
    return (dist_rad[:,0] * EARTH_R)

def safe_unzip(zip_path: Path, dest_dir: Path):
    if not zip_path.exists(): return
    dest_dir.mkdir(parents=True, exist_ok=True)
    log(f"📦 ZIP açılıyor: {zip_path} → {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for m in zf.infolist():
            out = dest_dir / m.filename
            out.parent.mkdir(parents=True, exist_ok=True)
            if m.is_dir():
                out.mkdir(parents=True, exist_ok=True); continue
            with zf.open(m, "r") as src, open(out, "wb") as dst:
                dst.write(src.read())
    log("✅ ZIP çıkarma tamam.")

# ---------------- Config ----------------
BASE_DIR     = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data"))
ARTIFACT_ZIP = Path(os.getenv("ARTIFACT_ZIP", "artifact/sf-crime-pipeline-output.zip"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifact_unzipped"))

GRID_IN   = Path(os.getenv("FR_GRID_DAILY_IN",  "fr_crime_grid_daily.csv"))
GRID_OUT  = Path(os.getenv("FR_GRID_DAILY_OUT", "fr_crime_grid_daily.csv"))
EV_IN     = Path(os.getenv("FR_EVENTS_DAILY_IN","fr_crime_events_daily.csv"))
EV_OUT    = Path(os.getenv("FR_EVENTS_DAILY_OUT","fr_crime_events_daily.csv"))

POLICE_FILE = os.getenv("POLICE_FILE", "sf_police_stations.csv")
GOV_FILE    = os.getenv("GOV_FILE",    "sf_government_buildings.csv")
BLOCKS_CSV  = os.getenv("BLOCKS_CSV",  "blocks_centroids.csv")  # GEOID, centroid_lat, centroid_lon (önerilen)
NEAR_M      = float(os.getenv("NEAR_THRESHOLD_M", "300"))

POLICE_CANDS = [ARTIFACT_DIR / POLICE_FILE, BASE_DIR / POLICE_FILE, Path(POLICE_FILE)]
GOV_CANDS    = [ARTIFACT_DIR / GOV_FILE,    BASE_DIR / GOV_FILE,    Path(GOV_FILE)]
BLOCKS_CANDS = [ARTIFACT_DIR / BLOCKS_CSV,  BASE_DIR / BLOCKS_CSV,  Path(BLOCKS_CSV)]

# ---------------- Run ----------------
def main() -> int:
    # 0) artifact varsa aç
    safe_unzip(ARTIFACT_ZIP, ARTIFACT_DIR)

    # 1) kaynakları yükle
    grid  = _read_csv(BASE_DIR / GRID_IN  if not GRID_IN.is_absolute()  else GRID_IN)
    ev    = _read_csv(BASE_DIR / EV_IN    if not EV_IN.is_absolute()    else EV_IN)
    pol_p = next((p for p in POLICE_CANDS if p.exists()), None)
    gov_p = next((p for p in GOV_CANDS    if p.exists()), None)
    blk_p = next((p for p in BLOCKS_CANDS if p.exists()), None)

    police = _prep_points(_read_csv(pol_p) if pol_p else pd.DataFrame())
    gov    = _prep_points(_read_csv(gov_p) if gov_p else pd.DataFrame())

    if pol_p is None: log("⚠️ Polis istasyonları bulunamadı; distance_to_police NaN kalabilir.")
    if gov_p is None: log("⚠️ Government binaları bulunamadı; distance_to_government_building NaN kalabilir.")
    if blk_p is None: log("⚠️ Blocks centroid CSV bulunamadı; centroid fallback kullanılamayabilir.")

    # 2) GRID: GEOID centroid → mesafeler
    if not grid.empty:
        grid = _ensure_date(grid)
        # GEOID
        if "GEOID" not in grid.columns:
            cand = [c for c in grid.columns if "geoid" in c.lower()]
            if not cand: raise RuntimeError("GRID’de GEOID yok.")
            grid.insert(0, "GEOID", _geoid11(grid[cand[0]]).astype("string"))
            grid.drop(columns=[c for c in cand if c != "GEOID"], inplace=True, errors="ignore")
        else:
            grid["GEOID"] = _geoid11(grid["GEOID"]).astype("string")

        # centroidleri al
        if blk_p is None:
            raise RuntimeError("GRID enrich için BLOCKS_CSV (GEOID, centroid_lat, centroid_lon) gerekli.")
        cent = _centroids_from_blocks(_read_csv(blk_p))
        grid = grid.merge(cent, on="GEOID", how="left", validate="many_to_one")

        src = grid[["centroid_lat","centroid_lon"]].to_numpy(dtype=float)
        pol_dist = _haversine_query(src, police[["latitude","longitude"]].to_numpy(dtype=float)) if not police.empty else np.full((len(src),), np.nan)
        gov_dist = _haversine_query(src, gov[["latitude","longitude"]].to_numpy(dtype=float))     if not gov.empty else np.full((len(src),), np.nan)

        # eski sütunlar varsa temizle
        drop_cols = ["distance_to_police","distance_to_government_building","is_near_police","is_near_government",
                     "distance_to_police_range","distance_to_government_building_range"]
        grid.drop(columns=[c for c in drop_cols if c in grid.columns], inplace=True, errors="ignore")

        grid["distance_to_police"] = np.round(pol_dist, 1)
        grid["distance_to_government_building"] = np.round(gov_dist, 1)
        grid["is_near_police"]      = (grid["distance_to_police"] <= NEAR_M).astype("Int64")
        grid["is_near_government"]  = (grid["distance_to_government_building"] <= NEAR_M).astype("Int64")
        grid["distance_to_police_range"] = _quantile_ranges(grid["distance_to_police"])
        grid["distance_to_government_building_range"] = _quantile_ranges(grid["distance_to_government_building"])

        # geçici kolon temizlik
        grid.drop(columns=["centroid_lat","centroid_lon"], inplace=True, errors="ignore")

        _safe_write_csv(grid, BASE_DIR / GRID_OUT if not GRID_OUT.is_absolute() else GRID_OUT)
    else:
        log("ℹ️ GRID bulunamadı → atlandı.")

    # 3) EVENTS: olay lat/lon → mesafeler; yoksa centroid fallback (varsa)
    if not ev.empty:
        # GEOID
        if "GEOID" in ev.columns:
            ev["GEOID"] = _geoid11(ev["GEOID"]).astype("string")
        else:
            cand = [c for c in ev.columns if "geoid" in c.lower()]
            if cand:
                ev.insert(0, "GEOID", _geoid11(ev[cand[0]]).astype("string"))

        lat_col = _find_col(ev.columns, ["latitude","lat","y"])
        lon_col = _find_col(ev.columns, ["longitude","lon","x"])
        ev["_lat_evt_"] = pd.to_numeric(ev[lat_col], errors="coerce") if lat_col else np.nan
        ev["_lon_evt_"] = pd.to_numeric(ev[lon_col], errors="coerce") if lon_col else np.nan

        # centroid fallback (varsa)
        if blk_p is not None and ("_lat_evt_" not in ev.columns or ev["_lat_evt_"].isna().any() or
                                  "_lon_evt_" not in ev.columns or ev["_lon_evt_"].isna().any()):
            cent = _centroids_from_blocks(_read_csv(blk_p))
            ev = ev.merge(cent, on="GEOID", how="left", validate="many_to_one")
            ev["_lat_evt_"] = ev["_lat_evt_"].fillna(ev["centroid_lat"])
            ev["_lon_evt_"] = ev["_lon_evt_"].fillna(ev["centroid_lon"])
            ev.drop(columns=["centroid_lat","centroid_lon"], inplace=True, errors="ignore")

        mask = ev["_lat_evt_"].notna() & ev["_lon_evt_"].notna()
        src = ev.loc[mask, ["_lat_evt_","_lon_evt_"]].to_numpy(dtype=float)

        pol_dist = _haversine_query(src, police[["latitude","longitude"]].to_numpy(dtype=float)) if not police.empty else np.full((len(src),), np.nan)
        gov_dist = _haversine_query(src, gov[["latitude","longitude"]].to_numpy(dtype=float))     if not gov.empty else np.full((len(src),), np.nan)

        # eski sütunları temizle
        drop_cols = ["distance_to_police","distance_to_government_building","is_near_police","is_near_government",
                     "distance_to_police_range","distance_to_government_building_range"]
        ev.drop(columns=[c for c in drop_cols if c in ev.columns], inplace=True, errors="ignore")

        ev.loc[mask, "distance_to_police"] = np.round(pol_dist, 1)
        ev.loc[mask, "distance_to_government_building"] = np.round(gov_dist, 1)
        ev["is_near_police"]     = (pd.to_numeric(ev.get("distance_to_police"), errors="coerce") <= NEAR_M).astype("Int64")
        ev["is_near_government"] = (pd.to_numeric(ev.get("distance_to_government_building"), errors="coerce") <= NEAR_M).astype("Int64")
        ev["distance_to_police_range"] = _quantile_ranges(ev["distance_to_police"])
        ev["distance_to_government_building_range"] = _quantile_ranges(ev["distance_to_government_building"])

        ev.drop(columns=["_lat_evt_","_lon_evt_"], inplace=True, errors="ignore")

        _safe_write_csv(ev, BASE_DIR / EV_OUT if not EV_OUT.is_absolute() else EV_OUT)
    else:
        log("ℹ️ EVENTS bulunamadı → atlandı.")

    # 4) Kısa önizleme
    try:
        cols = ["GEOID","distance_to_police","distance_to_government_building",
                "is_near_police","is_near_government",
                "distance_to_police_range","distance_to_government_building_range"]
        if not grid.empty:
            log("— GRID preview —")
            log(grid[[c for c in cols if c in grid.columns]].head(10).to_string(index=False))
        if not ev.empty:
            log("— EVENTS preview —")
            log(ev[[c for c in cols if c in ev.columns]].head(10).to_string(index=False))
    except Exception:
        pass

    log("✅ enrich_with_police_gov_daily.py tamam.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

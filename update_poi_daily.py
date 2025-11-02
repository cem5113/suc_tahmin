#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# update_poi_daily.py — GEOID-ONLY POI ENRICH (no date dependency)
# IN : daily_crime_05.csv
# OUT: daily_crime_06.csv
from __future__ import annotations
import os, ast, json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree

pd.options.mode.copy_on_write = True

# ── helpers ──────────────────────────────────────────────────────────────────
def log(msg: str): print(msg, flush=True)
def ensure_parent(path: str): Path(path).parent.mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, path: str):
    ensure_parent(path)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    log(f"💾 Kaydedildi: {path} (satır={len(df):,}, sütun={df.shape[1]})")

def normalize_geoid(s: pd.Series, L: int = 11) -> pd.Series:
    x = s.astype(str).str.extract(r"(\d+)", expand=False)
    return x.str[:L].str.zfill(L)

def read_geojson_robust(path: str) -> gpd.GeoDataFrame:
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"GeoJSON yok: {path}")
    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        # satır-satır JSON veya FeatureCollection fallback
        txt = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
        gj = None
        try:
            if "\n" in txt and txt.splitlines()[0].strip().startswith("{") and '"features"' not in txt:
                feats = [json.loads(line) for line in txt.splitlines() if line.strip()]
                gj = {"type":"FeatureCollection","features":feats}
            else:
                gj = json.loads(txt)
        except Exception as e2:
            raise ValueError(f"GeoJSON parse edilemedi: {e2}") from e
        if "features" not in gj:
            raise ValueError("FeatureCollection bekleniyordu.")
        gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
    # CRS normalize
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        try:
            if gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(4326)
        except Exception:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf

def parse_tags(val):
    if isinstance(val, dict): return val
    if isinstance(val, str):
        for loader in (json.loads, ast.literal_eval):
            try:
                x = loader(val);  return x if isinstance(x, dict) else {}
            except Exception:
                pass
    return {}

def extract_cat_sub_name(tags: dict):
    name = tags.get("name")
    for key in ("amenity","shop","leisure"):
        if key in tags and tags[key]:
            return key, tags[key], name
    return None, None, name

def make_labels(series: pd.Series, q=5):
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if vals.size == 0:
        def lab(_): return "Q1 (0-0)"
        return lab
    qs = np.quantile(vals, [i/q for i in range(q+1)])
    def lab(x):
        if pd.isna(x): return f"Q1 ({qs[0]:.1f}-{qs[1]:.1f})"
        for i in range(q):
            if x <= qs[i+1]:
                return f"Q{i+1} ({qs[i]:.1f}-{qs[i+1]:.1f})"
        return f"Q{q} ({qs[-2]:.1f}-{qs[-1]:.1f})"
    return lab

# ── config (no network) ─────────────────────────────────────────────────────
BASE_DIR   = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data")); BASE_DIR.mkdir(parents=True, exist_ok=True)
DAILY_IN   = Path(os.getenv("DAILY_IN",  str(BASE_DIR / "daily_crime_05.csv")))
DAILY_OUT  = Path(os.getenv("DAILY_OUT", str(BASE_DIR / "daily_crime_06.csv")))

# POI kaynakları (önce temiz CSV; yoksa yereldeki GeoJSON’dan üret)
POI_CLEAN_CSV = Path(os.getenv("POI_CLEAN_CSV", str(BASE_DIR / "sf_pois_cleaned_with_geoid.csv")))
POI_RISK_JSON = Path(os.getenv("POI_RISK_JSON", str(BASE_DIR / "risky_pois_dynamic.json")))
POI_GEOJSON   = Path(os.getenv("POI_GEOJSON",   str(BASE_DIR / "sf_pois.geojson")))

# Blok sınırları (GEOID üretimi için gerekli olabilir)
BLOCKS_GEOJSON = Path(os.getenv("SF_BLOCKS_GEOJSON", str(BASE_DIR / "sf_census_blocks_with_population.geojson")))

# ── 1) crime oku ────────────────────────────────────────────────────────────
if not DAILY_IN.exists():
    raise FileNotFoundError(f"❌ Girdi yok: {DAILY_IN}")
crime = pd.read_csv(DAILY_IN, low_memory=False)
if "GEOID" not in crime.columns:
    raise KeyError("❌ daily_crime_05.csv içinde 'GEOID' yok.")
crime["GEOID"] = normalize_geoid(crime["GEOID"], 11)
log(f"📥 daily_crime_05.csv — satır={len(crime):,}, uniq GEOID={crime['GEOID'].nunique():,}")

# ── 2) POI clean hazır mı? ─────────────────────────────────────────────────
def build_poi_clean(blocks_path: Path, poi_path: Path) -> pd.DataFrame:
    if not poi_path.exists():
        raise FileNotFoundError("❌ POI temiz CSV yok ve sf_pois.geojson da bulunamadı.")
    if not blocks_path.exists():
        raise FileNotFoundError("❌ Blok GeoJSON yok; POI→GEOID eşleme yapılamaz.")
    gdf = read_geojson_robust(str(poi_path))
    if "geometry" not in gdf.columns:
        if {"lon","lat"}.issubset(gdf.columns):
            gdf["geometry"] = gpd.points_from_xy(gdf["lon"], gdf["lat"])
        else:
            raise ValueError("POI verisi lon/lat veya geometry içermiyor.")

    if "tags" not in gdf.columns:
        gdf["tags"] = [{}]*len(gdf)
    gdf["tags"] = gdf["tags"].apply(parse_tags)
    triples = gdf["tags"].apply(extract_cat_sub_name)
    gdf[["poi_category","poi_subcategory","poi_name"]] = pd.DataFrame(triples.tolist(), index=gdf.index)

    blocks = read_geojson_robust(str(blocks_path))
    gcol = "GEOID" if "GEOID" in blocks.columns else next((c for c in blocks.columns if str(c).upper().startswith("GEOID")), None)
    if not gcol: raise KeyError("Block GeoJSON içinde GEOID yok.")
    blocks["GEOID"] = normalize_geoid(blocks[gcol], 11)

    try:
        joined = gpd.sjoin(gdf.to_crs(4326), blocks[["GEOID","geometry"]], how="left", predicate="within")
    except Exception:
        from shapely.strtree import STRtree
        tree = STRtree(list(blocks.geometry.values))
        idx_to_geoid = {id(g): ge for g, ge in zip(blocks.geometry.values, blocks["GEOID"])}
        gl = []
        for pt in gdf.geometry.values:
            hits = tree.query(pt, predicate="contains")
            gl.append(idx_to_geoid[id(hits[0])] if hits else None)
        joined = gdf.copy()
        joined["GEOID"] = gl

    df = pd.DataFrame(joined.drop(columns=["geometry"], errors="ignore"))
    if "id" not in df.columns:
        df["id"] = np.arange(len(df))
    df["GEOID"] = normalize_geoid(df["GEOID"], 11)
    keep = [c for c in ["id","lat","lon","latitude","longitude","poi_category","poi_subcategory","poi_name","GEOID"] if c in df.columns]
    df = df[keep].copy()
    safe_save_csv(df, str(POI_CLEAN_CSV))
    return df

if POI_CLEAN_CSV.exists():
    poi = pd.read_csv(POI_CLEAN_CSV, low_memory=False)
    # normalize columns
    if "lat" not in poi.columns and "latitude" in poi.columns:
        poi["lat"] = pd.to_numeric(poi["latitude"], errors="coerce")
    if "lon" not in poi.columns and "longitude" in poi.columns:
        poi["lon"] = pd.to_numeric(poi["longitude"], errors="coerce")
    poi["GEOID"] = normalize_geoid(poi.get("GEOID"), 11)
    log(f"ℹ️ Var olan POI clean kullanılacak: {POI_CLEAN_CSV} (satır={len(poi):,})")
else:
    poi = build_poi_clean(BLOCKS_GEOJSON, POI_GEOJSON)
    log(f"🧹 POI clean üretildi: {len(poi):,} satır")

# ── 3) Dinamik risk sözlüğü (opsiyonel, koordinat varsa) ───────────────────
def compute_dynamic_poi_risk(df_crime: pd.DataFrame, df_poi: pd.DataFrame, radius_m=300) -> dict:
    dfp = df_poi.copy()
    dfp["lat"] = pd.to_numeric(dfp.get("lat"), errors="coerce")
    dfp["lon"] = pd.to_numeric(dfp.get("lon"), errors="coerce")
    dfp = dfp.dropna(subset=["lat","lon"])
    if "poi_subcategory" in dfp.columns:
        dfp = dfp[~dfp["poi_subcategory"].isin(["police","ranger_station"])]

    dfc = df_crime.copy()
    dfc["latitude"]  = pd.to_numeric(dfc.get("latitude"), errors="coerce")
    dfc["longitude"] = pd.to_numeric(dfc.get("longitude"), errors="coerce")
    dfc = dfc.dropna(subset=["latitude","longitude"])

    if dfc.empty or dfp.empty:
        return {}

    crime_rad = np.radians(dfc[["latitude","longitude"]].values)
    poi_rad   = np.radians(dfp[["lat","lon"]].values)
    tree = BallTree(crime_rad, metric="haversine")
    r = radius_m / 6371000.0

    agg = defaultdict(list)
    for pt, t in zip(poi_rad, dfp["poi_subcategory"].fillna("")):
        if not t: continue
        idx = tree.query_radius([pt], r=r)[0]
        agg[t].append(len(idx))

    if not agg: return {}
    avg = {t: float(np.mean(v)) for t, v in agg.items()}
    v = list(avg.values()); vmin, vmax = min(v), max(v)
    if vmax - vmin < 1e-9:
        return {t: 1.5 for t in avg}
    return {t: round(3 * (x - vmin) / (vmax - vmin), 2) for t, x in avg.items()}

# risk sözlüğünü oku/üret
if POI_RISK_JSON.exists():
    try:
        poi_risk = json.loads(POI_RISK_JSON.read_text())
    except Exception:
        poi_risk = {}

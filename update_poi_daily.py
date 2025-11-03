#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# update_poi_daily.py — GEOID-ONLY POI ENRICH (no date dependency)
# IN : daily_crime_05.csv
# OUT: daily_crime_06.csv

from __future__ import annotations
import os, ast, json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import geopandas as gpd

# BallTree isteğe bağlı (dinamik risk için). Yoksa güvenli fallback.
try:
    from sklearn.neighbors import BallTree
    _BALLTREE_OK = True
except Exception:
    BallTree = None
    _BALLTREE_OK = False

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
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"GeoJSON yok: {p}")
    try:
        gdf = gpd.read_file(p)
    except Exception as e:
        txt = p.read_text(encoding="utf-8", errors="ignore").strip()
        gj = None
        try:
            if "\n" in txt and txt.splitlines()[0].strip().startswith("{") and '"features"' not in txt:
                feats = [json.loads(line) for line in txt.splitlines() if line.strip()]
                gj = {"type":"FeatureCollection","features":feats}
            else:
                gj = json.loads(txt)
        except Exception as e2:
            raise ValueError(f"GeoJSON parse edilemedi: {e2}") from e
        if not isinstance(gj, dict) or "features" not in gj:
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

def extract_cat_sub_name(tags: dict) -> Tuple[str|None, str|None, str|None]:
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

# Blok sınırları (GEOID üretimi için)
BLOCKS_GEOJSON = Path(os.getenv("SF_BLOCKS_GEOJSON", str(BASE_DIR / "sf_census_blocks_with_population.geojson")))

# "en çok görülen" alt-kategori kolon sayısı (şişmeyi önleme)
TOP_SUBCATS = int(os.getenv("POI_TOP_SUBCATS", "20"))

# ── 1) crime oku ────────────────────────────────────────────────────────────
if not (DAILY_IN.exists() and DAILY_IN.is_file()):
    raise FileNotFoundError(f"❌ Girdi yok: {DAILY_IN}")
crime = pd.read_csv(DAILY_IN, low_memory=False)
if "GEOID" not in crime.columns:
    raise KeyError("❌ daily_crime_05.csv içinde 'GEOID' yok.")
crime["GEOID"] = normalize_geoid(crime["GEOID"], 11)
crime_geoids = pd.Series(crime["GEOID"].unique(), name="GEOID")
log(f"📥 daily_crime_05.csv — satır={len(crime):,}, uniq GEOID={crime_geoids.nunique():,}")

# ── 2) POI clean hazır mı? değilse üret ─────────────────────────────────────
def build_poi_clean(blocks_path: Path, poi_path: Path) -> pd.DataFrame:
    if not (poi_path.exists() and poi_path.is_file()):
        raise FileNotFoundError("❌ POI temiz CSV yok ve sf_pois.geojson da bulunamadı.")
    if not (blocks_path.exists() and blocks_path.is_file()):
        raise FileNotFoundError("❌ Blok GeoJSON yok; POI→GEOID eşleme yapılamaz.")

    gdf = read_geojson_robust(str(poi_path))
    if "geometry" not in gdf.columns:
        if {"lon","lat"}.issubset(gdf.columns):
            gdf["geometry"] = gpd.points_from_xy(gdf["lon"], gdf["lat"])
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
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
        # Shapely STRtree fallback
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
    # koordinat kolonlarını normalize et
    if "lat" not in df.columns and "latitude" in df.columns:
        df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    if "lon" not in df.columns and "longitude" in df.columns:
        df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["GEOID"] = normalize_geoid(df["GEOID"], 11)

    keep = [c for c in ["id","lat","lon","latitude","longitude","poi_category","poi_subcategory","poi_name","GEOID"] if c in df.columns]
    df = df[keep].copy()
    safe_save_csv(df, str(POI_CLEAN_CSV))
    return df

if POI_CLEAN_CSV.exists() and POI_CLEAN_CSV.is_file():
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

# ── 3) Dinamik risk sözlüğü (opsiyonel) ─────────────────────────────────────
def compute_dynamic_poi_risk(df_crime: pd.DataFrame, df_poi: pd.DataFrame, radius_m=300) -> Dict[str, float]:
    """POI alt-kategorisi etrafındaki (radius_m) suç yoğunluğuna göre 0–3 ölçeğinde skorlar."""
    # POI koordinatları
    dfp = df_poi.copy()
    dfp["lat"] = pd.to_numeric(dfp.get("lat"), errors="coerce")
    dfp["lon"] = pd.to_numeric(dfp.get("lon"), errors="coerce")
    dfp = dfp.dropna(subset=["lat","lon"])
    if dfp.empty: return {}

    # Suç koordinatları (varsa)
    dfc = df_crime.copy()
    dfc["latitude"]  = pd.to_numeric(dfc.get("latitude"), errors="coerce")
    dfc["longitude"] = pd.to_numeric(dfc.get("longitude"), errors="coerce")
    dfc = dfc.dropna(subset=["latitude","longitude"])
    if dfc.empty or not _BALLTREE_OK:
        # Koordinat yoksa ya da BallTree yoksa fallback: alt-kategori frekansına dayalı normalize
        cnt = dfp["poi_subcategory"].value_counts()
        if cnt.empty: return {}
        v = cnt.astype(float)
        vmin, vmax = float(v.min()), float(v.max())
        if vmax - vmin < 1e-9:
            return {k: 1.5 for k in cnt.index}
        return {k: round(3 * (x - vmin) / (vmax - vmin), 2) for k, x in v.items()}

    # BallTree ile haversine arama
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
    vals = list(avg.values()); vmin, vmax = min(vals), max(vals)
    if vmax - vmin < 1e-9:
        return {t: 1.5 for t in avg}
    return {t: round(3 * (x - vmin) / (vmax - vmin), 2) for t, x in avg.items()}

# risk sözlüğü oku/üret
if POI_RISK_JSON.exists() and POI_RISK_JSON.is_file():
    try:
        poi_risk: Dict[str,float] = json.loads(POI_RISK_JSON.read_text())
        if not isinstance(poi_risk, dict):
            poi_risk = {}
    except Exception:
        poi_risk = {}
else:
    poi_risk = {}

if not poi_risk:
    log("ℹ️ POI risk sözlüğü yok → dinamik hesaplanıyor (radius=300m).")
    try:
        poi_risk = compute_dynamic_poi_risk(crime, poi, radius_m=int(os.getenv("POI_RISK_RADIUS_M", "300")))
    except Exception as e:
        log(f"⚠️ Dinamik risk hesaplanamadı: {e} — frekans tabanlı fallback kullanılacak.")
        cnt = poi["poi_subcategory"].value_counts()
        if not cnt.empty:
            v = cnt.astype(float); vmin, vmax = float(v.min()), float(v.max())
            poi_risk = {k: (1.5 if vmax - vmin < 1e-9 else round(3 * (x - vmin) / (vmax - vmin), 2))
                        for k, x in v.items()}
        else:
            poi_risk = {}

# ── 4) GEOID düzeyinde POI özetleri ────────────────────────────────────────
poi["poi_subcategory"] = poi["poi_subcategory"].fillna("unknown")
poi["GEOID"] = normalize_geoid(poi.get("GEOID"), 11)

# en sık TOP_SUBCATS alt-kategori + "other"
sub_counts = poi["poi_subcategory"].value_counts()
top_subs = list(sub_counts.head(max(1, TOP_SUBCATS)).index)

poi["__sub_keep__"] = poi["poi_subcategory"].where(poi["poi_subcategory"].isin(top_subs), other="__other__")

# toplam ve ağırlıklı risk
poi["__risk_w__"] = poi["poi_subcategory"].map(poi_risk).fillna(1.0)

# GEOID × alt_kategori sayıları (yalnızca top + other)
wide = (poi
        .groupby(["GEOID","__sub_keep__"], as_index=False)
        .size()
        .pivot_table(index="GEOID", columns="__sub_keep__", values="size", fill_value=0)
        .rename_axis(None, axis=1))

# kolon adları temizle
wide = wide.rename(columns={"__other__": "poi_other"})

# toplam adet
wide["poi_total_count"] = wide.sum(axis=1)

# risk skoru (tüm alt-kategorilerle, yalnızca top değil)
risk_geo = (poi.groupby(["GEOID","poi_subcategory"], as_index=False)
               .size()
               .rename(columns={"size":"cnt"}))
risk_geo["w"] = risk_geo["poi_subcategory"].map(poi_risk).fillna(1.0)
risk_geo["w_cnt"] = risk_geo["cnt"] * risk_geo["w"]
poi_risk_geo = (risk_geo.groupby("GEOID", as_index=False)["w_cnt"].sum()
                        .rename(columns={"w_cnt":"poi_risk_score"}))

# birleştir
poi_feat = wide.reset_index().merge(poi_risk_geo, on="GEOID", how="left")
poi_feat["poi_risk_score"] = pd.to_numeric(poi_feat["poi_risk_score"], errors="coerce").fillna(0.0)

# kantil etiketleri
lab_cnt  = make_labels(poi_feat["poi_total_count"], q=5)
lab_risk = make_labels(poi_feat["poi_risk_score"], q=5)
poi_feat["poi_total_count_range"] = poi_feat["poi_total_count"].apply(lab_cnt)
poi_feat["poi_risk_score_range"]  = poi_feat["poi_risk_score"].apply(lab_risk)

# GEOID temizle/tekilleştir
poi_feat["GEOID"] = normalize_geoid(poi_feat["GEOID"], 11)
poi_feat = poi_feat.sort_values("GEOID").drop_duplicates("GEOID", keep="first")

# ── 5) CRIME ile merge ─────────────────────────────────────────────────────
before = crime.shape
out = crime.merge(poi_feat, on="GEOID", how="left", validate="many_to_one")

# boşları doldur
num_cols = ["poi_total_count","poi_risk_score"]
num_cols += [c for c in poi_feat.columns if c.startswith("poi_") and c.endswith(tuple(top_subs + ["other"])) and c not in num_cols]
for c in num_cols:
    if c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

log(f"🔗 CRIME ⨯ POI: {before} → {out.shape} (BallTree={'ok' if _BALLTREE_OK else 'yok'})")

# ── 6) yaz ─────────────────────────────────────────────────────────────────
safe_save_csv(out, str(DAILY_OUT))
try:
    log(out.head(8).to_string(index=False))
except Exception:
    pass

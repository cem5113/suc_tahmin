#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enrich_with_poi_daily.py  (REVIZE - tür-bilinçli fill + doğru centroid CRS)

Ne yapar?
- POI (OSM) → GEOID eşle (blocks geojson ile), türleri çıkar, dinamik risk sözlüğü üret (opsiyonel),
  GEOID seviyesinde özet + 300/600/900m buffer sayıları / risk toplamları üretir.
- Sonra bunları hem GRID (GEOID×date) hem de EVENTS (olay satırları) dosyalarına merge eder.
- Günlük akış: saatlik YOK. POI özellikleri GEOID seviyesinde sabit → tüm günlere/olaylara aynen taşınır.

ENV (varsayılanlar):
  CRIME_DATA_DIR         (crime_prediction_data)

  FR_GRID_DAILY_IN       (fr_crime_grid_daily.csv)
  FR_GRID_DAILY_OUT      (fr_crime_grid_daily.csv)

  FR_EVENTS_DAILY_IN     (fr_crime_events_daily.csv)
  FR_EVENTS_DAILY_OUT    (fr_crime_events_daily.csv)

  # POI & Blocks kaynakları (en az biri bulunmalı; ilk bulunan kullanılır)
  POI_GEOJSON            (sf_pois.geojson)
  BLOCKS_GEOJSON         (sf_census_blocks_with_population.geojson)
  POI_CLEAN_CSV          (sf_pois_cleaned_with_geoid.csv)   # varsa direkt kullanılır/yeniden yazılabilir
  POI_RISK_JSON          (risky_pois_dynamic.json)          # dinamik risk sözlüğü (opsiyonel üretim)

  # Artifacts (varsa)
  ARTIFACT_ZIP           (artifact/sf-crime-pipeline-output.zip)
  ARTIFACT_DIR           (artifact_unzipped)
"""

from __future__ import annotations
import os, json, ast, zipfile
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

pd.options.mode.copy_on_write = True

# ============================== Utils ==============================
def log(msg: str): print(msg, flush=True)

def _digits_only(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)", expand=False).fillna("")

def _geoid11(s: pd.Series) -> pd.Series:
    return _digits_only(s).str.replace(" ", "", regex=False).str.zfill(11).str[:11]

def _read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        log(f"ℹ️ Dosya yok: {p}")
        return pd.DataFrame()
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

# ============================== Config ==============================
BASE_DIR     = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data"))
ARTIFACT_ZIP = Path(os.getenv("ARTIFACT_ZIP", "artifact/sf-crime-pipeline-output.zip"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifact_unzipped"))

GRID_IN   = Path(os.getenv("FR_GRID_DAILY_IN",  "fr_crime_grid_daily.csv"))
GRID_OUT  = Path(os.getenv("FR_GRID_DAILY_OUT", "fr_crime_grid_daily.csv"))
EV_IN     = Path(os.getenv("FR_EVENTS_DAILY_IN","fr_crime_events_daily.csv"))
EV_OUT    = Path(os.getenv("FR_EVENTS_DAILY_OUT","fr_crime_events_daily.csv"))

POI_GEOJSON  = os.getenv("POI_GEOJSON",  "sf_pois.geojson")
BLOCKS_GEOJSON = os.getenv("BLOCKS_GEOJSON","sf_census_blocks_with_population.geojson")
POI_CLEAN_CSV  = os.getenv("POI_CLEAN_CSV","sf_pois_cleaned_with_geoid.csv")
POI_RISK_JSON  = os.getenv("POI_RISK_JSON","risky_pois_dynamic.json")

POI_PATHS       = [ARTIFACT_DIR / POI_GEOJSON, BASE_DIR / POI_GEOJSON, Path(POI_GEOJSON)]
BLOCKS_PATHS    = [ARTIFACT_DIR / BLOCKS_GEOJSON, BASE_DIR / BLOCKS_GEOJSON, Path(BLOCKS_GEOJSON)]
POI_CLEAN_PATHS = [BASE_DIR / POI_CLEAN_CSV, ARTIFACT_DIR / POI_CLEAN_CSV, Path(POI_CLEAN_CSV)]

# ============================== Geo helpers ==============================
def _ensure_crs(gdf, target="EPSG:4326"):
    if gdf.crs is None:
        return gdf.set_crs(target, allow_override=True)
    s = (gdf.crs.to_string() if hasattr(gdf.crs, "to_string") else str(gdf.crs)).upper()
    if s.endswith("CRS84"):  # CRS84 == 4326
        return gdf.set_crs("EPSG:4326", allow_override=True)
    if s != target:
        return gdf.to_crs(target)
    return gdf

def _parse_tags(val):
    if isinstance(val, dict): return val
    if isinstance(val, str):
        for loader in (json.loads, ast.literal_eval):
            try:
                x = loader(val);  return x if isinstance(x, dict) else {}
            except Exception:
                pass
    return {}

def _extract_cat_sub_name(tags: dict):
    name = tags.get("name")
    for key in ("amenity", "shop", "leisure"):
        if key in tags and tags[key]:
            return key, tags[key], name
    return None, None, name

def _read_geojson(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    return _ensure_crs(gdf, "EPSG:4326")

# ============================== POI clean + GEOID ==============================
def load_or_build_poi_clean(blocks: gpd.GeoDataFrame, poi_path: Path, poi_clean_candidates: list[Path]) -> pd.DataFrame:
    # 1) varsa temiz CSV’yi kullan
    for p in poi_clean_candidates:
        if p.exists():
            log(f"ℹ️ Var olan temiz POI CSV kullanılacak: {p}")
            df = pd.read_csv(p, low_memory=False)
            # normalize
            if "lat" not in df.columns and "latitude" in df.columns:
                df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
            if "lon" not in df.columns and "longitude" in df.columns:
                df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
            if "poi_subcategory" not in df.columns:
                df["poi_subcategory"] = df.get("poi_category", "Unknown").astype(str)
            df["GEOID"] = _geoid11(df.get("GEOID"))
            return df
    # 2) yoksa geojson’dan üret
    log(f"📍 POI temizleniyor ve GEOID eşleniyor: {poi_path}")
    gdf = _read_geojson(poi_path)
    if "tags" not in gdf.columns:
        gdf["tags"] = [{}]*len(gdf)
    gdf["tags"] = gdf["tags"].apply(_parse_tags)
    triples = gdf["tags"].apply(_extract_cat_sub_name)
    gdf[["poi_category","poi_subcategory","poi_name"]] = pd.DataFrame(triples.tolist(), index=gdf.index)

    if "geometry" not in gdf.columns:
        if {"lon","lat"}.issubset(gdf.columns):
            gdf["geometry"] = gpd.points_from_xy(gdf["lon"], gdf["lat"])
        else:
            raise ValueError("GeoJSON 'geometry' veya 'lon/lat' içermiyor.")
    gdf = _ensure_crs(gdf, "EPSG:4326")
    gdf["lon"] = gdf.get("lon", pd.Series(index=gdf.index, dtype=float)).fillna(gdf.geometry.x)
    gdf["lat"] = gdf.get("lat", pd.Series(index=gdf.index, dtype=float)).fillna(gdf.geometry.y)

    # spatial join
    joined = gpd.sjoin(gdf, blocks[["GEOID","geometry"]], how="left", predicate="within")
    keep = [c for c in ["id","lat","lon","poi_category","poi_subcategory","poi_name","GEOID"] if c in joined.columns]
    df = joined[keep].copy()
    if "id" not in df.columns:
        df["id"] = np.arange(len(df))
    df = df.dropna(subset=["lat","lon"])
    df["GEOID"] = _geoid11(df["GEOID"])

    # kaydet
    outp = next((BASE_DIR / POI_CLEAN_CSV,),)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)
    log(f"✅ POI clean kaydedildi: {outp}  (satır={len(df):,})")
    return df

# ============================== Dinamik risk (opsiyonel) ==============================
def compute_dynamic_poi_risk(poi_df: pd.DataFrame, crime_points: pd.DataFrame, radius_m=300) -> dict:
    """POI alt-kategorileri için yakınındaki suç sayısına göre (ortalama) skor = [0..3] ölçeğinde normalize."""
    dfp = poi_df.copy()
    dfp["lat"] = pd.to_numeric(dfp.get("lat"), errors="coerce")
    dfp["lon"] = pd.to_numeric(dfp.get("lon"), errors="coerce")
    dfp = dfp.dropna(subset=["lat","lon"])

    dfc = crime_points.copy()
    dfc["latitude"]  = pd.to_numeric(dfc.get("latitude"), errors="coerce")
    dfc["longitude"] = pd.to_numeric(dfc.get("longitude"), errors="coerce")
    dfc = dfc.dropna(subset=["latitude","longitude"])

    if dfc.empty or dfp.empty:
        return {}

    crime_rad = np.radians(dfc[["latitude","longitude"]].values)
    poi_rad   = np.radians(dfp[["lat","lon"]].values)
    tree = BallTree(crime_rad, metric="haversine")
    r = radius_m / 6371000.0

    counts = []
    for pt, t in zip(poi_rad, dfp["poi_subcategory"].fillna("").astype(str)):
        if not t: continue
        idx = tree.query_radius([pt], r=r)[0]
        counts.append((t, len(idx)))

    if not counts: return {}
    agg = defaultdict(list)
    for t, c in counts: agg[t].append(c)
    avg = {t: float(np.mean(v)) for t, v in agg.items()}

    v = list(avg.values()); vmin, vmax = min(v), max(v)
    if vmax - vmin < 1e-9:
        return {t: 1.5 for t in avg}
    return {t: round(3 * (x - vmin) / (vmax - vmin), 2) for t, x in avg.items()}

# ============================== Yardımcı: tür-bilinçli doldurma ==============================
def _fillna_typed(df: pd.DataFrame, fills: dict[str, float | int | str]) -> pd.DataFrame:
    """Kategorik sütunlara doğrudan 0/0.0 yazmamak için tür-bilinçli doldurma."""
    out = df.copy()
    for c in out.columns:
        if c in fills:
            val = fills[c]
            s = out[c]
            if is_numeric_dtype(s):
                out[c] = pd.to_numeric(s, errors="coerce").fillna(val)
            elif is_categorical_dtype(s):
                # kategorik: val string değilse stringe çevir
                sval = str(val)
                if sval not in s.cat.categories:
                    s = s.cat.add_categories([sval])
                out[c] = s.fillna(sval)
            else:
                out[c] = s.fillna(val if isinstance(val, str) else str(val))
    return out

# ============================== GEOID-level + buffer ==============================
def geoid_poi_features(blocks: gpd.GeoDataFrame, poi_df: pd.DataFrame, poi_risk: dict,
                       radii=(300,600,900)) -> pd.DataFrame:
    # GEOID centroidleri (projeksiyonda hesapla → geri döndür)
    bl = blocks.copy()
    bl["GEOID"] = _geoid11(bl["GEOID"])
    bl = _ensure_crs(bl, "EPSG:4326")
    bl_m = bl.to_crs(3857)
    centroids_m = bl_m.dissolve(by="GEOID", as_index=False, dropna=False)["geometry"].centroid
    cent = gpd.GeoDataFrame(
        {"GEOID": bl_m.dissolve(by="GEOID", as_index=False)["GEOID"]},
        geometry=centroids_m, crs=bl_m.crs
    ).to_crs(4326)
    cent["cent_lat"] = cent.geometry.y
    cent["cent_lon"] = cent.geometry.x

    dfp = poi_df.copy()
    dfp["GEOID"] = _geoid11(dfp["GEOID"])
    dfp["lat"] = pd.to_numeric(dfp["lat"], errors="coerce")
    dfp["lon"] = pd.to_numeric(dfp["lon"], errors="coerce")
    dfp = dfp.dropna(subset=["lat","lon"])

    # risk sütunu
    dfp["__risk__"] = dfp["poi_subcategory"].astype(str).map(poi_risk).fillna(0.0)

    # dominant type
    def _mode(arr):
        arr = [a for a in arr if pd.notna(a) and str(a) != ""]
        if not arr: return "No_POI"
        return Counter(arr).most_common(1)[0][0]

    grp = dfp.groupby("GEOID", dropna=False)
    base = pd.DataFrame({
        "GEOID": grp.size().index,
        "poi_total_count": grp.size().values,
        "poi_risk_score": grp["__risk__"].sum().values,
        "poi_dominant_type": grp["poi_subcategory"].agg(_mode).values
    })

    # range etiketleri
    base["poi_total_count_range"] = pd.cut(
        base["poi_total_count"], bins=5,
        labels=[f"Q{i}" for i in range(1,6)], duplicates="drop"
    )
    base["poi_risk_score_range"] = pd.cut(
        base["poi_risk_score"], bins=5,
        labels=[f"Q{i}" for i in range(1,6)], duplicates="drop"
    )

    # buffer (BallTree haversine)
    poi_rad  = np.radians(dfp[["lat","lon"]].values)
    cent_rad = np.radians(cent[["cent_lat","cent_lon"]].values)
    if len(poi_rad) == 0:
        # hiç POI yoksa sıfırla
        buf = cent[["GEOID"]].copy()
        for r_m in radii:
            buf[f"poi_count_{r_m}m"] = 0
            buf[f"poi_risk_{r_m}m"]  = 0.0
    else:
        tree = BallTree(poi_rad, metric="haversine")
        poi_risk_vec = dfp["__risk__"].to_numpy()
        buf = cent[["GEOID"]].copy()
        for r_m in radii:
            idxs = tree.query_radius(cent_rad, r=r_m/6371000.0, count_only=False)
            buf[f"poi_count_{r_m}m"] = np.array([len(ix) for ix in idxs], dtype=int)
            buf[f"poi_risk_{r_m}m"]  = np.array([poi_risk_vec[ix].sum() if len(ix) else 0.0 for ix in idxs], dtype=float)

    out = cent[["GEOID"]].merge(base, on="GEOID", how="left").merge(buf, on="GEOID", how="left")

    # tür-bilinçli doldurma (Categorical hatası yaşamamak için)
    fills = {"poi_total_count":0, "poi_risk_score":0.0, "poi_dominant_type":"No_POI",
             "poi_total_count_range":"Q1", "poi_risk_score_range":"Q1"}
    for r_m in radii:
        fills[f"poi_count_{r_m}m"] = 0
        fills[f"poi_risk_{r_m}m"]  = 0.0
    out = _fillna_typed(out, fills)

    # tipler
    for c in [x for x in out.columns if x.startswith("poi_count_")]:
        out[c] = pd.to_numeric(out[c], downcast="integer")
    for c in [x for x in out.columns if x.startswith("poi_risk_") or x in ("poi_risk_score",)]:
        out[c] = pd.to_numeric(out[c], downcast="float")
    return out

# ============================== Enrich (GRID / EVENTS) ==============================
def enrich_df_with_poi(df: pd.DataFrame, geoid_poi: pd.DataFrame, is_grid: bool) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()
    # date normalizasyonu (görünürlük; içerik sabit)
    if is_grid:
        out = _ensure_date(out)
    # GEOID tek ve 11 hane
    if "GEOID" not in out.columns:
        cand = [c for c in out.columns if "geoid" in c.lower()]
        if not cand: raise RuntimeError("Hedef tabloda GEOID kolonu yok.")
        out.insert(0, "GEOID", _geoid11(out[cand[0]]).astype("string"))
        out.drop(columns=[c for c in cand if c != "GEOID"], inplace=True, errors="ignore")
    else:
        out["GEOID"] = _geoid11(out["GEOID"]).astype("string")

    # çakışmaları temizle (aynı isimli eski POI kolonu varsa)
    drop_cols = ["poi_total_count","poi_risk_score","poi_dominant_type",
                 "poi_total_count_range","poi_risk_score_range"]
    drop_cols += [c for c in out.columns if c.startswith("poi_count_") or c.startswith("poi_risk_")]
    out.drop(columns=[c for c in drop_cols if c in out.columns], inplace=True, errors="ignore")

    out = out.merge(geoid_poi, on="GEOID", how="left", validate="many_to_one")
    return out

# ============================== Run ==============================
def main() -> int:
    # 0) artifact varsa aç
    safe_unzip(ARTIFACT_ZIP, ARTIFACT_DIR)

    # 1) kaynakları oku
    grid = _read_csv(BASE_DIR / GRID_IN) if not GRID_IN.is_absolute() else _read_csv(GRID_IN)
    ev   = _read_csv(BASE_DIR / EV_IN)   if not EV_IN.is_absolute()  else _read_csv(EV_IN)

    poi_path    = next((p for p in POI_PATHS if Path(p).exists()), None)
    blocks_path = next((p for p in BLOCKS_PATHS if Path(p).exists()), None)
    if not poi_path:    raise FileNotFoundError(f"❌ POI GeoJSON yok: {POI_GEOJSON} (artifact/BASE/local)")
    if not blocks_path: raise FileNotFoundError(f"❌ Blocks GeoJSON yok: {BLOCKS_GEOJSON} (artifact/BASE/local)")

    blocks = _read_geojson(Path(blocks_path))
    if "GEOID" not in blocks.columns: raise ValueError("Blocks içinde GEOID yok.")
    blocks["GEOID"] = _geoid11(blocks["GEOID"])

    # 2) POI clean yükle/üret
    poi_clean = load_or_build_poi_clean(blocks, Path(poi_path), POI_CLEAN_PATHS)

    # 3) Dinamik POI risk sözlüğü (opsiyonel)
    crime_points = ev if not ev.empty else grid
    try:
        risk_dict = compute_dynamic_poi_risk(poi_clean, crime_points, radius_m=300)
    except Exception:
        risk_dict = {}

    # cachele
    try:
        (BASE_DIR / POI_RISK_JSON).parent.mkdir(parents=True, exist_ok=True)
        with open(BASE_DIR / POI_RISK_JSON, "w") as f: json.dump(risk_dict, f, indent=2)
        log(f"🧾 Risk sözlüğü yazıldı: {BASE_DIR / POI_RISK_JSON}  (count={len(risk_dict)})")
    except Exception:
        pass

    # 4) GEOID-level + buffer
    geoid_poi = geoid_poi_features(blocks, poi_clean, risk_dict, radii=(300,600,900))

    # 5) Enrich & yaz
    if not grid.empty:
        grid2 = enrich_df_with_poi(grid, geoid_poi, is_grid=True)
        _safe_write_csv(grid2, BASE_DIR / GRID_OUT if not GRID_OUT.is_absolute() else GRID_OUT)
    else:
        log("ℹ️ GRID boş/bulunamadı → atlandı.")

    if not ev.empty:
        ev2 = enrich_df_with_poi(ev, geoid_poi, is_grid=False)
        _safe_write_csv(ev2, BASE_DIR / EV_OUT if not EV_OUT.is_absolute() else EV_OUT)
    else:
        log("ℹ️ EVENTS boş/bulunamadı → atlandı.")

    # 6) kısa önizleme
    try:
        cols = ["GEOID","poi_total_count","poi_risk_score","poi_dominant_type",
                "poi_count_300m","poi_count_600m","poi_count_900m",
                "poi_risk_300m","poi_risk_600m","poi_risk_900m"]
        if not grid.empty:
            log("— GRID preview —")
            log(grid2[[c for c in cols if c in grid2.columns]].head(10).to_string(index=False))
        if not ev.empty:
            log("— EVENTS preview —")
            log(ev2[[c for c in cols if c in ev2.columns]].head(10).to_string(index=False))
    except Exception:
        pass

    log("✅ enrich_with_poi_daily.py tamam.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

# update_poi_fr.py
import os, ast, json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree

pd.options.mode.copy_on_write = True

# ---------- LOG ----------
def log_shape(df, label):
    r, c = df.shape
    print(f"📊 {label}: {r} satır × {c} sütun")

def log_delta(before_shape, after_shape, label):
    br, bc = before_shape
    ar, ac = after_shape
    print(f"🔗 {label}: {br}×{bc} → {ar}×{ac} (Δr={ar-br}, Δc={ac-bc})")

try:
    from shapely.strtree import STRtree
except Exception:
    STRtree = None

# ---------- PATHS / IO ----------
BASE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")

# Crime input auto-detect (FR first, then SF)
CAND_CRIME_IN = [
    os.path.join(BASE_DIR, "fr_crime_05.csv"),
    "fr_crime_05.csv",
]
CRIME_IN = next((p for p in CAND_CRIME_IN if os.path.exists(p)), None)
if not CRIME_IN:
    raise FileNotFoundError("❌ Suç girdisi bulunamadı: fr_crime_05.csv")

CRIME_OUT = os.path.join(BASE_DIR, "fr_crime_06.csv")

# POI kaynakları
POI_GEOJSON_1 = os.path.join(BASE_DIR, "sf_pois.geojson")
POI_GEOJSON_2 = "sf_pois.geojson"  # fallback
BLOCK_PATH_1  = os.path.join(BASE_DIR, "sf_census_blocks_with_population.geojson")
BLOCK_PATH_2  = "sf_census_blocks_with_population.geojson"  # fallback
POI_CLEAN_CSV = os.path.join(BASE_DIR, "sf_pois_cleaned_with_geoid.csv")
POI_RISK_JSON = os.path.join(BASE_DIR, "risky_pois_dynamic.json")

Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

# ---------- HELPERS ----------
def _ensure_parent(path: str):
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)

def _safe_save_csv(df: pd.DataFrame, path: str):
    try:
        _ensure_parent(path)
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"❌ Kaydetme hatası: {path}\n{e}")
        df.to_csv(path + ".bak", index=False)
        print(f"📁 Yedek oluşturuldu: {path}.bak")

def _read_geojson_robust(path: str) -> gpd.GeoDataFrame:
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"GeoJSON yok: {path}")
    try:
        gdf = gpd.read_file(path)
        return _ensure_crs(gdf, "EPSG:4326")
    except Exception as e:
        print(f"⚠️ gpd.read_file başarısız ({path}): {e}")
        txt = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
        gj = None
        try:
            if "\n" in txt and txt.splitlines()[0].strip().startswith("{") and '"features"' not in txt:
                feats = [json.loads(line) for line in txt.splitlines() if line.strip()]
                gj = {"type": "FeatureCollection", "features": feats}
            else:
                gj = json.loads(txt)
        except Exception as e2:
            raise ValueError(f"GeoJSON parse edilemedi: {e2}")
        if "features" not in gj:
            raise ValueError("GeoJSON FeatureCollection bekleniyordu (features yok).")
        gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
        return _ensure_crs(gdf, "EPSG:4326")

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

def _normalize_geoid(series: pd.Series, target_len: int = 11) -> pd.Series:
    s = series.astype(str).str.extract(r"(\d+)")[0]
    return s.str[:target_len].str.zfill(target_len)

def _make_dynamic_labels(series: pd.Series, bin_count=5):
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if vals.size == 0:
        def lab(_): return "Q1 (0-0)"
        return lab
    qs = np.quantile(vals, [i/bin_count for i in range(bin_count+1)])
    def lab(x):
        if pd.isna(x): return f"Q1 ({qs[0]:.1f}-{qs[1]:.1f})"
        for i in range(bin_count):
            if x <= qs[i+1]:
                return f"Q{i+1} ({qs[i]:.1f}-{qs[i+1]:.1f})"
        return f"Q{bin_count} ({qs[-2]:.1f}-{qs[-1]:.1f})"
    return lab

def _pick_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

# ---------- 1) POI CLEAN + GEOID ----------
def build_poi_clean_with_geoid(blocks_path: str, poi_geojson_path: str) -> pd.DataFrame:
    print("📍 POI okunuyor → kategoriler çıkarılıyor → GEOID atanıyor...")
    if poi_geojson_path is None or not os.path.exists(poi_geojson_path):
        raise FileNotFoundError("❌ POI GeoJSON bulunamadı.")
    gdf = _read_geojson_robust(poi_geojson_path)

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

    if blocks_path is None or not os.path.exists(blocks_path):
        raise FileNotFoundError("❌ Nüfus blokları GeoJSON bulunamadı.")
    blocks = _read_geojson_robust(blocks_path)
    if "GEOID" not in blocks.columns:
        raise ValueError("Block dosyasında 'GEOID' yok.")
    # Centroidleri 11 hane GE OID'e indir
    blocks["GEOID"] = _normalize_geoid(blocks["GEOID"], 11)

    try:
        joined = gpd.sjoin(gdf, blocks[["GEOID","geometry"]], how="left", predicate="within")
    except Exception as e:
        print("⚠️ gpd.sjoin başarısız, STRtree fallback →", e)
        if STRtree is None:
            raise RuntimeError("Shapely STRtree yok. 'shapely>=2.0' kurun veya rtree yükleyin.")
        geoms = list(blocks.geometry.values)
        tree = STRtree(geoms)
        geom_id_to_geoid = {id(g): geoid for g, geoid in zip(geoms, blocks["GEOID"])}
        geoid_list = []
        for pt in gdf.geometry.values:
            cands = tree.query(pt, predicate="contains")
            geoid_list.append(geom_id_to_geoid[id(cands[0])] if cands else None)
        joined = gdf.copy();  joined["GEOID"] = geoid_list

    keep = [c for c in ["id","lat","lon","poi_category","poi_subcategory","poi_name","GEOID"] if c in joined.columns]
    df = joined[keep].copy()
    if "id" not in df.columns:
        df["id"] = np.arange(len(df))
    df = df.dropna(subset=["lat","lon"])
    df["GEOID"] = _normalize_geoid(df["GEOID"], 11)

    _safe_save_csv(df, POI_CLEAN_CSV)
    print(f"✅ Kaydedildi: {POI_CLEAN_CSV}  |  Satır: {len(df):,}")
    try: print(df.head(5).to_string(index=False))
    except: pass
    return df

# ---------- 2) DİNAMİK RİSK (opsiyonel) ----------
def compute_dynamic_poi_risk(df_crime: pd.DataFrame, df_poi: pd.DataFrame, radius_m=300) -> dict:
    dfp = df_poi.copy()
    dfp["lat"] = pd.to_numeric(dfp.get("lat"), errors="coerce")
    dfp["lon"] = pd.to_numeric(dfp.get("lon"), errors="coerce")
    dfp = dfp.dropna(subset=["lat","lon"])
    if "poi_subcategory" in dfp.columns:
        dfp = dfp[~dfp["poi_subcategory"].isin(["police", "ranger_station"])]

    dfc = df_crime.copy()
    dfc["latitude"]  = pd.to_numeric(dfc.get("latitude"), errors="coerce")
    dfc["longitude"] = pd.to_numeric(dfc.get("longitude"), errors="coerce")
    dfc = dfc.dropna(subset=["latitude","longitude"])

    print(f"POI noktaları: {len(dfp):,} | Suç noktaları: {len(dfc):,}")
    if dfc.empty or dfp.empty:
        print("⚠️ Risk için yeterli nokta yok (koordinat eksik). Boş sözlük yazılacak.")
        _ensure_parent(POI_RISK_JSON)
        with open(POI_RISK_JSON, "w") as f: json.dump({}, f, indent=2)
        return {}

    crime_rad = np.radians(dfc[["latitude","longitude"]].values)
    poi_rad   = np.radians(dfp[["lat","lon"]].values)
    tree = BallTree(crime_rad, metric="haversine")
    r = radius_m / 6371000.0

    poi_types = dfp["poi_subcategory"].fillna("")
    counts = []
    for pt, t in zip(poi_rad, poi_types):
        if not t: continue
        idx = tree.query_radius([pt], r=r)[0]
        counts.append((t, len(idx)))

    if not counts:
        _ensure_parent(POI_RISK_JSON)
        with open(POI_RISK_JSON, "w") as f: json.dump({}, f, indent=2)
        return {}

    agg = defaultdict(list)
    for t, c in counts: agg[t].append(c)
    avg = {t: float(np.mean(v)) for t, v in agg.items()}

    v = list(avg.values()); vmin, vmax = min(v), max(v)
    if vmax - vmin < 1e-9:
        norm = {t: 1.5 for t in avg}
    else:
        norm = {t: round(3 * (x - vmin) / (vmax - vmin), 2) for t, x in avg.items()}

    _ensure_parent(POI_RISK_JSON)
    with open(POI_RISK_JSON, "w") as f: json.dump(norm, f, indent=2)

    print("🔝 İlk 15 alt-kategori (skora göre):")
    for k, s in sorted(norm.items(), key=lambda x: -x[1])[:15]:
        print(f"  {k:<24} → {s:.2f}")
    return norm

# ---------- 3) GEOID ÖZET + BUFFER (300/600/900m) ----------
def build_geoid_level_poi_features_with_buffers(blocks_path: str,
                                                df_poi: pd.DataFrame,
                                                poi_risk: dict,
                                                radii_m=(300, 600, 900)) -> pd.DataFrame:
    """
    ÇIKTI (GEOID=11 hane):
      - poi_total_count, poi_risk_score, poi_dominant_type, *_range
      - poi_count_{r}m, poi_risk_{r}m  (r ∈ {300,600,900})
    """
    # --- GEOID normalize
    dfp = df_poi.copy()
    dfp["GEOID"] = _normalize_geoid(dfp.get("GEOID"), 11)
    dfp["lat"] = pd.to_numeric(dfp.get("lat"), errors="coerce")
    dfp["lon"] = pd.to_numeric(dfp.get("lon"), errors="coerce")
    dfp = dfp.dropna(subset=["lat","lon"])

    # --- static risk sütunu
    sub = dfp.get("poi_subcategory", "").astype(str)
    dfp["__risk__"] = sub.map(poi_risk).fillna(0.0)

    # --- dominant type (mod)
    def _mode(arr):
        arr = [a for a in arr if pd.notna(a) and a != ""]
        if not arr: return "No_POI"
        c = Counter(arr)
        return c.most_common(1)[0][0]

    # --- GEOID polygon centroidleri
    if blocks_path is None or not os.path.exists(blocks_path):
        raise FileNotFoundError("❌ Nüfus blokları GeoJSON bulunamadı.")
    blocks = _read_geojson_robust(blocks_path)
    if "GEOID" not in blocks.columns:
        raise ValueError("Block dosyasında 'GEOID' yok.")
    blocks["GEOID"] = _normalize_geoid(blocks["GEOID"], 11)
    # Her GEOID için centroid
    centroids = blocks.dissolve(by="GEOID", as_index=False, dropna=False)["geometry"].centroid
    cent_gdf = gpd.GeoDataFrame({"GEOID": blocks.dissolve(by="GEOID", as_index=False)["GEOID"]},
                                geometry=centroids, crs="EPSG:4326")
    cent_gdf["cent_lat"] = cent_gdf.geometry.y
    cent_gdf["cent_lon"] = cent_gdf.geometry.x

    # --- GEOID bazlı temel özetler
    grp = dfp.groupby("GEOID", dropna=False)
    base = pd.DataFrame({
        "GEOID": grp.size().index,
        "poi_total_count": grp.size().values,
        "poi_risk_score": grp["__risk__"].sum().values,
        "poi_dominant_type": grp["poi_subcategory"].agg(_mode).values
    })

    # --- range etiketleri
    lab_cnt  = _make_dynamic_labels(base["poi_total_count"])
    lab_risk = _make_dynamic_labels(base["poi_risk_score"])
    base["poi_total_count_range"] = base["poi_total_count"].apply(lab_cnt)
    base["poi_risk_score_range"]  = base["poi_risk_score"].apply(lab_risk)

    # --- BUFFER sorguları (BallTree, haversine)
    poi_rad = np.radians(dfp[["lat","lon"]].values)
    cent_rad = np.radians(cent_gdf[["cent_lat","cent_lon"]].values)
    tree = BallTree(poi_rad, metric="haversine")

    # hazırlık: risk vector
    poi_risk_vec = dfp["__risk__"].to_numpy()

    buf_df = cent_gdf[["GEOID"]].copy()
    for r_m in radii_m:
        r = r_m / 6371000.0
        # tüm centroidler için komşuları getir
        idxs = tree.query_radius(cent_rad, r=r, count_only=False)
        counts = np.array([len(ix) for ix in idxs], dtype=int)
        risks  = np.array([poi_risk_vec[ix].sum() if len(ix) else 0.0 for ix in idxs], dtype=float)
        buf_df[f"poi_count_{r_m}m"] = counts
        buf_df[f"poi_risk_{r_m}m"]  = risks

    # --- birleştir
    out = cent_gdf[["GEOID"]].merge(base, on="GEOID", how="left").merge(buf_df, on="GEOID", how="left")
    # fillna
    fill_vals = {
        "poi_total_count": 0,
        "poi_risk_score": 0.0,
        "poi_dominant_type": "No_POI",
        "poi_total_count_range": "Q1 (0-0)",
        "poi_risk_score_range":  "Q1 (0-0)",
    }
    for col in [c for c in out.columns if c.startswith("poi_count_")]:
        fill_vals[col] = 0
    for col in [c for c in out.columns if c.startswith("poi_risk_")]:
        fill_vals[col] = 0.0
    out = out.fillna(fill_vals)

    log_shape(out, "POI (GEOID-özet + buffer)")
    return out

# ---------- 4) CRIME ⨯ POI MERGE ----------
def enrich_crime(df_crime: pd.DataFrame, geoid_poi: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [  # eski sütunlar/yenileri
        "poi_total_count","poi_risk_score","poi_dominant_type",
        "poi_total_count_range","poi_risk_score_range"
    ]
    # buffer sütun adları da çakışırsa temizle
    drop_cols += [c for c in df_crime.columns if c.startswith("poi_count_") or c.startswith("poi_risk_")]

    out = df_crime.copy()
    out["GEOID"] = _normalize_geoid(out.get("GEOID"), 11)
    out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")

    before = out.shape
    out = out.merge(geoid_poi, on="GEOID", how="left")
    # fillna güvenlik
    fill_vals = {
        "poi_total_count": 0,
        "poi_risk_score": 0.0,
        "poi_dominant_type": "No_POI",
        "poi_total_count_range": "Q1 (0-0)",
        "poi_risk_score_range":  "Q1 (0-0)",
    }
    for c in [c for c in geoid_poi.columns if c.startswith("poi_count_")]:
        fill_vals[c] = 0
    for c in [c for c in geoid_poi.columns if c.startswith("poi_risk_")]:
        fill_vals[c] = 0.0
    out = out.fillna(fill_vals)

    log_delta(before, out.shape, "CRIME ⨯ POI (GEOID+buffers)")
    return out

# ---------- MAIN ----------
if __name__ == "__main__":
    print("🚀 Başlıyor (GEOID + BUFFER POI Enrich: 300/600/900m)...")
    # Crime
    df_crime = pd.read_csv(CRIME_IN, low_memory=False)
    if "GEOID" not in df_crime.columns:
        raise KeyError("❌ Suç verisinde GEOID yok. GEOID olmadan merge yapılamaz.")
    df_crime["GEOID"] = _normalize_geoid(df_crime["GEOID"], 11)
    log_shape(df_crime, "CRIME (POI enrich öncesi)")

    # Paths
    blocks_path = _pick_existing(BLOCK_PATH_1, BLOCK_PATH_2)
    poi_geojson = _pick_existing(POI_GEOJSON_1, POI_GEOJSON_2)

    # POI clean hazır mı?
    if os.path.exists(POI_CLEAN_CSV):
        print("ℹ️ Var olan temiz POI CSV kullanılacak:", POI_CLEAN_CSV)
        df_poi = pd.read_csv(POI_CLEAN_CSV, low_memory=False)
        # normalize
        if "lat" not in df_poi.columns and "latitude" in df_poi.columns:
            df_poi["lat"] = pd.to_numeric(df_poi["latitude"], errors="coerce")
        if "lon" not in df_poi.columns and "longitude" in df_poi.columns:
            df_poi["lon"] = pd.to_numeric(df_poi["longitude"], errors="coerce")
        if "poi_subcategory" not in df_poi.columns:
            guess = df_poi.get("poi_category", "Unknown")
            df_poi["poi_subcategory"] = pd.Series(guess, dtype=str)
        df_poi["GEOID"] = _normalize_geoid(df_poi.get("GEOID"), 11)
    else:
        df_poi = build_poi_clean_with_geoid(blocks_path, poi_geojson)

    log_shape(df_poi, "POI clean")

    # Risk sözlüğü
    try:
        risk_dict = compute_dynamic_poi_risk(df_crime, df_poi, radius_m=300)
    except Exception as e:
        print(f"⚠️ Risk sözlüğü üretilemedi: {e}; boş sözlük kullanılacak.")
        risk_dict = {}
    print(f"🧪 Risk sözlüğü boyutu: {len(risk_dict)} alt-kategori")

    # GEOID + buffer özetleri
    geoid_poi = build_geoid_level_poi_features_with_buffers(
        blocks_path=blocks_path,
        df_poi=df_poi,
        poi_risk=risk_dict,
        radii_m=(300, 600, 900)
    )

    # Suçu zenginleştir
    before = df_crime.shape
    out_df = enrich_crime(df_crime, geoid_poi)
    log_delta(before, out_df.shape, "CRIME ⨯ POI (final)")

    # Kaydet
    _safe_save_csv(out_df, CRIME_OUT)
    log_shape(out_df, "CRIME (POI enrich sonrası)")
    print(f"✅ Yazıldı: {CRIME_OUT}  |  Satır: {len(out_df):,}")

    # Örnek
    try:
        cols = ["GEOID","poi_total_count","poi_risk_score","poi_dominant_type",
                "poi_count_300m","poi_count_600m","poi_count_900m",
                "poi_risk_300m","poi_risk_600m","poi_risk_900m"]
        cols = [c for c in cols if c in out_df.columns]
        print(out_df[cols].head(3).to_string(index=False))
    except Exception as e:
        print(f"(info) Örnek yazdırılamadı: {e}")

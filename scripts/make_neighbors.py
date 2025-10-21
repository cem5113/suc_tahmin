#!/usr/bin/env python3
from __future__ import annotations
import os, re, sys, math
import pandas as pd
from pathlib import Path
from datetime import timedelta

# Geo stack (poligon için)
try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:
    gpd = None

CRIME_DIR = Path(os.environ.get("CRIME_DATA_DIR", "crime_prediction_data"))
GEOID_LEN = int(os.environ.get("GEOID_LEN", "11"))

# Davranış bayrakları
STRATEGY   = os.environ.get("NEIGHBOR_STRATEGY", "queen").lower()   # queen|rook|knn
MIN_DEG    = int(os.environ.get("MIN_NEIGHBOR_DEG", "3"))
KNN_K      = int(os.environ.get("KNN_K", "5"))
OUT_PATH   = Path(os.environ.get("NEIGHBOR_FILE", str(CRIME_DIR / "neighbors.csv")))

# Olası poligon/katman adayları
POLY_HINT  = os.environ.get("NEIGHBOR_POLY", "")
POLY_CANDIDATES = [p for p in [
    POLY_HINT,
    str(CRIME_DIR / "sf_grid.geojson"),
    str(CRIME_DIR / "sf_crime_grid.geojson"),
    str(CRIME_DIR / "sf_cells.geojson"),
    "sf_grid.geojson", "sf_crime_grid.geojson", "sf_cells.geojson",
] if p]

GRID_CSV = Path(os.environ.get("STACKING_DATASET", str(CRIME_DIR / "sf_crime_grid_full_labeled.csv")))

# Yardımcı fonksiyonlar
def _norm_geoid(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.extract(r"(\d+)", expand=False)
              .str.zfill(GEOID_LEN))

def _find_geoid_col(df: pd.DataFrame) -> str | None:
    low = {re.sub(r"[^a-z0-9]","", c.lower()): c for c in df.columns}
    return low.get("geoid") or low.get("geoid10") or low.get("geographyid") or low.get("tractce")

def _find_latlon_cols(df: pd.DataFrame):
    low = {re.sub(r"[^a-z0-9]","", c.lower()): c for c in df.columns}
    lat = low.get("lat") or low.get("latitude") or low.get("centroidlat")
    lon = low.get("lon") or low.get("lng") or low.get("longitude") or low.get("centroidlon")
    return lat, lon

def _symmetrize(df):
    rev = df.rename(columns={"geoid":"neighbor","neighbor":"geoid"})
    return (pd.concat([df, rev], ignore_index=True)
              .drop_duplicates()
              .query("geoid != neighbor"))

def _ensure_min_degree(df, geos, k=KNN_K, min_deg=MIN_DEG):
    deg = df.groupby("geoid")["neighbor"].nunique()
    need = set(deg.index[deg < min_deg])
    if not need:
        return df

    import numpy as np
    from numpy.linalg import norm
    idx = {g:i for i,g in enumerate(geos["geoid"])}
    XY = geos[["x","y"]].to_numpy()

    rows = []
    for g in need:
        i = idx[g]
        d = norm(XY - XY[i], axis=1)
        order = np.argsort(d)
        picks = []
        for j in order[1: 1+k*2]:
            gj = geos.iloc[j]["geoid"]
            if gj != g:
                picks.append(gj)
            if len(picks) >= k:
                break
        rows += [(g, n) for n in picks]

    knn_df = pd.DataFrame(rows, columns=["geoid","neighbor"])
    all_df = _symmetrize(pd.concat([df, knn_df], ignore_index=True).drop_duplicates())
    return all_df

def _neighbors_from_polygons(poly_path: str) -> pd.DataFrame | None:
    if gpd is None:
        return None
    try:
        gdf = gpd.read_file(poly_path)
    except Exception:
        return None

    gcol = _find_geoid_col(gdf)
    if not gcol or "geometry" not in gdf.columns:
        return None

    gdf = gdf[[gcol, "geometry"]].rename(columns={gcol:"geoid"}).copy()
    gdf["geoid"] = _norm_geoid(gdf["geoid"])
    gdf = gdf[~gdf["geometry"].is_empty & gdf["geometry"].notna()].reset_index(drop=True)

    predicate = "touches"
    pairs = gdf.sindex.query_bulk(gdf.geometry, predicate=predicate)
    i, j = pairs
    m = pd.DataFrame({"i": i, "j": j})
    m = m[m["i"] < m["j"]]
    df = pd.DataFrame({
        "geoid":   gdf.loc[m["i"], "geoid"].to_numpy(),
        "neighbor":gdf.loc[m["j"], "geoid"].to_numpy(),
    })
    df = _symmetrize(df.drop_duplicates())
    return df

def _centroid_table(poly_df: pd.DataFrame | None, grid_csv: Path) -> pd.DataFrame | None:
    if poly_df is not None and gpd is not None:
        c = poly_df.copy()
        c["x"] = c["geometry"].centroid.x
        c["y"] = c["geometry"].centroid.y
        return pd.DataFrame({"geoid": c["geoid"], "x": c["x"], "y": c["y"]})

    if grid_csv.exists():
        df = pd.read_csv(grid_csv, low_memory=False, dtype=str)
        gcol = _find_geoid_col(df)
        lat, lon = _find_latlon_cols(df)
        if gcol and lat and lon:
            tmp = df[[gcol, lat, lon]].rename(columns={gcol:"geoid", lat:"lat", lon:"lon"})
            tmp["geoid"] = _norm_geoid(tmp["geoid"])
            tmp["x"] = pd.to_numeric(tmp["lon"], errors="coerce")
            tmp["y"] = pd.to_numeric(tmp["lat"], errors="coerce")
            tmp = tmp[["geoid","x","y"]].dropna()
            return tmp
    return None


# === ANA AKIŞ ===
def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Poligonlardan komşuluk oluştur
    poly_df = None
    for p in POLY_CANDIDATES:
        if not p:
            continue
        try:
            dfp = _neighbors_from_polygons(p)
            if dfp is not None and len(dfp) > 0:
                poly_df = gpd.read_file(p) if gpd is not None else None
                nb = dfp
                break
        except Exception:
            continue
    else:
        nb = pd.DataFrame(columns=["geoid","neighbor"])

    # 2️⃣ Centroid tablosu
    cent = _centroid_table(poly_df, GRID_CSV)

    # 3️⃣ Eksik derece tamamla (KNN)
    if cent is not None:
        nb = _ensure_min_degree(nb, cent, k=KNN_K, min_deg=MIN_DEG)

    # 4️⃣ Son temizlik ve kaydetme
    nb["geoid"]    = _norm_geoid(nb["geoid"])
    nb["neighbor"] = _norm_geoid(nb["neighbor"])
    nb = (_symmetrize(nb)
            .drop_duplicates()
            .sort_values(["geoid","neighbor"]))
    nb.to_csv(OUT_PATH, index=False, encoding="utf-8")

    deg = nb.groupby("geoid")["neighbor"].nunique()
    print(f"neighbors.csv yazıldı → {OUT_PATH} | n_edges={len(nb)} | n_nodes={len(deg)} | "
          f"min_deg={int(deg.min()) if len(deg) else 0} | mean_deg={round(float(deg.mean()),2) if len(deg) else 0.0}")

    # 5️⃣ Komşuluk bazlı zenginleştirme (sf_crime_08 → sf_crime_09)
    sf_path = CRIME_DIR / "sf_crime_08.csv"
    dst_path = CRIME_DIR / "sf_crime_09.csv"

    if not sf_path.exists():
        print(f"⚠️ {sf_path} bulunamadı, sf_crime_09 oluşturulmadı.")
        return

    print(f"▶︎ Komşuluk bazlı zenginleştirme başlıyor: {sf_path.name} + {OUT_PATH.name} → {dst_path.name}")

    sf = pd.read_csv(sf_path, parse_dates=["datetime"], low_memory=False)
    sf["geoid"] = sf["geoid"].astype(str).str.zfill(GEOID_LEN)
    nb["geoid"] = nb["geoid"].astype(str).str.zfill(GEOID_LEN)
    nb["neighbor"] = nb["neighbor"].astype(str).str.zfill(GEOID_LEN)

    sf_crimes = sf[sf["Y_label"] == 1][["geoid", "datetime"]].copy()

    def neighbor_stats(base_time, geoid):
        neighbors = nb.loc[nb["geoid"] == geoid, "neighbor"].tolist()
        if not neighbors:
            return (0, 0, 0)
        t1 = base_time - timedelta(hours=24)
        t3 = base_time - timedelta(hours=72)
        t7 = base_time - timedelta(days=7)
        recent = sf_crimes[(sf_crimes["datetime"] <= base_time) & (sf_crimes["geoid"].isin(neighbors))]
        n24 = recent[recent["datetime"] >= t1].shape[0]
        n72 = recent[recent["datetime"] >= t3].shape[0]
        n7d = recent[recent["datetime"] >= t7].shape[0]
        return (n24, n72, n7d)

    records = []
    for _, row in sf.iterrows():
        g = row["geoid"]
        t = row["datetime"]
        n24, n72, n7d = neighbor_stats(t, g)
        records.append((g, t, n24, n72, n7d))

    nei_feats = pd.DataFrame(records, columns=["geoid", "datetime",
                                               "neighbor_crime_24h", "neighbor_crime_72h", "neighbor_crime_7d"])
    sf9 = sf.merge(nei_feats, on=["geoid", "datetime"], how="left")
    sf9.to_csv(dst_path, index=False)
    print(f"✅ {dst_path.name} oluşturuldu → {len(sf9):,} satır, {sf9.shape[1]} sütun")
    print(sf9.head(5))


if __name__ == "__main__":
    main()

# update_911_fr.py  — FR hattı için 911 tabanlı mekânsal özellikler
# Değişiklik: sf_crime_L.csv'den koordinat beklemek yerine
#             suç noktalarını grid (tercih) ya da event (sf_crime_y.csv) içinden okur.

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple, Optional, List

import pandas as pd
import geopandas as gpd


# --------------------------------------------------------- yardımcılar
def log(msg: str) -> None:
    """Basit stdout logger."""
    print(msg, file=sys.stdout, flush=True)


def _pick_lat_lon(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    lat = next((c for c in ["latitude", "lat", "y", "centroid_lat", "_lat_"] if c in df.columns), None)
    lon = next((c for c in ["longitude", "long", "lon", "x", "centroid_lon", "_lon_"] if c in df.columns), None)
    return lat, lon


def _ensure_point_gdf(df: pd.DataFrame, lat_col: str, lon_col: str) -> gpd.GeoDataFrame:
    d = df.copy()
    d[lat_col] = pd.to_numeric(d[lat_col], errors="coerce")
    d[lon_col] = pd.to_numeric(d[lon_col], errors="coerce")
    d = d.dropna(subset=[lat_col, lon_col])
    return gpd.GeoDataFrame(
        d,
        geometry=gpd.points_from_xy(d[lon_col], d[lat_col]),
        crs="EPSG:4326",
    )


def _to_metric(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Metre tabanlı projeksiyona çevirir."""
    try:
        return gdf.to_crs(3857)  # Web Mercator (metre)
    except Exception:
        return gdf.to_crs(32610)  # UTM Zone 10N (SF çevresi)


def _distance_nearest(src_m: gpd.GeoDataFrame, pts_m: gpd.GeoDataFrame) -> pd.Series:
    """Kaynak noktalara en yakın 911 noktasının mesafesi (metre)."""
    if pts_m.empty:
        return pd.Series([pd.NA] * len(src_m), index=src_m.index, dtype="float")

    try:
        j = gpd.sjoin_nearest(
            src_m[["geometry"]],
            pts_m[["geometry"]],
            how="left",
            distance_col="_dist",
        )
        # sjoin_nearest indexleri korur; sıraya göre hizalayalım
        return j["_dist"].reindex(src_m.index)
    except Exception:
        # yavaş ama güvenli fallback
        vals: List[float] = []
        P = list(pts_m.geometry.values)
        for g in src_m.geometry.values:
            if not P:
                vals.append(pd.NA)
            else:
                vals.append(min(g.distance(p) for p in P))
        return pd.Series(vals, index=src_m.index, dtype="float")


def _bin_distance(m) -> str:
    if pd.isna(m):
        return "none"
    try:
        m = float(m)
    except Exception:
        return "none"
    if m <= 300:
        return "≤300m"
    if m <= 600:
        return "300–600m"
    if m <= 900:
        return "600–900m"
    return ">900m"


def _count_within(src_m: gpd.GeoDataFrame, pts_m: gpd.GeoDataFrame, r: int) -> pd.Series:
    """Kaynak noktaların r (metre) tamponu içinde kaç 911 noktası var?"""
    if pts_m.empty:
        return pd.Series(0, index=src_m.index, dtype="int32")

    buf = src_m.buffer(r)
    gbuf = gpd.GeoDataFrame(src_m[[]], geometry=buf, crs=src_m.crs)

    # points (left) ∈ buffer (right) eşleşmesi
    j = gpd.sjoin(
        pts_m[["geometry"]],
        gbuf.rename_geometry("geometry"),
        how="left",
        predicate="within",
    )
    counts = j.groupby("index_right").size()

    out = pd.Series(0, index=src_m.index, dtype="int32")
    out.loc[counts.index] = counts.values
    return out


# --------------------------------------------------------- sabitler & yollar
SAVE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# Suç noktaları için KOORDİNAT içeren aday kaynaklar
CRIME_POINT_SOURCES = [
    Path(SAVE_DIR) / "sf_crime_grid_full_labeled.csv",  # tercih: grid (lat/lon var)
    Path(SAVE_DIR) / "sf_crime_y.csv",                  # fallback: event-level
    Path(".")        / "sf_crime_grid_full_labeled.csv",
    Path(".")        / "sf_crime_y.csv",
]

CRIME_OUT = Path(SAVE_DIR) / "fr_crime_01.csv"

# 911 ham nokta dosya adayları (öncelik _y.csv)
SF911_CANDIDATES = [
    Path(SAVE_DIR) / "sf_911_last_5_year_y.csv",
    Path(".") / "sf_911_last_5_year_y.csv",
    Path(SAVE_DIR) / "sf_911_last_5_year.csv",
    Path(".") / "sf_911_last_5_year.csv",
]

# Buffer yarıçapları (metre)
BUFFERS_M = [300, 600, 900]


def _load_crime_points() -> tuple[pd.DataFrame, gpd.GeoDataFrame, str, str]:
    """
    Koordinat içeren bir suç kaynağı (grid veya event) döndürür.
    Çıktı: (crime_df_raw, crime_gdf_wgs84, lat_col, lon_col)
    """
    for p in CRIME_POINT_SOURCES:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as e:
            log(f"⚠️ Okuma hatası ({p.name}): {e}")
            continue

        lat_c, lon_c = _pick_lat_lon(df)
        if not lat_c or not lon_c:
            log(f"⚠️ {p.name}: lat/lon yok, atlandı.")
            continue

        # kaba SF BBOX (opsiyonel filtre)
        df = df.copy()
        df[lat_c] = pd.to_numeric(df[lat_c], errors="coerce")
        df[lon_c] = pd.to_numeric(df[lon_c], errors="coerce")
        df = df.dropna(subset=[lat_c, lon_c])
        df = df[(df[lat_c].between(37.5, 38.2)) & (df[lon_c].between(-123.2, -122.0))]
        if df.empty:
            log(f"⚠️ {p.name}: SF BBOX içinde nokta yok, atlandı.")
            continue

        gdf = _ensure_point_gdf(df, lat_c, lon_c)
        if not gdf.empty:
            log(f"📥 Suç kaynağı: {p} (satır={len(gdf):,})")
            return df, gdf, lat_c, lon_c

    raise SystemExit(
        "❌ Suç noktası kaynağı bulunamadı. 'sf_crime_grid_full_labeled.csv' veya 'sf_crime_y.csv' gerekli."
    )


# --------------------------------------------------------- ana akış
def main() -> None:
    # 1) Suç noktaları (KOORDİNATLI kaynaklardan)
    crime_df, g_crime, lat_c, lon_c = _load_crime_points()
    # satır kimliği — geri birleştirmek için
    crime_df = crime_df.reset_index(drop=False).rename(columns={"index": "__row_id"})
    # gdf'ye aynı kimliği bindir
    if "__row_id" not in g_crime.columns and len(g_crime) == len(crime_df):
        g_crime = g_crime.copy()
        g_crime["__row_id"] = crime_df["__row_id"].values

    g_crime_m = _to_metric(g_crime)

    # 2) 911 ham nokta seçimi (öncelik _y.csv)
    src911: Optional[pd.DataFrame] = None
    picked_path: Optional[Path] = None

    for p in SF911_CANDIDATES:
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            lat911, lon911 = _pick_lat_lon(df)
            if lat911 and lon911:
                src911 = df
                picked_path = p
                log(f"📥 911 kaynağı: {p} (satır={len(df):,})")
                break
            else:
                log(f"⚠️ {p.name}: lat/lon yok, atlandı.")

    if src911 is None:
        log("❌ 911 ham nokta CSV bulunamadı veya lat/lon içermiyor.")
        sys.exit(1)

    # kaba SF BBOX filtresi (opsiyonel)
    lat911, lon911 = _pick_lat_lon(src911)
    src911 = src911.copy()
    src911[lat911] = pd.to_numeric(src911[lat911], errors="coerce")
    src911[lon911] = pd.to_numeric(src911[lon911], errors="coerce")
    src911 = src911.dropna(subset=[lat911, lon911])
    src911 = src911[
        (src911[lat911].between(37.5, 38.2)) & (src911[lon911].between(-123.2, -122.0))
    ]

    g_911 = _ensure_point_gdf(src911, lat911, lon911)
    g_911_m = _to_metric(g_911)
    log(f"🗺️ Suç noktası: {len(g_crime_m):,} | 911 noktası: {len(g_911_m):,}")

    # 3) en yakın 911 mesafesi + buffer sayıları
    dist_min = _distance_nearest(g_crime_m, g_911_m)
    g_crime_m["911_dist_min_m"] = pd.to_numeric(dist_min, errors="coerce")
    g_crime_m["911_dist_min_range"] = g_crime_m["911_dist_min_m"].apply(_bin_distance)

    for r in BUFFERS_M:
        g_crime_m[f"911_cnt_{r}m"] = _count_within(g_crime_m, g_911_m, r)

    # 4) geri DataFrame’e dön ve birleştir (__row_id ile)
    feat_cols = (
        ["__row_id", "911_dist_min_m", "911_dist_min_range"]
        + [f"911_cnt_{r}m" for r in BUFFERS_M]
    )
    feats = pd.DataFrame(g_crime_m[feat_cols].copy())
    merged = crime_df.merge(feats, on="__row_id", how="left")

    # tip/doldurma
    for c in [f"911_cnt_{r}m" for r in BUFFERS_M]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0).astype("int32")
    merged["911_dist_min_m"] = pd.to_numeric(merged["911_dist_min_m"], errors="coerce")
    merged["911_dist_min_range"] = merged["911_dist_min_range"].fillna("none")

    # 5) yaz
    out = merged.drop(columns=["__row_id"], errors="ignore")
    out.to_csv(CRIME_OUT, index=False)
    log(f"✅ Yazıldı: {CRIME_OUT} | satır={len(out):,}")

    # mini önizleme
    try:
        cols = [
            "GEOID",
            lat_c, lon_c,
            "911_cnt_300m", "911_cnt_600m", "911_cnt_900m",
            "911_dist_min_m", "911_dist_min_range",
        ]
        view_cols = [c for c in cols if c in out.columns]
        if view_cols:
            print(out[view_cols].head(5).to_string(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()

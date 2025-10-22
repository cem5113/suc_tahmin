# update_911_fr.py  — FR hattı için 911 tabanlı mekânsal özellikler (GEOID tabanlı, lat/lon gerektirmez)
# Değişiklik: sf_crime_L.csv'den koordinat beklemek yerine suç noktalarını grid (tercih) ya da event (sf_crime_y.csv) içinden okur.
# Bu sürüm yalnızca GEOID ve neighbors.csv (GEOID, neighbor) ile çalışır; metrik tamponlar (300/600/900m) komşuluk halkalarına (1/2/3 hop) yaklaşık eşlenir.

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional

import pandas as pd


# =======================================================================
# Yardımcı fonksiyonlar
# =======================================================================

def log(msg: str) -> None:
    """Basit stdout logger."""
    print(msg, file=sys.stdout, flush=True)


def pick_column_case_insensitive(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Kolon adlarını küçük harfe indirerek adaylardan birini bulur ve orijinal adını döndürür."""
    columns_lower_to_original = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        candidate_lower = candidate.lower()
        if candidate_lower in columns_lower_to_original:
            return columns_lower_to_original[candidate_lower]
    return None


def ensure_geoid_column(df: pd.DataFrame, name_hint: str = "") -> Tuple[pd.DataFrame, str]:
    """Veride GEOID kolonunu bulur; yoksa hata verir. GEOID değerlerini stringe çevirir."""
    geoid_column = pick_column_case_insensitive(df, ["GEOID", "geoid", "geoid10", "GEOID10"])
    if not geoid_column:
        raise SystemExit(f"❌ {name_hint} içinde GEOID kolonu bulunamadı.")
    df_out = df.copy()
    df_out[geoid_column] = df_out[geoid_column].astype(str)
    return df_out, geoid_column


def load_first_existing_csv(paths: List[Path], human_readable_name: str) -> Tuple[pd.DataFrame, Path]:
    """Verilen yol adaylarından mevcut olan ilk CSV'yi okur ve döndürür."""
    for path in paths:
        if path.exists():
            try:
                df = pd.read_csv(path, low_memory=False)
                log(f"📥 Yüklendi: {path} (satır={len(df):,})")
                return df, path
            except Exception as exc:
                log(f"⚠️ Okunamadı ({path}): {exc}")
    raise SystemExit(f"❌ Girdi bulunamadı: {human_readable_name}")


def load_neighbors_edge_list(candidates: List[Path], geoid_pool: Set[str]) -> Dict[str, Set[str]]:
    """
    neighbors.csv biçimi: iki sütun — GEOID, neighbor
    Kenar listesi simetrik kabul edilir (u-v varsa v-u da eklenir).
    Grid (geoid_pool) dışında kalan düğümler filtrelenir.
    """
    for path in candidates:
        if not path.exists():
            continue
        try:
            neighbors_df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            log(f"⚠️ Komşuluk dosyası okunamadı ({path}): {exc}")
            continue

        neighbors_df, geoid_col = ensure_geoid_column(neighbors_df, path.name)
        neighbor_col = pick_column_case_insensitive(neighbors_df, ["neighbor", "NEIGHBOR"])
        if not neighbor_col:
            log(f"⚠️ {path.name} içinde 'neighbor' sütunu bulunamadı.")
            continue

        neighbors_df[neighbor_col] = neighbors_df[neighbor_col].astype(str)

        # Sadece grid'te bulunan çiftleri tut
        neighbors_df = neighbors_df[
            neighbors_df[geoid_col].isin(geoid_pool) | neighbors_df[neighbor_col].isin(geoid_pool)
        ].copy()

        adjacency: Dict[str, Set[str]] = {}

        for geoid_value, neighbor_value in neighbors_df[[geoid_col, neighbor_col]].itertuples(index=False):
            if geoid_value in geoid_pool and neighbor_value in geoid_pool:
                adjacency.setdefault(geoid_value, set()).add(neighbor_value)
                adjacency.setdefault(neighbor_value, set()).add(geoid_value)

        # Grid’te olup komşuluğu olmayanlara boş set tanımla
        for geoid_value in geoid_pool:
            adjacency.setdefault(geoid_value, set())

        log(f"✅ Komşuluk yüklendi: {path} (düğüm={len(adjacency):,})")
        return adjacency

    log("ℹ️ neighbors.csv bulunamadı. Sadece aynı GEOID üzerinden hesap yapılacak (komşu halkaları boş).")
    return {geoid_value: set() for geoid_value in geoid_pool}


def rings_for_source(adjacency: Dict[str, Set[str]], source_geoid: str, max_hops: int = 3) -> List[Set[str]]:
    """
    Belirli bir GEOID için komşuluk halkalarını üretir.
    ring[0] = {source}
    ring[1] = 1 hop komşular
    ring[2] = 2 hop komşular
    ring[3] = 3 hop komşular
    """
    rings: List[Set[str]] = [set([source_geoid])]
    visited: Set[str] = set([source_geoid])
    frontier: Set[str] = set([source_geoid])
    for _ in range(1, max_hops + 1):
        next_frontier: Set[str] = set()
        for node in frontier:
            next_frontier |= adjacency.get(node, set())
        next_frontier -= visited
        rings.append(next_frontier)
        visited |= next_frontier
        frontier = next_frontier
    return rings


def nearest_hops_to_911(
    geoid_value: str,
    geoids_with_911: Set[str],
    adjacency: Dict[str, Set[str]],
    max_hops: int = 3
) -> Optional[int]:
    """Bir GEOID’in en yakın 911 içeren GEOID’e halka (hop) cinsinden uzaklığı. Yoksa None."""
    if geoid_value in geoids_with_911:
        return 0
    visited: Set[str] = set([geoid_value])
    frontier: Set[str] = set([geoid_value])
    for hop in range(1, max_hops + 1):
        next_frontier: Set[str] = set()
        for node in frontier:
            next_frontier |= adjacency.get(node, set())
        next_frontier -= visited
        if not next_frontier:
            return None
        if next_frontier & geoids_with_911:
            return hop
        visited |= next_frontier
        frontier = next_frontier
    return None


def hop_to_meters(hop_value: Optional[int], hop_to_meter_map: Dict[int, int]) -> Optional[int]:
    """Hop değerini yaklaşık metreye çevirir. Haritada yoksa lineer genişletir."""
    if hop_value is None:
        return None
    if hop_value <= 0:
        return 0
    if hop_value in hop_to_meter_map:
        return hop_to_meter_map[hop_value]
    max_key = max(hop_to_meter_map)
    return int(hop_to_meter_map[max_key] * (hop_value / max_key))


def bin_distance_from_meters(meters: Optional[int]) -> str:
    """Metreyi istenen kategori etiketine dönüştürür."""
    if meters is None:
        return "none"
    if meters <= 300:
        return "≤300m"
    if meters <= 600:
        return "300–600m"
    if meters <= 900:
        return "600–900m"
    return ">900m"


# =======================================================================
# Sabitler ve dosya yolları
# =======================================================================

SAVE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# Suç satırları için aday kaynaklar (GEOID içermesi yeterli)
CRIME_POINT_SOURCES: List[Path] = [
    Path(SAVE_DIR) / "sf_crime_grid_full_labeled.csv",  # tercih
    Path(SAVE_DIR) / "sf_crime_y.csv",                  # fallback
    Path(".")        / "sf_crime_grid_full_labeled.csv",
    Path(".")        / "sf_crime_y.csv",
]

# 911 veri adayları (öncelik event seviye _y.csv)
SF911_CANDIDATES: List[Path] = [
    Path(SAVE_DIR) / "sf_911_last_5_year_y.csv",
    Path(".")        / "sf_911_last_5_year_y.csv",
    Path(SAVE_DIR) / "sf_911_last_5_year.csv",
    Path(".")        / "sf_911_last_5_year.csv",
]

# Komşuluk (iki sütun: GEOID, neighbor)
NEIGHBOR_FILE_CANDIDATES: List[Path] = [
    Path(SAVE_DIR) / "neighbors.csv",
    Path(".")        / "neighbors.csv",
    # alternatif isimler
    Path(SAVE_DIR) / "geoid_neighbors.csv",
    Path(".")        / "geoid_neighbors.csv",
]

# Çıktı
CRIME_OUT_PATH: Path = Path(SAVE_DIR) / "fr_crime_01.csv"

# Hop → metre yaklaşık eşlemesi (grid ölçeğinize göre güncellenebilir)
HOP_TO_METER_MAP: Dict[int, int] = {1: 300, 2: 600, 3: 900}
MAX_HOPS: int = max(HOP_TO_METER_MAP)


# =======================================================================
# Ana akış
# =======================================================================

def main() -> None:
    # 1) Suç satırlarını yükle (GEOID şart)
    crime_df_raw, crime_path = load_first_existing_csv(CRIME_POINT_SOURCES, "crime grid/event")
    crime_df, crime_geoid_col = ensure_geoid_column(crime_df_raw, crime_path.name)

    # Geri join işlemi için satır kimliği ekle
    crime_df = crime_df.reset_index(drop=False).rename(columns={"index": "__row_id"})
    grid_geoids: Set[str] = set(crime_df[crime_geoid_col].unique().tolist())
    log(f"🧩 GEOID (grid): {len(grid_geoids):,}")

    # 2) 911 verisini yükle ve grid'te olmayanları filtrele
    df911_raw, path911 = load_first_existing_csv(SF911_CANDIDATES, "911 data")
    df911, geoid_911_col = ensure_geoid_column(df911_raw, path911.name)
    df911 = df911[df911[geoid_911_col].isin(grid_geoids)].copy()

    # GEOID başına toplam 911 olayı say
    counts_by_geoid = (
        df911.groupby(geoid_911_col).size().rename("cnt_911").to_frame().reset_index()
    )
    geoid_to_911_count: Dict[str, int] = dict(
        zip(counts_by_geoid[geoid_911_col], counts_by_geoid["cnt_911"])
    )
    geoids_with_any_911: Set[str] = {k for k, v in geoid_to_911_count.items() if v > 0}
    log(f"☎️ 911 içeren GEOID sayısı: {len(geoids_with_any_911):,}")

    # 3) Komşuluk grafiğini yükle
    adjacency: Dict[str, Set[str]] = load_neighbors_edge_list(NEIGHBOR_FILE_CANDIDATES, grid_geoids)

    # 4) Her GEOID için halka kümelerini hazırla
    rings_cache: Dict[str, List[Set[str]]] = {}
    for geoid_value in grid_geoids:
        rings_cache[geoid_value] = rings_for_source(adjacency, geoid_value, max_hops=MAX_HOPS)

    # 5) Halkalara göre kümülatif kapsam kümelerini ve özellikleri hesapla
    feature_rows: List[Dict[str, object]] = []
    for geoid_value in grid_geoids:
        rings_for_this: List[Set[str]] = rings_cache[geoid_value]  # [ring0, ring1, ring2, ring3]
        cumulative_sets: List[Set[str]] = []
        accumulator_set: Set[str] = set()
        for ring_set in rings_for_this[: MAX_HOPS + 1]:
            accumulator_set |= ring_set
            cumulative_sets.append(set(accumulator_set))

        # Hop→metre eşlemesine göre kümülatif 911 sayıları
        feature_counts_for_geo: Dict[str, int] = {}
        for hop_value, meter_value in HOP_TO_METER_MAP.items():
            if hop_value < len(cumulative_sets):
                scope_geoids = cumulative_sets[hop_value]
            else:
                scope_geoids = cumulative_sets[-1]
            total_911_in_scope: int = 0
            for scope_geoid in scope_geoids:
                total_911_in_scope += geoid_to_911_count.get(scope_geoid, 0)
            feature_counts_for_geo[f"911_cnt_{meter_value}m"] = int(total_911_in_scope)

        # En yakın 911 içeren GEOID’e hop cinsinden mesafe ve yaklaşık metre
        nearest_hop_value: Optional[int] = nearest_hops_to_911(
            geoid_value, geoids_with_any_911, adjacency, max_hops=MAX_HOPS
        )
        nearest_meters_value: Optional[int] = hop_to_meters(nearest_hop_value, HOP_TO_METER_MAP)
        nearest_bin_label: str = bin_distance_from_meters(nearest_meters_value)

        feature_rows.append(
            {
                crime_geoid_col: geoid_value,
                **feature_counts_for_geo,
                "911_dist_min_m": nearest_meters_value if nearest_meters_value is not None else pd.NA,
                "911_dist_min_range": nearest_bin_label,
            }
        )

    geoid_level_features = pd.DataFrame(feature_rows)

    # 6) Suç satırlarına GEOID ile join et
    merged_output = crime_df.merge(geoid_level_features, on=crime_geoid_col, how="left")

    # Tip düzeltmeleri ve doldurma
    for meters_value in HOP_TO_METER_MAP.values():
        col_name = f"911_cnt_{meters_value}m"
        if col_name in merged_output.columns:
            merged_output[col_name] = (
                pd.to_numeric(merged_output[col_name], errors="coerce")
                .fillna(0)
                .astype("int32")
            )

    if "911_dist_min_m" in merged_output.columns:
        merged_output["911_dist_min_m"] = pd.to_numeric(
            merged_output["911_dist_min_m"], errors="coerce"
        )

    if "911_dist_min_range" in merged_output.columns:
        merged_output["911_dist_min_range"] = merged_output["911_dist_min_range"].fillna("none")

    # 7) Yaz
    final_out = merged_output.drop(columns=["__row_id"], errors="ignore")
    final_out.to_csv(CRIME_OUT_PATH, index=False)
    log(f"✅ Yazıldı: {CRIME_OUT_PATH} | satır={len(final_out):,}")

    # 8) Mini önizleme
    preview_columns_order: List[str] = [
        crime_geoid_col,
        "911_cnt_300m",
        "911_cnt_600m",
        "911_cnt_900m",
        "911_dist_min_m",
        "911_dist_min_range",
    ]
    columns_to_view = [c for c in preview_columns_order if c in final_out.columns]
    if columns_to_view:
        try:
            print(final_out[columns_to_view].head(5).to_string(index=False))
        except Exception:
            pass


if __name__ == "__main__":
    main()

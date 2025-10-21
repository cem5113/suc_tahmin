from __future__ import annotations
from typing import Optional, Union, Dict, List, Tuple, Any

import streamlit as st
import pandas as pd
import requests
import re    
import os, json, subprocess, sys
from pathlib import Path
import io, zipfile
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np  # ▶️ eklendi: aşağıda np.nanmean vb. kullanılıyor

# --- Forensic rapor yardımcı (varsa import et, yoksa stub kullan) ---
try:
    from scripts.forensic_report import build_forensic_report
except Exception:
    def build_forensic_report(**kwargs):
        return None

st.set_page_config(page_title="Veri Güncelleme", layout="wide")

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()
DATA_DIR = ROOT / "crime_prediction_data"
SCRIPTS_DIR = ROOT / "scripts"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
SEARCH_DIRS = [SCRIPTS_DIR, ROOT]

PIPELINE = [
    {"name": "update_crime",      "alts": ["build_crime_grid", "crime_grid_build"]},
    {"name": "update_911",        "alts": ["enrich_911"]},
    {"name": "update_311",        "alts": ["enrich_311"]},
    {"name": "update_population", "alts": ["enrich_population"]},
    {"name": "update_bus",        "alts": ["enrich_bus"]},
    {"name": "update_train",      "alts": ["enrich_train"]},
    {"name": "update_poi",        "alts": ["pipeline_make_sf_crime_06", "app_poi_to_06", "enrich_poi"]},
    {"name": "update_police_gov", "alts": ["enrich_police_gov_06_to_07", "enrich_police_gov", "enrich_police"]},
    {"name": "update_weather",    "alts": ["enrich_weather"]},
]

# --- AĞIR BAĞIMLILIKLAR İÇİN LAZY IMPORT ---
def _load_ml_deps():
    try:
        import shap
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.inspection import PartialDependenceDisplay
        from lightgbm import LGBMClassifier, LGBMRegressor
        from lime.lime_tabular import LimeTabularExplainer
        return {
            "np": np,  # global np'i döndürüyoruz
            "shap": shap,
            "TimeSeriesSplit": TimeSeriesSplit,
            "roc_auc_score": roc_auc_score,
            "brier_score_loss": brier_score_loss,
            "mean_absolute_error": mean_absolute_error,
            "CalibratedClassifierCV": CalibratedClassifierCV,
            "PartialDependenceDisplay": PartialDependenceDisplay,
            "LGBMClassifier": LGBMClassifier,
            "LGBMRegressor": LGBMRegressor,
            "LimeTabularExplainer": LimeTabularExplainer,
        }
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", "bir paket")
        st.error(
            f"🧱 Gerekli paket eksik: **{missing}**. "
            "Lütfen sol menüden **'0) Gereklilikleri yükle'** düğmesini kullanın "
            "ve kurulumdan sonra **Rerun** yapın."
        )
        st.stop()

def pick_url(key: str, default: str) -> str:
    # Öncelik: 1) st.secrets  2) ENV  3) default
    try:
        if key in st.secrets and st.secrets[key]:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)

CRIME_CSV_LATEST = pick_url(
    "CRIME_CSV_URL",
    "https://github.com/cem5113/crime_prediction_data/releases/latest/download/sf_crime_y.csv",
)

RAW_911_URL = pick_url(
    "RAW_911_URL",
    "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year_y.csv",
)

SF311_URL = pick_url(
    "SF311_URL",
    "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.2/sf_311_last_5_years_y.csv",
)

# CSV-ONLY: Nüfus verisi yerel dosyadan okunacak
DEFAULT_POP_CSV = str((Path(os.environ.get("CRIME_DATA_DIR", "crime_prediction_data")) / "sf_population.csv").resolve())
POPULATION_PATH = pick_url("POPULATION_PATH", DEFAULT_POP_CSV)

# Güvenlik: URL verilirse reddet (CSV-only mod)
if re.match(r"^https?://", str(POPULATION_PATH), flags=re.I):
    # URL kabul etmiyoruz; varsayılan yerel yola düş
    POPULATION_PATH = DEFAULT_POP_CSV

os.environ["POPULATION_PATH"] = str(POPULATION_PATH)

# ⬇️ 911 artımlı çekim için API ayarları
SF911_API_URL       = pick_url("SF911_API_URL", "https://data.sfgov.org/resource/2zdj-bwza.json")
SF911_AGENCY_FILTER = pick_url("SF911_AGENCY_FILTER", "agency like '%Police%'")  # boş string verirsen filtre kalkar
SF911_API_TOKEN     = pick_url("SF911_API_TOKEN", "") 

# Çocuk süreçlerin (update_*.py) de aynı değerleri görmesi için ENV’e yaz
os.environ["CRIME_CSV_URL"] = CRIME_CSV_LATEST
os.environ["RAW_911_URL"]   = RAW_911_URL
os.environ["SF311_URL"]     = SF311_URL
os.environ["GEOID_LEN"] = os.environ.get("GEOID_LEN", "11")

GEOID_LEN = int(os.environ.get("GEOID_LEN", "11"))

def _norm_geoid(s: pd.Series, L: int = GEOID_LEN) -> pd.Series:
    return (
        s.astype(str)
         .str.extract(r"(\d+)", expand=False)  # sadece rakamlar
         .str[:L]                               # hedef uzunluğa kes (örn. 11)
         .str.zfill(L)                          # baştaki sıfırları doldur
    )

os.environ["SF911_API_URL"]     = SF911_API_URL
os.environ["SF911_AGENCY_FILTER"] = SF911_AGENCY_FILTER
if SF911_API_TOKEN:
    os.environ["SF911_API_TOKEN"] = SF911_API_TOKEN
SOCS_APP_TOKEN = st.secrets.get("SOCS_APP_TOKEN", os.environ.get("SOCS_APP_TOKEN", ""))
if SOCS_APP_TOKEN:
    os.environ["SOCS_APP_TOKEN"] = SOCS_APP_TOKEN  # alt katman scriptler için

# LATEST veya 2022/2023 gibi belirli yıl
os.environ["ACS_YEAR"] = st.secrets.get("ACS_YEAR", os.environ.get("ACS_YEAR", "LATEST"))

# Virgülle filtre (boş bırak = tüm kategoriler). Örn: "population,median_income,education"
os.environ["DEMOG_WHITELIST"] = st.secrets.get(
    "DEMOG_WHITELIST",
    os.environ.get("DEMOG_WHITELIST", "")
)

# --- GitHub Actions entegrasyonu (manual tetik & artifact indirme) ---
GITHUB_REPO = os.environ.get("GITHUB_REPO", "cem5113/crime_prediction_data")   # owner/repo
GITHUB_WORKFLOW = os.environ.get("GITHUB_WORKFLOW", "full_pipeline.yml")       # .github/workflows/...

def _gh_headers():
    token = st.secrets.get("GH_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        raise RuntimeError("GH_TOKEN gerekli (Streamlit secrets veya env).")
    return {"Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json"}

def fetch_file_from_latest_artifact(pick_names: list[str], artifact_name="sf-crime-pipeline-output") -> bytes | None:
    """Son başarılı run'dan artifact içindeki pick_names listesinde geçen ilk dosyayı döndürür (bytes)."""
    runs_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs?per_page=20"
    runs = requests.get(runs_url, headers=_gh_headers(), timeout=30).json()
    run_ids = [r["id"] for r in runs.get("workflow_runs", []) if r.get("conclusion") == "success"]
    for rid in run_ids:
        arts_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{rid}/artifacts"
        arts = requests.get(arts_url, headers=_gh_headers(), timeout=30).json().get("artifacts", [])
        for a in arts:
            if a.get("name") == artifact_name and not a.get("expired", False):
                dl = requests.get(a["archive_download_url"], headers=_gh_headers(), timeout=60)
                zf = zipfile.ZipFile(io.BytesIO(dl.content))
                names = zf.namelist()
                # sonuna/suffix'e bakarak eşle
                for pick in pick_names:
                    # hem tam path hem de sadece dosya adı için dene
                    candidates = [pick, f"crime_prediction_data/{pick}"]
                    for c in candidates:
                        if c in names:
                            return zf.read(c)
                # Suffix eşleşmesi (daha toleranslı)
                for n in names:
                    if any(n.endswith(p) for p in pick_names):
                        return zf.read(n)
    return None

def dispatch_workflow(persist: str = "artifact", force: bool = True) -> dict:
    """Actions workflow’u tetikle (persist: artifact|commit|none, force: 07:00 kapısını bypass)."""
    import json as _json
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{GITHUB_WORKFLOW}/dispatches"
    payload = {
        "ref": "main",
        "inputs": {
            "persist": persist,
            "force": "true" if force else "false"  # 🔴 booleanlar string olmalı
        }
    }
    r = requests.post(url, headers=_gh_headers(), data=_json.dumps(payload), timeout=30)
    return {"ok": r.status_code in (204, 201), "status": r.status_code, "text": r.text}

def _get_last_run_by_workflow():
    """full_pipeline.yml için en son run (1 adet)"""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{GITHUB_WORKFLOW}/runs?per_page=1"
    r = requests.get(url, headers=_gh_headers(), timeout=30)
    if r.status_code != 200:
        return None, r.status_code, r.text
    arr = r.json().get("workflow_runs", [])
    return (arr[0] if arr else None), 200, ""

def _render_last_run_status(container):
    if not (st.secrets.get("GH_TOKEN") or os.environ.get("GH_TOKEN")):
        container.info("GH_TOKEN yok; GitHub durumunu okuyamıyorum.")
        return
    try:
        run, code, msg = _get_last_run_by_workflow()
        if not run:
            container.info("Bu workflow için run bulunamadı.")
            return
        status = run.get("status")         # queued | in_progress | completed
        concl  = run.get("conclusion") or "-"  # success | failure | cancelled | -
        started = run.get("run_started_at")
        html_url = run.get("html_url")
        container.markdown(
            f"**Son koşum:** `{status}` / `{concl}` · başlama: `{started}`  ·  [GitHub’da aç]({html_url})"
        )
    except Exception as e:
        container.warning(f"Durum okunamadı: {e}")

def fetch_latest_artifact_df() -> Optional[pd.DataFrame]:
    """Son başarılı run’daki artifact içinden sf_crime_08.csv’yi getir."""
    try:
        runs_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs?per_page=20"
        runs = requests.get(runs_url, headers=_gh_headers(), timeout=30).json()
        run_ids = [r["id"] for r in runs.get("workflow_runs", []) if r.get("conclusion") == "success"]
        for rid in run_ids:
            arts_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{rid}/artifacts"
            arts = requests.get(arts_url, headers=_gh_headers(), timeout=30).json().get("artifacts", [])
            for a in arts:
                if a.get("name") == "sf-crime-pipeline-output" and not a.get("expired", False):
                    dl = requests.get(a["archive_download_url"], headers=_gh_headers(), timeout=60)
                    zf = zipfile.ZipFile(io.BytesIO(dl.content))
                    for pick in ("crime_prediction_data/sf_crime_08.csv", "sf_crime_08.csv"):
                        if pick in zf.namelist():
                            with zf.open(pick) as f:
                                df = pd.read_csv(f, low_memory=False)
                                if "date" in df.columns:
                                    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
                                elif "datetime" in df.columns:
                                    df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date
                                return df
        return None
    except Exception as e:
        st.warning(f"Artifact indirilemedi: {e}")
        return None

def load_sf_crime_08(local_path: Path) -> Optional[pd.DataFrame]:
    """Önce yerel dosyayı dene; yoksa artifact’tan çek. Ardından crime_mix varsa grid ile merge et."""
    def _normalize_date_cols(df: pd.DataFrame) -> pd.DataFrame:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        elif "datetime" in df.columns and "date" not in df.columns:
            df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date
        return df

    # 1) Kaynağı yükle
    df: Optional[pd.DataFrame] = None
    try:
        if local_path.exists():
            df = pd.read_csv(local_path, low_memory=False)
            df = _normalize_date_cols(df)
    except Exception as e:
        st.warning(f"Yerel sf_crime_08.csv okunamadı: {e}")

    if df is None:
        df = fetch_latest_artifact_df()
        if df is None:
            return None
        df = _normalize_date_cols(df)

    # 2) Opsiyonel: crime_mix merge (grid)
    try:
        _out_dir  = local_path.parent
        _grid_path = _out_dir / "sf_crime_grid_full_labeled.csv"
        if _grid_path.exists():
            grid = pd.read_csv(_grid_path, dtype={"GEOID": str}, low_memory=False)

            keys = ["GEOID", "season", "day_of_week", "event_hour"]
            if set(keys).issubset(grid.columns) and "crime_mix" in grid.columns and set(keys).issubset(df.columns):

                df["GEOID"]   = _norm_geoid(df["GEOID"])
                grid["GEOID"] = _norm_geoid(grid["GEOID"])

                for c in ["day_of_week", "event_hour"]:
                    df[c]   = pd.to_numeric(df[c], errors="coerce").astype("Int64")
                    grid[c] = pd.to_numeric(grid[c], errors="coerce").astype("Int64")

                merged = df.merge(
                    grid[keys + ["crime_mix"]],
                    on=keys, how="left", suffixes=("", "_grid"), validate="many_to_one"
                )

                # var ise boşları doldur
                if "crime_mix_grid" in merged.columns:
                    if "crime_mix" not in merged.columns:
                        merged["crime_mix"] = ""
                    merged["crime_mix"] = merged["crime_mix"].astype(str)
                    merged["crime_mix"] = merged["crime_mix"].where(
                        merged["crime_mix"].str.len() > 0,
                        merged["crime_mix_grid"].fillna("")
                    )
                    merged = merged.drop(columns=["crime_mix_grid"], errors="ignore")
                df = merged
        else:
            print(f"crime_mix merge atlandı: grid bulunamadı → {_grid_path}")
    except Exception as _e:
        print(f"crime_mix merge uyarısı: {_e}")

    return df

# --- Rare class grouping helper ---
def _group_rare_labels(
    df: pd.DataFrame,
    col: str,
    min_prop: Optional[float] = None,
    min_count: Optional[int] = None,
    other_label: str = "Other",
    out_stats_path: Optional[Path] = None,
) -> pd.Series:
    """
    col içindeki nadir etiketleri Other altında toplar.
    Öncelik: min_prop (oran) -> min_count (mutlak).
    """
    if col not in df.columns:
        return pd.Series([None] * len(df), index=df.index)

    s = df[col].astype(str).str.strip()
    total = len(s)
    vc = s.value_counts(dropna=False)

    # eşik seçimi: env > param > varsayılan
    env_prop = os.environ.get("RARE_MIN_PROP")
    env_count = os.environ.get("RARE_MIN_COUNT")
    if min_prop is None and env_prop:
        try: min_prop = float(env_prop)
        except: pass
    if min_count is None and env_count:
        try: min_count = int(env_count)
        except: pass

    # varsayılan eşikler: %1 veya 200
    if min_prop is None and min_count is None:
        min_prop, min_count = 0.01, 200

    rare_mask = pd.Series(False, index=vc.index)
    if min_prop is not None:
        rare_mask |= (vc / max(total, 1)) < float(min_prop)
    if min_count is not None:
        rare_mask |= vc < int(min_count)

    rare_values = set(vc[rare_mask].index)

    grouped = s.where(~s.isin(rare_values), other_label)

    # İsteğe bağlı: özet kaydet
    if out_stats_path is not None:
        out_stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_df = pd.DataFrame({
            col: vc.index,
            "count": vc.values,
            "prop": vc.values / max(total, 1),
            "is_rare": vc.index.map(lambda v: v in rare_values)
        })
        try:
            stats_df.to_csv(out_stats_path, index=False)
        except Exception:
            pass

    return grouped

def clean_and_save_crime_09(input_obj="sf_crime_08.csv", output_path="sf_crime_09.csv"):
    # input_obj hem DataFrame hem de dosya yolu olabilir
    if isinstance(input_obj, pd.DataFrame):
        df = input_obj.copy()
    else:
        df = pd.read_csv(input_obj, dtype={"GEOID": str})
    if "GEOID" in df.columns:
        # sadece rakamları al ve GEOID_LEN’e (env) göre pad et
        target_len = int(os.environ.get("GEOID_LEN", "11"))
        df["GEOID"] = (
            df["GEOID"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .str.zfill(target_len)
        )
    if "category" in df.columns:
        df["category"] = df["category"].astype(str).str.strip().str.title()

    # --- Rare class grouping (category / subcategory) ---
    try:
        out_dir = Path(output_path).parent if isinstance(output_path, str) else Path(".")
        if "category" in df.columns:
            df["category_grouped"] = _group_rare_labels(
                df, "category",
                # Env ile override edilebilir; yoksa varsayılan %1 veya 200
                min_prop=None, min_count=None,
                other_label="Other",
                out_stats_path=out_dir / "rare_stats_category.csv"
            )

        if "subcategory" in df.columns:
            # subcategory varsa onu da grupla (daha agresif olabilir)
            df["subcategory"] = df["subcategory"].astype(str).str.strip().str.title()
            df["subcategory_grouped"] = _group_rare_labels(
                df, "subcategory",
                min_prop=None, min_count=None,
                other_label="Other",
                out_stats_path=out_dir / "rare_stats_subcategory.csv"
            )

        # Streamlit log (varsa)
        try:
            st.caption("🔎 Rare grouping uygulandı (category/subcategory). İstatistikler CSV olarak kaydedildi.")
        except Exception:
            pass
    except Exception as _e:
        try:
            st.warning(f"Rare grouping atlandı: {str(_e)}")
        except Exception:
            print(f"Rare grouping atlandı: {_e}")

    # --- Yardımcı dönüştürücüler
    def to_int(df, col, default=0):
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(default)
                .round()
                .astype("Int64")  # nullable int; isterseniz .astype(int) yapabilirsiniz
            )

    def to_float(df, col, default=0.0):
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(default)
                .astype(float)
            )

    # --- 1) Sayaç kolonları (int gibi)
    int_count_cols = [
        "crime_count",
        "911_request_count_hour_range",
        "911_request_count_daily(before_24_hours)",
        "311_request_count",
        "bus_stop_count",
        "train_stop_count",
        "poi_total_count",
    ]
    for c in int_count_cols:
        to_int(df, c, default=0)

    # --- 1-b) Risk skoru (float kalsın)
    to_float(df, "poi_risk_score", default=0.0)

    # --- 2) Binary kolonlar (0/1)
    def to_binary(df, col):
        if col in df.columns:
            m = {
                # TRUE varyantları
                "true": 1, "t": 1, "yes": 1, "y": 1, "1": 1, "evet": 1,
                # FALSE varyantları
                "false": 0, "f": 0, "no": 0, "n": 0, "0": 0, "hayır": 0, "hayir": 0
            }
            # bool değerleri yakala -> sonra string normalize et
            s = df[col].replace({True: 1, False: 0})
            s = s.astype(str).str.strip().str.lower().map(m)
            df[col] = pd.to_numeric(s, errors="coerce").fillna(0).astype("Int64")
    
    # --- 2) Binary kolonlar (0/1)
    for c in ["is_near_police", "is_near_government"]:
        to_binary(df, c)

    # --- 3) Mesafe kolonları (float; NaN -> 9999)
    for c in ["distance_to_bus", "distance_to_train", "distance_to_police", "distance_to_government_building"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(9999.0).astype(float)

    # --- 4) Range kolonları (int kategoriler)
    for c in ["bus_stop_count_range", "train_stop_count_range", "poi_total_count_range", "poi_risk_score_range"]:
        to_int(df, c, default=0)

    for c in ["distance_to_bus_range", "distance_to_train_range", "distance_to_police_range", "distance_to_government_building_range"]:
        if c in df.columns:
            # max kategoriye doldur (yoksa 3)
            s = pd.to_numeric(df[c], errors="coerce")
            max_cat = int(s.max(skipna=True)) if pd.notna(s.max(skipna=True)) else 3
            df[c] = s.fillna(max_cat).round().astype("Int64")

    # --- 5) Nüfus (median)
    if "population" in df.columns:
        df["population"] = pd.to_numeric(df["population"], errors="coerce")
        median_pop = df["population"].median(skipna=True)
        if pd.isna(median_pop):
            median_pop = 0
        df["population"] = df["population"].fillna(median_pop)

    # --- 6) POI dominant type (NaN -> "None")
    if "poi_dominant_type" in df.columns:
        df["poi_dominant_type"] = df["poi_dominant_type"].fillna("None").astype(str)

    # --- 7) Tarih alanı normalize (varsa)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    elif "datetime" in df.columns and "date" not in df.columns:
        df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date

    # --- 7.5) Near-repeat: son 7/14 gün aynı tür olay sayısı (GEOID × category_grouped)
    try:
        if {"date","GEOID"}.issubset(df.columns):
            # kategori alanı (grupladıysan onu kullan; yoksa 'category' veya 'subcategory')
            cat_col = "category_grouped" if "category_grouped" in df.columns else (
                      "subcategory_grouped" if "subcategory_grouped" in df.columns else
                      ("category" if "category" in df.columns else None))
            if cat_col:
                tmp = df[["date","GEOID",cat_col,"crime_count"]].copy()
                tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.date
                # günlük agg
                g = (tmp
                     .groupby(["GEOID",cat_col,"date"], as_index=False)["crime_count"]
                     .sum())
                # tam tarih eksenine oturt
                g["date"] = pd.to_datetime(g["date"])
                g = g.sort_values(["GEOID",cat_col,"date"])
                # her grup için yuvarlanan pencere (geçmiş 7/14 gün)
                def _roll_counts(x):
                    x = x.set_index("date").asfreq("D", fill_value=0)
                    x["nr_7d"]  = x["crime_count"].rolling("7D").sum().shift(1)   # önceki 7 gün
                    x["nr_14d"] = x["crime_count"].rolling("14D").sum().shift(1)  # önceki 14 gün
                    return x.reset_index()

                g2 = (g.groupby(["GEOID", cat_col])
                        .apply(_roll_counts)
                        .reset_index(level=[0,1])
                        .reset_index(drop=True))
                g2["date"] = g2["date"].dt.date

                df = df.merge(
                    g2[["GEOID",cat_col,"date","nr_7d","nr_14d"]],
                    on=["GEOID",cat_col,"date"], how="left"
                )
                for c in ["nr_7d","nr_14d"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)
    except Exception as _e:
        print(f"near-repeat uyarı: {_e}")

    # --- 7.7) Dışsal değişkenleri tür-eşlemeli ekleme (örnek şablon)
    try:
        # 1) Hava: df'de halihazırda 'temp','wind_speed','precip' vs varsa doğrudan kullanılır.
        # 2) Etkinlik & Toplu taşıma: olayla aynı GEOID ve tarihe düşen sayaçlar/mesafeler zaten varsa,
        #    tür eşleme ile kolon seçimini özelleştirebilirsin.
        ext_map_path = Path(DATA_DIR) / "crime_type_externals_map.json"  # opsiyonel
        if ext_map_path.exists():
            with open(ext_map_path, "r", encoding="utf-8") as f:
                type_map = json.load(f)  # {"Theft": ["precip","bus_ridership"], ...}
            key_col = "category_grouped" if "category_grouped" in df.columns else (
                      "category" if "category" in df.columns else None)
            if key_col:
                # her satır için o türe ait kolonlar -> basit toplam/ortalama ile tek bir skor
                def _ext_score(row):
                    cols = type_map.get(str(row[key_col]), [])
                    vals = []
                    for c in cols:
                        if c in df.columns:
                            try:
                                vals.append(float(row.get(c, 0)))
                            except:
                                pass
                    return float(np.nanmean(vals)) if len(vals)>0 else np.nan
                df["externals_type_score"] = df.apply(_ext_score, axis=1)
                df["externals_type_score"] = df["externals_type_score"].fillna(0.0).astype(float)
    except Exception as _e:
        print(f"dışsal değişken uyarı: {_e}")

    # (Kaydın hemen öncesi/sonrası) hızlı vitrin:
    preview_cols = [c for c in ["nr_7d","nr_14d","nei_7d_sum","externals_type_score"] if c in df.columns]
    if preview_cols:
        st.caption("🧩 Yeni mekânsal-zamansal özellikler (ilk 20 satır):")
        st.dataframe(df[preview_cols].head(20))

    # --- 8) Kaydet
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ {output_path} kaydedildi. Satır sayısı: {len(df)}")
    return df

# Streamlit UI

st.title("📦 Günlük Suç Tahmin Zenginleştirme ve Güncelleme Paneli")

with st.sidebar:
    st.markdown("### GitHub Actions")

    VARIANTS = ["default", "fr"]
    variant = st.selectbox(
        "Pipeline varyantı",
        VARIANTS,
        index=0,
        help="default: update_*.py • fr: update_*.fr.py / update_*_fr.py"
    )
    os.environ["PIPELINE_VARIANT"] = variant

    # Çıktı saklama modu
    persist = st.selectbox(
        "Çıktıyı saklama modu",
        ["artifact", "commit", "none"],
        index=0,
        help="artifact: repo’yu bozmadan sakla • commit: repo’ya yaz • none: sadece log"
    )

    # 07:00 kapısını bypass et (manuel tetiklemelerde önerilir)
    force_bypass = st.checkbox(
        "07:00 kapısını yok say (force)",
        value=True,
        help="İşaretli ise saat filtresi devre dışı kalır ve pipeline her saatte çalışır."
    )

    # Son run durum kutusu
    status_box = st.empty()
    _render_last_run_status(status_box)

    # Butonlar
    col_run, col_refresh = st.columns(2)
    with col_run:
        if st.button("🚀 Full pipeline’ı Actions’ta çalıştır"):
            if not (st.secrets.get("GH_TOKEN") or os.environ.get("GH_TOKEN")):
                st.error("GH_TOKEN tanımlı değil (Streamlit secrets veya env).")
            else:
                try:
                    r = dispatch_workflow(persist=persist, force=force_bypass)
                    if r["ok"]:
                        st.success(f"Workflow tetiklendi (persist={persist}, force={force_bypass}). Runs’ı kontrol et.")
                    else:
                        st.error(f"Tetikleme başarısız: {r['status']} {r['text']}")
                except Exception as e:
                    st.error(f"Hata: {e}")

    with col_refresh:
        if st.button("📡 Son durumu yenile"):
            _render_last_run_status(status_box)
            
    with st.sidebar.expander("ACS Ayarları (Demografi)"):
        # Varsayılanlar (ENV > mevcut değer > fallback)
        acs_year_default = os.environ.get("ACS_YEAR", "LATEST")
        whitelist_default = os.environ.get("DEMOG_WHITELIST", "")
        level_default = os.environ.get("CENSUS_GEO_LEVEL", "auto")
    
        # 1) ACS yılı
        acs_year_in = st.text_input(
            label="ACS_YEAR (LATEST veya YYYY)",
            value=str(acs_year_default or "LATEST"),
            key="acs_year_in",
            help="5-year ACS için en son yılı kullanmak genelde uygundur."
        )
    
        # 2) Demografi whitelist
        whitelist_in = st.text_input(
            label="DEMOG_WHITELIST (virgüllü; boş = hepsi)",
            value=str(whitelist_default or ""),
            key="demog_whitelist_in",
            help='Örn: "population,median_income,education". Metin eşleşmesiyle filtreler.'
        )
    
        # 3) GEO seviye seçimi
        levels = ["auto", "tract", "blockgroup", "block"]
        try:
            idx = levels.index(level_default) if level_default in levels else 0
        except Exception:
            idx = 0
        level_in = st.selectbox(
            "CENSUS_GEO_LEVEL",
            levels,
            index=idx,
            key="census_geo_level_in",
            help="Nüfus GEOID eşleşme seviyesi. `auto` çoğu durumda yeterlidir."
        )
        os.environ["CENSUS_GEO_LEVEL"] = level_in
    
        # 4) Nüfus CSV yolu (YEREL dosya; URL reddedilir)
        pop_default = os.environ.get("POPULATION_PATH", str(POPULATION_PATH))
        pop_url_in = st.text_input(
            label="POPULATION_PATH (YEREL CSV YOLU)",
            value=str(pop_default or ""),
            key="population_path_in",
            help="Örn: crime_prediction_data/sf_population.csv (URL kabul edilmez)."
        )
    
        # ENV’e yaz – doğrulamalar
        # ACS_YEAR: 'LATEST' veya 4 haneli yıl
        _v = str(acs_year_in).strip()
        if _v.upper() == "LATEST":
            os.environ["ACS_YEAR"] = "LATEST"
        else:
            _digits = re.sub(r"\D", "", _v)
            os.environ["ACS_YEAR"] = _digits if len(_digits) == 4 else "LATEST"
    
        os.environ["DEMOG_WHITELIST"] = str(whitelist_in or "")
    
        if re.match(r"^https?://", str(pop_url_in), flags=re.I):
            st.error("CSV-only mod: URL kabul edilmez. Yerel bir CSV yolu girin.")
        else:
            os.environ["POPULATION_PATH"] = pop_url_in or str(POPULATION_PATH)

def _mask_token(u: str) -> str:
    try:
        return re.sub(r'(\$\$app_token=)[^&]+', r'\1•••', str(u))
    except:
        return str(u)

# -----------------------------------------------------------------------------
# İndirilebilir kaynaklar (önizleme için)
# -----------------------------------------------------------------------------
DOWNLOADS = {
    "Suç Taban CSV (Release latest)": {
        "url": CRIME_CSV_LATEST,
        "path": str(DATA_DIR / "sf_crime_y.csv"),
    },
    "Tahmin Grid Verisi (GEOID × Zaman + Y_label)": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_crime_grid_full_labeled.csv",
        "path": str(DATA_DIR / "sf_crime_grid_full_labeled.csv"),
        "allow_artifact": True,
        "artifact_picks": ["sf_crime_grid_full_labeled.csv"],
    },
    "911 Çağrıları (özet)": {
        "url": RAW_911_URL,
        "path": str(DATA_DIR / "sf_911_last_5_year_y.csv"),
    },
    "311 Çağrıları (özet)": {
        "url": SF311_URL,
        "path": str(DATA_DIR / "sf_311_last_5_years_y.csv"),
    },
    "Otobüs Durakları": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_bus_stops_with_geoid.csv",
        "path": str(DATA_DIR / "sf_bus_stops_with_geoid.csv"),
    },
    "Tren Durakları": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_train_stops_with_geoid.csv",
        "path": str(DATA_DIR / "sf_train_stops_with_geoid.csv"),
    },
    "POI GeoJSON": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_pois.geojson",
        "path": str(DATA_DIR / "sf_pois.geojson"),
        "is_json": True,
    },
    "Nüfus Verisi": {
        "url": "",  # indirme yok
        "path": str(DATA_DIR / "sf_population.csv"),
        "local_src": str(POPULATION_PATH),   # buradan KOPYALA
        "is_local_csv": True,                # işaret
    },
    "POI Risk Skorları": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/risky_pois_dynamic.json",
        "path": str(DATA_DIR / "risky_pois_dynamic.json"),
        "is_json": True,
    },
    "Polis İstasyonları": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_police_stations.csv",
        "path": str(DATA_DIR / "sf_police_stations.csv"),
    },
    "Devlet Binaları": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_government_buildings.csv",
        "path": str(DATA_DIR / "sf_government_buildings.csv"),
    },
    "Hava Durumu": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_weather_5years_y.csv",
        "path": str(DATA_DIR / "sf_weather_5years_y.csv"),
    },
}

def _human_bytes(n: int) -> str:
    if n is None: return "-"
    step = 1024.0
    for u in ["B","KB","MB","GB","TB"]:
        if n < step: return f"{n:.0f} {u}" if u=="B" else f"{n:.1f} {u}"
        n /= step
    return f"{n:.1f} PB"

def _fmt_dt(ts: Optional[float]) -> str:
    if ts is None: return "-"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def _age_str(ts: Optional[float]) -> str:
    if ts is None: return "-"
    delta = datetime.now().timestamp() - ts
    if delta < 60:   return f"{int(delta)} sn"
    if delta < 3600: return f"{int(delta//60)} dk"
    if delta < 86400:return f"{int(delta//3600)} sa"
    return f"{int(delta//86400)} g"

def list_files_sorted(
    include: Optional[List[Union[str, Path]]] = None,
    base_dir: Optional[Path] = None,
    pattern: str = "*.csv",
    ascending: bool = True,
    include_missing: bool = True,
) -> pd.DataFrame:
    """
    Belirtilen dosyaları 'son değiştirme zamanı'na göre sırala.
    - include verilirse bu tam yol listesini kullanır.
    - verilmezse base_dir (varsayılan DATA_DIR) içinde pattern ile glob yapar.
    """
    bdir = base_dir or DATA_DIR
    rows: List[Dict[str, Any]] = []

    # Varsayılan adaylar: DOWNLOADS[path] + pipeline çıktı dosyaları
    if include is None:
        include = []
        for prefix in ["sf", "fr"]:
            include += [str(bdir / f"{prefix}_crime_{i:02d}.csv") for i in range(1, 10)]
            include += [str(bdir / f"{prefix}_crime_y.csv")]
        include += [str(bdir / "sf_crime_grid_full_labeled.csv")]

    # Ayrıca glob ile genişlet
    for p in bdir.glob(pattern):
        include.append(str(p))

    seen = set()
    for x in include:
        p = Path(x)
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)

        exists = p.exists()
        try:
            st_ = p.stat() if exists else None
            mtime = st_.st_mtime if st_ else None
            size  = st_.st_size  if st_ else None
        except Exception:
            mtime, size = None, None

        if exists or include_missing:
            rows.append({
                "file": p.name,
                "path": str(p),
                "exists": bool(exists),
                "size": _human_bytes(size),
                "modified": _fmt_dt(mtime),
                "age": _age_str(mtime),
                "_mtime": mtime,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("_mtime", ascending=ascending, na_position="last").drop(columns=["_mtime"])
    return df

# 0) (Opsiyonel) requirements yükleme
st.markdown("### 0) (Opsiyonel) Gereklilikleri yükle")
if st.button("📦 requirements.txt yükle"):
    try:
        req = ROOT / "requirements.txt"
        if req.exists():
            out = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            st.code(out.stdout or "")
            if out.returncode == 0:
                st.success("✅ Gereklilikler yüklendi.")
            else:
                st.error("❌ Kurulumda hata!")
                st.code(out.stderr or "")
        else:
            st.warning("⚠️ requirements.txt bulunamadı.")
    except Exception as e:
        st.error(f"Kurulum çağrısı başarısız: {e}")

# -----------------------------------------------------------------------------
# Yardımcı: indir & önizle
# -----------------------------------------------------------------------------
def download_and_preview(name, url, file_path, is_json=False, allow_artifact_fallback=False, artifact_picks=None):
    st.markdown(f"### 🔹 {name}")
    # URL’yi ekranda maskeli göster
    st.caption(f"URL: {_mask_token(url)}")
    ok = False
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        if is_json:
            Path(file_path).write_text(r.text, encoding="utf-8")
        else:
            with open(file_path, "wb") as f:
                f.write(r.content)
        ok = True
    except Exception as e:
        st.warning(f"Raw indirme başarısız: {e}")
    if not ok and allow_artifact_fallback:
        try:
            blob = fetch_file_from_latest_artifact(artifact_picks or [os.path.basename(file_path)])
            if blob:
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(blob)
                ok = True
                st.info("Dosya artifact'tan alındı.")
        except Exception as e:
            st.warning(f"Artifact fallback başarısız: {e}")

    if not ok:
        st.error(f"❌ {name} indirilemedi.")
        return

    # Önizleme
    try:
        if is_json:
            try:
                data = json.loads(Path(file_path).read_text(encoding="utf-8"))
                if isinstance(data, dict): st.json(data)
                elif isinstance(data, list): st.json(data[:3])
                else: st.code(str(data)[:1000])
            except Exception as e:
                st.code(Path(file_path).read_text(encoding="utf-8")[:2000])
        else:
            head = pd.read_csv(file_path, nrows=3)
            cols = pd.read_csv(file_path, nrows=0).columns.tolist()
            st.dataframe(head)
            st.caption(f"📌 Sütunlar: {cols}")
        st.success("✅ İndirildi.")
    except Exception as e:
        st.info("Önizleme başarısız; dosya indirildi.")
        st.code(f"Önizleme hatası: {e}")

st.markdown("### 1) (Opsiyonel) Verileri indir ve önizle")
if st.button("📥 Verileri İndir ve Önizle (İlk 3 Satır)"):
    for name, info in DOWNLOADS.items():
        download_and_preview(
            name,
            info["url"],
            info["path"],
            is_json=info.get("is_json", False),
            allow_artifact_fallback=info.get("allow_artifact", False),
            artifact_picks=info.get("artifact_picks"),
        )
    st.success("✅ İndirme tamamlandı.")

st.markdown("### 1.5) Dosyaları tarihe göre sırala")

def convert_csv_dir_to_parquet(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.csv",
    compression: str = "zstd",
    stats: bool = True
) -> pd.DataFrame:
    """
    input_dir altındaki CSV dosyalarını output_dir altına Parquet'e çevirir.
    Varsayılan: zstd sıkıştırma, stats=True ise dosya boyutu ve satır sayısı özetlenir.
    Tercihen polars kullanır; yoksa pandas+pyarrow ile devam eder.
    Dönen DataFrame: kaynak dosya, hedef dosya, satır sayısı, byte bilgisi.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # Polars mevcutsa onu tercih et
    try:
        import polars as pl
        use_polars = True
    except Exception:
        use_polars = False

    # Pandas için pyarrow şart
    if not use_polars:
        try:
            import pyarrow  # noqa: F401
        except Exception:
            raise RuntimeError(
                "Ne polars ne de pyarrow mevcut. Lütfen 'pip install polars pyarrow' kurun."
            )

    for p in sorted(Path(input_dir).glob(pattern)):
        if not p.is_file():
            continue
        out = Path(output_dir) / (p.stem + ".parquet")
        try:
            if use_polars:
                df_pl = pl.read_csv(str(p))
                df_pl.write_parquet(str(out), compression=compression)
                n_rows = df_pl.height
            else:
                df_pd = pd.read_csv(p, low_memory=False)
                df_pd.to_parquet(out, compression=compression, index=False, engine="pyarrow")
                n_rows = len(df_pd)

            src_sz = p.stat().st_size if p.exists() else None
            dst_sz = out.stat().st_size if out.exists() else None
            rows.append({
                "src": str(p.name),
                "dst": str(out.name),
                "rows": n_rows,
                "src_size": src_sz,
                "dst_size": dst_sz,
            })
        except Exception as e:
            rows.append({
                "src": str(p.name),
                "dst": str(out.name),
                "rows": None,
                "src_size": None,
                "dst_size": None,
                "error": str(e),
            })

    res = pd.DataFrame(rows)
    if stats and not res.empty:
        try:
            res["src_size_mb"] = (res["src_size"].astype("float") / (1024**2)).round(3)
            res["dst_size_mb"] = (res["dst_size"].astype("float") / (1024**2)).round(3)
            res["ratio"] = (res["dst_size"].astype("float") / res["src_size"].astype("float")).round(3)
        except Exception:
            pass
    return res

colA, colB, colC = st.columns([1,1,2])
with colA:
    order = st.radio("Sıralama", ["Eski → Yeni", "Yeni → Eski"], horizontal=True, index=0)
with colB:
    show_missing = st.checkbox("Eksikleri de göster", value=True)
with colC:
        patt = st.text_input(
            "Desen (glob)", "*.csv",
            help="Örn: sf_crime_*.csv",
            key="glob_list_files"        
        )

asc = (order == "Eski → Yeni")
df_files = list_files_sorted(pattern=patt, ascending=asc, include_missing=show_missing)
if df_files.empty:
    st.info("Eşleşen dosya yok.")
else:
    st.dataframe(df_files, use_container_width=True)

st.markdown("### 1.6) CSV → Parquet dönüştür")
with st.expander("🔄 CSV’leri Parquet’e çevir (zstd)"):
    in_dir = st.text_input(
        "Girdi klasörü", value=str(DATA_DIR),
        help="Örn: crime_prediction_data/",
        key="csv2parquet_in_dir"                 # ✅
    )
    out_dir = st.text_input(
        "Çıktı klasörü", value=str(ROOT / "parquet_out"),
        help="Örn: parquet_out/",
        key="csv2parquet_out_dir"                # ✅
    )
    patt_in = st.text_input(
        "Desen (glob)", "*.csv",
        help="Örn: sf_crime_*.csv",
        key="csv2parquet_glob"                   # ✅ 
    )
    comp = st.selectbox(
        "Sıkıştırma",
        ["zstd", "snappy", "gzip", "brotli", "uncompressed"],
        index=0,
        key="csv2parquet_codec"                  # ✅
    )
    want_stats = st.checkbox(
        "Özet/stats üret", value=True,
        key="csv2parquet_stats"                  # ✅
    )

    if st.button("🧰 Dönüştür (CSV → Parquet)", key="csv2parquet_run"):  # ✅
        try:
            res = convert_csv_dir_to_parquet(
                input_dir=Path(in_dir),
                output_dir=Path(out_dir),
                pattern=patt_in,
                compression=comp,
                stats=want_stats
            )
            if res.empty:
                st.info("Eşleşen CSV bulunamadı.")
            else:
                st.success("Dönüşüm tamamlandı.")
                st.dataframe(res)
        except Exception as e:
            st.error(f"Dönüşüm hatası: {e}")

with st.expander("🔎 Tanı: Etkin URL/ENV değerleri"):
    st.write("CRIME_CSV_URL (env):", os.environ.get("CRIME_CSV_URL"))
    st.write("RAW_911_URL (env):", os.environ.get("RAW_911_URL"))
    st.write("SF311_URL (env):", os.environ.get("SF311_URL"))

if st.button("♻️ Streamlit cache temizle"):
    try:
        st.cache_data.clear()
        st.success("Cache temizlendi.")
    except Exception as e:
        st.warning(f"Cache temizlenemedi: {e}")

# -----------------------------------------------------------------------------
# Script bul/çalıştır
# -----------------------------------------------------------------------------
def _candidate_names(base: str, locale: str) -> List[str]:
    # arama sırası: locale > default
    if locale and locale != "default":
        return [
            f"{base}.fr.py",      # update_xxx.fr.py
            f"{base}_fr.py",      # update_xxx_fr.py
            f"{base}.{locale}.py",
            f"{base}-{locale}.py",
            f"{base}.py",         # en sonda default'a düş
        ]
    else:
        return [f"{base}.py"]

def resolve_script(entry: dict, locale: str = "default") -> Optional[Path]:
    # 1) asıl ad için
    for cand in _candidate_names(entry["name"], locale):
        p = ensure_script(cand)
        if p:
            return p
    # 2) alternatif adlar için
    for alt in entry.get("alts", []):
        for cand in _candidate_names(alt, locale):
            pp = ensure_script(cand)
            if pp:
                return pp
    return None

def run_script(path: Path) -> bool:
    st.write(f"▶️ {path.name} çalıştırılıyor…")
    placeholder = st.empty()
    lines = []
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(path)],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                lines.append(line.rstrip())
                lines = lines[-400:]
                placeholder.code("\n".join(lines))
        rc = proc.wait()
        if rc == 0:
            st.success(f"✅ {path.name} tamamlandı")
            return True
        else:
            st.error(f"❌ {path.name} hata verdi (exit={rc})")
            return False
    except Exception as e:
        st.error(f"🚨 {path.name} çağrılamadı: {e}")
        return False

# -----------------------------------------------------------------------------
# 2) Güncelleme ve Zenginleştirme
# -----------------------------------------------------------------------------
st.markdown("### 2) Güncelleme ve Zenginleştirme (01 → 09)")
if st.button("⚙️ Güncelleme ve Zenginleştirme (01 → 09)"):
    with st.spinner("⏳ Scriptler çalıştırılıyor..."):
        all_ok = True
        for entry in PIPELINE:
            sp = resolve_script(entry, locale=os.environ.get("PIPELINE_VARIANT", "default"))
            if not sp:
                st.warning(f"⏭️ {entry['name']} bulunamadı/indirilemedi, atlanıyor.")
                all_ok = False
                continue
            ok = run_script(sp)
            all_ok = all_ok and ok
    if all_ok:
        st.success("🎉 Pipeline bitti: Tüm adımlar başarıyla tamamlandı.")
    else:
        st.warning("ℹ️ Pipeline tamamlandı; eksik/hatalı adımlar var. Logları kontrol edin.")

st.markdown("### 3) Güncel _08 → _09 üret (sf + fr)")

for prefix in ["sf", "fr"]:
    st.subheader(f"🔹 {prefix.upper()} akışı")

    # Eğer 09 zaten varsa üretim adımını atlayalım; yoksa üretelim
    p09 = DATA_DIR / f"{prefix}_crime_09.csv"
    if not p09.exists():
        _ = process_city_to_09(prefix, DATA_DIR)

    # Önizleme: 08 ve 09
    p08 = DATA_DIR / f"{prefix}_crime_08.csv"
    if p08.exists():
        try:
            st.markdown(f"**{p08.name} — ilk 20 satır**")
            st.dataframe(pd.read_csv(p08, nrows=20, low_memory=False), use_container_width=True)
        except Exception as e:
            st.info(f"{p08.name} önizlenemedi: {e}")

    if p09.exists():
        try:
            st.markdown(f"**{p09.name} — ilk 20 satır**")
            st.dataframe(pd.read_csv(p09, nrows=20, low_memory=False), use_container_width=True)
        except Exception as e:
            st.info(f"{p09.name} önizlenemedi: {e}")
            
def load_city_crime_08(prefix: str, data_dir: Path) -> Optional[pd.DataFrame]:
    """{prefix}_crime_08.csv'yi yükler ve date kolonunu normalize eder."""
    path = data_dir / f"{prefix}_crime_08.csv"
    if prefix.lower() == "sf":
        return load_sf_crime_08(path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        elif "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date
        return df
    except Exception as e:
        st.warning(f"{path.name} okunamadı: {e}")
        return None


def process_city_to_09(prefix: str, data_dir: Path) -> Optional[pd.DataFrame]:
    """{prefix}_crime_08 → {prefix}_crime_08_clean → (neighbors) → {prefix}_crime_09 üretir."""
    df08 = load_city_crime_08(prefix, data_dir)
    if df08 is None:
        st.info(f"{prefix}_crime_08.csv bulunamadı.")
        return None

    out_clean = data_dir / f"{prefix}_crime_08_clean.csv"
    clean_and_save_crime_09(df08, str(out_clean))
    st.success(f"✅ {out_clean.name} kaydedildi.")

    graph_script = resolve_script({"name": "update_neighbors_graph.py", "alts": ["neighbors_graph.py"]})
    feat_script  = resolve_script({"name": "update_neighbors.py",        "alts": []})

    # prefix'e özel neighbors varsa onu kullan; yoksa genel
    neighbor_file_pref = data_dir / f"{prefix}_neighbors.csv"
    neighbor_file_gen  = data_dir / "neighbors.csv"
    neighbor_file_use  = neighbor_file_pref if neighbor_file_pref.exists() else neighbor_file_gen

    if not neighbor_file_use.exists() and graph_script:
        ok_graph = run_script(graph_script)
        st.success("🗺️ neighbors.csv üretildi.") if ok_graph else st.warning("neighbors graph başarısız.")

    if feat_script:
        os.environ["NEIGHBOR_FILE"]        = os.environ.get("NEIGHBOR_FILE", str(neighbor_file_use))
        os.environ["NEIGHBOR_INPUT_CSV"]   = str(out_clean)
        os.environ["NEIGHBOR_OUTPUT_CSV"]  = str(data_dir / f"{prefix}_crime_09.csv")
        os.environ["NEIGHBOR_WINDOW_DAYS"] = os.environ.get("NEIGHBOR_WINDOW_DAYS", "7")
        os.environ["NEIGHBOR_LAG_DAYS"]    = os.environ.get("NEIGHBOR_LAG_DAYS", "1")

        ok_feat = run_script(feat_script)
        if ok_feat:
            st.success(f"🧩 {prefix}_crime_09.csv üretildi (nei_7d_sum eklendi).")
            try:
                return pd.read_csv(data_dir / f"{prefix}_crime_09.csv", low_memory=False)
            except Exception:
                return None
        else:
            st.warning("update_neighbors.py çalıştırılamadı; logu kontrol edin.")
    else:
        st.info("update_neighbors.py bulunamadı (scripts klasörüne ekleyin).")
    return None

        # -------------------------------
        # Global — Sayı (kuantil regresyon)
        # -------------------------------
        with tabs[1]:
            if 0.5 in q_models:
                st.caption("Kuantil (q=0.5) LightGBMRegressor için SHAP — ilk 10")
                expl_reg = shap.TreeExplainer(q_models[0.5])
                sample_idx_reg = X_cnt.sample(min(1500, len(X_cnt)), random_state=42).index
                shap_vals_reg = expl_reg.shap_values(X_cnt.loc[sample_idx_reg])
                mean_abs_r = np.abs(shap_vals_reg).mean(axis=0)
                top_idx_r = np.argsort(mean_abs_r)[::-1][:10]
                top_df_r = pd.DataFrame({
                    "özellik": X_cnt.columns[top_idx_r],
                    "önem(Mean|SHAP|)": mean_abs_r[top_idx_r]
                })
                st.dataframe(top_df_r, use_container_width=True)
                st.bar_chart(top_df_r.set_index("özellik"))
            else:
                st.info("Kuantil regresyon modeli bulunamadı.")

        # -------------------------------
        # Local — tek satır açıklaması
        # -------------------------------
        with tabs[2]:
            st.caption("Seçtiğin satır için sınıf (Y>0) olasılığı ve özellik katkıları")
            idx = st.number_input("Satır indexi", min_value=0, max_value=int(len(X_all)-1), value=0, step=1)
            x_row = X_all.iloc[[idx]]
            p_row = float(clf.predict_proba(x_row)[:,1])
            exp_row = float(df09.loc[x_row.index, "pred_expected"]) if "pred_expected" in df09.columns else np.nan
            st.write(f"**P(Y>0)** = {p_row:.3f} | **Beklenen sayı** ≈ {exp_row:.2f}")

            shap_row = expl_clf.shap_values(x_row)
            shap_row_pos = shap_row[1][0] if isinstance(shap_row, list) else shap_row[0]
            contrib = pd.DataFrame({
                "özellik": x_row.columns,
                "değer": x_row.iloc[0].values,
                "katkı(SHAP)": shap_row_pos
            }).sort_values("katkı(SHAP)", key=np.abs, ascending=False).head(15)
            st.dataframe(contrib, use_container_width=True)

        # -------------------------------
        # PDP / ICE (kritik özellikler)
        # -------------------------------
        with tabs[3]:
            st.caption("Marjinal etki (PDP). Sınıf modeli (Y>0, target=1) üzerinde.")
            candidates = [c for c in ["event_hour","nei_7d_sum","nr_7d","bus_stop_count","poi_risk_score"]
                          if c in X_occ.columns]
            feats = st.multiselect("PDP için özellik(ler) seç", options=candidates, default=candidates[:2])
            if len(feats) > 0:
                for f in feats[:3]:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    PartialDependenceDisplay.from_estimator(
                        clf_tree, X_occ, [f], kind="average", target=1, ax=ax
                    )
                    ax.set_title(f"PDP — {f}")
                    st.pyplot(fig, clear_figure=True)
            else:
                st.info("Listeden en az bir özellik seç.")

        with tabs[4]:
            st.caption("Seçilen iki sınıf için özet kart (global SHAP ilk 5)")
            if cat_col:
                pick_a = st.selectbox("Kart A sınıfı", sorted(df09[cat_col].dropna().unique()), key="cardA")
                pick_b = st.selectbox("Kart B sınıfı", sorted(df09[cat_col].dropna().unique()), key="cardB")

                def _topk_for_class(pick, k=5):
                    sub_idx = df09.loc[sample_idx][df09.loc[sample_idx, cat_col] == pick].index
                    if len(sub_idx) < 20:
                        return pd.DataFrame({"özellik": [], "önem": []})
                    shap_sub = expl_clf.shap_values(X_occ.loc[sub_idx])
                    shap_sub_pos = shap_sub[1] if isinstance(shap_sub, list) else shap_sub
                    mabs = np.abs(shap_sub_pos).mean(axis=0)
                    top = np.argsort(mabs)[::-1][:k]
                    return pd.DataFrame({"özellik": X_occ.columns[top], "önem": mabs[top]})

                colA, colB = st.columns(2)
                with colA:
                    st.markdown(f"**{pick_a} — Top 5 etken**")
                    st.dataframe(_topk_for_class(pick_a), use_container_width=True)
                with colB:
                    st.markdown(f"**{pick_b} — Top 5 etken**")
                    st.dataframe(_topk_for_class(pick_b), use_container_width=True)
            else:
                st.info("category_grouped / subcategory_grouped yoksa sınıf kartları oluşturulamaz.")
else:
    available_09 = {p: DATA_DIR / f"{p}_crime_09.csv" for p in ["sf", "fr"] if (DATA_DIR / f"{p}_crime_09.csv").exists()}
    if not available_09:
        st.markdown("### 5) Hızlı Model")
        st.info("Model eğitmek için önce sf_crime_09.csv veya fr_crime_09.csv’nin üretilmiş olması gerekiyor.")
        st.stop()
    
    st.markdown("### 5) Hızlı Model (ZI/Hurdle + Quantile + Kalibrasyon)")
    pick_city = st.selectbox("Model verisi (09)", list(available_09.keys()), index=0, format_func=lambda x: x.upper())
    try:
        df09 = pd.read_csv(available_09[pick_city], low_memory=False)
    except Exception as e:
        st.warning(f"{pick_city}_crime_09.csv okunamadı: {e}")
        st.stop()
    st.markdown("### 5) Hızlı Model")
    st.info("Model eğitmek için önce sf_crime_09.csv’nin üretilmiş olması gerekiyor.")

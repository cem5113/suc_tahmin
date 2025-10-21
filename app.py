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

# -----------------------------------------------------------------------------
# Global Kurulum
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# AĞIR BAĞIMLILIKLAR İÇİN LAZY IMPORT
# -----------------------------------------------------------------------------
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
            "np": np,
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

# -----------------------------------------------------------------------------
# Yardımcılar
# -----------------------------------------------------------------------------
def pick_url(key: str, default: str) -> str:
    # Öncelik: 1) st.secrets  2) ENV  3) default
    try:
        if key in st.secrets and st.secrets[key]:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)

def _mask_token(u: str) -> str:
    try:
        return re.sub(r'(\$\$app_token=)[^&]+', r'\1•••', str(u))
    except Exception:
        return str(u)

def _human_bytes(n: int) -> str:
    if n is None:
        return "-"
    step = 1024.0
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < step:
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= step
    return f"{n:.1f} PB"

def _fmt_dt(ts: Optional[float]) -> str:
    if ts is None:
        return "-"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def _age_str(ts: Optional[float]) -> str:
    if ts is None:
        return "-"
    delta = datetime.now().timestamp() - ts
    if delta < 60:
        return f"{int(delta)} sn"
    if delta < 3600:
        return f"{int(delta // 60)} dk"
    if delta < 86400:
        return f"{int(delta // 3600)} sa"
    return f"{int(delta // 86400)} g"

def ensure_script(filename: str) -> Optional[Path]:
    """
    Verilen dosya adını SEARCH_DIRS içinde arar; bulursa Path döner, yoksa None.
    """
    cand = filename if filename.endswith(".py") else f"{filename}.py"
    for base in SEARCH_DIRS:
        p = Path(base) / cand
        if p.exists() and p.is_file():
            return p
    return None

def _candidate_names(base: str, locale: str) -> List[str]:
    if locale and locale != "default":
        return [
            f"{base}.fr.py",
            f"{base}_fr.py",
            f"{base}.{locale}.py",
            f"{base}-{locale}.py",
            f"{base}.py",
        ]
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
    lines: List[str] = []
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
# ENV ve URL’ler
# -----------------------------------------------------------------------------
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
    POPULATION_PATH = DEFAULT_POP_CSV

os.environ["POPULATION_PATH"] = str(POPULATION_PATH)

# ⬇️ 911 artımlı çekim için API ayarları
SF911_API_URL       = pick_url("SF911_API_URL", "https://data.sfgov.org/resource/2zdj-bwza.json")
SF911_AGENCY_FILTER = pick_url("SF911_AGENCY_FILTER", "agency like '%Police%'")
SF911_API_TOKEN     = pick_url("SF911_API_TOKEN", "")

# Çocuk süreçlerin de aynı değerleri görmesi için ENV
os.environ["CRIME_CSV_URL"] = CRIME_CSV_LATEST
os.environ["RAW_911_URL"]   = RAW_911_URL
os.environ["SF311_URL"]     = SF311_URL
os.environ["GEOID_LEN"]     = os.environ.get("GEOID_LEN", "11")

GEOID_LEN = int(os.environ.get("GEOID_LEN", "11"))

def _norm_geoid(s: pd.Series, L: int = GEOID_LEN) -> pd.Series:
    return (
        s.astype(str)
         .str.extract(r"(\d+)", expand=False)
         .str[:L]
         .str.zfill(L)
    )

os.environ["SF911_API_URL"]       = SF911_API_URL
os.environ["SF911_AGENCY_FILTER"] = SF911_AGENCY_FILTER
if SF911_API_TOKEN:
    os.environ["SF911_API_TOKEN"] = SF911_API_TOKEN

SOCS_APP_TOKEN = st.secrets.get("SOCS_APP_TOKEN", os.environ.get("SOCS_APP_TOKEN", ""))
if SOCS_APP_TOKEN:
    os.environ["SOCS_APP_TOKEN"] = SOCS_APP_TOKEN

# LATEST veya 2022/2023 gibi belirli yıl
os.environ["ACS_YEAR"] = st.secrets.get("ACS_YEAR", os.environ.get("ACS_YEAR", "LATEST"))

# Virgülle filtre (boş bırak = tüm kategoriler)
os.environ["DEMOG_WHITELIST"] = st.secrets.get(
    "DEMOG_WHITELIST",
    os.environ.get("DEMOG_WHITELIST", "")
)

# --- GitHub Actions entegrasyonu ---
GITHUB_REPO = os.environ.get("GITHUB_REPO", "cem5113/crime_prediction_data")
GITHUB_WORKFLOW = os.environ.get("GITHUB_WORKFLOW", "full_pipeline.yml")

ARTIFACT_NAMES = [
    "crime-pipeline-output",       # ✅ yeni
    "sf-crime-pipeline-output",    # (geri uyum)
]

def _gh_headers():
    token = st.secrets.get("GH_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        raise RuntimeError("GH_TOKEN gerekli (Streamlit secrets veya env).")
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

def fetch_file_from_latest_artifact(pick_names: List[str], artifact_names: Optional[List[str]] = None) -> bytes | None:
    names_to_try = artifact_names or ARTIFACT_NAMES
    runs_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs?per_page=20"
    runs = requests.get(runs_url, headers=_gh_headers(), timeout=30).json()
    run_ids = [r["id"] for r in runs.get("workflow_runs", []) if r.get("conclusion") == "success"]

    for rid in run_ids:
        arts_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{rid}/artifacts"
        arts = requests.get(arts_url, headers=_gh_headers(), timeout=30).json().get("artifacts", [])
        for a in arts:
            if a.get("expired", False):
                continue
            if a.get("name") not in names_to_try:
                continue
            dl = requests.get(a["archive_download_url"], headers=_gh_headers(), timeout=60)
            zf = zipfile.ZipFile(io.BytesIO(dl.content))
            names = zf.namelist()

            # 1) Doğrudan dosya araması
            for pick in pick_names:
                for cand in (pick, f"crime_prediction_data/{pick}"):
                    if cand in names:
                        return zf.read(cand)

            # 2) "paquet_run_*.zip" içinden ara (artifact ZIP'i içinde ikinci bir zip olabilir)
            paquet_inner = [n for n in names if re.search(r"^paquet_run_\d+\.zip$", n)]
            for pin in paquet_inner:
                with zf.open(pin) as inner_blob:
                    with zipfile.ZipFile(io.BytesIO(inner_blob.read())) as inner_zip:
                        inner_names = inner_zip.namelist()
                        for pick in pick_names:
                            # hem düz ad hem de fr_eda/ altından eşleşme
                            exacts = [pick, f"fr_eda/{pick}", f"crime_prediction_data/{pick}"]
                            for ex in exacts:
                                if ex in inner_names:
                                    return inner_zip.read(ex)
                        # adın sonu eşleşsin (fallback)
                        for n in inner_names:
                            if any(n.endswith(p) for p in pick_names):
                                return inner_zip.read(n)

            # 3) Son-çare: isim sonu eşleşmesi
            for n in names:
                if any(n.endswith(p) for p in pick_names):
                    return zf.read(n)
    return None

def _resolve_workflow_id(target: str):
    import requests, os
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows?per_page=100"
    r = requests.get(url, headers=_gh_headers(), timeout=30); r.raise_for_status()
    ws = r.json().get("workflows", [])
    # 1) Dosya adıyla eşleşme
    for w in ws:
        if os.path.basename(str(w.get("path",""))) == target:
            return w.get("id")
    # 2) Görünen adla eşleşme (yedek)
    for w in ws:
        if str(w.get("name","")).strip().lower() == target.strip().lower():
            return w.get("id")
    return None

def dispatch_workflow(persist="artifact", force=True, ref="main", variant=None, top_k=None):
    import json as _json, requests
    target = os.environ.get("GITHUB_WORKFLOW", "full_pipeline.yml")
    wid = _resolve_workflow_id(target)
    if not wid:
        return {"ok": False, "status": 404, "text": f"Workflow bulunamadı: {target}"}
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{wid}/dispatches"
    inputs = {"persist": persist, "force": "true" if force else "false"}
    if variant: inputs["variant"] = str(variant)
    if top_k:   inputs["top_k"]   = str(top_k)
    payload = {"ref": ref, "inputs": inputs}
    r = requests.post(url, headers=_gh_headers(), data=_json.dumps(payload), timeout=30)
    return {"ok": r.status_code in (204, 201), "status": r.status_code, "text": r.text}

def diag_workflow(target: str | None = None, ref: str | None = None):
    """Workflow dispatch 422 tanısı: dosya adı/ad, branch ve YAML içeriğini kontrol eder (PyYAML yoksa YAML analizi atlanır)."""
    import os, base64, urllib.parse
    import requests
    try:
        import yaml  # PyYAML yoksa except'e düşer ve YAML analizi atlanır
    except Exception:
        yaml = None

    target = target or os.environ.get("GITHUB_WORKFLOW", "full_pipeline.yml")
    ref = ref or "main"

    st.write(f"🎯 Hedef workflow: `{target}` · ref: `{ref}`")

    # 1) Workflow listesini çek
    url_list = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows?per_page=100"
    r = requests.get(url_list, headers=_gh_headers(), timeout=30)
    if r.status_code != 200:
        st.error(f"Workflow listesi okunamadı: {r.status_code} {r.text}")
        return
    ws = r.json().get("workflows", [])
    st.caption(f"Toplam {len(ws)} workflow bulundu.") if ws else st.warning("Repo’da hiç workflow görünmüyor.")
    if ws:
        st.code("\n".join([f"- {w.get('name')}  | id={w.get('id')}  | path={w.get('path')}" for w in ws]) or "-", language="text")

    # 2) ID çöz: önce dosya adıyla, olmazsa görünen adla
    wid, wmeta = None, None
    for w in ws:
        if os.path.basename(str(w.get("path",""))) == target:
            wid, wmeta = w.get("id"), w
            break
    if not wid:
        for w in ws:
            if str(w.get("name","")).strip().lower() == target.strip().lower():
                wid, wmeta = w.get("id"), w
                break
    if not wid:
        st.error(f"❌ Hedef workflow bulunamadı: `{target}` (dosya adı ya da görünen ad eşleşmedi)")
        return
    st.success(f"✅ Çözülen workflow id: {wid}")

    # 3) YAML içeriğini getir ve göster
    url_w = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{wid}"
    rw = requests.get(url_w, headers=_gh_headers(), timeout=30)
    if rw.status_code != 200:
        st.warning(f"Workflow ayrıntısı okunamadı: {rw.status_code} {rw.text}")
        return
    wpath = rw.json().get("path", "")
    st.write(f"🗂️ Dosya yolu: `{wpath}`")

    url_contents = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{urllib.parse.quote(wpath)}?ref={urllib.parse.quote(ref)}"
    rc = requests.get(url_contents, headers=_gh_headers(), timeout=30)
    if rc.status_code != 200 or rc.json().get("encoding") != "base64":
        st.error(f"YAML okunamadı: {rc.status_code} {rc.text[:200]}")
        return
    yaml_text = base64.b64decode(rc.json()["content"]).decode("utf-8", errors="replace")
    st.code(yaml_text, language="yaml")

    # 4) (Opsiyonel) YAML parse: workflow_dispatch var mı?
    if yaml:
        try:
            y = yaml.safe_load(yaml_text) or {}
            on_section = y.get("on") or {}
            has_dispatch = False
            inputs_info = {}
            if isinstance(on_section, dict):
                if "workflow_dispatch" in on_section:
                    has_dispatch = True
                    wd = on_section.get("workflow_dispatch") or {}
                    if isinstance(wd, dict):
                        inputs_info = wd.get("inputs", {}) or {}
            elif isinstance(on_section, list):
                has_dispatch = "workflow_dispatch" in on_section

            if has_dispatch:
                st.success("✅ Bu workflow YAML'ında `workflow_dispatch` tanımlı.")
            else:
                st.error("❌ Bu workflow YAML'ında `workflow_dispatch` YOK. 422’nin ana nedeni bu.")

            if inputs_info:
                st.write("🧩 `workflow_dispatch.inputs` tanımı:")
                st.json(inputs_info)
            else:
                st.caption("Bu workflow için özel inputs tanımı yok (veya boş).")
        except Exception as e:
            st.warning(f"YAML parse edilemedi: {e} (PyYAML sürümü/format)")

    else:
        st.info("PyYAML yüklü değil; YAML analizi atlandı. (İstersen requirements.txt → PyYAML ekleyebilirsin)")

    # 5) Branch kontrolü
    url_branch = f"https://api.github.com/repos/{GITHUB_REPO}/branches/{urllib.parse.quote(ref)}"
    rb = requests.get(url_branch, headers=_gh_headers(), timeout=30)
    st.success(f"✅ Branch mevcut: {ref}") if rb.status_code == 200 else st.warning(f"⚠️ Branch bulunamadı: {ref} ({rb.status_code})")

    # 6) Örnek dispatch payload bilgisini göster
    st.write({"ref": ref, "inputs": {"persist": "artifact", "force": "true"}})
    st.caption(f"Endpoint: /repos/{GITHUB_REPO}/actions/workflows/{wid}/dispatches")

def _get_last_run_by_workflow():
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
        status = run.get("status")
        concl  = run.get("conclusion") or "-"
        started = run.get("run_started_at")
        html_url = run.get("html_url")
        container.markdown(
            f"**Son koşum:** `{status}` / `{concl}` · başlama: `{started}`  ·  [GitHub’da aç]({html_url})"
        )
    except Exception as e:
        container.warning(f"Durum okunamadı: {e}")

def fetch_latest_artifact_df() -> Optional[pd.DataFrame]:
    try:
        runs_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs?per_page=20"
        runs = requests.get(runs_url, headers=_gh_headers(), timeout=30).json()
        run_ids = [r["id"] for r in runs.get("workflow_runs", []) if r.get("conclusion") == "success"]

        for rid in run_ids:
            arts_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{rid}/artifacts"
            arts = requests.get(arts_url, headers=_gh_headers(), timeout=30).json().get("artifacts", [])
            for a in arts:
                if a.get("expired", False) or a.get("name") not in ARTIFACT_NAMES:
                    continue
                dl = requests.get(a["archive_download_url"], headers=_gh_headers(), timeout=60)
                zf = zipfile.ZipFile(io.BytesIO(dl.content))

                # Önce düzden dene
                for pick in ("crime_prediction_data/sf_crime_08.csv", "sf_crime_08.csv"):
                    if pick in zf.namelist():
                        with zf.open(pick) as f:
                            df = pd.read_csv(f, low_memory=False)
                            if "date" in df.columns:
                                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
                            elif "datetime" in df.columns:
                                df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date
                            return df

                # paquet_run_*.zip içinden dene
                for n in zf.namelist():
                    if re.search(r"^paquet_run_\d+\.zip$", n):
                        with zf.open(n) as inner_blob:
                            with zipfile.ZipFile(io.BytesIO(inner_blob.read())) as inner_zip:
                                for pick in ("sf_crime_08.csv", "crime_prediction_data/sf_crime_08.csv"):
                                    if pick in inner_zip.namelist():
                                        with inner_zip.open(pick) as f:
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

# -----------------------------------------------------------------------------
# 08 → 09 Dönüşüm Yardımcıları
# -----------------------------------------------------------------------------
def _group_rare_labels(
    df: pd.DataFrame,
    col: str,
    min_prop: Optional[float] = None,
    min_count: Optional[int] = None,
    other_label: str = "Other",
    out_stats_path: Optional[Path] = None,
) -> pd.Series:
    if col not in df.columns:
        return pd.Series([None] * len(df), index=df.index)

    s = df[col].astype(str).str.strip()
    total = len(s)
    vc = s.value_counts(dropna=False)

    env_prop = os.environ.get("RARE_MIN_PROP")
    env_count = os.environ.get("RARE_MIN_COUNT")
    if min_prop is None and env_prop:
        try:
            min_prop = float(env_prop)
        except Exception:
            pass
    if min_count is None and env_count:
        try:
            min_count = int(env_count)
        except Exception:
            pass

    if min_prop is None and min_count is None:
        min_prop, min_count = 0.01, 200

    rare_mask = pd.Series(False, index=vc.index)
    if min_prop is not None:
        rare_mask |= (vc / max(total, 1)) < float(min_prop)
    if min_count is not None:
        rare_mask |= vc < int(min_count)

    rare_values = set(vc[rare_mask].index)
    grouped = s.where(~s.isin(rare_values), other_label)

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

def clean_and_save_crime_09(input_obj: Union[str, pd.DataFrame] = "sf_crime_08.csv", output_path: str = "sf_crime_09.csv"):
    # input_obj hem DataFrame hem de dosya yolu olabilir
    if isinstance(input_obj, pd.DataFrame):
        df = input_obj.copy()
    else:
        df = pd.read_csv(input_obj, dtype={"GEOID": str})

    # 🛠 FIX-1: GEOID normalizasyonu (zfill kaldırıldı, sadece sayısal çekirdek)
    if "GEOID" in df.columns:
        df["GEOID"] = (
            df["GEOID"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
        )

    if "category" in df.columns:
        df["category"] = df["category"].astype(str).str.strip().str.title()

    # Rare class grouping
    try:
        out_dir = Path(output_path).parent if isinstance(output_path, str) else Path(".")
        if "category" in df.columns:
            df["category_grouped"] = _group_rare_labels(
                df, "category", min_prop=None, min_count=None,
                other_label="Other", out_stats_path=out_dir / "rare_stats_category.csv"
            )

        if "subcategory" in df.columns:
            df["subcategory"] = df["subcategory"].astype(str).str.strip().str.title()
            df["subcategory_grouped"] = _group_rare_labels(
                df, "subcategory", min_prop=None, min_count=None,
                other_label="Other", out_stats_path=out_dir / "rare_stats_subcategory.csv"
            )

        try:
            st.caption("🔎 Rare grouping uygulandı (category/subcategory). İstatistikler CSV olarak kaydedildi.")
        except Exception:
            pass
    except Exception as _e:
        try:
            st.warning(f"Rare grouping atlandı: {str(_e)}")
        except Exception:
            print(f"Rare grouping atlandı: {_e}")

    # Tip dönüştürücüler
    def to_int(df_, col, default=0):
        if col in df_.columns:
            df_[col] = (
                pd.to_numeric(df_[col], errors="coerce")
                .fillna(default)
                .round()
                .astype("Int64")
            )

    def to_float(df_, col, default=0.0):
        if col in df_.columns:
            df_[col] = (
                pd.to_numeric(df_[col], errors="coerce")
                .fillna(default)
                .astype(float)
            )

    # Sayaç kolonları
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

    # Risk skoru
    to_float(df, "poi_risk_score", default=0.0)

    # Binary kolonlar
    def to_binary(df_, col):
        if col in df_.columns:
            m = {
                "true": 1, "t": 1, "yes": 1, "y": 1, "1": 1, "evet": 1,
                "false": 0, "f": 0, "no": 0, "n": 0, "0": 0, "hayır": 0, "hayir": 0
            }
            s = df_[col].replace({True: 1, False: 0})
            s = s.astype(str).str.strip().str.lower().map(m)
            df_[col] = pd.to_numeric(s, errors="coerce").fillna(0).astype("Int64")

    for c in ["is_near_police", "is_near_government"]:
        to_binary(df, c)

    # Mesafe kolonları
    for c in ["distance_to_bus", "distance_to_train", "distance_to_police", "distance_to_government_building"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(9999.0).astype(float)

    # Range kolonları (int kategoriler)
    for c in ["bus_stop_count_range", "train_stop_count_range", "poi_total_count_range", "poi_risk_score_range"]:
        to_int(df, c, default=0)

    for c in ["distance_to_bus_range", "distance_to_train_range", "distance_to_police_range", "distance_to_government_building_range"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            max_cat = int(s.max(skipna=True)) if pd.notna(s.max(skipna=True)) else 3
            df[c] = s.fillna(max_cat).round().astype("Int64")

    # Nüfus (median ile doldur)
    if "population" in df.columns:
        df["population"] = pd.to_numeric(df["population"], errors="coerce")
        median_pop = df["population"].median(skipna=True)
        median_pop = 0 if pd.isna(median_pop) else median_pop
        df["population"] = df["population"].fillna(median_pop)

    # POI dominant type
    if "poi_dominant_type" in df.columns:
        # 🛠 FIX-2: Boş stringleri de None say, sonra "None" ile doldur
        df["poi_dominant_type"] = (
            df["poi_dominant_type"]
            .replace({"": np.nan})
            .fillna("None")
            .astype(str)
        )

    # Tarih normalize
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    elif "datetime" in df.columns and "date" not in df.columns:
        df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date

    # 🛠 FIX-3: Near-repeat güvenli hale getirildi (crime_count yoksa Y_label'dan türet)
    try:
        need = {"date", "GEOID"}
        has_crime_count = "crime_count" in df.columns

        if not has_crime_count and "Y_label" in df.columns:
            df["crime_count"] = (
                pd.to_numeric(df["Y_label"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
            has_crime_count = True

        if need.issubset(df.columns) and has_crime_count:
            # kategori kolonu öncelik sırası
            cat_col = None
            for cc in ["category_grouped", "subcategory_grouped", "category"]:
                if cc in df.columns:
                    cat_col = cc
                    break

            if cat_col:
                tmp = df[["date", "GEOID", cat_col, "crime_count"]].copy()
                tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.date
                g = (tmp.groupby(["GEOID", cat_col, "date"], as_index=False)["crime_count"].sum())
                g["date"] = pd.to_datetime(g["date"])
                g = g.sort_values(["GEOID", cat_col, "date"])

                def _roll_counts(x):
                    x = x.set_index("date").asfreq("D", fill_value=0)
                    x["nr_7d"]  = x["crime_count"].rolling("7D").sum().shift(1)
                    x["nr_14d"] = x["crime_count"].rolling("14D").sum().shift(1)
                    return x.reset_index()

                g2 = (g.groupby(["GEOID", cat_col]).apply(_roll_counts).reset_index(level=[0, 1]).reset_index(drop=True))
                g2["date"] = g2["date"].dt.date

                df = df.merge(
                    g2[["GEOID", cat_col, "date", "nr_7d", "nr_14d"]],
                    on=["GEOID", cat_col, "date"], how="left"
                )
                for c in ["nr_7d", "nr_14d"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)
    except Exception as _e:
        print(f"near-repeat uyarı: {_e}")

    # Dışsal değişken örnek skoru
    try:
        ext_map_path = Path(DATA_DIR) / "crime_type_externals_map.json"
        if ext_map_path.exists():
            with open(ext_map_path, "r", encoding="utf-8") as f:
                type_map = json.load(f)
            key_col = "category_grouped" if "category_grouped" in df.columns else (
                      "category" if "category" in df.columns else None)
            if key_col:
                def _ext_score(row):
                    cols = type_map.get(str(row[key_col]), [])
                    vals = []
                    for c in cols:
                        if c in df.columns:
                            try:
                                vals.append(float(row.get(c, 0)))
                            except Exception:
                                pass
                    return float(np.nanmean(vals)) if len(vals) > 0 else np.nan
                df["externals_type_score"] = df.apply(_ext_score, axis=1)
                df["externals_type_score"] = df["externals_type_score"].fillna(0.0).astype(float)
    except Exception as _e:
        print(f"dışsal değişken uyarı: {_e}")

    preview_cols = [c for c in ["nr_7d", "nr_14d", "nei_7d_sum", "externals_type_score"] if c in df.columns]
    if preview_cols:
        try:
            st.caption("🧩 Yeni mekânsal-zamansal özellikler (ilk 20 satır):")
            st.dataframe(df[preview_cols].head(20))
        except Exception:
            pass

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ {output_path} kaydedildi. Satır sayısı: {len(df)}")
    return df

def load_sf_crime_08(local_path: Path) -> Optional[pd.DataFrame]:
    """Önce yerel dosyayı dene; yoksa artifact’tan çek. Ardından crime_mix varsa grid ile merge et."""
    def _normalize_date_cols(df_: pd.DataFrame) -> pd.DataFrame:
        if "date" in df_.columns:
            df_["date"] = pd.to_datetime(df_["date"], errors="coerce").dt.date
        elif "datetime" in df_.columns and "date" not in df_.columns:
            df_["date"] = pd.to_datetime(df_["datetime"], errors="coerce").dt.date
        return df_

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

    # crime_mix merge (grid) — opsiyonel
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
    """{prefix}_crime_08 → temizle (_08_clean) → neighbors → {prefix}_crime_09 üretir."""
    df08 = load_city_crime_08(prefix, data_dir)
    if df08 is None:
        st.info(f"{prefix}_crime_08.csv bulunamadı.")
        return None

    out_clean = data_dir / f"{prefix}_crime_08_clean.csv"
    clean_and_save_crime_09(df08, str(out_clean))
    st.success(f"✅ {out_clean.name} kaydedildi.")

    graph_script = resolve_script({"name": "update_neighbors_graph", "alts": ["update_neighbors_graph.py", "neighbors_graph", "neighbors_graph.py"]})
    feat_script  = resolve_script({"name": "update_neighbors",       "alts": ["update_neighbors.py"]})

    # prefix'e özel neighbors varsa onu kullan; yoksa genel
    neighbor_file_pref = data_dir / f"{prefix}_neighbors.csv"
    neighbor_file_gen  = data_dir / "neighbors.csv"
    neighbor_file_use  = neighbor_file_pref if neighbor_file_pref.exists() else neighbor_file_gen

    if not neighbor_file_use.exists() and graph_script:
        ok_graph = run_script(graph_script)
        st.success("🗺️ neighbors.csv üretildi.") if ok_graph else st.warning("neighbors graph başarısız.")
        if ok_graph:
            neighbor_file_use = neighbor_file_gen  # grafikten sonra genel üretildi varsayalım

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

# -----------------------------------------------------------------------------
# Dosya Listeleme / Dönüştürme Yardımcıları
# -----------------------------------------------------------------------------
def list_files_sorted(
    include: Optional[List[Union[str, Path]]] = None,
    base_dir: Optional[Path] = None,
    pattern: str = "*.csv",
    ascending: bool = True,
    include_missing: bool = True,
) -> pd.DataFrame:
    bdir = base_dir or DATA_DIR
    rows: List[Dict[str, Any]] = []

    if include is None:
        include = []
        for prefix in ["sf", "fr"]:
            include += [str(bdir / f"{prefix}_crime_{i:02d}.csv") for i in range(1, 10)]
            include += [str(bdir / f"{prefix}_crime_y.csv")]
        include += [str(bdir / "sf_crime_grid_full_labeled.csv")]
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

def convert_csv_dir_to_parquet(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.csv",
    compression: str = "zstd",
    stats: bool = True
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    try:
        import polars as pl
        use_polars = True
    except Exception:
        use_polars = False

    if not use_polars:
        try:
            import pyarrow  # noqa: F401
        except Exception:
            raise RuntimeError("Ne polars ne de pyarrow mevcut. Lütfen 'pip install polars pyarrow' kurun.")

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

# -----------------------------------------------------------------------------
# İndirilebilir kaynaklar
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
        "url": "",
        "path": str(DATA_DIR / "sf_population.csv"),
        "local_src": str(POPULATION_PATH),
        "is_local_csv": True,
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
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_weather_5years.csv",
        "path": str(DATA_DIR / "sf_weather_5years.csv"),
    },
}

def download_and_preview(name, url, file_path, is_json=False, allow_artifact_fallback=False, artifact_picks=None):
    st.markdown(f"### 🔹 {name}")
    st.caption(f"URL: {_mask_token(url)}")
    ok = False

    # Yerel kopya (Nüfus) — CSV-only mod
    meta = DOWNLOADS.get(name, {})
    if meta.get("is_local_csv"):
        try:
            src = Path(meta["local_src"])
            dst = Path(file_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                dst.write_bytes(src.read_bytes())
                ok = True
                st.info("Yerel CSV kopyalandı.")
            else:
                st.warning(f"Yerel dosya bulunamadı: {src}")
        except Exception as e:
            st.warning(f"Yerel kopya hatası: {e}")

    if not ok and url:
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
                if isinstance(data, dict):
                    st.json(data)
                elif isinstance(data, list):
                    st.json(data[:3])
                else:
                    st.code(str(data)[:1000])
            except Exception:
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

# -----------------------------------------------------------------------------
# UI — Başlık ve Sidebar
# -----------------------------------------------------------------------------
st.title("📦 Günlük Suç Tahmin Zenginleştirme ve Güncelleme Paneli")

with st.sidebar:
    st.markdown("### GitHub Actions")

with st.expander("🕵️ Workflow tanı (dispatch)"):
    wf_in = st.text_input(
        "Workflow (dosya adı veya görünen ad)",
        os.environ.get("GITHUB_WORKFLOW", "full_pipeline.yml")
    )
    ref_in = st.text_input("Ref (branch/tag)", "main")
    if st.button("Tanıyı Çalıştır"):
        diag_workflow(target=wf_in, ref=ref_in)

    with st.expander("🕵️ Workflow tanı (dispatch)"):
        wf_in = st.text_input("Workflow (dosya adı veya görünen ad)",
                              os.environ.get("GITHUB_WORKFLOW", "full_pipeline.yml"))
        ref_in = st.text_input("Ref (branch/tag)", "main")
        if st.button("Tanıyı Çalıştır"):
            diag_workflow(target=wf_in, ref=ref_in)

    VARIANTS = ["default", "fr"]
    variant = st.selectbox(
        "Pipeline varyantı",
        VARIANTS, index=0,
        help="default: update_*.py • fr: update_*.fr.py / update_*_fr.py"
    )
    os.environ["PIPELINE_VARIANT"] = variant

    persist = st.selectbox(
        "Çıktıyı saklama modu",
        ["artifact", "commit", "none"], index=0,
        help="artifact: repo’yu bozmadan sakla • commit: repo’ya yaz • none: sadece log"
    )

    force_bypass = st.checkbox(
        "07:00 kapısını yok say (force)",
        value=True,
        help="İşaretli ise saat filtresi devre dışı kalır ve pipeline her saatte çalışır."
    )

    status_box = st.empty()
    _render_last_run_status(status_box)

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
        acs_year_default = os.environ.get("ACS_YEAR", "LATEST")
        whitelist_default = os.environ.get("DEMOG_WHITELIST", "")
        level_default = os.environ.get("CENSUS_GEO_LEVEL", "auto")

        acs_year_in = st.text_input(
            label="ACS_YEAR (LATEST veya YYYY)",
            value=str(acs_year_default or "LATEST"),
            key="acs_year_in",
            help="5-year ACS için en son yılı kullanmak genelde uygundur."
        )

        whitelist_in = st.text_input(
            label="DEMOG_WHITELIST (virgüllü; boş = hepsi)",
            value=str(whitelist_default or ""),
            key="demog_whitelist_in",
            help='Örn: "population,median_income,education". Metin eşleşmesiyle filtreler.'
        )

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

        pop_default = os.environ.get("POPULATION_PATH", str(POPULATION_PATH))
        pop_url_in = st.text_input(
            label="POPULATION_PATH (YEREL CSV YOLU)",
            value=str(pop_default or ""),
            key="population_path_in",
            help="Örn: crime_prediction_data/sf_population.csv (URL kabul edilmez)."
        )

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

# -----------------------------------------------------------------------------
# 0) (Opsiyonel) requirements yükleme
# -----------------------------------------------------------------------------
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
# 1) (Opsiyonel) Verileri indir ve önizle
# -----------------------------------------------------------------------------
st.markdown("### 1) (Opsiyonel) Verileri indir ve önizle")
if st.button("📥 Verileri İndir ve Önizle (İlk 3 Satır)"):
    for name, info in DOWNLOADS.items():
        download_and_preview(
            name,
            info.get("url", ""),
            info["path"],
            is_json=info.get("is_json", False),
            allow_artifact_fallback=info.get("allow_artifact", False),
            artifact_picks=info.get("artifact_picks"),
        )
    st.success("✅ İndirme tamamlandı.")

# -----------------------------------------------------------------------------
# 1.5) Dosyaları tarihe göre sırala
# -----------------------------------------------------------------------------
st.markdown("### 1.5) Dosyaları tarihe göre sırala")
colA, colB, colC = st.columns([1, 1, 2])
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

# -----------------------------------------------------------------------------
# 1.6) CSV → Parquet dönüştür
# -----------------------------------------------------------------------------
st.markdown("### 1.6) CSV → Parquet dönüştür")
with st.expander("🔄 CSV’leri Parquet’e çevir (zstd)"):
    in_dir = st.text_input(
        "Girdi klasörü", value=str(DATA_DIR),
        help="Örn: crime_prediction_data/",
        key="csv2parquet_in_dir"
    )
    out_dir = st.text_input(
        "Çıktı klasörü", value=str(ROOT / "parquet_out"),
        help="Örn: parquet_out/",
        key="csv2parquet_out_dir"
    )
    patt_in = st.text_input(
        "Desen (glob)", "*.csv",
        help="Örn: sf_crime_*.csv",
        key="csv2parquet_glob"
    )
    comp = st.selectbox(
        "Sıkıştırma",
        ["zstd", "snappy", "gzip", "brotli", "uncompressed"],
        index=0,
        key="csv2parquet_codec"
    )
    want_stats = st.checkbox(
        "Özet/stats üret", value=True,
        key="csv2parquet_stats"
    )

    if st.button("🧰 Dönüştür (CSV → Parquet)", key="csv2parquet_run"):
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

# -----------------------------------------------------------------------------
# Tanı ve Cache
# -----------------------------------------------------------------------------
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
# 2) Güncelleme ve Zenginleştirme (01 → 09)
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

# -----------------------------------------------------------------------------
# 3) Güncel _08 → _09 üret (sf + fr)
# -----------------------------------------------------------------------------
st.markdown("### 3) Güncel _08 → _09 üret (sf + fr)")
if st.button("🧪 _08'i temizle ve _09 üret (sf & fr)"):
    for prefix in ["sf", "fr"]:
        st.subheader(f"🔹 {prefix.upper()} akışı")
        _ = process_city_to_09(prefix, DATA_DIR)

# Önizleme: 08 ve 09 dosyaları (varsa)
for prefix in ["sf", "fr"]:
    st.subheader(f"📄 Önizleme — {prefix.upper()}")
    p08 = DATA_DIR / f"{prefix}_crime_08.csv"
    p09 = DATA_DIR / f"{prefix}_crime_09.csv"
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

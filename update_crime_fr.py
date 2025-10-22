# update_crime_fr.py  (event-based fr_crime.csv)
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import shutil

# --------------------------
# Ayarlar (ENV ile özelleştirilebilir)
# --------------------------
EVENTS_PATH = Path(os.getenv("FR_EVENTS_PATH", "sf_crime.csv"))  # olay bazlı kaynak (suç_id içermeli)
LABEL_PATH  = Path(os.getenv("FR_LABEL_PATH", "sf_crime_L.csv"))  # grid etiket kaynağı (ya da sf_crime_L)
OUT_PATH    = Path(os.getenv("FR_OUT_PATH",   "fr_crime.csv"))   # hedef: olay bazlı çıktı
MIRROR_DIR  = Path(os.getenv("FR_MIRROR_DIR", "crime_prediction_data"))

# Potansiyel eşleşme anahtarları (öncelik sırasıyla)
FULL_KEYS = ["GEOID", "season", "day_of_week", "event_hour"]
GEOID_ONLY = ["GEOID"]

# Çekilecek etiket kolonu adı
YCOL = os.getenv("FR_YCOL", "Y_label")

# --------------------------
# Yardımcılar
# --------------------------
def safe_read_csv(p: Path) -> pd.DataFrame:
    if not p.exists(): return pd.DataFrame()
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception as e:
        print(f"⚠️ Okunamadı: {p} → {e}")
        return pd.DataFrame()

def safe_save_csv(df: pd.DataFrame, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(p)
    print(f"💾 Kaydedildi: {p}  ({len(df):,} satır, {df.shape[1]} sütun)")

def normalize_event_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "GEOID" in df: df["GEOID"] = df["GEOID"].astype(str).str.extract(r"(\d+)")[0].str.zfill(11)
    for c in ("day_of_week","event_hour"):
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df

def normalize_label_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "GEOID" in df: df["GEOID"] = df["GEOID"].astype(str).str.extract(r"(\d+)")[0].str.zfill(11)
    for c in ("day_of_week","event_hour", YCOL):
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df

def dedupe_labels(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Aynı anahtar için birden fazla Y varsa 1'i tercih et; yoksa 0."""
    if not all(k in df.columns for k in keys+[YCOL]): 
        return df.drop_duplicates(subset=[c for c in keys if c in df.columns])
    agg = df.groupby(keys, as_index=False)[YCOL].max()  # 1 > 0
    return agg

def try_merge(events: pd.DataFrame, labels: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    keys = [k for k in keys if k in events.columns and k in labels.columns]
    if not keys:
        raise ValueError("Eşleşme için ortak anahtar bulunamadı.")
    L = dedupe_labels(labels, keys)
    merged = events.merge(L[keys+[YCOL]], on=keys, how="left")
    return merged, keys

# --------------------------
# Akış
# --------------------------
def main():
    # 1) Olay verisini oku
    events = safe_read_csv(EVENTS_PATH)
    if events.empty:
        print(f"❌ Olay verisi boş veya yok: {EVENTS_PATH}")
        return 0
    if "crime_id" not in events.columns:
        print("⚠️ Uyarı: 'crime_id' kolonu bulunamadı. Yine de olay bazlı devam edilecek.")
    events = normalize_event_df(events)
    base_len = len(events)

    # 2) Etiket gridini oku
    labels = safe_read_csv(LABEL_PATH)
    if labels.empty:
        print(f"❌ Etiket kaynağı boş veya yok: {LABEL_PATH}")
        return 0
    labels = normalize_label_df(labels)

    # 3) Önce tam anahtar, olmazsa yalnızca GEOID ile eşle
    used_keys = None
    try:
        out, used_keys = try_merge(events, labels, FULL_KEYS)
        print(f"🔗 Eşleşme anahtarları: {used_keys} (tam anahtar)")
    except Exception:
        out, used_keys = try_merge(events, labels, GEOID_ONLY)
        print(f"🔗 Eşleşme anahtarları: {used_keys} (yalnızca GEOID)")

    # 4) Eşleşemeyen satırlar hakkında bilgi
    missing = out[YCOL].isna().sum()
    if missing > 0:
        rate = round(missing / base_len * 100, 2)
        print(f"ℹ️ Eşleşemeyen olay satırı: {missing:,} (%{rate}) → {YCOL} NaN. NaN'ları 0'a çeviriyorum.")
        out[YCOL] = out[YCOL].fillna(0).astype(int)

    # 5) İz bilgisi
    out["fr_snapshot_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    out["fr_label_keys"]  = "+".join(used_keys)

    # 6) Kaydet & mirror
    safe_save_csv(out, OUT_PATH)
    try:
        MIRROR_DIR.mkdir(exist_ok=True)
        shutil.copy2(OUT_PATH, MIRROR_DIR / OUT_PATH.name)
        print(f"📦 Mirror kopya: {MIRROR_DIR / OUT_PATH.name}")
    except Exception as e:
        print(f"ℹ️ Mirror kopya atlandı/başarısız: {e}")

    # 7) Hızlı özet
    if YCOL in out.columns:
        vc = out[YCOL].value_counts(normalize=True).mul(100).round(2)
        print("\n📊 Y_label oranları (%):")
        print(vc.to_string())

    # 8) Temel kalite kontrolleri
    if "crime_id" in out.columns:
        dup = out["crime_id"].duplicated().sum()
        if dup:
            print(f"⚠️ Uyarı: {dup} adet tekrar eden crime_id var.")
        else:
            print("✅ crime_id benzersiz görünüyor.")

    return 0

if __name__ == "__main__":
    try:
        code = main()
        raise SystemExit(code if isinstance(code, int) else 0)
    except Exception as e:
        print(f"⚠️ FR derleme sırasında yakalanmamış hata: {e}")
        raise SystemExit(0)

# update_crime_fr.py
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import shutil

# --------------------------
# Basit ayarlar (ENV ile özelleştirilebilir)
# --------------------------
SRC_PATH   = Path(os.getenv("FR_SRC_PATH", "sf_crime_grid_full_labeled.csv"))          # kaynak (zenginleştirilmiş) grid
OUT_PATH   = Path(os.getenv("FR_OUT_PATH", "fr_crime.csv"))                 # çıktı
MIRROR_DIR = Path(os.getenv("FR_MIRROR_DIR", "crime_prediction_data"))  # opsiyonel ayna klasörü
KEYS       = ["GEOID","season","day_of_week","event_hour"]                  # grid anahtarı
# Yeni veriyi mevcut veriye göre tercih et
PREFER_NEW = os.getenv("FR_PREFER_NEW", "1").lower() in ("1","true","yes","on")

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

def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    if "GEOID" in df: df["GEOID"] = df["GEOID"].astype(str).str.extract(r"(\d+)")[0].str.zfill(11)
    for c in ("day_of_week","event_hour","crime_count","Y_label"):
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    if "crime_mix" in df: df["crime_mix"] = df["crime_mix"].fillna("").astype(str)
    return df

def coalesce(left, right, cols):
    """Sağdaki (new) doluysa onu kullan, değilse soldaki (old)."""
    out = left.copy()
    for c in cols:
        if c in right.columns:
            r = right[c]
            l = out.get(c)
            if l is None: out[c] = r
            else:
                out[c] = np.where(r.notna() & (r.astype(str) != "nan"), r, l)
    return out

# --------------------------
# Akış
# --------------------------
def main():
    if not SRC_PATH.exists():
        raise SystemExit(f"❌ Kaynak bulunamadı: {SRC_PATH}")

    new_df = safe_read_csv(SRC_PATH)
    if new_df.empty:
        raise SystemExit("❌ Kaynak boş görünüyor.")

    new_df = normalize_types(new_df.copy())
    new_df["fr_snapshot_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    old_df = safe_read_csv(OUT_PATH)
    if old_df.empty:
        # İlk kurulum: direkt yaz
        out = new_df
    else:
        old_df = normalize_types(old_df.copy())
        # Anahtar kümeyi garanti et
        for col in KEYS:
            if col not in new_df.columns:
                raise SystemExit(f"❌ Kaynakta eksik anahtar sütun: {col}")
        # Birleştir (outer): aynı anahtar için yeniyi tercih et
        all_cols = sorted(set(old_df.columns) | set(new_df.columns))
        merged = old_df.merge(new_df, on=KEYS, how="outer", suffixes=("_old","_new"), indicator=True)

        # Hangi kolonları koordine edeceğiz? (anahtarlar hariç hepsi)
        value_cols_old = [c for c in merged.columns if c.endswith("_old")]
        base_names = [c[:-4] for c in value_cols_old]
        value_cols_new = [f"{b}_new" for b in base_names if f"{b}_new" in merged.columns]

        # çıkış iskeleti
        out = merged[KEYS].copy()
        # coalesce: PREFER_NEW varsa new → old, yoksa old → new
        if PREFER_NEW:
            for base in base_names:
                ao, an = f"{base}_old", f"{base}_new"
                if an in merged.columns and ao in merged.columns:
                    out[base] = merged[an].where(merged[an].notna(), merged[ao])
                elif an in merged.columns:
                    out[base] = merged[an]
                else:
                    out[base] = merged.get(ao)
        else:
            for base in base_names:
                ao, an = f"{base}_old", f"{base}_new"
                if ao in merged.columns and an in merged.columns:
                    out[base] = merged[ao].where(merged[ao].notna(), merged[an])
                elif ao in merged.columns:
                    out[base] = merged[ao]
                else:
                    out[base] = merged.get(an)

        # 'new_df'te olup 'old_df'te hiç olmayan yeni kolonlar
        extra_new_cols = [c for c in new_df.columns if c not in KEYS and c not in base_names]
        for c in extra_new_cols:
            if f"{c}_new" in merged.columns:
                out[c] = merged[f"{c}_new"]
            elif f"{c}_old" in merged.columns:
                out[c] = merged[f"{c}_old"]

        out = normalize_types(out)
        # tutarlı sıralama
        order = KEYS + [c for c in out.columns if c not in KEYS]
        out = out[order].sort_values(KEYS).reset_index(drop=True)

    # yaz ve mirror’a kopyala
    safe_save_csv(out, OUT_PATH)

    try:
        MIRROR_DIR.mkdir(exist_ok=True)
        shutil.copy2(OUT_PATH, MIRROR_DIR / OUT_PATH.name)
        print(f"📦 Mirror kopya: {MIRROR_DIR / OUT_PATH.name}")
    except Exception as e:
        print(f"ℹ️ Mirror kopya atlandı/başarısız: {e}")

    # hızlı özet
    if "Y_label" in out:
        vc = out["Y_label"].value_counts(normalize=True).mul(100).round(2)
        print("\n📊 Y_label oranları (%):")
        print(vc.to_string())

if __name__ == "__main__":
    main()

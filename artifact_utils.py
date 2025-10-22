# artifact_utils.py
from __future__ import annotations
import os, io, zipfile, glob
from pathlib import Path
from typing import Iterable, Optional, Union, List, Dict, Any
import pandas as pd

# ---------- Config (ENV) ----------
ARTIFACT_ZIP = Path(os.getenv("ARTIFACT_ZIP", "artifact/sf-crime-pipeline-output.zip"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifact_unzipped"))
# Varsayılan fallback dizinleri: crime_prediction_data ve çalışma dizini
FALLBACK_DIRS = [p for p in (os.getenv("FALLBACK_DIRS", "crime_prediction_data,.").split(",")) if p]
FALLBACK_DIRS = [Path(p.strip()) for p in FALLBACK_DIRS]
OUTPUT_DIR = Path(os.getenv("FR_OUTPUT_DIR", str(ARTIFACT_DIR)))

# ---------- Logging ----------
def log(msg: str): print(msg, flush=True)

# ---------- ZIP (safe extract) ----------
def _is_within(directory: Path, target: Path) -> bool:
    try:
        return str(target.resolve()).startswith(str(directory.resolve()))
    except Exception:
        return False

def ensure_unzipped(zip_path: Path = ARTIFACT_ZIP, dest_dir: Path = ARTIFACT_DIR) -> None:
    """
    ZIP varsa güvenli şekilde bir kez çıkarır. Yoksa sessizce geçer.
    mtime karşılaştırmasıyla gereksiz tekrar açmayı önler.
    """
    if not zip_path.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)

    stamp = dest_dir / ".unzipped_from"
    current_src = str(zip_path.resolve())
    need_extract = True
    if stamp.exists():
        old = stamp.read_text().strip()
        try:
            need_extract = (old != current_src) or (
                zip_path.stat().st_mtime > (dest_dir / ".unzipped_time").stat().st_mtime
            )
        except Exception:
            need_extract = True

    if not need_extract:
        return

    log(f"📦 ZIP açılıyor: {zip_path} → {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            out_path = dest_dir / info.filename
            if not _is_within(dest_dir, out_path.parent):
                raise RuntimeError(f"Zip path outside target dir engellendi: {info.filename}")
            if info.is_dir():
                out_path.mkdir(parents=True, exist_ok=True)
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
    (dest_dir / ".unzipped_from").write_text(current_src)
    (dest_dir / ".unzipped_time").write_text("ok")
    log("✅ ZIP çıkarma tamam.")

# ---------- Path resolution ----------
PathLike = Union[str, Path]

def _as_list(x: Union[PathLike, Iterable[PathLike]]) -> List[Path]:
    if isinstance(x, (str, Path)):
        return [Path(x)]
    return [Path(p) for p in x]

def first_existing(paths: Iterable[PathLike]) -> Optional[Path]:
    for p in _as_list(paths):
        if p.exists():
            return p
    return None

def _glob_first(root: Path, pattern: str) -> Optional[Path]:
    # Alt klasörlerde arar (örn: **/fr_crime.csv)
    matches = sorted(root.glob(f"**/{pattern}"))
    return matches[0] if matches else None

def find_path(name_or_candidates: Union[str, Path, Iterable[PathLike]],
              must_exist: bool = True) -> Optional[Path]:
    """
    - Tek bir dosya adı (örn 'fr_crime.csv') verirseniz:
        1) ARTIFACT_DIR/**/name
        2) Her fallback_dir/**/name
    - Aday listesi verirseniz sırayla ilk bulunan döner.
    """
    ensure_unzipped()  # önce zip'i hazırla

    # Liste ise doğrudan sırayla bak
    if not isinstance(name_or_candidates, (str, Path)):
        return first_existing(name_or_candidates) if not must_exist else _require(first_existing(name_or_candidates))

    name = Path(name_or_candidates).name  # sadece dosya adını al
    # 1) artifact içinde ara
    p = _glob_first(ARTIFACT_DIR, name)
    if p: return p
    # 2) fallback dizinlerinde ara
    for root in FALLBACK_DIRS:
        p = _glob_first(root, name)
        if p: return p

    if must_exist:
        raise FileNotFoundError(f"Bulunamadı: {name} (artifact ve fallback dizinlerde)")
    return None

def _require(p: Optional[Path]) -> Path:
    if p is None:
        raise FileNotFoundError("Gerekli dosya bulunamadı.")
    return p

# ---------- CSV IO (atomic save) ----------
def read_csv(name_or_candidates: Union[str, Path, Iterable[PathLike]], **pd_kwargs) -> pd.DataFrame:
    """
    `find_path` ile yolu çözer, sonra pandas read_csv çağırır.
    """
    path = find_path(name_or_candidates, must_exist=True)
    return pd.read_csv(path, **({"low_memory": False} | pd_kwargs))

def save_csv(df: pd.DataFrame, out_name: str, output_dir: PathLike = OUTPUT_DIR) -> Path:
    """
    df'yi output_dir altında out_name ile atomik kaydeder.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / out_name
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_path)
    log(f"💾 Kaydedildi: {out_path} (satır={len(df):,}, sütun={df.shape[1]})")
    return out_path

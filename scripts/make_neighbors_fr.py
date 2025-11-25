import pandas as pd
from collections import defaultdict
from pathlib import Path

# ---------- paths ----------
DATA_DIR = Path(".")  # gerekirse repo içindeki klasöre göre değiştir
crime_path = DATA_DIR / "fr_crime_08.csv"
neighbors_path = DATA_DIR / "neighbors.csv"
out_path = DATA_DIR / "fr_crime_09.csv"

# ---------- load crimes ----------
df = pd.read_csv(crime_path)

# datetime oluştur / parse et
if "datetime" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
else:
    # date + time sütunları varsa birleştir
    if "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str),
                                        errors="coerce")
    else:
        raise ValueError("Ne 'datetime' ne de ('date','time') sütunları bulunamadı.")

# GEOID kontrolü
if "GEOID" not in df.columns:
    raise ValueError("fr_crime_08.csv içinde 'GEOID' sütunu yok.")

# NaT olanları at (istersen farklı ele al)
df = df.dropna(subset=["datetime"]).copy()

# ---------- load neighbors ----------
nbr_raw = pd.read_csv(neighbors_path)

# neighbors.csv ilk iki kolonu al (isim ne olursa olsun)
if nbr_raw.shape[1] < 2:
    raise ValueError("neighbors.csv en az iki kolona sahip olmalı (GEOID, neighbor).")

col_a, col_b = nbr_raw.columns[:2]
edges = nbr_raw[[col_a, col_b]].dropna()

# adjacency list (undirected)
adj = defaultdict(set)
for a, b in edges.itertuples(index=False):
    adj[a].add(b)
    adj[b].add(a)

# ---------- helper: rolling counts on neighbor events ----------
# df'yi sıralayıp indeks üzerinden geri map'leyelim
df = df.sort_values(["GEOID", "datetime"]).reset_index(drop=False).rename(columns={"index": "_orig_idx"})

# her GEOID için olay zamanları listesi
events_by_geo = {g: sub[["datetime"]].copy() for g, sub in df.groupby("GEOID")}

# sonuçları burada toplayacağız (orijinal indexe göre)
neighbor_24h = pd.Series(0, index=df.index, dtype="int64")
neighbor_72h = pd.Series(0, index=df.index, dtype="int64")
neighbor_7d  = pd.Series(0, index=df.index, dtype="int64")

# pencereler
win_24h = pd.Timedelta(hours=24)
win_72h = pd.Timedelta(hours=72)
win_7d  = pd.Timedelta(days=7)

for geo, sub in df.groupby("GEOID", sort=False):
    neighs = list(adj.get(geo, []))
    if len(neighs) == 0:
        continue

    # komşu GEOID’lerin bütün eventlerini bir araya getir
    neigh_events = []
    for n in neighs:
        if n in events_by_geo:
            neigh_events.append(events_by_geo[n])
    if not neigh_events:
        continue

    neigh_df = pd.concat(neigh_events, ignore_index=True).sort_values("datetime")
    neigh_times = neigh_df["datetime"]

    # hedef GEOID olayları
    target_times = sub["datetime"]

    # rolling sayım için komşu olaylarını time-index yap
    neigh_df = neigh_df.set_index("datetime")
    # 1'lik seri (her komşu olay = 1)
    ones = pd.Series(1, index=neigh_df.index)

    # hedef zamanları komşu serisine göre asof ile geri bakıp kümülatif fark alacağız
    # önce komşu olayların kümülatif toplamı
    csum = ones.cumsum()

    # yardımcı fonksiyon: [t-win, t] aralığındaki komşu olay sayısı
    def count_in_window(t_series, window):
        left = t_series - window

        # csum'u t ve left için asof'la çek
        right_val = pd.merge_asof(
            pd.DataFrame({"t": t_series}).sort_values("t"),
            csum.rename("csum").reset_index().sort_values("datetime"),
            left_on="t", right_on="datetime",
            direction="backward"
        )["csum"].fillna(0).to_numpy()

        left_val = pd.merge_asof(
            pd.DataFrame({"t": left}).sort_values("t"),
            csum.rename("csum").reset_index().sort_values("datetime"),
            left_on="t", right_on="datetime",
            direction="backward"
        )["csum"].fillna(0).to_numpy()

        return (right_val - left_val).astype("int64")

    idx = sub.index
    neighbor_24h.loc[idx] = count_in_window(target_times, win_24h)
    neighbor_72h.loc[idx] = count_in_window(target_times, win_72h)
    neighbor_7d.loc[idx]  = count_in_window(target_times, win_7d)

# ---------- assign back / save ----------
df["neighbor_crime_24h"] = neighbor_24h
df["neighbor_crime_72h"] = neighbor_72h
df["neighbor_crime_7d"]  = neighbor_7d

# orijinal sıralamaya dön
df = df.sort_values("_orig_idx").drop(columns=["_orig_idx"])

df.to_csv(out_path, index=False)
print(f"Saved: {out_path}")

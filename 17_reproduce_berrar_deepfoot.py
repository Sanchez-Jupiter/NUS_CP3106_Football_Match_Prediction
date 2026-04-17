"""
17_reproduce_berrar_deepfoot.py
───────────────────────────────
Reproduce: Berrar, Lopes & Dubitzky (2024)
"A data and knowledge-driven framework for developing
 machine learning models to predict soccer match outcomes"
 Machine Learning, 113:8165–8204

Method
------
Knowledge-driven feature engineering from historical match scores ONLY.
Each match → 6 features (total feature set):
  scr_h / con_h / rank_h  — home team's avg goals scored / conceded / normalised rank
  scr_a / con_a / rank_a  — away team's same stats
All computed from each team's most recent *n* matches ("super league" view).

Models:  k-NN · ANN (MLP) · Gaussian Naive Bayes
Metrics: RPS (primary, lower=better) · RMSE (score) · Accuracy · F1-macro
Eval:    Strict temporal holdout (80 / 20 by date)
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings, json, os, time

warnings.filterwarnings("ignore")

# ═══════════════════ Configuration ═══════════════════
DATA_PATH    = "data/processed/pretrain_dataset.csv"
REPORT_DIR   = "reports"
N_CANDIDATES = [6, 10, 15, 20, 30, 40]        # recency depths to explore
K_CANDIDATES = [5, 10, 20, 30, 50]            # k values for k-NN
TEST_RATIO   = 0.20
SEED         = 42
FEAT_COLS    = ["scr_h", "con_h", "rank_h", "scr_a", "con_a", "rank_a"]

os.makedirs(REPORT_DIR, exist_ok=True)


# ═══════════════════ Metrics ═══════════════════
def rps_avg(y_true, y_prob):
    """Average Ranked Probability Score for 3-class ordinal (H, D, A).
    Lower is better.  Eq.(2) in Berrar et al."""
    hot = {"H": np.array([1, 0, 0]),
           "D": np.array([0, 1, 0]),
           "A": np.array([0, 0, 1])}
    total = 0.0
    for i in range(len(y_true)):
        cum_p = np.cumsum(y_prob[i])
        cum_y = np.cumsum(hot[y_true[i]])
        total += 0.5 * np.sum((cum_p[:2] - cum_y[:2]) ** 2)
    return total / len(y_true)


def rmse_score(true_h, true_a, pred_h, pred_a):
    """Score-prediction RMSE (Berrar definition — Eq.(1))."""
    return np.sqrt(np.mean((true_h - pred_h) ** 2 + (true_a - pred_a) ** 2))


# ═══════════════════ Feature Engineering ═══════════════════
def build_features_fast(df, n, compute_rank=True):
    """Build 6-feature total set. Optimised single-pass implementation.

    When compute_rank=False (n-search mode), rank features are set to 0.5.
    When compute_rank=True, rank is approximated as normalised
    points-per-game from the last n matches (higher = better, 0–1 range).
    This is faster than building a full league table and preserves the
    cardinal strength signal that ordinal rank would compress.
    """
    df = df.sort_values("date").reset_index(drop=True)

    # ── pre-extract arrays for speed (avoid DataFrame overhead) ──
    dates     = df["date"].values
    ht_arr    = df["home_team"].values
    at_arr    = df["away_team"].values
    gh_arr    = df["goals_home"].values.astype(float)
    ga_arr    = df["goals_away"].values.astype(float)
    res_arr   = df["result"].values

    # per-team histories: plain Python lists (append-only)
    t_gf  = defaultdict(list)   # goals FOR  per match
    t_ga  = defaultdict(list)   # goals AGAINST per match
    t_pts = defaultdict(list)   # points (3/1/0)

    out_dates, out_ht, out_at = [], [], []
    out_scr_h, out_con_h, out_rank_h = [], [], []
    out_scr_a, out_con_a, out_rank_a = [], [], []
    out_res, out_gh, out_ga = [], [], []

    prev_date = None
    batch_idx = []                # indices of matches on current date

    def _flush(batch):
        """Compute features for a batch of matches sharing one date,
        then update histories."""
        nonlocal prev_date

        for i in batch:
            ht, at = ht_arr[i], at_arr[i]
            if len(t_gf[ht]) >= n and len(t_gf[at]) >= n:
                h_scr = sum(t_gf[ht][-n:]) / n
                h_con = sum(t_ga[ht][-n:]) / n
                a_scr = sum(t_gf[at][-n:]) / n
                a_con = sum(t_ga[at][-n:]) / n
                if compute_rank:
                    h_rank = sum(t_pts[ht][-n:]) / (3.0 * n)
                    a_rank = sum(t_pts[at][-n:]) / (3.0 * n)
                else:
                    h_rank = a_rank = 0.5

                out_dates.append(dates[i])
                out_ht.append(ht);           out_at.append(at)
                out_scr_h.append(h_scr);     out_con_h.append(h_con)
                out_rank_h.append(h_rank)
                out_scr_a.append(a_scr);     out_con_a.append(a_con)
                out_rank_a.append(a_rank)
                out_res.append(res_arr[i])
                out_gh.append(gh_arr[i]);    out_ga.append(ga_arr[i])

        # update histories after the matchday
        for i in batch:
            ht, at = ht_arr[i], at_arr[i]
            gh, ga = gh_arr[i], ga_arr[i]
            t_gf[ht].append(gh);  t_ga[ht].append(ga)
            t_gf[at].append(ga);  t_ga[at].append(gh)
            pts_h = 3.0 if gh > ga else (1.0 if gh == ga else 0.0)
            pts_a = 3.0 if ga > gh else (1.0 if gh == ga else 0.0)
            t_pts[ht].append(pts_h); t_pts[at].append(pts_a)

    for idx in range(len(df)):
        cur_date = dates[idx]
        if cur_date != prev_date and batch_idx:
            _flush(batch_idx)
            batch_idx = []
        batch_idx.append(idx)
        prev_date = cur_date

    if batch_idx:
        _flush(batch_idx)

    return pd.DataFrame({
        "date": out_dates, "home_team": out_ht, "away_team": out_at,
        "scr_h": out_scr_h, "con_h": out_con_h, "rank_h": out_rank_h,
        "scr_a": out_scr_a, "con_a": out_con_a, "rank_a": out_rank_a,
        "result": out_res, "goals_home": out_gh, "goals_away": out_ga,
    })


# ═══════════════════ Helpers ═══════════════════
def ordered_prob(clf, X):
    """Return predict_proba columns in [H, D, A] order."""
    prob = clf.predict_proba(X)
    cls  = list(clf.classes_)
    idx  = [cls.index(c) for c in ["H", "D", "A"]]
    return prob[:, idx]


# ═══════════════════ Main ═══════════════════
def main():
    t0 = time.time()
    print("=" * 72)
    print("  Berrar et al. (2024) 'DeepFoot' Framework — Reproduction")
    print("  Features from historical scores only · strict temporal holdout")
    print("=" * 72)

    # ── 1. Load ──
    use_cols = ["date", "home_team", "away_team",
                "goals_home", "goals_away", "result"]
    df = pd.read_csv(DATA_PATH, usecols=use_cols).dropna(subset=use_cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"\nData  : {len(df):,} matches · {df['home_team'].nunique()} teams")
    print(f"Range : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Result: {df['result'].value_counts().to_dict()}")

    # ── 2. Optimal n (Pearson correlation approach, §5 of paper) ──
    print(f"\n{'─' * 72}")
    print("Step 1 · Determining optimal recency depth n  (Pearson method)")
    print(f"{'─' * 72}")
    best_n, best_r = None, -np.inf
    n_stats = []
    for n in N_CANDIDATES:
        fdf = build_features_fast(df, n, compute_rank=False)
        if len(fdf) < 200:
            print(f"  n={n:3d}  →  {len(fdf):5d} samples (too few, skip)")
            continue
        actual_gd   = fdf["goals_home"].values - fdf["goals_away"].values
        expected_gd = ((fdf["scr_h"] - fdf["con_h"]) -
                       (fdf["scr_a"] - fdf["con_a"])).values
        corr = np.corrcoef(actual_gd, expected_gd)[0, 1]
        n_stats.append({"n": n, "samples": len(fdf),
                        "pearson_r": round(corr, 4)})
        mark = "  ← best" if corr > best_r else ""
        if corr > best_r:
            best_r, best_n = corr, n
        print(f"  n={n:3d}  →  {len(fdf):5d} samples   "
              f"r = {corr:.4f}{mark}")
    print(f"\n  ✔ Selected n = {best_n}  (r = {best_r:.4f})")

    # ── 3. Build final feature set ──
    print(f"\n{'─' * 72}")
    print(f"Step 2 · Building features  (n = {best_n})")
    print(f"{'─' * 72}")
    feat = build_features_fast(df, best_n, compute_rank=True).sort_values("date").reset_index(drop=True)
    print(f"  Usable matches: {len(feat):,}")
    print(f"  Feature stats:")
    print(feat[FEAT_COLS].describe().round(3).to_string())

    # ── 4. Temporal split ──
    sp = int(len(feat) * (1 - TEST_RATIO))
    tr, te = feat.iloc[:sp].copy(), feat.iloc[sp:].copy()
    print(f"\n  Train : {len(tr):,}  (→ {tr['date'].max().date()})")
    print(f"  Test  : {len(te):,}  ({te['date'].min().date()} →)")
    for lbl in ["H", "D", "A"]:
        print(f"    {lbl}: train {(tr['result']==lbl).sum()}"
              f"  test {(te['result']==lbl).sum()}")

    X_tr, X_te      = tr[FEAT_COLS].values, te[FEAT_COLS].values
    y_tr, y_te      = tr["result"].values,  te["result"].values
    gh_tr, ga_tr    = tr["goals_home"].values, tr["goals_away"].values
    gh_te, ga_te    = te["goals_home"].values, te["goals_away"].values

    scaler = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)

    # encode labels for sklearn compatibility (MLP early_stopping)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(["A", "D", "H"])
    y_tr_enc = le.transform(y_tr)
    y_te_enc = le.transform(y_te)

    results = []

    # ── 5. Reference: Null model ──
    print(f"\n{'─' * 72}")
    print("Step 3 · Reference model")
    print(f"{'─' * 72}")
    ph = np.mean(y_tr == "H")
    pd_ = np.mean(y_tr == "D")
    pa = np.mean(y_tr == "A")
    null_prob = np.tile([ph, pd_, pa], (len(y_te), 1))
    null_rps  = rps_avg(y_te, null_prob)
    null_pred = np.full(len(y_te), "H")
    null_rmse = rmse_score(gh_te, ga_te,
                           np.full(len(gh_te), gh_tr.mean()),
                           np.full(len(ga_te), ga_tr.mean()))
    null_acc  = accuracy_score(y_te, null_pred)
    null_f1   = f1_score(y_te, null_pred, average="macro", zero_division=0)
    results.append(dict(model="Null (avg distribution)", RPS=null_rps,
                        RMSE=null_rmse, Acc=null_acc, F1=null_f1))
    print(f"  Null model  →  RPS={null_rps:.4f}   RMSE={null_rmse:.4f}   "
          f"Acc={null_acc:.4f}")

    # ── 6. ML Models ──
    print(f"\n{'─' * 72}")
    print("Step 4 · Machine learning models")
    print(f"{'─' * 72}")

    best_preds = {}  # model_name → predictions (for confusion matrix)

    # 6a  k-NN
    print("\n  ▸ k-Nearest Neighbors")
    for k in K_CANDIDATES:
        if k >= len(Xs_tr):
            continue
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        knn.fit(Xs_tr, y_tr)
        pred = knn.predict(Xs_te)
        prob = ordered_prob(knn, Xs_te)
        rps  = rps_avg(y_te, prob)
        acc  = accuracy_score(y_te, pred)
        f1   = f1_score(y_te, pred, average="macro")
        # score regression
        kr_h = KNeighborsRegressor(n_neighbors=k).fit(Xs_tr, gh_tr)
        kr_a = KNeighborsRegressor(n_neighbors=k).fit(Xs_tr, ga_tr)
        rmse = rmse_score(gh_te, ga_te,
                          kr_h.predict(Xs_te), kr_a.predict(Xs_te))
        name = f"k-NN (k={k})"
        results.append(dict(model=name, RPS=rps, RMSE=rmse, Acc=acc, F1=f1))
        best_preds[name] = pred
        print(f"    k={k:3d}  →  RPS={rps:.4f}  RMSE={rmse:.4f}  "
              f"Acc={acc:.4f}  F1={f1:.4f}")

    # 6b  ANN (MLP)
    print("\n  ▸ Feed-forward Neural Network (MLP)")
    ann_cfgs = [(64, 32), (128, 64), (256, 128, 64)]
    for hid in ann_cfgs:
        mlp = MLPClassifier(hidden_layer_sizes=hid, max_iter=500,
                            random_state=SEED, early_stopping=True,
                            validation_fraction=0.1)
        mlp.fit(Xs_tr, y_tr_enc)
        pred_enc = mlp.predict(Xs_te)
        pred = le.inverse_transform(pred_enc)
        prob = mlp.predict_proba(Xs_te)
        # reorder columns to [H, D, A]
        cls = list(le.classes_)      # ['A', 'D', 'H']
        idx = [cls.index(c) for c in ["H", "D", "A"]]
        prob_ordered = prob[:, idx]
        rps  = rps_avg(y_te, prob_ordered)
        acc  = accuracy_score(y_te, pred)
        f1   = f1_score(y_te, pred, average="macro")
        # score regression
        reg = MLPRegressor(hidden_layer_sizes=hid, max_iter=500,
                           random_state=SEED, early_stopping=True)
        reg.fit(Xs_tr, np.c_[gh_tr, ga_tr])
        pg  = reg.predict(Xs_te)
        rmse = rmse_score(gh_te, ga_te, pg[:, 0], pg[:, 1])
        tag  = "-".join(map(str, hid))
        name = f"ANN ({tag})"
        results.append(dict(model=name, RPS=rps, RMSE=rmse, Acc=acc, F1=f1))
        best_preds[name] = pred
        print(f"    {tag:14s}→  RPS={rps:.4f}  RMSE={rmse:.4f}  "
              f"Acc={acc:.4f}  F1={f1:.4f}")

    # 6c  Gaussian Naive Bayes
    print("\n  ▸ Gaussian Naive Bayes")
    nb = GaussianNB().fit(Xs_tr, y_tr)
    pred = nb.predict(Xs_te)
    prob = ordered_prob(nb, Xs_te)
    rps  = rps_avg(y_te, prob)
    acc  = accuracy_score(y_te, pred)
    f1   = f1_score(y_te, pred, average="macro")
    # score via expected value under class-conditional mean
    avg_by = {}
    for c in ["H", "D", "A"]:
        m = y_tr == c
        avg_by[c] = (gh_tr[m].mean(), ga_tr[m].mean())
    s_h = prob @ np.array([avg_by["H"][0], avg_by["D"][0], avg_by["A"][0]])
    s_a = prob @ np.array([avg_by["H"][1], avg_by["D"][1], avg_by["A"][1]])
    rmse_nb = rmse_score(gh_te, ga_te, s_h, s_a)
    results.append(dict(model="Naive Bayes", RPS=rps, RMSE=rmse_nb,
                        Acc=acc, F1=f1))
    best_preds["Naive Bayes"] = pred
    print(f"    {'':14s}→  RPS={rps:.4f}  RMSE={rmse_nb:.4f}  "
          f"Acc={acc:.4f}  F1={f1:.4f}")

    # ── 7. Summary ──
    rdf = pd.DataFrame(results).sort_values("RPS")
    print(f"\n{'=' * 72}")
    print("  RESULTS  (sorted by RPS — lower is better)")
    print(f"{'=' * 72}")
    print(rdf.to_string(index=False, float_format="%.4f"))

    # confusion matrix for best ML model (lowest RPS, excluding Null)
    ml_rows = rdf[rdf["model"] != "Null (avg distribution)"]
    if not ml_rows.empty:
        bname = ml_rows.iloc[0]["model"]
        bp    = best_preds.get(bname, None)
        if bp is not None:
            cm = confusion_matrix(y_te, bp, labels=["H", "D", "A"])
            print(f"\n  Best ML model: {bname}")
            print(f"  Confusion matrix (rows=true, cols=pred):")
            print(f"  {'':8s} Pred_H  Pred_D  Pred_A")
            for i, lab in enumerate(["H", "D", "A"]):
                print(f"  True_{lab}  {cm[i,0]:6d}  {cm[i,1]:6d}  "
                      f"{cm[i,2]:6d}")

    # ── 8. Save reports ──
    rdf.to_csv(os.path.join(REPORT_DIR, "17_berrar_deepfoot_results.csv"),
               index=False)

    lines = [
        "=" * 72,
        "Berrar et al. (2024) DeepFoot Reproduction Report",
        "=" * 72,
        f"Data        : {len(df):,} matches, {df['home_team'].nunique()} teams",
        f"Date range  : {df['date'].min().date()} → {df['date'].max().date()}",
        f"Optimal n   : {best_n}  (Pearson r = {best_r:.4f})",
        f"Features    : {FEAT_COLS}",
        f"Usable      : {len(feat):,} matches (after n-match warm-up)",
        f"Train / Test: {len(tr):,} / {len(te):,} (temporal 80/20)",
        "",
        "Paper context:",
        "  - Uses match scores ONLY (no shots / corners / cards / odds)",
        "  - RPS is the primary evaluation metric (lower = better)",
        "  - Paper's best ML model: k-NN  RPS ≈ 0.211  (44 leagues, 736 matches)",
        "  - Bookmaker odds baseline: RPS ≈ 0.206",
        "",
        "Our results:",
        rdf.to_string(index=False, float_format="%.4f"),
        "",
        f"Elapsed: {time.time() - t0:.1f}s",
    ]
    with open(os.path.join(REPORT_DIR, "17_berrar_deepfoot_report.txt"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    report_json = {
        "paper": "Berrar, Lopes & Dubitzky (2024)",
        "title": ("A data and knowledge-driven framework for developing "
                  "ML models to predict soccer match outcomes"),
        "method": "Knowledge-driven features from historical scores only",
        "optimal_n": int(best_n),
        "pearson_r": round(float(best_r), 4),
        "n_search": n_stats,
        "features": FEAT_COLS,
        "data_matches": len(df),
        "feature_matches": len(feat),
        "train_size": len(tr),
        "test_size": len(te),
        "train_end": str(tr["date"].max().date()),
        "test_start": str(te["date"].min().date()),
        "results": json.loads(rdf.to_json(orient="records")),
    }
    with open(os.path.join(REPORT_DIR, "17_berrar_deepfoot_report.json"),
              "w") as f:
        json.dump(report_json, f, indent=2, default=str)

    print(f"\nReports → {REPORT_DIR}/17_berrar_deepfoot_*")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

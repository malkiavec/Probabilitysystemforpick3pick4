# shift_based_prediction.py
# Run: streamlit run shift_based_prediction.py

import io
import itertools as it
import math
from collections import Counter, defaultdict
from functools import lru_cache
from typing import List, Tuple, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Optional: only import plotting when used
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except Exception:
    HAS_PLOTTING = False

# Optional: Hungarian algorithm for optimal multiset mapping
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# -----------------------------
# Utility Functions
# -----------------------------
def _digits_only(s: str) -> List[int]:
    return [int(ch) for ch in s if ch.isdigit()]

def to_tuple(x, n_digits: Optional[int] = None) -> Tuple[int, ...]:
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        digs = [int(v) for v in x]
    else:
        s = str(x)
        digs = _digits_only(s)
    if n_digits is not None:
        if len(digs) > n_digits:
            digs = digs[-n_digits:]
        elif len(digs) < n_digits:
            digs = [0] * (n_digits - len(digs)) + digs
    return tuple(digs)

def tuple_to_str(t: Tuple[int, ...]) -> str:
    return "".join(str(int(d)) for d in t)

def greedy_multiset_mapping(a: Tuple[int, ...], b: Tuple[int, ...]) -> List[Tuple[int, int]]:
    ca, cb = Counter(a), Counter(b)
    pairs: List[Tuple[int, int]] = []
    for d in range(10):
        m = min(ca[d], cb[d])
        if m:
            pairs.extend((d, d) for _ in range(m))
            ca[d] -= m
            cb[d] -= m
    rem_a, rem_b = [], []
    for d in range(10):
        if ca[d] > 0:
            rem_a.extend([d] * ca[d])
        if cb[d] > 0:
            rem_b.extend([d] * cb[d])
    rem_a.sort()
    rem_b.sort()
    pairs.extend(zip(rem_a, rem_b))
    return pairs

def optimal_multiset_mapping(a: Tuple[int, ...], b: Tuple[int, ...], costs: Dict[Tuple[int, int], float]) -> List[Tuple[int, int]]:
    if not HAS_SCIPY:
        return greedy_multiset_mapping(a, b)
    na, nb = len(a), len(b)
    if na != nb:
        if na < nb:
            a = a + tuple([-1] * (nb - na))
        else:
            b = b + tuple([-1] * (na - nb))
    if len(a) > 8:
        return greedy_multiset_mapping(a, b)
    A = list(a)
    B = list(b)
    n = len(A)
    C = np.zeros((n, n), dtype=float)
    worst = 20.0
    for i, x in enumerate(A):
        for j, y in enumerate(B):
            if x == -1 or y == -1:
                C[i, j] = worst
            else:
                C[i, j] = costs.get((x, y), worst)
    r, c = linear_sum_assignment(C)
    pairs = []
    for i, j in zip(r, c):
        xi, yj = A[i], B[j]
        if xi == -1 or yj == -1:
            continue
        pairs.append((xi, yj))
    return pairs

def extract_digit_transitions(draws: List[Tuple[int, ...]], lag: int, mapping="greedy") -> Counter:
    trans: Counter = Counter()
    if not draws or lag <= 0:
        return trans
    if mapping == "optimal":
        base_costs = {(x, y): (0.0 if x == y else 1.0) for x in range(10) for y in range(10)}
    for i in range(len(draws) - lag):
        a, b = draws[i], draws[i + lag]
        if mapping == "optimal":
            pairs = optimal_multiset_mapping(a, b, base_costs)
        else:
            pairs = greedy_multiset_mapping(a, b)
        for x, y in pairs:
            trans[(x, y)] += 1
    return trans

def normalize_matrix(cnt: Counter, alpha: float = 0.5) -> Dict[Tuple[int, int], float]:
    totals = Counter()
    for (x, y), c in cnt.items():
        totals[x] += c
    probs: Dict[Tuple[int, int], float] = {}
    for x in range(10):
        row_total = totals[x]
        denom = row_total + alpha * 10
        if denom <= 0:
            for y in range(10):
                probs[(x, y)] = 1.0 / 10.0
            continue
        for y in range(10):
            c = cnt.get((x, y), 0.0)
            probs[(x, y)] = (c + alpha) / denom
    return probs

def transition_matrix_to_df(trans: Counter) -> pd.DataFrame:
    mat = np.zeros((10, 10), dtype=float)
    row_totals = np.zeros(10, dtype=float)
    for (x, y), c in trans.items():
        mat[x, y] += c
        row_totals[x] += c
    for x in range(10):
        s = row_totals[x]
        if s > 0:
            mat[x] /= s
        else:
            mat[x][:] = 0.1
    df = pd.DataFrame(mat, index=[f"{i}" for i in range(10)], columns=[f"{j}" for j in range(10)])
    return df

def apply_positionless_transitions(
    seed: Tuple[int, ...],
    probs: Dict[Tuple[int, int], float],
    top_k: int = 3,
) -> List[Tuple[int, ...]]:
    choices_per_digit: List[List[int]] = []
    for v in seed:
        dist = [(y, probs.get((v, y), 0.0)) for y in range(10)]
        dist.sort(key=lambda t: t[1], reverse=True)
        top = [y for y, _ in dist[: max(1, top_k)]]
        if v not in top:
            top = [v] + top[:-1]
        choices_per_digit.append(top)
    raw = it.product(*choices_per_digit)
    outs = {tuple(r) for r in raw}
    return list(outs)

def score_by_transition_likelihood(
    cand: Tuple[int, ...],
    seed: Tuple[int, ...],
    probs: Dict[Tuple[int, int], float],
    mapping: str = "greedy",
) -> float:
    if mapping == "optimal" and HAS_SCIPY:
        costs = {}
        for x in range(10):
            for y in range(10):
                p = max(probs.get((x, y), 1e-12), 1e-12)
                costs[(x, y)] = -math.log(p)
        pairs = optimal_multiset_mapping(seed, cand, costs)
    else:
        pairs = greedy_multiset_mapping(seed, cand)
    score = 0.0
    for x, y in pairs:
        p = max(probs.get((x, y), 1e-12), 1e-12)
        score += math.log(p)
    return score

@st.cache_data(show_spinner=False)
def parse_draws_from_df(df: pd.DataFrame, n_digits: int, hist_col: Optional[str]) -> List[Tuple[int, ...]]:
    if hist_col and hist_col in df.columns:
        col = df[hist_col]
    else:
        col = df.iloc[:, 0]
    out: List[Tuple[int, ...]] = []
    for val in col.dropna():
        digs = to_tuple(val, n_digits=n_digits)
        if len(digs) == n_digits and all(0 <= d <= 9 for d in digs):
            out.append(digs)
    return out

@st.cache_data(show_spinner=False)
def compute_transitions(
    draws: List[Tuple[int, ...]],
    recent_window: int,
    max_lag: int,
    lag_weights: Iterable[float],
    mapping: str,
) -> Counter:
    windowed = draws[-recent_window:] if recent_window > 0 else draws
    all_trans = Counter()
    for lag, w in zip(range(1, max_lag + 1), lag_weights):
        if w <= 0:
            continue
        trans = extract_digit_transitions(windowed, lag, mapping=mapping)
        if w != 1.0:
            trans = Counter({k: v * w for k, v in trans.items()})
        all_trans.update(trans)
    return all_trans

def is_match(pred: Tuple[int,...], actual: Tuple[int,...], positionless: bool) -> bool:
    if positionless:
        return Counter(pred) == Counter(actual)
    return pred == actual

def generate_predictions_for_seed(
    seed: Tuple[int, ...],
    cnt: Counter,
    alpha: float,
    top_k: int,
    mapping_flag: str,
    num_preds: int,
) -> List[Tuple[int, ...]]:
    probs = normalize_matrix(cnt, alpha=alpha)
    candidates = apply_positionless_transitions(seed, probs, top_k)
    scored = [
        (cand, score_by_transition_likelihood(cand, seed, probs, mapping=mapping_flag))
        for cand in candidates
    ]
    scored.sort(key=lambda t: t[1], reverse=True)
    return [c for c, _ in scored[:num_preds]]

@st.cache_data(show_spinner=False)
def backtest(
    draws: List[Tuple[int, ...]],
    recent_window: int,
    max_lag: int,
    lag_weights: List[float],
    alpha: float,
    top_k: int,
    mapping_flag: str,
    num_preds: int,
    positionless_match: bool = False,
    start_min_history: int = 30,
) -> pd.DataFrame:
    rows = []
    for t in range(start_min_history, len(draws)):
        history = draws[max(0, t - recent_window):t] if recent_window > 0 else draws[:t]
        if len(history) < 2:
            continue
        all_trans = Counter()
        for lag, w in zip(range(1, max_lag + 1), lag_weights):
            if w <= 0:
                continue
            trans = extract_digit_transitions(history, lag, mapping=mapping_flag)
            if w != 1.0:
                trans = Counter({k: v * w for k, v in trans.items()})
            all_trans.update(trans)
        seed = history[-1]
        preds = generate_predictions_for_seed(
            seed=seed,
            cnt=all_trans,
            alpha=alpha,
            top_k=top_k,
            mapping_flag=mapping_flag,
            num_preds=num_preds,
        )
        actual = draws[t]
        hit = any(is_match(p, actual, positionless_match) for p in preds)
        rows.append(
            {
                "t": t,
                "seed": tuple_to_str(seed),
                "actual": tuple_to_str(actual),
                "hit": int(hit),
                "preds": [tuple_to_str(p) for p in preds],
            }
        )
    return pd.DataFrame(rows)

# -----------------------------
# Paste Box for Draws
# -----------------------------
def parse_pasted_draws(paste_data, n_digits):
    draws = []
    lines = paste_data.splitlines()
    for line in lines:
        # Optionally split further if user puts commas/spaces
        for part in line.replace(',', ' ').split():
            digs = to_tuple(part, n_digits=n_digits)
            if len(digs) == n_digits and all(0 <= d <= 9 for d in digs):
                draws.append(digs)
    return draws

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="Shift-Based Prediction System", layout="wide")
st.title("⚙️ Shift-Based Prediction System (Pick 3 / Pick 4)")
st.caption(
    "Learns digit-to-digit transitions (not position-bound), detects speed/skip patterns, and predicts next draws."
)

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("History Input")

paste_data = st.sidebar.text_area(
    "Paste draw history here (one draw per line, e.g. 654 or 0972):",
    height=150,
    help="Paste draws as plain text, one per line or separated by commas/spaces."
)

hist_file = st.sidebar.file_uploader(
    "Or upload history CSV (one column with draws like 654 or 09724)", type=["csv"]
)
hist_col = st.sidebar.text_input("Draw column name (optional)")

if st.sidebar.checkbox("Use sample data", value=False, help="Loads a small synthetic sample."):
    sample = pd.DataFrame({"draw": ["012", "124", "245", "457", "579", "791", "913", "135", "357", "579"]})
    hist_file = io.BytesIO(sample.to_csv(index=False).encode("utf-8"))
    hist_col = "draw"
    paste_data = ""  # clear paste data if using sample

mode = st.sidebar.selectbox("Game", ["Pick 3", "Pick 4"], index=1)
n_digits = 3 if mode == "Pick 3" else 4

recent_window = st.sidebar.slider("Recent window for speed detection (draws)", 5, 500, 100, 5)
max_lag = st.sidebar.slider("Max skip (lag) to analyze", 1, 10, 3, 1)

st.sidebar.subheader("Lag weighting")
weight_scheme = st.sidebar.selectbox("Weights", ["Uniform", "Linear decay", "Exponential decay"], index=0)
if weight_scheme == "Uniform":
    lag_weights = [1.0] * max_lag
elif weight_scheme == "Linear decay":
    lag_weights = [max(0.1, (max_lag - (i)) / max_lag) for i in range(max_lag)]
else:
    decay = st.sidebar.slider("Exponential decay factor", 0.5, 0.99, 0.85, 0.01)
    lag_weights = [decay**i for i in range(max_lag)]

per_digit_topk = st.sidebar.slider("Top-K targets per digit", 1, 10, 3, 1)

mapping_mode = st.sidebar.selectbox(
    "Digit mapping mode", ["Greedy (fast)", "Optimal (Hungarian, slower)"], index=0
)
mapping_flag = "optimal" if mapping_mode.startswith("Optimal") else "greedy"

alpha = st.sidebar.slider("Smoothing α (Laplace=1.0)", 0.0, 2.0, 0.5, 0.1)

play_mode = st.sidebar.radio("Prediction set size", ["10 picks", "20 picks", "30 picks", "50 picks"])
num_preds = int(play_mode.split()[0])

if HAS_PLOTTING:
    do_heatmaps = st.sidebar.checkbox("Show heatmaps", value=True)
else:
    do_heatmaps = False

# -----------------------------
# Main Logic
# -----------------------------
draws: List[Tuple[int, ...]] = []

if paste_data.strip():
    draws = parse_pasted_draws(paste_data, n_digits)
elif hist_file is not None:
    try:
        df = pd.read_csv(hist_file)
        draws = parse_draws_from_df(df, n_digits=n_digits, hist_col=hist_col)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

if draws:
    st.subheader("Data Summary")
    st.write(f"Total parsed draws: {len(draws)}; Unique: {len(set(draws))}")
    if len(draws) < 20:
        st.warning("Very few draws detected; results may be unstable.")

    st.subheader("Transition Analysis")
    all_trans = compute_transitions(
        draws=draws,
        recent_window=recent_window,
        max_lag=max_lag,
        lag_weights=lag_weights,
        mapping=mapping_flag,
    )

    df_mat = transition_matrix_to_df(all_trans)
    st.markdown("Top outgoing transitions per digit (P(y|x))")

    top_rows = []
    for x in range(10):
        row = [(y, df_mat.iloc[x, y]) for y in range(10)]
        row.sort(key=lambda t: t[1], reverse=True)
        for y, p in row[:3]:
            top_rows.append({"from": x, "to": y, "prob": round(float(p), 4)})
    st.dataframe(pd.DataFrame(top_rows))

    if do_heatmaps:
        if len(draws) <= 10000:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(df_mat, annot=False, cmap="Blues", cbar=True, ax=ax)
            ax.set_title("Weighted Transition Heatmap (P(y|x))")
            st.pyplot(fig)
        else:
            st.info("Heatmap skipped for large datasets.")

    probs = normalize_matrix(all_trans, alpha=alpha)

    st.subheader("Predictions")
    last_draw = draws[-1]
    st.write(f"Last draw: {tuple_to_str(last_draw)}")

    candidates = apply_positionless_transitions(last_draw, probs, per_digit_topk)
    scored = [
        (cand, score_by_transition_likelihood(cand, last_draw, probs, mapping=mapping_flag))
        for cand in candidates
    ]
    scored.sort(key=lambda t: t[1], reverse=True)

    preds = [tuple_to_str(c) for c, _ in scored[:num_preds]]
    st.write(preds)

    # Backtest UI
    st.subheader("Backtest (walk-forward)")
    col_bt1, col_bt2, col_bt3 = st.columns(3)
    with col_bt1:
        positionless_bt = st.checkbox("Positionless match", value=False)
    with col_bt2:
        min_hist = st.number_input("Min history to start", min_value=10, max_value=500, value=50, step=10)
    with col_bt3:
        run_bt = st.button("Run backtest")

    if run_bt and draws and len(draws) > min_hist:
        bt_df = backtest(
            draws=draws,
            recent_window=recent_window,
            max_lag=max_lag,
            lag_weights=lag_weights,
            alpha=alpha,
            top_k=per_digit_topk,
            mapping_flag=mapping_flag,
            num_preds=num_preds,
            positionless_match=positionless_bt,
            start_min_history=int(min_hist),
        )
        if bt_df.empty:
            st.info("Not enough data to backtest with current settings.")
        else:
            total = len(bt_df)
            hits = int(bt_df["hit"].sum())
            hit_rate = hits / total if total else 0.0
            st.metric("Backtest hit rate", f"{hit_rate:.1%}", help=f"{hits}/{total} hits")

            bt_df["rolling_hit"] = bt_df["hit"].rolling(20, min_periods=1).mean()
            st.line_chart(bt_df.set_index("t")[["rolling_hit"]])
            st.dataframe(bt_df.tail(20))
            st.download_button(
                "Download backtest results",
                bt_df.to_csv(index=False).encode("utf-8"),
                "backtest.csv",
                mime="text/csv",
            )

    # Store predictions in session
    st.session_state.setdefault("last_seed", None)
    st.session_state.setdefault("last_preds", [])

    if st.session_state["last_seed"] != tuple_to_str(last_draw):
        st.session_state["last_seed"] = tuple_to_str(last_draw)
        st.session_state["last_preds"] = preds

    # Check live prediction hit
    if len(draws) >= 2:
        prev_seed_str = st.session_state.get("last_seed")
        try:
            idx = [tuple_to_str(d) for d in draws].index(prev_seed_str)
            if idx + 1 < len(draws):
                actual_today = tuple_to_str(draws[idx + 1])
                hit_live = actual_today in st.session_state.get("last_preds", [])
                st.write(f"Live check for next draw after {prev_seed_str}: actual={actual_today}, hit={hit_live}")
        except ValueError:
            pass

        csv_bytes = pd.Series(preds, name="prediction").to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv_bytes, "predictions.csv", mime="text/csv")

        with st.expander("Why these predictions?"):
            top5 = scored[: min(5, len(scored))]
            expl_rows = []
            for cand, sc in top5:
                pairs = greedy_multiset_mapping(last_draw, cand) if mapping_flag == "greedy" else greedy_multiset_mapping(
                    last_draw, cand
                )
                parts = []
                for x, y in pairs:
                    p = probs.get((x, y), 1e-12)
                    parts.append(f"{x}->{y} (p={p:.3f})")
                expl_rows.append({"candidate": tuple_to_str(cand), "score": round(sc, 4), "pairs": ", ".join(parts)})
            st.dataframe(pd.DataFrame(expl_rows))

else:
    st.info("Upload your history CSV or paste draws to start analysis, or enable sample data.")

# -----------------------------
# Notes
# -----------------------------
# - Positionless mapping is heuristic; 'Optimal' mode can slightly alter rankings.
# - Smoothing alpha prevents zero-probability log penalties and stabilizes small samples.
# - Lag weights emphasize certain skips; try exponential decay for more recency focus.                    

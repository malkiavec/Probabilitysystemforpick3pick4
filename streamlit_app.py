# shift_based_prediction.py
# Run: streamlit run shift_based_prediction.py

import itertools as it
import math
from collections import Counter
from typing import List, Tuple, Dict

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

st.set_page_config(page_title="Shift-Based Prediction System", layout="wide")
st.title("⚙️ Shift-Based Prediction System (Pick 3 / Pick 4)")
st.caption("Learns digit-to-digit transitions (not position-bound), detects speed/skip patterns, and predicts next draws.")

# -----------------------------
# Utilities
# -----------------------------

def to_tuple(x) -> Tuple[int, ...]:
    if isinstance(x, (list, tuple)):
        return tuple(int(v) for v in x)
    s = str(x)
    digs = [int(ch) for ch in s if ch.isdigit()]
    return tuple(digs)


def tuple_to_str(t: Tuple[int, ...]) -> str:
    return ''.join(str(d) for d in t)


def greedy_multiset_mapping(a: Tuple[int, ...], b: Tuple[int, ...]):
    ca, cb = Counter(a), Counter(b)
    pairs = []
    for d in range(10):
        m = min(ca[d], cb[d])
        for _ in range(m):
            pairs.append((d, d))
            ca[d] -= 1
            cb[d] -= 1
    rem_a = []
    rem_b = []
    for d in range(10):
        if ca[d] > 0:
            rem_a.extend([d]*ca[d])
        if cb[d] > 0:
            rem_b.extend([d]*cb[d])
    for x, y in zip(sorted(rem_a), sorted(rem_b)):
        pairs.append((x, y))
    return pairs


def extract_digit_transitions(draws: List[Tuple[int, ...]], lag: int) -> Counter:
    trans = Counter()
    for i in range(len(draws) - lag):
        a, b = draws[i], draws[i+lag]
        pairs = greedy_multiset_mapping(a, b)
        for x, y in pairs:
            trans[(x, y)] += 1
    return trans


def normalize_matrix(cnt: Counter) -> Dict[Tuple[int, int], float]:
    totals = Counter()
    for (x, y), c in cnt.items():
        totals[x] += c
    probs = {}
    for (x, y), c in cnt.items():
        denom = totals[x] if totals[x] else 1
        probs[(x, y)] = c / denom
    return probs


def transition_matrix_to_df(trans: Counter) -> pd.DataFrame:
    mat = np.zeros((10, 10), dtype=float)
    for (x, y), c in trans.items():
        mat[x, y] += c
    for x in range(10):
        row_sum = mat[x].sum()
        if row_sum > 0:
            mat[x] = mat[x] / row_sum
    df = pd.DataFrame(mat, index=[f"{i}" for i in range(10)], columns=[f"{j}" for j in range(10)])
    return df


def apply_positionless_transitions(seed: Tuple[int, ...], probs: Dict[Tuple[int,int], float], top_k: int = 3) -> List[Tuple[int, ...]]:
    choices_per_digit = []
    for v in seed:
        dist = [(y, probs.get((v, y), 0.0)) for y in range(10)]
        dist.sort(key=lambda t: t[1], reverse=True)
        top = [y for y, _ in dist[:max(1, top_k)]]
        if v not in top:
            top = [v] + top[:-1]
        choices_per_digit.append(top)
    raw = list(it.product(*choices_per_digit))
    outs = set(tuple(r) for r in raw)
    return list(outs)


def score_by_transition_likelihood(cand: Tuple[int, ...], seed: Tuple[int, ...], probs: Dict[Tuple[int,int], float]) -> float:
    pairs = greedy_multiset_mapping(seed, cand)
    score = 0.0
    for x, y in pairs:
        p = probs.get((x, y), 1e-9)
        score += math.log(p + 1e-12)
    return score

# -----------------------------
# Sidebar Controls
# -----------------------------

st.sidebar.header("History Input")
mode = st.sidebar.selectbox("Game", ["Pick 3", "Pick 4"], index=1)
n_digits = 3 if mode == "Pick 3" else 4

hist_file = st.sidebar.file_uploader("Upload history CSV (one column with draws like 654 or 9724)", type=["csv"])
hist_col = st.sidebar.text_input("Draw column name (optional)")

recent_window = st.sidebar.slider("Recent window for speed detection (draws)", 5, 100, 20, 1)
max_lag = st.sidebar.slider("Max skip (lag) to analyze", 1, 5, 3, 1)

per_digit_topk = st.sidebar.slider("Top-K targets per digit", 1, 10, 3, 1)

play_mode = st.sidebar.radio("Prediction set size", ["10 picks", "20 picks", "30 picks"])
num_preds = int(play_mode.split()[0])

# -----------------------------
# Main Logic
# -----------------------------

draws = []
if hist_file is not None:
    df = pd.read_csv(hist_file)
    if hist_col and hist_col in df.columns:
        col = df[hist_col]
    else:
        col = df.iloc[:,0]
    for val in col.dropna():
        digs = to_tuple(val)
        if len(digs) == n_digits:
            draws.append(digs)

if draws:
    st.subheader("Transition Analysis")
    all_trans = Counter()
    for lag in range(1, max_lag+1):
        trans = extract_digit_transitions(draws[-recent_window:], lag)
        all_trans.update(trans)
        st.markdown(f"**Lag {lag} transitions**")
        df_mat = transition_matrix_to_df(trans)
        st.dataframe(df_mat)

        if HAS_PLOTTING:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.heatmap(df_mat, annot=False, cmap="Blues", cbar=True, ax=ax)
            ax.set_title(f"Lag {lag} Transition Heatmap")
            st.pyplot(fig)

    probs = normalize_matrix(all_trans)

    st.subheader("Predictions")
    last_draw = draws[-1]
    st.write(f"Last draw: {tuple_to_str(last_draw)}")

    candidates = apply_positionless_transitions(last_draw, probs, per_digit_topk)
    scored = [(cand, score_by_transition_likelihood(cand, last_draw, probs)) for cand in candidates]
    scored.sort(key=lambda t: t[1], reverse=True)

    preds = [tuple_to_str(c) for c, _ in scored[:num_preds]]
    st.write(preds)

    st.download_button("Download Predictions", pd.Series(preds).to_csv(index=False), "predictions.csv")

else:
    st.info("Upload your history CSV to start analysis.")

# edat_pathB.py  — EDAT Regime Engine (Python) for Sierra  [fixed tailer]
# pip install numpy pandas scipy hmmlearn

import os, time, csv, math, sys
import numpy as np
from collections import deque
from pathlib import Path
from hmmlearn.hmm import GaussianHMM

# ---- paths ----
# If you're writing 1-minute bars, change MNQ_1s.csv -> MNQ_1m.csv
DATA = Path(r"C:\SierraChart\Data\EDAT\MNQ_1m.csv")
OUT  = Path(r"C:\SierraChart\Data\EDAT\edat_regime_MNQ.csv")
MODEL_DIR = Path(r"C:\SierraChart\Data\EDAT\models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---- utils ----
def rolling_entropy(x, bins=20):
    if len(x) < 2: return 0.0
    hist, _ = np.histogram(x, bins=bins)
    p = hist.astype(float); s = p.sum()
    if s <= 0: return 0.0
    p /= s; p = p[p > 0]
    return float(-(p * np.log(p)).sum())

class Roller:
    def __init__(self, n): self.n=n; self.q=deque(maxlen=n)
    def add(self,x): self.q.append(float(x))
    def mean(self): return float(np.mean(self.q)) if self.q else 0.0
    def std(self):  return float(np.std(self.q))  if self.q else 0.0
    def sum(self):  return float(np.sum(self.q))  if self.q else 0.0

# ---- robust tailer (no csv.reader on file; no tell() after next()) ----
def tail_csv(path: Path, sleep=0.25):
    # Wait for file to exist
    while not path.exists():
        time.sleep(sleep)
    with path.open('r', newline='') as f:
        # Read header (wait if empty)
        header = None
        while header is None:
            first = f.readline()
            if not first:
                time.sleep(sleep)
                f.seek(0)
                continue
            header = [h.strip().lower() for h in first.strip().split(',')]

        # Stream subsequent lines forever
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                time.sleep(sleep)
                f.seek(pos)
                continue
            parts = [p.strip() for p in line.strip().split(',')]
            if len(parts) < 6:
                continue
            try:
                ts = parts[0]
                o,h,l,c,v = map(float, parts[1:6])
            except Exception:
                continue
            yield ts,o,h,l,c,v

# ---- feature builder (minute-native windows in BARS) ----
def build_features(stream):
    # windows in BARS
    w_dp  = 60
    w_phi = 20
    w_rv  = 120
    w_hl  = 60

    dpw  = Roller(w_dp)
    phiw = Roller(w_phi)
    r2w  = Roller(w_rv)  # dp^2
    hlw  = Roller(w_hl)

    prev_close = None
    hi = lo = None

    for ts,o,h,l,c,v in stream:
        if prev_close is not None:
            dp = c - prev_close
            dpw.add(dp)
            r2w.add(dp*dp)
            phiw.add(abs(dp)*v)        # φ contribution
        prev_close = c

        # HL tracking
        hi = c if hi is None else max(hi, h)
        lo = c if lo is None else min(lo, l)
        hlw.add(hi - lo)

        eps  = dpw.std()
        rv   = r2w.sum()
        # use dp std as rough scale; add HL mean for turbulence flavor
        rv_s = r2w.std() if len(r2w.q) else 1e-6
        hlm  = hlw.mean()
        zeta = ((rv + hlm) - (r2w.mean() + hlm)) / (rv_s + 1e-9)

        phi  = phiw.sum()
        eta  = 1.0 / (1.0 + (dpw.std() or 1e-9))
        omega = 0.0
        if len(dpw.q) > 2:
            omega = dpw.q[-1] - dpw.q[-2]   # accel proxy on minutes
        S = rolling_entropy(list(dpw.q), bins=20)

        yield {
            'ts': ts, 'o':o,'h':h,'l':l,'c':c,'v':v,
            'eps':eps, 'zeta':zeta, 'phi':phi, 'eta':eta, 'omega':omega, 'S':S
        }

def run():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        with OUT.open('w', newline='') as f:
            csv.writer(f).writerow(
                "ts,state_id,state_name,p0,p1,p2,p3,tau,eps,zeta,phi,omega,eta,S".split(',')
            )

    n_states = 4
    min_train = 600    # bars before first fit
    refit_every = 60   # bars between refits
    hmm = None
    scale = None
    buf = []
    tau = 0
    last_state = None
    bar_i = 0

    for rec in build_features(tail_csv(DATA)):
        bar_i += 1
        buf.append([rec['eps'], rec['zeta'], rec['phi'], rec['eta'], rec['omega'], rec['S']])

        # fit/refit
        if (hmm is None and len(buf) >= min_train) or (hmm is not None and bar_i % refit_every == 0):
            X = np.array(buf[-2000:])  # sliding window
            X_tr = X.copy()
            X_tr[:,2] = np.log1p(np.maximum(0.0, X[:,2]))  # log φ
            stds = X_tr.std(axis=0) + 1e-6
            means = X_tr.mean(axis=0)
            X_tr = (X_tr - means) / stds
            hmm = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=200, random_state=7)
            hmm.fit(X_tr)
            scale = (means, stds)

        # predict
        if hmm is not None and scale is not None:
            x = np.array([[rec['eps'], rec['zeta'], rec['phi'], rec['eta'], rec['omega'], rec['S']]])
            x[:,2] = np.log1p(np.maximum(0.0, x[:,2]))
            x = (x - scale[0]) / scale[1]
            logprob, post = hmm.score_samples(x)
            p = post[0]
            state = int(np.argmax(p))
        else:
            p = np.array([0.25,0.25,0.25,0.25])
            state = 0

        tau = tau + 1 if state == last_state else 1
        last_state = state

        names = {0:'Range', 1:'Trans', 2:'TrendUp', 3:'TrendDn'}

        with OUT.open('a', newline='') as f:
            csv.writer(f).writerow([
                rec['ts'], state, names.get(state, f'S{state}'),
                f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}", f"{p[3]:.4f}",
                tau,
                f"{rec['eps']:.6f}", f"{rec['zeta']:.6f}", f"{rec['phi']:.3f}",
                f"{rec['omega']:.6f}", f"{rec['eta']:.6f}", f"{rec['S']:.6f}"
            ])

if __name__ == '__main__':
    if not DATA.exists():
        sys.exit(f"Input file not found: {DATA}. Enable 'Write Bar And Study Data To File' on your chart.")
    run()

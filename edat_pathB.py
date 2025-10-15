import os, time, csv, math, sys
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path
from hmmlearn.hmm import GaussianHMM

DATA = Path(r"C:\SierraChart\Data\EDAT\MNQ_1s.csv")         # input from Sierra (1s or 1m ok)
OUT  = Path(r"C:\SierraChart\Data\EDAT\edat_regime_MNQ.csv")
MODEL_DIR = Path(r"C:\SierraChart\Data\EDAT\models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- utilities ----------------

def rolling_entropy(x, bins=20):
    if len(x) < 2: return 0.0
    hist, _ = np.histogram(x, bins=bins)
    p = hist / np.maximum(1, hist.sum())
    p = p[p>0]
    return float(-(p*np.log(p)).sum())

class Roller:
    def __init__(self, n): self.n=n; self.q=deque(maxlen=n)
    def add(self,x): self.q.append(float(x))
    def mean(self): return float(np.mean(self.q)) if self.q else 0.0
    def std(self):  return float(np.std(self.q))  if self.q else 0.0
    def sum(self):  return float(np.sum(self.q))  if self.q else 0.0

# ---------------- ingest ----------------

def tail_csv(path: Path, sleep=0.25):
    with path.open('r', newline='') as f:
        rdr = csv.reader(f)
        header = next(rdr)  # assume exists
        pos = f.tell()
        while True:
            line = f.readline()
            if not line:
                time.sleep(sleep)
                f.seek(pos)
                continue
            pos = f.tell()
            row = next(csv.reader([line]))
            # Expect: DateTime, Open, High, Low, Close, Volume, ...
            try:
                ts = row[0]
                o,h,l,c,v = map(float, row[1:6])
            except Exception:
                continue
            yield ts,o,h,l,c,v

# ---------------- model ----------------

def build_features(stream, is_minute=True):
    """Yield dict per bar with features and prices."""
    # windows in BARS (minute‑native)
    w_dp  = 60
    w_phi = 20
    w_rv  = 120
    w_hl  = 60
    
    dpw  = Roller(w_dp)
    phiw = Roller(w_phi)
    r2w  = Roller(w_rv)  # dp^2
    hlw  = Roller(w_hl)
    pw_prev = None

    hi = lo = None

    for ts,o,h,l,c,v in stream:
        if pw_prev is not None:
            dp = c - pw_prev
            dpw.add(dp)
            r2w.add(dp*dp)
            phiw.add(abs(dp)*v)        # φ contribution
        pw_prev = c

        # HL tracking for entropy proxy
        hi = c if hi is None else max(hi, h)
        lo = c if lo is None else min(lo, l)
        hlw.add(hi - lo)

        # metrics
        eps  = dpw.std()
        rv   = r2w.sum()                 # realized variance sum
        hlm  = hlw.mean()                # HL mean
        zeta = ( (rv + hlm) - (r2w.mean()+hlm) ) / (r2w.std() + 1e-9)  # coarse zscore
        phi  = phiw.sum()
        eta  = 1.0 / (1.0 + (dpw.std() or 1e-9))
        # ω minute proxy = second difference
        omega = 0.0
        if len(dpw.q) > 2:
            omega = dpw.q[-1] - dpw.q[-2]
        # S entropy on recent closes
        S = rolling_entropy(list(dpw.q), bins=20)

        yield {
            'ts': ts, 'o':o,'h':h,'l':l,'c':c,'v':v,
            'eps':eps, 'zeta':zeta, 'phi':phi, 'eta':eta, 'omega':omega, 'S':S
        }

# ---------------- regime engine ----------------

def run():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        with OUT.open('w', newline='') as f:
            csv.writer(f).writerow(
                "ts,state_id,state_name,p0,p1,p2,p3,tau,eps,zeta,phi,omega,eta,S".split(',')
            )

    # HMM config
    n_states = 4
    min_train = 600   # bars required before first fit
    refit_every = 60  # refit frequency (bars)

    hmm = None
    buf = []
    tau = 0
    last_state = None

    # stream
    bar_i = 0
    for rec in build_features(tail_csv(DATA)):
        bar_i += 1
        buf.append([rec['eps'], rec['zeta'], rec['phi'], rec['eta'], rec['omega'], rec['S']])

        # fit/refit
        if (hmm is None and len(buf) >= min_train) or (hmm is not None and bar_i % refit_every == 0):
            X = np.array(buf[-2000:])  # cap window
            # log-transform skewed features
            X_tr = X.copy()
            X_tr[:,2] = np.log1p(np.maximum(0.0, X[:,2]))  # phi
            # scale roughly by standard deviation to help HMM
            stds = X_tr.std(axis=0) + 1e-6
            X_tr = (X_tr - X_tr.mean(axis=0)) / stds
            hmm = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=200, random_state=7)
            hmm.fit(X_tr)
            scale = (X_tr.mean(axis=0), stds)
        # predict
        if hmm is not None:
            x = np.array([[rec['eps'], rec['zeta'], rec['phi'], rec['eta'], rec['omega'], rec['S']]])
            x[:,2] = np.log1p(np.maximum(0.0, x[:,2]))
            x = (x - scale[0]) / scale[1]
            # state probabilities via score_samples
            logprob, post = hmm.score_samples(x)
            p = post[0]
            state = int(np.argmax(p))
        else:
            p = np.array([0.25,0.25,0.25,0.25])
            state = 0

        # τ run-length
        tau = tau + 1 if state == last_state else 1
        last_state = state

        # name mapping (editable)
        names = {
            0: 'Range',
            1: 'Trans',
            2: 'TrendUp',
            3: 'TrendDn'
        }
        # heuristic polarity: use omega/dp sign to split 2/3 later if desired

        # write row
        with OUT.open('a', newline='') as f:
            wr = csv.writer(f)
            wr.writerow([
                rec['ts'], state, names.get(state, f'S{state}'),
                f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}", f"{p[3]:.4f}",
                tau,
                f"{rec['eps']:.6f}", f"{rec['zeta']:.6f}", f"{rec['phi']:.3f}", f"{rec['omega']:.6f}", f"{rec['eta']:.6f}", f"{rec['S']:.6f}"
            ])

if __name__ == '__main__':
    if not DATA.exists():
        sys.exit(f"Input file not found: {DATA}. Enable 'Write Bar And Study Data To File' on your chart.")
    run()

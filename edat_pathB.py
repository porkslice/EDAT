# edat_pathB.py — EDAT Regime Engine (Path B) [minute-native, robust tail, fast fit]
# pip install numpy scipy hmmlearn

import time, csv, sys
import numpy as np
from collections import deque
from pathlib import Path
from hmmlearn.hmm import GaussianHMM

# --- PATHS (match Sierra's writer exactly) ---
DATA = Path(r"C:\SierraChart\Data\EDAT\MNQ_1m.csv")   # <- use MNQ_1s.csv if you write seconds
OUT  = Path(r"C:\SierraChart\Data\EDAT\edat_regime_MNQ.csv")

# ---------- utils ----------
def rolling_entropy(x, bins=20):
    if len(x) < 2: return 0.0
    hist, _ = np.histogram(x, bins=bins)
    s = hist.sum()
    if s <= 0: return 0.0
    p = hist / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

class Roller:
    def __init__(self, n): self.q=deque(maxlen=n)
    def add(self,x): self.q.append(float(x))
    def mean(self): return float(np.mean(self.q)) if self.q else 0.0
    def std(self):  return float(np.std(self.q))  if self.q else 0.0
    def sum(self):  return float(np.sum(self.q))  if self.q else 0.0
    def __len__(self): return len(self.q)

def safe_append(path: Path, row, retries=12, wait=0.25):
    for _ in range(retries):
        try:
            with path.open('a', newline='') as f:
                csv.writer(f).writerow(row)
            return True
        except PermissionError:
            time.sleep(wait)
    return False

# ---------- robust tailer (NO csv.reader on the file) ----------
def tail_csv(path: Path, sleep=0.25):
    print(f"[EDAT] Tailing: {path}")
    while not path.exists():
        print("[EDAT] Waiting for input file…"); time.sleep(sleep)
    with path.open('r', newline='') as f:
        # read header (or first line) once
        header = None
        while header is None:
            first = f.readline()
            if not first:
                time.sleep(sleep); f.seek(0); continue
            header = [h.strip().lower() for h in first.strip().split(',')]
            print(f"[EDAT] Header/First: {header}")

        bars = 0
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                time.sleep(sleep); f.seek(pos); continue
            parts = [p.strip() for p in line.strip().split(',')]
            if len(parts) < 6:  # need at least ts, O,H,L,C,V
                continue
            try:
                ts = parts[0]
                o,h,l,c,v = map(float, parts[1:6])
            except Exception:
                continue
            bars += 1
            if bars % 10 == 0:
                print(f"[EDAT] Bar {bars} @ {ts} c={c} v={v}")
            yield ts,o,h,l,c,v

# ---------- features (minute-native windows in BARS) ----------
def build_features_from_rows(rows):
    w_dp, w_phi, w_rv, w_hl = 60, 20, 120, 60
    dpw, phiw, r2w, hlw = Roller(w_dp), Roller(w_phi), Roller(w_rv), Roller(w_hl)
    prev = None; hi = lo = None
    for ts,o,h,l,c,v in rows:
        if prev is not None:
            dp = c - prev
            dpw.add(dp); r2w.add(dp*dp); phiw.add(abs(dp)*v)
        prev = c
        hi = c if hi is None else max(hi, h)
        lo = c if lo is None else min(lo, l)
        hlw.add(hi - lo)

        eps  = dpw.std()
        rv   = r2w.sum()
        rv_s = (np.std(r2w.q) if len(r2w) else 1e-6)
        hlm  = hlw.mean()
        zeta = ((rv + hlm) - (np.mean(r2w.q) if len(r2w) else 0.0 + hlm)) / (rv_s + 1e-9)
        phi  = phiw.sum()
        eta  = 1.0 / (1.0 + (eps or 1e-9))
        omega = (dpw.q[-1] - dpw.q[-2]) if len(dpw) > 2 else 0.0
        S = rolling_entropy(list(dpw.q), bins=20)

        yield {'ts':ts,'eps':eps,'zeta':zeta,'phi':phi,'eta':eta,'omega':omega,'S':S}

def preload_rows(path: Path, N=2000):
    if not path.exists(): return []
    lines = path.read_text().strip().splitlines()
    out=[]
    for line in lines[-N:]:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 6: continue
        try:
            ts = parts[0]; o,h,l,c,v = map(float, parts[1:6])
            out.append((ts,o,h,l,c,v))
        except Exception:
            continue
    return out

# ---------- main loop ----------
def run():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        with OUT.open('w', newline='') as f:
            csv.writer(f).writerow("ts,state_id,state_name,p0,p1,p2,p3,tau,eps,zeta,phi,omega,eta,S".split(','))
        print(f"[EDAT] Created {OUT}")

    n_states = 4
    min_train = 120     # faster first fit
    refit_every = 30

    hmm = None; scale = None; buf=[]; tau=0; last_state=None; bar_i=0

    # Bootstrap from existing file so we can fit immediately
    boot = preload_rows(DATA, N=2000)
    if boot:
        feats = list(build_features_from_rows(boot))
        for r in feats:
            buf.append([r['eps'], r['zeta'], r['phi'], r['eta'], r['omega'], r['S']])
        if len(buf) >= min_train:
            X = np.array(buf[-2000:])
            X_tr = X.copy(); X_tr[:,2] = np.log1p(np.maximum(0.0, X[:,2]))  # log φ
            stds = X_tr.std(axis=0) + 1e-6; means = X_tr.mean(axis=0)
            X_tr = (X_tr - means) / stds
            hmm = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=200, random_state=7)
            hmm.fit(X_tr)
            scale = (means, stds)
            print(f"[EDAT] HMM prefit on {len(X_tr)} bars")

    # Live tail
    for rec in build_features_from_rows(tail_csv(DATA)):
        bar_i += 1
        buf.append([rec['eps'], rec['zeta'], rec['phi'], rec['eta'], rec['omega'], rec['S']])

        # fit/refit
        if (hmm is None and len(buf) >= min_train) or (hmm is not None and bar_i % refit_every == 0):
            X = np.array(buf[-2000:])
            X_tr = X.copy(); X_tr[:,2] = np.log1p(np.maximum(0.0, X[:,2]))
            stds = X_tr.std(axis=0) + 1e-6; means = X_tr.mean(axis=0)
            X_tr = (X_tr - means) / stds
            hmm = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=200, random_state=7)
            hmm.fit(X_tr)
            scale = (means, stds)
            print(f"[EDAT] HMM fitted on {len(X_tr)} bars")

        # predict (or provisional pre-HMM)
        if hmm is not None and scale is not None:
            x = np.array([[rec['eps'], rec['zeta'], rec['phi'], rec['eta'], rec['omega'], rec['S']]])
            x[:,2] = np.log1p(np.maximum(0.0, x[:,2]))
            x = (x - scale[0]) / scale[1]
            _, post = hmm.score_samples(x)
            p = post[0]
            state = int(np.argmax(p))
        else:
            z, e = rec['zeta'], rec['eta']
            if z > 1.5 and e < 0.6: state = 2   # Trend (Up placeholder)
            elif z < 0.2 and e > 0.8: state = 0 # Range
            else: state = 1                      # Transitional
            p = np.array([0.0,0.0,0.0,0.0]); p[state] = 1.0

        tau = tau + 1 if state == last_state else 1
        last_state = state
        names = {0:'Range',1:'Trans',2:'TrendUp',3:'TrendDn'}

        row = [
            rec['ts'], state, names.get(state, f'S{state}'),
            f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}", f"{p[3]:.4f}",
            tau,
            f"{rec['eps']:.6f}", f"{rec['zeta']:.6f}", f"{rec['phi']:.3f}",
            f"{rec['omega']:.6f}", f"{rec['eta']:.6f}", f"{rec['S']:.6f}"
        ]
        ok = safe_append(OUT, row)
        if not ok:
            print("[EDAT] WARN: output locked (Excel?). Will retry.")
        elif bar_i % 10 == 0:
            print(f"[EDAT] Wrote bar {bar_i}: state={state} tau={tau}")

if __name__ == '__main__':
    if not DATA.exists():
        sys.exit(f"[EDAT] Input not found: {DATA}")
    run()

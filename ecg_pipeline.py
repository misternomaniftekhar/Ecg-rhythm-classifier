"""ecg_pipeline.py - shared signal-processing + data utilities.

Research / educational use only. Not a medical device.
"""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
from scipy import signal as sps

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
FS = 360                      # all signals are resampled to the MIT-BIH rate
PRE, POST = 90, 110           # samples before / after the R peak (250 ms / 306 ms)
BEAT_LEN = PRE + POST         # 200 samples per beat
N_RR = 4                      # RR-interval side features

CLASSES = ["N", "S", "V", "F", "Q"]          # AAMI EC57 beat classes
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
CLASS_LABELS = {
    "N": "N - Normal / bundle-branch beat",
    "S": "S - Supraventricular ectopic beat",
    "V": "V - Ventricular ectopic beat",
    "F": "F - Fusion beat",
    "Q": "Q - Paced / unknown beat",
}
CLASS_COLORS = {"N": "#2e8b57", "S": "#f59e0b", "V": "#dc2626",
                "F": "#7c3aed", "Q": "#6b7280"}
MODEL_TITLES = {"cnn": "1D-CNN", "lstm": "BiLSTM", "transformer": "Transformer"}

# MIT-BIH annotation symbol -> AAMI class
AAMI_MAP = {
    **{s: "N" for s in ["N", "L", "R", "e", "j"]},
    **{s: "S" for s in ["A", "a", "J", "S"]},
    **{s: "V" for s in ["V", "E"]},
    "F": "F",
    **{s: "Q" for s in ["/", "f", "Q"]},
}

# Inter-patient split (de Chazal et al., 2004); paced records 102,104,107,217 excluded
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
       213, 214, 219, 221, 222, 228, 231, 232, 233, 234]


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def resample_to_fs(x, fs, target: int = FS):
    """Polyphase resample to `target` Hz."""
    x = np.asarray(x, dtype=float)
    fs_i = int(round(fs))
    if fs_i == target:
        return x
    g = gcd(fs_i, target)
    return sps.resample_poly(x, target // g, fs_i // g)


def preprocess(x, fs: float = FS, lo: float = 0.5, hi: float = 40.0, notch_hz=None):
    """Median removal -> Butterworth band-pass (zero phase) -> optional mains notch."""
    x = np.nan_to_num(np.asarray(x, dtype=float))
    x = x - np.median(x)
    hi = min(hi, 0.45 * fs)
    if len(x) > 100:
        sos = sps.butter(4, [lo, hi], btype="band", fs=fs, output="sos")
        y = sps.sosfiltfilt(sos, x)
        if notch_hz and notch_hz < fs / 2:
            b, a = sps.iirnotch(notch_hz, 30.0, fs=fs)
            y = sps.filtfilt(b, a, y)
    else:
        y = x
    return y


def normalize(x):
    x = np.asarray(x, dtype=float)
    return x / (np.std(x) + 1e-8)


# ----------------------------------------------------------------------------
# R-peak detection (Pan-Tompkins style) + beat extraction
# ----------------------------------------------------------------------------
def detect_r_peaks(x, fs: float = FS):
    """Return sample indices of detected R peaks in a pre-filtered ECG."""
    x = np.asarray(x, dtype=float)
    if len(x) < int(2 * fs):
        return np.array([], dtype=int)
    sos = sps.butter(2, [5.0, min(15.0, 0.45 * fs)], btype="band", fs=fs, output="sos")
    f = sps.sosfiltfilt(sos, x)
    sq = np.gradient(f) ** 2
    w = max(1, int(0.12 * fs))
    integ = np.convolve(sq, np.ones(w) / w, mode="same")
    thr = 0.15 * np.percentile(integ, 99)
    cand, _ = sps.find_peaks(integ, height=thr, distance=int(0.25 * fs))
    half = int(0.08 * fs)
    refined = []
    for p in cand:
        lo, hi = max(0, p - half), min(len(x), p + half + 1)
        refined.append(lo + int(np.argmax(np.abs(x[lo:hi]))))
    refined = np.unique(refined)
    # enforce a 200 ms refractory period, keep the larger deflection
    keep = []
    for p in refined:
        if keep and p - keep[-1] < int(0.2 * fs):
            if abs(x[p]) > abs(x[keep[-1]]):
                keep[-1] = p
        else:
            keep.append(p)
    return np.array(keep, dtype=int)


def rr_features(peaks, fs: float = FS):
    """[prev RR, next RR, prev/local-mean, next/local-mean] for every beat (seconds)."""
    peaks = np.asarray(peaks)
    n = len(peaks)
    if n < 2:
        return np.tile(np.array([0.8, 0.8, 1.0, 1.0], dtype=np.float32), (n, 1))
    rr = np.diff(peaks) / fs
    prev = np.concatenate([[rr[0]], rr])
    nxt = np.concatenate([rr, [rr[-1]]])
    loc = np.array([prev[max(0, i - 9): i + 1].mean() for i in range(n)])
    feats = np.stack([np.clip(prev, 0.2, 2.0), np.clip(nxt, 0.2, 2.0),
                      np.clip(prev / loc, 0.3, 3.0), np.clip(nxt / loc, 0.3, 3.0)], axis=1)
    return feats.astype(np.float32)


def extract_beats(x, peaks, fs: float = FS):
    """Cut a fixed window around every R peak. Returns (beats, rr_feats, keep_mask)."""
    peaks = np.asarray(peaks, dtype=int)
    rr = rr_features(peaks, fs)
    keep = (peaks - PRE >= 0) & (peaks + POST <= len(x))
    if not keep.any():
        return (np.zeros((0, BEAT_LEN), np.float32), np.zeros((0, N_RR), np.float32), keep)
    beats = np.stack([x[p - PRE: p + POST] for p in peaks[keep]]).astype(np.float32)
    return beats, rr[keep], keep


# ----------------------------------------------------------------------------
# Segment-level rhythm summary (rule-based, interpretable)
# ----------------------------------------------------------------------------
def rhythm_summary(all_peaks, labels, fs: float = FS):
    """Summarise a segment from its R peaks (RR statistics) and predicted beat classes."""
    peaks = np.asarray(all_peaks)
    out = dict(n_beats=int(len(peaks)), hr=np.nan, mean_rr=np.nan, sdnn=np.nan,
               rmssd=np.nan, irregularity=np.nan, ectopic_fraction=np.nan,
               label="Too few beats to summarise")
    if len(peaks) < 4:
        return out
    rr = np.diff(peaks) / fs
    mean_rr = float(rr.mean())
    out.update(hr=60.0 / mean_rr, mean_rr=mean_rr, sdnn=float(rr.std()),
               rmssd=float(np.sqrt(np.mean(np.diff(rr) ** 2))))
    out["irregularity"] = out["rmssd"] / mean_rr
    labels = list(labels)
    v_frac = np.mean([l == "V" for l in labels]) if labels else 0.0
    ect = np.mean([l in ("S", "V", "F") for l in labels]) if labels else 0.0
    out["ectopic_fraction"] = float(ect)
    if v_frac >= 0.3:
        out["label"] = "Frequent ventricular ectopy"
    elif out["irregularity"] > 0.15 and ect < 0.3:
        out["label"] = "Irregularly irregular RR pattern (AF-like)"
    elif ect >= 0.1:
        out["label"] = "Underlying rhythm with ectopic beats"
    elif out["hr"] < 60:
        out["label"] = "Regular, slow (bradycardia-like)"
    elif out["hr"] > 100:
        out["label"] = "Regular, fast (tachycardia-like)"
    else:
        out["label"] = "Regular, normal rate (sinus-like)"
    return out


def match_annotations(peaks, ann_samples, ann_symbols, tol: int = 36):
    """Map detected peaks to the nearest reference annotation (AAMI class) or None."""
    out = [None] * len(peaks)
    ann_samples = np.asarray(ann_samples)
    if len(ann_samples) == 0:
        return out
    idx = np.searchsorted(ann_samples, peaks)
    for i, p in enumerate(peaks):
        best, bd = None, tol + 1
        for j in (idx[i] - 1, idx[i]):
            if 0 <= j < len(ann_samples):
                d = abs(int(ann_samples[j]) - int(p))
                if d < bd:
                    best, bd = j, d
        if best is not None and bd <= tol:
            out[i] = AAMI_MAP.get(str(ann_symbols[best]))
    return out


# ----------------------------------------------------------------------------
# Data access (PhysioNet via wfdb)
# ----------------------------------------------------------------------------
def load_record(rec, data_dir="data/mitdb", pn_dir="mitdb"):
    """Download (if needed) and read one MIT-BIH record. Returns sig, fs, ann samples, symbols."""
    import wfdb
    rec = str(rec)
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    if not (d / f"{rec}.hea").exists():
        wfdb.dl_files(pn_dir, str(d), [f"{rec}.hea", f"{rec}.dat", f"{rec}.atr"])
    r = wfdb.rdrecord(str(d / rec))
    ann = wfdb.rdann(str(d / rec), "atr")
    names = [n.upper() for n in r.sig_name]
    ch = names.index("MLII") if "MLII" in names else 0
    return r.p_signal[:, ch], r.fs, np.asarray(ann.sample), list(ann.symbol)


def build_dataset(records, data_dir="data/mitdb", cache=None, verbose=True):
    """Beat matrix X (n,200), RR feats R (n,4), labels y (n,), record ids, R-peak positions."""
    if cache and Path(cache).exists():
        z = np.load(cache)
        return z["X"], z["R"], z["y"], z["rec"], z["pos"]
    Xs, Rs, ys, recs, poss = [], [], [], [], []
    for rec in records:
        sig, fs, samp, sym = load_record(rec, data_dir)
        x = normalize(preprocess(resample_to_fs(sig, fs), FS))
        mask = np.array([s in AAMI_MAP for s in sym])
        peaks = np.asarray(samp)[mask]
        labels = np.array([CLASS_IDX[AAMI_MAP[s]] for s, m in zip(sym, mask) if m])
        beats, rr, keep = extract_beats(x, peaks)
        Xs.append(beats); Rs.append(rr); ys.append(labels[keep])
        recs.append(np.full(keep.sum(), int(rec))); poss.append(peaks[keep])
        if verbose:
            print(f"  record {rec}: {keep.sum()} beats")
    X, R = np.concatenate(Xs), np.concatenate(Rs)
    y, rec, pos = np.concatenate(ys), np.concatenate(recs), np.concatenate(poss)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, R=R, y=y, rec=rec, pos=pos)
    return X, R, y, rec, pos


def fetch_physionet(db: str, record: str, start_s: float = 0, dur_s: float = 60):
    """Stream a slice of channel 0 straight from PhysioNet (needs internet)."""
    import wfdb
    fs = wfdb.rdheader(record, pn_dir=db).fs
    r = wfdb.rdrecord(record, pn_dir=db, sampfrom=int(start_s * fs),
                      sampto=int((start_s + dur_s) * fs), channels=[0])
    return r.p_signal[:, 0], fs


# ----------------------------------------------------------------------------
# Synthetic ECG (offline fallback demo / tests)
# ----------------------------------------------------------------------------
def synthetic_ecg(duration=60, fs=FS, hr=72, irregular=False, seed=0):
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    x = np.zeros(n)
    cur, times = 0.5, []
    while cur < duration - 0.5:
        times.append(cur)
        rr = 60 / hr * (1 + (rng.normal(0, 0.25) if irregular else rng.normal(0, 0.02)))
        cur += max(0.35, rr)

    def g(c, a, w):
        return a * np.exp(-0.5 * ((t - c) / w) ** 2)

    for bt in times:
        if not irregular:
            x += g(bt - 0.16, 0.12, 0.025)           # P wave (absent in AF-like demo)
        x += g(bt - 0.025, -0.12, 0.010) + g(bt, 1.0, 0.012) + g(bt + 0.028, -0.25, 0.012)
        x += g(bt + 0.25, 0.30, 0.05)                # T wave
    x += 0.05 * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.02, n)
    return x

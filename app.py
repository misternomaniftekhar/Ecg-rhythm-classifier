"""ECG Rhythm Classification - Streamlit prototype.

Load ECG -> waveform -> preprocessing -> selected segment -> predicted rhythm.
Research / educational use only. NOT a medical device; no clinical claims.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import signal as sps

import ecg_pipeline as ep

try:
    import models as ml
    TORCH_ERR = ""
except Exception as exc:                       # torch missing -> degrade gracefully
    ml, TORCH_ERR = None, str(exc)

ROOT = Path(__file__).parent
DEMO_DIR, MODEL_DIR, ART_DIR = ROOT / "demo_data", ROOT / "models", ROOT / "artifacts"
MAX_SECONDS = 600

st.set_page_config(page_title="ECG Rhythm Classifier", page_icon="🫀", layout="wide")


# ----------------------------------------------------------------------------
# Cached helpers
# ----------------------------------------------------------------------------
def available_models():
    if ml is None:
        return []
    return [k for k in ep.MODEL_TITLES if (MODEL_DIR / f"{k}.pt").exists()]


@st.cache_resource(show_spinner=False)
def get_model(key):
    return ml.load_model(key, MODEL_DIR / f"{key}.pt")


@st.cache_data(show_spinner=False)
def demo_index():
    p = DEMO_DIR / "index.json"
    return json.loads(p.read_text()) if p.exists() else []


@st.cache_data(show_spinner=False)
def load_demo(fname):
    z = np.load(DEMO_DIR / fname, allow_pickle=False)
    return dict(sig=z["signal"].astype(float), fs=float(z["fs"]),
                ann_samples=z["ann_samples"], ann_symbols=z["ann_symbols"])


@st.cache_data(show_spinner="Running preprocessing + beat classification...")
def analyse(sig, fs, lo, hi, notch_hz, model_key):
    raw = ep.resample_to_fs(sig, fs)
    filt = ep.normalize(ep.preprocess(raw, ep.FS, lo, hi, notch_hz))
    all_peaks = ep.detect_r_peaks(filt)
    beats, rr, keep = ep.extract_beats(filt, all_peaks)
    probs = None
    if model_key and len(beats):
        probs = ml.predict_proba(get_model(model_key), beats, rr)
    return dict(raw=ep.normalize(raw - np.median(raw)), filt=filt, all_peaks=all_peaks,
                peaks=all_peaks[keep], beats=beats, rr=rr, probs=probs)


@st.cache_data(show_spinner=False)
def load_metrics():
    p = ART_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def parse_upload(f):
    name = f.name.lower()
    if name.endswith(".npy"):
        arr = np.load(f, allow_pickle=False)
    else:
        arr = pd.read_csv(f, header=None, sep=None, engine="python").apply(pd.to_numeric, errors="coerce")
        arr = arr.dropna(axis=0, how="all").dropna(axis=1, how="all").to_numpy(dtype=float)
    arr = np.asarray(arr, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


# ----------------------------------------------------------------------------
# Sidebar: data source + parameters
# ----------------------------------------------------------------------------
st.sidebar.title("🫀 ECG Rhythm Classifier")
src = st.sidebar.radio("ECG source", ["Demo recordings", "Synthetic ECG", "Upload file", "PhysioNet (online)"])
record = None

if src == "Demo recordings":
    idx = demo_index()
    if idx:
        pick = st.sidebar.selectbox("Recording", idx, format_func=lambda d: d["title"])
        record = {**load_demo(pick["file"]), "title": pick["title"]}
    else:
        st.sidebar.warning("No demo_data/ found. Run `python prepare_demo.py` (or `train.py`), "
                           "or use the Synthetic ECG source.")
elif src == "Synthetic ECG":
    kind = st.sidebar.selectbox("Pattern", ["Regular rhythm", "Irregular (AF-like) rhythm"])
    hr = st.sidebar.slider("Heart rate (bpm)", 40, 150, 72)
    record = dict(sig=ep.synthetic_ecg(60, ep.FS, hr, irregular=kind.startswith("Irregular")),
                  fs=float(ep.FS), ann_samples=np.array([]), ann_symbols=np.array([]),
                  title=f"Synthetic ECG - {kind}")
elif src == "Upload file":
    up = st.sidebar.file_uploader("CSV / TXT / NPY (one value per sample)", type=["csv", "txt", "npy"])
    fs_in = st.sidebar.number_input("Sampling rate (Hz)", 50.0, 2000.0, 360.0, step=10.0)
    if up is not None:
        try:
            arr = parse_upload(up)
            col = st.sidebar.selectbox("Signal column", list(range(arr.shape[1]))) if arr.shape[1] > 1 else 0
            record = dict(sig=arr[:, col], fs=float(fs_in), ann_samples=np.array([]),
                          ann_symbols=np.array([]), title=f"Uploaded: {up.name}")
        except Exception as exc:
            st.sidebar.error(f"Could not read file: {exc}")
else:
    db = st.sidebar.selectbox("Database", ["mitdb", "afdb", "nsrdb"])
    rid = st.sidebar.text_input("Record", {"mitdb": "100", "afdb": "04015", "nsrdb": "16265"}[db])
    s0 = st.sidebar.number_input("Start (s)", 0, 3000, 0)
    dur = st.sidebar.number_input("Duration (s)", 10, 120, 60)
    if st.sidebar.button("Fetch from PhysioNet"):
        try:
            with st.spinner("Downloading..."):
                sig, fs = ep.fetch_physionet(db, rid, s0, dur)
            st.session_state["physionet"] = dict(sig=sig, fs=float(fs), ann_samples=np.array([]),
                                                 ann_symbols=np.array([]), title=f"PhysioNet {db}/{rid}")
        except Exception as exc:
            st.sidebar.error(f"Download failed: {exc}")
    record = st.session_state.get("physionet")

st.sidebar.markdown("---")
st.sidebar.subheader("Preprocessing")
lo = st.sidebar.slider("High-pass cut-off (Hz)", 0.1, 2.0, 0.5, 0.1)
hi = st.sidebar.slider("Low-pass cut-off (Hz)", 20, 100, 40, 5)
notch = st.sidebar.selectbox("Mains notch", ["Off", "50 Hz", "60 Hz"])
notch_hz = None if notch == "Off" else float(notch.split()[0])

models_avail = available_models()
st.sidebar.subheader("Classifier")
model_key = st.sidebar.selectbox("Model", models_avail, format_func=ep.MODEL_TITLES.get) if models_avail else None
thr = st.sidebar.slider("Confidence threshold", 0.0, 0.99, 0.6, 0.01,
                        help="Beats predicted below this confidence are flagged 'uncertain'.")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("ECG Rhythm Classification")
st.caption("Load ECG → waveform → preprocessing → selected segment → predicted rhythm")
st.warning("⚠️ Research / educational prototype only. This is not a medical device and must not be used "
           "for diagnosis or any clinical decision.")

if record is None:
    st.info("Choose an ECG source in the sidebar to begin.")
    st.stop()

sig, fs = record["sig"][: int(MAX_SECONDS * record["fs"])], record["fs"]
res = analyse(sig, fs, lo, hi, notch_hz, model_key)
duration = len(res["filt"]) / ep.FS
t_all = np.arange(len(res["filt"])) / ep.FS

c1, c2, c3 = st.columns(3)
c1.metric("Recording", record["title"][:40])
c2.metric("Duration", f"{duration:.1f} s @ {fs:.0f} Hz")
c3.metric("R peaks detected", len(res["all_peaks"]))

tab_wave, tab_pre, tab_seg, tab_eval, tab_about = st.tabs(
    ["1 · Waveform", "2 · Preprocessing", "3 · Segment & prediction", "4 · Evaluation", "About & ethics"])

# ---- 1. Waveform ----------------------------------------------------------
with tab_wave:
    fig = go.Figure(go.Scattergl(x=t_all, y=res["raw"], mode="lines", line=dict(width=1), name="raw"))
    fig.update_layout(title="Raw ECG (median-centred, z-scored)", xaxis_title="Time (s)",
                      yaxis_title="Amplitude (a.u.)", height=340, margin=dict(t=40, b=10))
    st.plotly_chart(fig)

# ---- 2. Preprocessing -----------------------------------------------------
with tab_pre:
    w = min(10.0, duration)
    m = t_all <= w
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_all[m], y=res["raw"][m], name="raw", line=dict(width=1, color="#9ca3af")))
    fig.add_trace(go.Scatter(x=t_all[m], y=res["filt"][m], name="filtered", line=dict(width=1.5, color="#2563eb")))
    pk = res["all_peaks"][res["all_peaks"] < w * ep.FS]
    fig.add_trace(go.Scatter(x=pk / ep.FS, y=res["filt"][pk], mode="markers", name="R peaks",
                             marker=dict(color="#dc2626", size=8, symbol="triangle-down")))
    fig.update_layout(title=f"First {w:.0f} s: raw vs. band-pass filtered ({lo}-{hi} Hz)",
                      xaxis_title="Time (s)", height=340, margin=dict(t=40, b=10))
    st.plotly_chart(fig)
    if len(res["raw"]) > 512:
        f1_, p1 = sps.welch(res["raw"], ep.FS, nperseg=min(1024, len(res["raw"])))
        f2_, p2 = sps.welch(res["filt"], ep.FS, nperseg=min(1024, len(res["filt"])))
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=f1_, y=p1, name="raw", line=dict(color="#9ca3af")))
        fig2.add_trace(go.Scatter(x=f2_, y=p2, name="filtered", line=dict(color="#2563eb")))
        fig2.update_layout(title="Power spectral density (Welch)", xaxis_title="Frequency (Hz)",
                           yaxis_title="PSD", yaxis_type="log", xaxis_range=[0, 100], height=300,
                           margin=dict(t=40, b=10))
        st.plotly_chart(fig2)
    st.caption("Pipeline: median removal → zero-phase Butterworth band-pass → optional notch → z-score → "
               "Pan-Tompkins-style R-peak detection → 200-sample beat windows + RR-interval features.")

# ---- 3. Segment & prediction ------------------------------------------------
with tab_seg:
    cA, cB = st.columns(2)
    win = cA.slider("Segment length (s)", 3, int(max(4, min(30, duration))), int(min(10, duration)))
    start = cB.slider("Segment start (s)", 0.0, float(max(duration - win, 0.0)), 0.0, 1.0)
    a, b = int(start * ep.FS), int((start + win) * ep.FS)

    peaks, probs = res["peaks"], res["probs"]
    in_seg = (peaks >= a) & (peaks < b)
    seg_peaks = peaks[in_seg]
    all_in_seg = res["all_peaks"][(res["all_peaks"] >= a) & (res["all_peaks"] < b)]

    pred = conf = None
    if probs is not None and len(probs):
        pred = np.array([ep.CLASSES[i] for i in probs.argmax(1)])
        conf = probs.max(1)
    seg_pred = pred[in_seg] if pred is not None else None
    seg_conf = conf[in_seg] if conf is not None else None
    truth = ep.match_annotations(seg_peaks, record["ann_samples"], record["ann_symbols"])

    fig = go.Figure(go.Scatter(x=t_all[a:b], y=res["filt"][a:b], mode="lines",
                               line=dict(width=1.5, color="#374151"), name="ECG"))
    if seg_pred is None:
        fig.add_trace(go.Scatter(x=seg_peaks / ep.FS, y=res["filt"][seg_peaks], mode="markers",
                                 name="R peaks", marker=dict(color="#dc2626", size=9)))
    else:
        certain = seg_conf >= thr
        for c in ep.CLASSES:
            mk = (seg_pred == c) & certain
            if mk.any():
                fig.add_trace(go.Scatter(x=seg_peaks[mk] / ep.FS, y=res["filt"][seg_peaks[mk]], mode="markers",
                                         name=ep.CLASS_LABELS[c], marker=dict(size=11, color=ep.CLASS_COLORS[c])))
        if (~certain).any():
            mk = ~certain
            fig.add_trace(go.Scatter(x=seg_peaks[mk] / ep.FS, y=res["filt"][seg_peaks[mk]], mode="markers",
                                     name=f"Uncertain (<{thr:.2f})",
                                     marker=dict(size=11, color="#9ca3af", symbol="x")))
    fig.update_layout(title="Selected segment with beat-level predictions", xaxis_title="Time (s)",
                      height=380, margin=dict(t=40, b=10), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig)

    summ = ep.rhythm_summary(all_in_seg, list(seg_pred) if seg_pred is not None else [])
    st.subheader("Predicted rhythm for this segment")
    st.success(f"**{summ['label']}**")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Beats", summ["n_beats"])
    k2.metric("Heart rate", "–" if np.isnan(summ["hr"]) else f"{summ['hr']:.0f} bpm")
    k3.metric("SDNN", "–" if np.isnan(summ["sdnn"]) else f"{summ['sdnn'] * 1000:.0f} ms")
    k4.metric("RMSSD", "–" if np.isnan(summ["rmssd"]) else f"{summ['rmssd'] * 1000:.0f} ms")
    k5.metric("Ectopic fraction", "–" if np.isnan(summ["ectopic_fraction"]) else f"{summ['ectopic_fraction']:.0%}")
    st.caption("Segment label = interpretable rules on RR statistics + AI beat classes "
               "(AF-like = high RR irregularity with few ectopic beats). It is a screening-style summary, "
               "not a diagnosis.")

    if seg_pred is None:
        st.info("No trained model found in `models/` - showing signal processing only. "
                "Train models with `python train.py` (see the Evaluation tab).")
    else:
        d1, d2 = st.columns([1, 2])
        counts = pd.Series(seg_pred).value_counts().reindex(ep.CLASSES, fill_value=0)
        d1.plotly_chart(px.bar(x=counts.index, y=counts.values, color=counts.index,
                               color_discrete_map=ep.CLASS_COLORS,
                               labels=dict(x="Class", y="Beats"), title="Beat classes").update_layout(
            showlegend=False, height=300, margin=dict(t=40, b=10)))
        table = pd.DataFrame(dict(time_s=(seg_peaks / ep.FS).round(2), predicted=seg_pred,
                                  confidence=seg_conf.round(3), uncertain=seg_conf < thr,
                                  reference=[t or "–" for t in truth]))
        d2.dataframe(table, height=300)
        ref = [(p, t) for p, t in zip(seg_pred, truth) if t]
        if ref:
            st.caption(f"Agreement with reference annotations in this segment: "
                       f"{np.mean([p == t for p, t in ref]):.0%} of {len(ref)} matched beats.")

        st.subheader("Inspect a single beat")
        if len(seg_peaks):
            i = st.selectbox("Beat #", range(len(seg_peaks)), format_func=lambda j: f"{j + 1} @ {seg_peaks[j] / ep.FS:.2f} s")
            gi = int(np.where(in_seg)[0][i])
            e1, e2 = st.columns(2)
            tb = (np.arange(ep.BEAT_LEN) - ep.PRE) / ep.FS * 1000
            e1.plotly_chart(go.Figure(go.Scatter(x=tb, y=res["beats"][gi], line=dict(color="#2563eb"))).update_layout(
                title="Beat window", xaxis_title="ms from R peak", height=280, margin=dict(t=40, b=10)))
            e2.plotly_chart(px.bar(x=ep.CLASSES, y=probs[gi], color=ep.CLASSES, color_discrete_map=ep.CLASS_COLORS,
                                   labels=dict(x="Class", y="Probability"), title="Class probabilities").update_layout(
                showlegend=False, yaxis_range=[0, 1], height=280, margin=dict(t=40, b=10)))

# ---- 4. Evaluation ----------------------------------------------------------
with tab_eval:
    M = load_metrics()
    if M is None:
        st.info("No `artifacts/metrics.json` yet. Train locally / in Colab with `python train.py`, commit the "
                "`models/` and `artifacts/` folders, and redeploy.")
        if ml is None:
            st.error(f"PyTorch could not be imported: {TORCH_ERR}")
        elif st.button("Run quick demo training here (few minutes, small subset)"):
            with st.spinner("Downloading MIT-BIH records and training..."):
                proc = subprocess.run([sys.executable, "train.py", "--quick"], capture_output=True,
                                      text=True, cwd=ROOT)
            st.code((proc.stdout[-2500:] + proc.stderr[-1500:]) or "(no output)")
            st.cache_data.clear(); st.cache_resource.clear()
            st.success("Finished - reload the page to use the new models." if proc.returncode == 0
                       else "Training failed - see log above.")
    else:
        meta = M["meta"]
        st.caption(f"Protocol: **{meta['protocol']}** · run: {meta['run']} · epochs: {meta['epochs']} · "
                   f"train {meta['n_train']:,} / val {meta['n_val']:,} / test {meta['n_test']:,} beats")
        summary = pd.DataFrame([dict(Model=m["title"], Accuracy=m["accuracy"], MacroF1=m["macro_f1"],
                                     WeightedF1=m["weighted_f1"], TrainSeconds=m["train_seconds"])
                                for m in M["models"].values()])
        st.subheader("Model comparison (held-out DS2 patients)")
        st.dataframe(summary.style.format({"Accuracy": "{:.3f}", "MacroF1": "{:.3f}", "WeightedF1": "{:.3f}"}))

        sel = st.selectbox("Inspect model", list(M["models"]), format_func=ep.MODEL_TITLES.get, key="evalsel")
        m = M["models"][sel]
        l, r = st.columns(2)
        pc = pd.DataFrame(m["per_class"]).T.rename(columns=dict(recall="sensitivity"))
        l.markdown("**Per-class metrics**")
        l.dataframe(pc.style.format({"precision": "{:.3f}", "sensitivity": "{:.3f}", "f1": "{:.3f}",
                                     "specificity": "{:.3f}", "support": "{:.0f}"}))
        cm = np.array(m["confusion_matrix"])
        norm = st.toggle("Row-normalise confusion matrix", True)
        cmv = cm / np.maximum(cm.sum(1, keepdims=True), 1) if norm else cm
        r.plotly_chart(px.imshow(cmv, x=ep.CLASSES, y=ep.CLASSES, text_auto=".2f" if norm else "d",
                                 color_continuous_scale="Blues", labels=dict(x="Predicted", y="True"),
                                 title="Confusion matrix").update_layout(height=380, margin=dict(t=40, b=10)))

        g1, g2 = st.columns(2)
        rob = pd.DataFrame(m["robustness"]).T.reset_index().rename(columns=dict(index="Noise"))
        g1.plotly_chart(px.bar(rob, x="Noise", y="macro_f1", title="Robustness: macro-F1 vs. added noise (SNR)",
                               range_y=[0, 1]).update_layout(height=300, margin=dict(t=40, b=10)))
        unc = pd.DataFrame(m["uncertainty"]).dropna()
        fu = go.Figure()
        fu.add_trace(go.Scatter(x=unc["threshold"], y=unc["accuracy"], name="Accuracy (kept beats)"))
        fu.add_trace(go.Scatter(x=unc["threshold"], y=unc["coverage"], name="Coverage"))
        fu.update_layout(title="Uncertainty: confidence threshold trade-off", xaxis_title="Threshold",
                         height=300, margin=dict(t=40, b=10))
        g2.plotly_chart(fu)

        st.subheader("Error analysis")
        ep_csv = ART_DIR / f"errors_{sel}.csv"
        if ep_csv.exists():
            err = pd.read_csv(ep_csv)
            st.write(f"{len(err):,} misclassified test beats.")
            pairs = err.groupby(["true", "pred"]).size().reset_index(name="count").sort_values("count", ascending=False)
            e1, e2 = st.columns(2)
            e1.markdown("**Most common confusions**"); e1.dataframe(pairs.head(10))
            e2.markdown("**Confident mistakes (highest confidence first)**")
            e2.dataframe(err.sort_values("confidence", ascending=False).head(15))
            by_rec = err.groupby("record").size().sort_values(ascending=False).head(8)
            st.markdown("**Records contributing most errors:** " + ", ".join(f"{k} ({v})" for k, v in by_rec.items()))
        with st.expander("Training curves"):
            h = pd.DataFrame(m["history"])
            st.plotly_chart(px.line(h, x="epoch", y=["train_loss", "val_macro_f1"]))

# ---- About ------------------------------------------------------------------
with tab_about:
    st.markdown("""
### What this prototype does
1. **Load** an ECG (demo, synthetic, upload, or PhysioNet) and resample to 360 Hz.
2. **Preprocess**: baseline/noise removal with a zero-phase band-pass (+ optional mains notch).
3. **Segment**: Pan-Tompkins-style R-peak detection, 200-sample beat windows, RR-interval features.
4. **Classify** each beat into AAMI classes **N, S, V, F, Q** with a 1D-CNN, BiLSTM or Transformer, then
   summarise the selected segment's rhythm with transparent RR-statistics rules.

### Data
MIT-BIH Arrhythmia, MIT-BIH Atrial Fibrillation and MIT-BIH Normal Sinus Rhythm databases from
[PhysioNet](https://physionet.org) (Goldberger et al., 2000; Moody & Mark, 2001). Models are trained with an
**inter-patient** split (DS1 train / DS2 test) so metrics are not inflated by patient leakage.

### Responsible AI
* Research and education only - **no clinical decision-making claims**.
* Classes are heavily imbalanced (S, F, Q are rare); look at per-class recall, not accuracy alone.
* Performance can drop on other devices, leads, sampling rates, noise levels and populations.
* Use the confidence threshold to flag uncertain beats instead of trusting every prediction.
""")

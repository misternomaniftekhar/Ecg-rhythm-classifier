# 🫀 ECG Rhythm Classification (Biomedical Signal Processing)

Prototype pipeline: **Load ECG → waveform → preprocessing → selected segment → predicted rhythm**.
Beat-level classification (AAMI classes N, S, V, F, Q) with a **1D-CNN, BiLSTM and Transformer**, plus an interpretable
segment-level rhythm summary (RR statistics: rate, SDNN, RMSSD, AF-like irregularity).

> ⚠️ **Research / educational use only. Not a medical device. No clinical decision-making claims.**

## Features
- Waveform viewer (raw), preprocessing view (band-pass, optional 50/60 Hz notch, PSD before/after)
- Pan-Tompkins-style R-peak detection, 200-sample beat windows, RR-interval features
- Choose a segment (3-30 s) → beat-level colour-coded predictions, uncertain-beat flagging (confidence threshold)
- Single-beat inspector with class probabilities
- Evaluation tab: per-class precision / recall(sensitivity) / specificity / F1, confusion matrix, model comparison,
  robustness to noise (SNR), confidence-threshold trade-off, error analysis
- Sources: bundled demo recordings, synthetic ECG, file upload (CSV/TXT/NPY), live PhysioNet download

## Project structure
```
app.py             Streamlit app
ecg_pipeline.py    filtering, R-peak detection, beat extraction, rhythm rules, data loading
models.py          1D-CNN / BiLSTM / Transformer (PyTorch)
train.py           trains all models, writes models/ + artifacts/ (metrics, errors) + demo_data/
prepare_demo.py    creates small offline demo recordings from PhysioNet
requirements.txt
```
After training you will also have `models/*.pt`, `artifacts/metrics.json`, `artifacts/errors_*.csv`, `demo_data/`.

## 1. Train (Google Colab with GPU is easiest)
```bash
pip install -r requirements.txt
python train.py              # full: DS1 train / DS2 test, 15 epochs, 3 models
python train.py --quick      # fast sanity check
```
Protocol: **inter-patient split** (DS1 train, DS2 test; paced records excluded) so there is no patient leakage.
Class imbalance handled with sqrt-inverse-frequency loss weights. Commit the generated `models/`, `artifacts/` and
`demo_data/` folders (all small).

## 2. Run locally
```bash
streamlit run app.py
```

## 3. Push to GitHub
```bash
git init && git add . && git commit -m "ECG rhythm classifier"
git branch -M main
git remote add origin https://github.com/<your-username>/ecg-rhythm-classifier.git
git push -u origin main
```

## 4. Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io → **New app** → select the repo, branch `main`, main file `app.py`.
2. Advanced settings → Python **3.11** (or 3.12).
3. Deploy. `requirements.txt` installs the CPU build of PyTorch.

If `models/` is missing the app still runs (signal processing + rhythm rules) and offers a quick in-app training demo.

## Data & references
- MIT-BIH Arrhythmia, MIT-BIH Atrial Fibrillation and MIT-BIH Normal Sinus Rhythm databases, PhysioNet.
- Goldberger et al., *PhysioBank, PhysioToolkit, and PhysioNet*, Circulation 101(23), 2000.
- Moody & Mark, *The impact of the MIT-BIH Arrhythmia Database*, IEEE EMB Magazine 20(3), 2001.
- de Chazal et al., *Automatic classification of heartbeats using ECG morphology and heartbeat interval features*, IEEE TBME 2004 (DS1/DS2 split).

## Limitations
Rare classes (S, F, Q) are hard; models trained on MIT-BIH lead II may not transfer to other devices, leads or
sampling rates. The AF-like label is a heuristic on RR irregularity, not a trained AF detector.

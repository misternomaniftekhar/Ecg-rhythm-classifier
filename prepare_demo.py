"""prepare_demo.py - cut short demo snippets from PhysioNet so the app works offline.

Usage:  python prepare_demo.py
Creates demo_data/*.npz and demo_data/index.json (small enough to commit to GitHub).
Data: PhysioNet (ODC-By licence) - MIT-BIH Arrhythmia, AF and Normal Sinus Rhythm DBs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import ecg_pipeline as ep

DEMO_MITDB = [
    ("100", "Mostly normal beats, a few ectopic beats"),
    ("119", "Frequent ventricular ectopy"),
    ("200", "Frequent ventricular ectopy"),
    ("207", "Complex arrhythmias"),
    ("208", "Ventricular ectopy and fusion beats"),
    ("210", "Mixed ectopy, irregular rhythm"),
    ("221", "Irregular rhythm with ventricular ectopy"),
    ("233", "Ventricular ectopy"),
]


def prepare_demo(data_dir="data", out_dir="demo_data", seconds=60):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = []

    for rec, note in DEMO_MITDB:
        try:
            sig, fs, samp, sym = ep.load_record(rec, f"{data_dir}/mitdb")
            n = int(seconds * fs)
            m = samp < n
            fname = f"mitdb_{rec}.npz"
            np.savez_compressed(out / fname, signal=sig[:n].astype(np.float32), fs=fs,
                                ann_samples=samp[m], ann_symbols=np.array(sym, dtype="<U4")[m])
            index.append(dict(file=fname, title=f"MIT-BIH Arrhythmia {rec} - {note}",
                              source="MIT-BIH Arrhythmia Database", fs=fs))
            print("demo:", fname)
        except Exception as exc:                       # keep going if one record fails
            print(f"skip mitdb {rec}: {exc}")

    extra = [("afdb", "04015", "MIT-BIH Atrial Fibrillation 04015 (AF episode)", "AF"),
             ("nsrdb", "16265", "MIT-BIH Normal Sinus Rhythm 16265", "NSR")]
    for db, rec, title, kind in extra:
        try:
            import wfdb
            fs = wfdb.rdheader(rec, pn_dir=db).fs
            start = int(300 * fs)
            if kind == "AF":
                ann = wfdb.rdann(rec, "atr", pn_dir=db)
                af = [s for s, a in zip(ann.sample, ann.aux_note) if a.startswith("(AFIB")]
                start = int(af[0]) if af else start
            r = wfdb.rdrecord(rec, pn_dir=db, sampfrom=start,
                              sampto=start + int(seconds * fs), channels=[0])
            fname = f"{db}_{rec}.npz"
            np.savez_compressed(out / fname, signal=r.p_signal[:, 0].astype(np.float32), fs=fs,
                                ann_samples=np.array([], dtype=int),
                                ann_symbols=np.array([], dtype="<U4"))
            index.append(dict(file=fname, title=title, source=db, fs=fs))
            print("demo:", fname)
        except Exception as exc:
            print(f"skip {db} {rec}: {exc}")

    (out / "index.json").write_text(json.dumps(index, indent=2))
    return index


if __name__ == "__main__":
    prepare_demo()

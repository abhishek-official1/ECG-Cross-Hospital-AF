"""Read-only adapters for the exported ECG experiments."""
from pathlib import Path
import importlib.util
import json
import sys
import threading
import numpy as np
import pandas as pd

SECTIONS = ("Study overview", "ECG prediction", "Cross-domain performance", "Model comparison", "Calibration & uncertainty", "Robustness")
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
_IMPORT_LOCK = threading.RLock()


def discover_runs(root):
    return sorted((p for p in Path(root).glob("ecg_results_*") if (p/"tables/all_metrics.csv").is_file() and (p/"bundle/bundle.json").is_file()), reverse=True)


def metadata(run):
    return json.loads((Path(run)/"bundle/bundle.json").read_text(encoding="utf-8"))


def table(run, name):
    p = Path(run)/"tables"/name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def operating_metrics(run):
    frame = table(run, "all_metrics_operating_points.csv")
    if frame.empty:
        frame = table(run, "all_metrics.csv")
        frame["operating_point"] = "youden"
    return frame


def predictions(run, tag, split):
    if tag not in set(table(run,"all_metrics.csv").tag) or split not in ("external","source_test"):
        raise ValueError("Unknown model or evaluation split")
    return pd.read_csv(Path(run)/"predictions"/f"{tag}_{split}.csv")


def confusion(frame, threshold):
    y=frame.label.to_numpy(dtype=int); pred=frame.p_af.to_numpy() >= threshold
    return [[int(((y==0)&~pred).sum()),int(((y==0)&pred).sum())],
            [int(((y==1)&~pred).sum()),int(((y==1)&pred).sum())]]


def calibration_bins(frame, count=15):
    p=frame.p_af.to_numpy(); y=frame.label.to_numpy()
    bins=np.minimum((np.clip(p,0,1)*count).astype(int),count-1)
    rows=[]
    for b in range(count):
        mask=bins==b
        if mask.any(): rows.append(dict(mean_probability=float(p[mask].mean()),observed_af=float(y[mask].mean()),n=int(mask.sum())))
    return pd.DataFrame(rows)


def validate_ecg(array, fs):
    x=np.asarray(array)
    if x.dtype.kind not in "fiu": raise ValueError("Upload a real numerical ECG array; objects and complex values are not supported.")
    if fs not in (100,500): raise ValueError("Choose 100 or 500 Hz.")
    if x.shape != (12,fs*10): raise ValueError(f"Expected shape (12, {fs*10}) for 10 seconds at {fs} Hz; received {x.shape}.")
    x=np.asarray(x,dtype=np.float64)
    if not np.isfinite(x).all(): raise ValueError("ECG contains nonfinite values.")
    if np.any(np.ptp(x,axis=1)<=1e-5): raise ValueError("Each lead must contain a nonflat signal.")
    if np.max(np.abs(x))>=100: raise ValueError("Implausible amplitude: convert the signal to millivolts (mV).")
    return x


def read_upload(upload,fs):
    upload.seek(0,2); size=upload.tell();upload.seek(0)
    if size>2_000_000: raise ValueError("File exceeds the 2 MB ECG upload limit.")
    # Check the header before allocation, including malicious shape declarations.
    version=np.lib.format.read_magic(upload)
    if version==(1,0): shape,_,dtype=np.lib.format.read_array_header_1_0(upload,max_header_size=10000)
    elif version==(2,0): shape,_,dtype=np.lib.format.read_array_header_2_0(upload,max_header_size=10000)
    else: raise ValueError("Use a NumPy .npy format version 1 or 2 file.")
    if shape!=(12,fs*10) or dtype.kind not in "fiu" or dtype.itemsize>8:
        raise ValueError(f"Expected a real numerical array of shape (12, {fs*10}).")
    upload.seek(0)
    return validate_ecg(np.load(upload,allow_pickle=False,max_header_size=10000),fs)


def load_predictor(run):
    folder=(Path(run)/"bundle").resolve()
    # Existing exports import a top-level `model`. Isolate each run's module.
    with _IMPORT_LOCK:
        old=sys.modules.get("model")
        try:
            spec=importlib.util.spec_from_file_location("model",folder/"model.py")
            model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
            sys.modules["model"]=model
            spec=importlib.util.spec_from_file_location("ecg_export_inference",folder/"inference.py")
            inference=importlib.util.module_from_spec(spec);spec.loader.exec_module(inference)
            return inference.Predictor(folder)
        finally:
            if old is None: sys.modules.pop("model",None)
            else: sys.modules["model"]=old


def stress_comparison(run):
    """The notebook stresses the first selected member, not the ensemble."""
    tag=metadata(run)["members"][0]["tag"]
    raw=table(run,"all_metrics.csv")
    clean=raw[(raw.tag==tag)&(raw.split=="external")].copy()
    clean["condition"]="clean"
    stress=table(run,"noise_stress.csv")
    stress["tag"]=tag
    columns=["tag","condition","auroc","auprc","sensitivity","specificity","f1","accuracy","brier","ece"]
    return pd.concat([clean,stress],ignore_index=True)[columns]

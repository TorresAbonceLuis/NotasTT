import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import pywt
from sklearn.preprocessing import StandardScaler

FS_DEFAULT = 44100
FRAME_SEC = 0.050
HOP_SEC   = 0.025
WAVELET   = "db8"
LEVEL     = 9
MIDI_MIN, MIDI_MAX = 21, 108  # A0..C8

def note_name(m):
    names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    return f"{names[m%12]}{(m//12)-1}"

def piano_key_centers():
    mids = np.arange(MIDI_MIN, MIDI_MAX+1)
    freqs = 440.0 * (2.0 ** ((mids - 69) / 12.0))
    return mids, freqs

def piano_key_edges():
    mids, centers = piano_key_centers()
    edges = np.sqrt(centers[:-1] * centers[1:])
    low  = centers[0] / np.sqrt(2**(1/12))
    high = centers[-1] * np.sqrt(2**(1/12))
    return np.concatenate([[low], edges, [high]])

KEY_EDGES = piano_key_edges()

def wpt_band_energies(x, fs, wavelet=WAVELET, level=LEVEL):
    wp = pywt.WaveletPacket(data=x, wavelet=wavelet, mode='symmetric', maxlevel=level)
    nodes = [n.path for n in wp.get_level(level, order='freq')]
    energies = np.array([float(np.sum(wp[n].data.astype(np.float64)**2)) for n in nodes], dtype=np.float64)
    B = 2 ** level
    band_bw = (fs / 2.0) / B
    idxs = np.arange(B)
    f_low  = idxs * band_bw
    f_high = (idxs + 1) * band_bw
    return energies, f_low, f_high

def aggregate_to_88(energies, f_low, f_high):
    feat = np.zeros(88, dtype=np.float64)
    for b in range(len(energies)):
        bl, bh = f_low[b], f_high[b]
        if bh <= KEY_EDGES[0] or bl >= KEY_EDGES[-1]:
            continue
        i_start = np.searchsorted(KEY_EDGES, bl, side='right') - 1
        i_end   = np.searchsorted(KEY_EDGES, bh, side='left')
        i_start = max(i_start, 0)
        i_end   = min(i_end, 88)
        if i_end <= i_start:
            continue
        for i in range(i_start, i_end):
            kl, kh = KEY_EDGES[i], KEY_EDGES[i+1]
            overlap = max(0.0, min(bh, kh) - max(bl, kl))
            if overlap > 0:
                feat[i] += energies[b] * (overlap / (bh - bl))
    return feat

def frame_audio(x, sr, frame_sec=FRAME_SEC, hop_sec=HOP_SEC):
    N = int(round(frame_sec * sr))
    H = int(round(hop_sec * sr))
    if len(x) < N: return np.empty((0, N), dtype=np.float32)
    n_frames = 1 + (len(x) - N) // H
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, N), strides=(x.strides[0]*H, x.strides[0]), writeable=False
    )
    return frames.copy()

def process_file(wav_path, midi_label, fs_expected=FS_DEFAULT):
    x, sr = sf.read(str(wav_path), dtype='float32', always_2d=False)
    if x.ndim > 1: x = x.mean(axis=1)
    if sr != fs_expected:
        raise ValueError(f"SR {sr} != esperado {fs_expected} en {wav_path}")
    frames = frame_audio(x, sr)
    if frames.shape[0] == 0:
        return np.empty((0,88), dtype=np.float32), np.empty((0,), dtype=np.int64)
    feats = []
    for f in frames:
        e, fl, fh = wpt_band_energies(f, sr)
        v = aggregate_to_88(e, fl, fh)
        v = np.log10(v + 1e-12).astype(np.float32)  # compresión log
        feats.append(v)
    X = np.vstack(feats)
    y = np.full((X.shape[0],), int(midi_label - MIDI_MIN), dtype=np.int64)
    return X, y

def load_split(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = df[df["filepath"].apply(lambda p: Path(p).suffix.lower()==".wav" and Path(p).exists())].copy()
    assert "midi" in df.columns and "filepath" in df.columns
    return df

def build_features(df, fs_expected, scaler=None, fit_scaler=False):
    X_all, y_all = [], []
    for _, r in df.iterrows():
        X, y = process_file(Path(r["filepath"]), int(r["midi"]), fs_expected)
        if X.size == 0: continue
        X_all.append(X); y_all.append(y)
    if not X_all:
        return np.empty((0,88), dtype=np.float32), np.empty((0,), dtype=np.int64), scaler
    X_all = np.vstack(X_all); y_all = np.concatenate(y_all)
    if fit_scaler:
        scaler = StandardScaler()
        scaler.fit(X_all)
    if scaler is not None:
        X_all = scaler.transform(X_all).astype(np.float32)
    return X_all, y_all, scaler

def main():
    global WAVELET, LEVEL, FRAME_SEC, HOP_SEC
    
    ap = argparse.ArgumentParser(description="Extracción WPT → 88 bins (A0..C8)")
    ap.add_argument("--splits_dir", type=Path, default=Path("splits"))
    ap.add_argument("--out_dir",    type=Path, default=Path("features_wpt"))
    ap.add_argument("--fs", type=int, default=FS_DEFAULT)
    ap.add_argument("--wavelet", type=str, default=WAVELET)
    ap.add_argument("--level", type=int, default=LEVEL)
    ap.add_argument("--frame", type=float, default=FRAME_SEC)
    ap.add_argument("--hop",   type=float, default=HOP_SEC)
    args = ap.parse_args()
    WAVELET = args.wavelet; LEVEL = args.level
    FRAME_SEC = args.frame; HOP_SEC = args.hop

    train_df = load_split(args.splits_dir / "train.csv")
    valid_df = load_split(args.splits_dir / "valid.csv")
    test_df  = load_split(args.splits_dir / "test.csv")

    X_train, y_train, scaler = build_features(train_df, args.fs, scaler=None, fit_scaler=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "X_train.npy", X_train)
    np.save(args.out_dir / "y_train.npy", y_train)
    with open(args.out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    X_valid, y_valid, _ = build_features(valid_df, args.fs, scaler=scaler, fit_scaler=False)
    X_test,  y_test,  _ = build_features(test_df,  args.fs, scaler=scaler, fit_scaler=False)
    np.save(args.out_dir / "X_valid.npy", X_valid); np.save(args.out_dir / "y_valid.npy", y_valid)
    np.save(args.out_dir / "X_test.npy",  X_test);  np.save(args.out_dir / "y_test.npy",  y_test)

    mids, _ = piano_key_centers()
    meta = {
        "fs": args.fs, "frame_sec": FRAME_SEC, "hop_sec": HOP_SEC,
        "wavelet": WAVELET, "level": LEVEL,
        "keys_midi": mids.tolist(),
        "keys_name": [note_name(m) for m in mids],
        "key_edges_hz": KEY_EDGES.tolist(),
        "X_shapes": {"train": list(X_train.shape), "valid": list(X_valid.shape), "test": list(X_test.shape)}
    }
    with open(args.out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("✅ Features listas en", args.out_dir.resolve())
    print("   X_train:", X_train.shape, "X_valid:", X_valid.shape, "X_test:", X_test.shape)

if __name__ == "__main__":
    main()
